"""NPU Manager v2.0 for NEMWAS Framework - OpenVINO Optimized for Debian Linux"""

import os
import subprocess
import logging
import psutil
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque

import openvino as ov
import openvino.properties as props
import openvino.properties.intel_gpu as intel_gpu
import openvino.properties.intel_cpu as intel_cpu

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types in OpenVINO."""
    CPU = "CPU"
    GPU = "GPU"
    GPU_0 = "GPU.0"
    GPU_1 = "GPU.1"
    NPU = "NPU"
    AUTO = "AUTO"
    MULTI = "MULTI"
    HETERO = "HETERO"
    BATCH = "BATCH"


@dataclass
class DeviceCapabilities:
    """Device capability information."""
    device_name: str
    full_name: str
    device_type: str
    available_memory: int  # in bytes
    max_batch_size: int
    supported_properties: List[str]
    optimal_precision: str  # FP32, FP16, INT8, INT4
    thermal_status: str
    compute_units: int
    frequency_mhz: int


class CoreAffinityManager:
    """Manages CPU core affinity for P-cores vs E-cores on Intel hybrid architectures."""

    def __init__(self):
        self.p_cores = []
        self.e_cores = []
        self._detect_core_types()

    def _detect_core_types(self):
        """Detect P-cores and E-cores on Intel hybrid CPUs."""
        try:
            # Read CPU topology
            cpu_count = psutil.cpu_count(logical=True)

            # Try to detect core types via cpuinfo
            result = subprocess.run(['lscpu', '-p=CPU,CORE,MAXMHZ'],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                core_frequencies = {}
                for line in result.stdout.strip().split('\n'):
                    if line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 3:
                        cpu_id = int(parts[0])
                        max_freq = int(parts[2]) if parts[2] else 0
                        core_frequencies[cpu_id] = max_freq

                # P-cores typically have higher frequency
                if core_frequencies:
                    avg_freq = sum(core_frequencies.values()) / len(core_frequencies)
                    self.p_cores = [cpu for cpu, freq in core_frequencies.items()
                                   if freq > avg_freq * 1.1]
                    self.e_cores = [cpu for cpu, freq in core_frequencies.items()
                                   if freq <= avg_freq * 1.1]

                    logger.info(f"Detected {len(self.p_cores)} P-cores: {self.p_cores}")
                    logger.info(f"Detected {len(self.e_cores)} E-cores: {self.e_cores}")

        except Exception as e:
            logger.warning(f"Could not detect P/E cores: {e}")
            # Fallback: assume first half are P-cores
            cpu_count = psutil.cpu_count(logical=False)
            self.p_cores = list(range(cpu_count // 2))
            self.e_cores = list(range(cpu_count // 2, cpu_count))

    def get_optimal_cores(self, workload_type: str) -> List[int]:
        """Get optimal cores for workload type."""
        if workload_type == "latency":
            return self.p_cores[:4] if self.p_cores else [0, 1, 2, 3]
        elif workload_type == "throughput":
            return self.e_cores if self.e_cores else list(range(4, 8))
        else:
            return self.p_cores[:2] + self.e_cores[:2]


class NPUManager:
    """Advanced NPU Manager with real device integration and monitoring."""

    def __init__(self, cache_dir: str = "./models/cache", enable_profiling: bool = True):
        self.core = ov.Core()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Core affinity manager
        self.affinity_manager = CoreAffinityManager()

        # Device discovery
        self.available_devices = self._discover_devices()
        self.device_capabilities = self._probe_device_capabilities()

        # Performance monitoring
        self.enable_profiling = enable_profiling
        self.performance_history = deque(maxlen=100)
        self._monitoring_thread = None

        if enable_profiling:
            self._start_monitoring()

        logger.info(f"NPU Manager v2.0 initialized")
        logger.info(f"Available devices: {list(self.available_devices.keys())}")

    def _discover_devices(self) -> Dict[str, Dict[str, Any]]:
        """Discover all available compute devices."""
        devices = {}

        try:
            available = self.core.available_devices

            for device in available:
                device_info = {"name": device, "available": True}

                # Get device-specific info
                if device == "CPU":
                    device_info.update(self._get_cpu_info())
                elif device.startswith("GPU"):
                    device_info.update(self._get_gpu_info(device))
                elif device == "NPU":
                    device_info.update(self._get_npu_info())

                devices[device] = device_info

        except Exception as e:
            logger.error(f"Error discovering devices: {e}")
            devices["CPU"] = {"name": "CPU", "available": True}

        return devices

    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get detailed CPU information."""
        info = {}
        try:
            # CPU properties
            info["full_name"] = self.core.get_property("CPU", props.device.full_name)
            info["num_streams"] = self.core.get_property("CPU", props.streams.num)
            info["affinity"] = self.core.get_property("CPU", props.affinity)

            # System info
            info["physical_cores"] = psutil.cpu_count(logical=False)
            info["logical_cores"] = psutil.cpu_count(logical=True)
            info["p_cores"] = len(self.affinity_manager.p_cores)
            info["e_cores"] = len(self.affinity_manager.e_cores)

            # Get CPU model
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if "model name" in line:
                        info["model"] = line.split(":")[1].strip()
                        break

        except Exception as e:
            logger.warning(f"Could not get full CPU info: {e}")

        return info

    def _get_gpu_info(self, device: str) -> Dict[str, Any]:
        """Get Intel GPU information."""
        info = {}
        try:
            info["full_name"] = self.core.get_property(device, props.device.full_name)

            # Try to get memory info via sysfs
            gpu_mem_path = Path("/sys/class/drm/card0/device/mem_info_vram_total")
            if gpu_mem_path.exists():
                with open(gpu_mem_path, 'r') as f:
                    info["memory_bytes"] = int(f.read().strip())

            # Intel GPU specific properties
            if hasattr(intel_gpu, 'memory_statistics'):
                try:
                    info["memory_stats"] = self.core.get_property(device, intel_gpu.memory_statistics)
                except:
                    pass

        except Exception as e:
            logger.warning(f"Could not get full GPU info: {e}")

        return info

    def _get_npu_info(self) -> Dict[str, Any]:
        """Get Intel NPU information."""
        info = {"type": "Intel NPU"}

        try:
            # Check for NPU device
            npu_device = Path("/dev/accel/accel0")
            info["device_exists"] = npu_device.exists()

            if info["device_exists"]:
                # Get device permissions
                stat = os.stat(npu_device)
                info["accessible"] = os.access(npu_device, os.R_OK)

                # Check kernel module
                result = subprocess.run(['lsmod'], capture_output=True, text=True)
                info["driver_loaded"] = 'intel_vpu' in result.stdout

                # Try to get NPU properties from OpenVINO
                try:
                    info["full_name"] = self.core.get_property("NPU", props.device.full_name)
                    supported = self.core.get_property("NPU", props.supported_properties)
                    info["supported_properties"] = list(supported)
                except:
                    pass

                # Get NPU status from dmesg
                result = subprocess.run(['sudo', 'dmesg'], capture_output=True, text=True)
                if result.returncode == 0:
                    npu_lines = [line for line in result.stdout.split('\n')
                               if any(x in line.lower() for x in ['vpu', 'npu', 'accel'])]
                    info["dmesg_entries"] = len(npu_lines)

        except Exception as e:
            logger.warning(f"Could not get NPU info: {e}")
            info["error"] = str(e)

        return info

    def _probe_device_capabilities(self) -> Dict[str, DeviceCapabilities]:
        """Probe detailed capabilities of each device."""
        capabilities = {}

        for device_name, device_info in self.available_devices.items():
            try:
                cap = DeviceCapabilities(
                    device_name=device_name,
                    full_name=device_info.get('full_name', device_name),
                    device_type=device_name.split('.')[0],
                    available_memory=self._get_device_memory(device_name),
                    max_batch_size=self._get_max_batch_size(device_name),
                    supported_properties=self._get_supported_properties(device_name),
                    optimal_precision=self._get_optimal_precision(device_name),
                    thermal_status="normal",
                    compute_units=self._get_compute_units(device_name),
                    frequency_mhz=self._get_frequency(device_name)
                )
                capabilities[device_name] = cap

            except Exception as e:
                logger.warning(f"Could not probe {device_name}: {e}")

        return capabilities

    def _get_device_memory(self, device: str) -> int:
        """Get available memory for device in bytes."""
        if device == "CPU":
            return psutil.virtual_memory().available
        elif device.startswith("GPU"):
            # Try multiple methods to get GPU memory
            try:
                # Method 1: sysfs
                for i in range(2):  # Check card0 and card1
                    mem_path = Path(f"/sys/class/drm/card{i}/device/mem_info_vram_total")
                    if mem_path.exists():
                        with open(mem_path, 'r') as f:
                            return int(f.read().strip())
            except:
                pass
            return 4 * 1024 * 1024 * 1024  # Default 4GB
        elif device == "NPU":
            return 4 * 1024 * 1024 * 1024  # NPU typically has 4GB
        return 0

    def _get_optimal_precision(self, device: str) -> str:
        """Determine optimal precision for device."""
        if device == "NPU":
            return "INT8"  # NPU optimized for INT8/INT4
        elif device.startswith("GPU"):
            return "FP16"  # GPU good for FP16
        else:
            return "FP32"  # CPU default

    def _get_supported_properties(self, device: str) -> List[str]:
        """Get list of supported properties for device."""
        try:
            supported = self.core.get_property(device, props.supported_properties)
            return list(supported)
        except:
            return []

    def _get_compute_units(self, device: str) -> int:
        """Get number of compute units."""
        if device == "CPU":
            return psutil.cpu_count(logical=True)
        elif device == "NPU":
            return 2  # Intel NPU typically has 2 compute units
        elif device.startswith("GPU"):
            # This would need GPU-specific queries
            return 96  # Typical for Intel Xe graphics
        return 1

    def _get_frequency(self, device: str) -> int:
        """Get device frequency in MHz."""
        if device == "CPU":
            try:
                freqs = psutil.cpu_freq()
                return int(freqs.max) if freqs else 3000
            except:
                return 3000
        elif device == "NPU":
            return 1800  # Typical NPU frequency
        elif device.startswith("GPU"):
            return 1550  # Typical GPU frequency
        return 0

    def compile_model(self,
                     model: Union[str, ov.Model],
                     device: Optional[str] = None,
                     config: Optional[Dict[str, Any]] = None) -> ov.CompiledModel:
        """Compile model with device-specific optimizations."""

        # Load model if path provided
        if isinstance(model, str):
            model = self.core.read_model(model)

        # Auto-select device if not specified
        if device is None:
            device = self.select_optimal_device(model)

        # Get device-specific config
        device_config = self._get_device_config(device, model)
        if config:
            device_config.update(config)

        logger.info(f"Compiling model on {device}")
        logger.debug(f"Config: {device_config}")

        try:
            # Set core properties
            if device == "CPU" and self.affinity_manager.p_cores:
                # Use P-cores for low latency
                os.environ['OMP_NUM_THREADS'] = str(len(self.affinity_manager.p_cores))
                cpu_mask = ','.join(map(str, self.affinity_manager.p_cores[:4]))
                os.environ['KMP_AFFINITY'] = f"granularity=fine,proclist=[{cpu_mask}],explicit"

            compiled_model = self.core.compile_model(model, device, device_config)

            # Log compilation success
            logger.info(f"Model compiled successfully on {device}")

            return compiled_model

        except Exception as e:
            logger.error(f"Failed to compile on {device}: {e}")
            if device != "CPU":
                logger.info("Falling back to CPU")
                return self.compile_model(model, "CPU", config)
            raise

    def _get_device_config(self, device: str, model: ov.Model) -> Dict[str, Any]:
        """Get optimized configuration for device and model."""
        config = {
            props.cache_dir: str(self.cache_dir),
        }

        # Analyze model characteristics
        model_size = self._estimate_model_size(model)
        is_llm = self._is_language_model(model)

        if device == "NPU":
            config.update({
                props.performance_mode: props.PerformanceMode.LATENCY,
                props.inference_num_threads: 1,
                "NPU_COMPILATION_MODE_PARAMS": "optimization-level=3",
                "NPU_TURBO_MODE": "TRUE",
                "VPUX_COMPILATION_MODE": "DefaultHW",
            })

        elif device == "CPU":
            # CPU optimization based on model type
            if is_llm:
                # LLM-specific optimizations
                config.update({
                    props.inference_num_threads: len(self.affinity_manager.p_cores),
                    props.affinity: props.Affinity.CORE,
                    props.performance_mode: props.PerformanceMode.LATENCY,
                    props.hint.scheduling_core_type: props.hint.SchedulingCoreType.PCORE_ONLY,
                    intel_cpu.denormals_optimization: True,
                })
            else:
                # General model optimizations
                config.update({
                    props.inference_num_threads: psutil.cpu_count(logical=False),
                    props.affinity: props.Affinity.NUMA,
                    props.performance_mode: props.PerformanceMode.THROUGHPUT,
                    props.streams.num: props.streams.Num.AUTO,
                })

        elif device.startswith("GPU"):
            config.update({
                props.performance_mode: props.PerformanceMode.THROUGHPUT,
                props.hint.num_requests: 2,
                props.cache_mode: props.CacheMode.OPTIMIZE_SIZE,
            })

            if hasattr(intel_gpu, 'hint'):
                config[intel_gpu.hint.host_task_priority] = intel_gpu.hint.HostTaskPriority.MEDIUM

        elif device == "AUTO":
            # AUTO device plugin configuration
            config.update({
                props.device.priorities: "NPU,GPU,CPU",
                props.performance_mode: props.PerformanceMode.CUMULATIVE_THROUGHPUT,
                props.multi.device_bind_buffer: True,
            })

        return config

    def select_optimal_device(self, model: Union[str, ov.Model]) -> str:
        """Intelligently select optimal device for model."""

        if isinstance(model, str):
            model = self.core.read_model(model)

        # Analyze model
        model_size = self._estimate_model_size(model)
        is_llm = self._is_language_model(model)
        has_int8_weights = self._has_quantized_weights(model)

        logger.info(f"Model analysis: size={model_size:.1f}MB, LLM={is_llm}, INT8={has_int8_weights}")

        # Decision logic
        if "NPU" in self.available_devices and self.available_devices["NPU"].get("device_exists", False):
            # NPU is ideal for quantized models under 4GB
            if has_int8_weights and model_size < 4000:
                return "NPU"
            # Also good for small-medium LLMs
            if is_llm and model_size < 2000:
                return "NPU"

        if any(gpu in self.available_devices for gpu in ["GPU", "GPU.0"]):
            # GPU for medium models or when NPU unavailable
            if model_size < 8000:
                return "GPU" if "GPU" in self.available_devices else "GPU.0"

        # Default to CPU
        return "CPU"

    def _estimate_model_size(self, model: ov.Model) -> float:
        """Estimate model size in MB."""
        total_params = 0

        for op in model.get_ordered_ops():
            if hasattr(op, 'get_element_type'):
                element_type = op.get_element_type()
                if hasattr(op, 'get_output_shape'):
                    shape = op.get_output_shape(0)
                    if shape:
                        params = np.prod(shape)
                        # Estimate bytes per element
                        if 'i8' in str(element_type).lower():
                            bytes_per_element = 1
                        elif 'f16' in str(element_type).lower():
                            bytes_per_element = 2
                        else:
                            bytes_per_element = 4
                        total_params += params * bytes_per_element

        return total_params / (1024 * 1024)

    def _is_language_model(self, model: ov.Model) -> bool:
        """Detect if model is a language model."""
        model_str = str(model)
        llm_indicators = ['transformer', 'attention', 'embedding', 'gpt', 'bert', 'llama']
        return any(indicator in model_str.lower() for indicator in llm_indicators)

    def _has_quantized_weights(self, model: ov.Model) -> bool:
        """Check if model has quantized weights."""
        for op in model.get_ordered_ops():
            if hasattr(op, 'get_element_type'):
                element_type = str(op.get_element_type()).lower()
                if 'i8' in element_type or 'i4' in element_type:
                    return True
        return False

    def benchmark_model(self,
                       model: Union[str, ov.Model],
                       devices: Optional[List[str]] = None,
                       num_iterations: int = 100,
                       batch_size: int = 1) -> Dict[str, Dict[str, float]]:
        """Comprehensive model benchmarking across devices."""

        if devices is None:
            devices = list(self.available_devices.keys())

        results = {}

        for device in devices:
            logger.info(f"Benchmarking on {device}...")
            try:
                compiled_model = self.compile_model(model, device)
                infer_request = compiled_model.create_infer_request()

                # Create dummy input
                input_layer = compiled_model.input(0)
                input_shape = list(input_layer.shape)
                if input_shape[0] == -1:
                    input_shape[0] = batch_size

                dummy_input = np.random.randn(*input_shape).astype(np.float32)

                # Warmup
                for _ in range(10):
                    infer_request.infer({0: dummy_input})

                # Benchmark
                latencies = []

                for _ in range(num_iterations):
                    start = time.perf_counter()
                    infer_request.infer({0: dummy_input})
                    latencies.append(time.perf_counter() - start)

                # Calculate statistics
                latencies_ms = [l * 1000 for l in latencies]
                results[device] = {
                    "mean_latency_ms": np.mean(latencies_ms),
                    "std_latency_ms": np.std(latencies_ms),
                    "p50_latency_ms": np.percentile(latencies_ms, 50),
                    "p90_latency_ms": np.percentile(latencies_ms, 90),
                    "p99_latency_ms": np.percentile(latencies_ms, 99),
                    "min_latency_ms": np.min(latencies_ms),
                    "max_latency_ms": np.max(latencies_ms),
                    "throughput_fps": 1000.0 / np.mean(latencies_ms),
                    "batch_size": batch_size,
                }

                # Add device utilization if monitoring enabled
                if self.enable_profiling:
                    results[device]["avg_utilization"] = self._get_device_utilization(device)

            except Exception as e:
                logger.error(f"Benchmark failed on {device}: {e}")
                results[device] = {"error": str(e)}

        return results

    def _start_monitoring(self):
        """Start background monitoring thread."""
        def monitor():
            while self.enable_profiling:
                metrics = {}

                # CPU metrics
                metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
                metrics['memory_percent'] = psutil.virtual_memory().percent

                # NPU metrics
                if "NPU" in self.available_devices:
                    metrics['npu_active'] = self._check_npu_active()

                # GPU metrics
                for gpu in ["GPU", "GPU.0", "GPU.1"]:
                    if gpu in self.available_devices:
                        gpu_util = self._get_gpu_utilization(gpu)
                        if gpu_util is not None:
                            metrics[f'{gpu}_percent'] = gpu_util

                self.performance_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics
                })

                time.sleep(1)

        self._monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self._monitoring_thread.start()

    def _check_npu_active(self) -> bool:
        """Check if NPU is currently active."""
        try:
            # Check if any process has /dev/accel/accel0 open
            result = subprocess.run(['lsof', '/dev/accel/accel0'],
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _get_gpu_utilization(self, device: str) -> Optional[float]:
        """Get Intel GPU utilization."""
        try:
            # Try intel_gpu_top
            result = subprocess.run(['intel_gpu_top', '-s', '1', '-n', '1'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                # Parse output for render usage
                for line in result.stdout.split('\n'):
                    if 'render' in line.lower():
                        parts = line.split()
                        for part in parts:
                            if '%' in part:
                                return float(part.strip('%'))
        except:
            pass
        return None

    def _get_device_utilization(self, device: str) -> float:
        """Get average device utilization from monitoring history."""
        if not self.performance_history:
            return 0.0

        recent_metrics = list(self.performance_history)[-10:]

        if device == "CPU":
            values = [m['metrics'].get('cpu_percent', 0) for m in recent_metrics]
        elif device == "NPU":
            values = [100.0 if m['metrics'].get('npu_active', False) else 0.0
                     for m in recent_metrics]
        elif device.startswith("GPU"):
            values = [m['metrics'].get(f'{device}_percent', 0) for m in recent_metrics]
        else:
            return 0.0

        return np.mean(values) if values else 0.0

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": time.time(),
            "devices": {},
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            }
        }

        # Per-device status
        for device_name, device_info in self.available_devices.items():
            device_status = {
                "available": device_info.get("available", False),
                "info": device_info,
            }

            if device_name in self.device_capabilities:
                cap = self.device_capabilities[device_name]
                device_status["capabilities"] = {
                    "memory_gb": cap.available_memory / (1024**3),
                    "optimal_precision": cap.optimal_precision,
                    "compute_units": cap.compute_units,
                    "frequency_mhz": cap.frequency_mhz,
                }

            if self.enable_profiling:
                device_status["utilization"] = self._get_device_utilization(device_name)

            status["devices"][device_name] = device_status

        return status

    def optimize_for_inference(self,
                             model_path: str,
                             target_device: str,
                             optimization_level: int = 2) -> str:
        """Optimize model for specific device inference."""

        output_name = f"{Path(model_path).stem}_{target_device.lower()}_opt{optimization_level}.xml"
        output_path = self.cache_dir / output_name

        if output_path.exists():
            logger.info(f"Using cached optimized model: {output_path}")
            return str(output_path)

        logger.info(f"Optimizing {model_path} for {target_device}...")

        # Load model
        model = self.core.read_model(model_path)

        # Apply device-specific optimizations
        if target_device == "NPU":
            # NPU prefers INT8/INT4 quantization
            # This would integrate with NNCF for quantization
            pass

        elif target_device == "CPU":
            # CPU optimizations
            if optimization_level >= 2:
                # Enable bfloat16 if supported
                pass

        # Save optimized model
        ov.save_model(model, output_path)
        logger.info(f"Optimized model saved to: {output_path}")

        return str(output_path)
