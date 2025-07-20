"""NPU Manager for hardware acceleration and model optimization"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import time
import asyncio
from dataclasses import dataclass

try:
    import openvino as ov
except ImportError:
    ov = None
    logging.warning("OpenVINO not installed. NPU acceleration will not be available.")

logger = logging.getLogger(__name__)


@dataclass
class NPUDevice:
    """NPU device information"""
    name: str
    type: str
    available: bool
    properties: Dict[str, Any]


class NPUManager:
    """Manages NPU resources and model deployment"""
    
    def __init__(self, config: Dict[str, Any], cache_dir: str = "./models/cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.core = None
        self.available_devices = []
        self.loaded_models = {}
        self.device_metrics = {}
        
        # Initialize OpenVINO
        self._initialize_openvino()
    
    def _initialize_openvino(self):
        """Initialize OpenVINO runtime"""
        if ov is None:
            logger.error("OpenVINO not available")
            return
            
        try:
            self.core = ov.Core()
            
            # Detect available devices
            devices = self.core.available_devices
            logger.info(f"Available devices: {devices}")
            
            for device in devices:
                device_info = NPUDevice(
                    name=device,
                    type=self._get_device_type(device),
                    available=True,
                    properties={}
                )
                
                # Get device properties
                try:
                    if device != "AUTO":
                        device_info.properties = {
                            "full_name": self.core.get_property(device, "FULL_DEVICE_NAME"),
                        }
                except Exception as e:
                    logger.warning(f"Could not get properties for {device}: {e}")
                
                self.available_devices.append(device)
            
            # Set NPU-specific properties if available
            if "NPU" in self.available_devices:
                self._configure_npu()
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO: {e}")
    
    def _get_device_type(self, device_name: str) -> str:
        """Get device type from name"""
        if "NPU" in device_name:
            return "NPU"
        elif "GPU" in device_name:
            return "GPU"
        elif "CPU" in device_name:
            return "CPU"
        else:
            return "UNKNOWN"
    
    def _configure_npu(self):
        """Configure NPU-specific settings"""
        try:
            npu_config = self.config.get("npu", {})
            
            # Set compilation mode
            compilation_mode = npu_config.get("compilation_mode", "LATENCY")
            self.core.set_property("NPU", {"PERFORMANCE_HINT": compilation_mode})
            
            # Enable turbo mode if configured
            if npu_config.get("turbo_mode", False):
                self.core.set_property("NPU", {"ENABLE_CPU_FALLBACK": "NO"})
            
            logger.info(f"NPU configured with mode: {compilation_mode}")
            
        except Exception as e:
            logger.warning(f"Could not configure NPU: {e}")
    
    def has_npu(self) -> bool:
        """Check if NPU is available"""
        return "NPU" in self.available_devices
    
    def load_model(self, model_path: Union[str, Path], 
                   device_preference: List[str] = None) -> Any:
        """Load a model with device preference"""
        model_path = Path(model_path)
        
        # Check if model already loaded
        model_key = str(model_path)
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if not ov or not self.core:
            logger.error("OpenVINO not initialized")
            return None
        
        try:
            # Read model
            model = self.core.read_model(model_path)
            
            # Select device
            device = self._select_device(device_preference)
            
            # Apply optimizations based on device
            if device == "NPU":
                model = self._optimize_for_npu(model)
            
            # Compile model
            compiled_model = self.core.compile_model(model, device)
            
            # Cache the model
            self.loaded_models[model_key] = compiled_model
            
            logger.info(f"Model loaded on {device}: {model_path.name}")
            return compiled_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def _select_device(self, preference: List[str] = None) -> str:
        """Select device based on preference and availability"""
        if not preference:
            preference = self.config.get("npu", {}).get("device_priority", ["NPU", "GPU", "CPU"])
        
        for device in preference:
            if device in self.available_devices:
                return device
        
        # Fallback to CPU
        return "CPU" if "CPU" in self.available_devices else "AUTO"
    
    def _optimize_for_npu(self, model: Any) -> Any:
        """Apply NPU-specific optimizations"""
        try:
            # Model would be optimized here
            # This is a placeholder for actual optimization logic
            logger.info("Applying NPU optimizations")
            return model
        except Exception as e:
            logger.warning(f"NPU optimization failed: {e}")
            return model
    
    async def generate(self, model: Any, prompt: str, 
                      max_new_tokens: int = 256,
                      temperature: float = 0.7) -> str:
        """Generate text using the model"""
        if not model:
            return "Model not loaded"
        
        try:
            # This is a simplified version
            # In reality, would handle tokenization and generation
            start_time = time.time()
            
            # Simulate async generation
            await asyncio.sleep(0.1)  # Placeholder for actual inference
            
            # Track metrics
            inference_time = time.time() - start_time
            self._update_metrics("inference_time", inference_time)
            
            # Return generated text (placeholder)
            return f"Generated response for: {prompt[:50]}..."
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {e}"
    
    def get_model_device(self, model_path: Union[str, Path]) -> str:
        """Get the device a model is loaded on"""
        model_key = str(Path(model_path))
        if model_key in self.loaded_models:
            # In real implementation, would get actual device
            return self._select_device()
        return "Not loaded"
    
    def get_device_metrics(self, device: str) -> Dict[str, Any]:
        """Get metrics for a specific device"""
        return self.device_metrics.get(device, {
            "utilization": 0.0,
            "memory_used": 0,
            "temperature": 0.0,
            "power": 0.0
        })
    
    def _update_metrics(self, metric: str, value: float):
        """Update device metrics"""
        device = self._select_device()
        if device not in self.device_metrics:
            self.device_metrics[device] = {}
        self.device_metrics[device][metric] = value
    
    def optimize_model_for_device(self, model_path: Path, 
                                 device: str,
                                 quantization_preset: str = "mixed") -> Path:
        """Optimize a model for specific device"""
        optimized_path = self.cache_dir / f"{model_path.stem}_{device.lower()}_{quantization_preset}.xml"
        
        if optimized_path.exists():
            logger.info(f"Using cached optimized model: {optimized_path}")
            return optimized_path
        
        try:
            # This would contain actual optimization logic
            logger.info(f"Optimizing {model_path} for {device} with preset: {quantization_preset}")
            
            # For now, return original path
            return model_path
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return model_path
    
    def clear_cache(self):
        """Clear model cache"""
        self.loaded_models.clear()
        logger.info("Model cache cleared")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            "available_devices": self.available_devices,
            "loaded_models": list(self.loaded_models.keys()),
            "has_npu": self.has_npu(),
            "openvino_available": ov is not None
        }
        
        if ov and self.core:
            info["openvino_version"] = ov.get_version()
        
        return info