#!/usr/bin/env python3
"""Comprehensive AI Stack Benchmark - Tests CPU, GPU, NPU with OpenVINO"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import psutil
import platform

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from npu_manager_v2 import NPUManager
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False
    print("Warning: OpenVINO not found. Some tests will be skipped.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class AIStackBenchmark:
    """Comprehensive AI stack benchmarking suite."""

    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.system_info = self._collect_system_info()
        self.results = {
            "system": self.system_info,
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {}
        }

        if HAS_OPENVINO:
            self.npu_manager = NPUManager(enable_profiling=True)

    def _collect_system_info(self):
        """Collect comprehensive system information."""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
        }

        # Get CPU model
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if "model name" in line:
                        info["cpu_model"] = line.split(":")[1].strip()
                        break
        except:
            pass

        # Check for Intel GPU
        try:
            import subprocess
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA' in line or 'Display' in line:
                        if 'Intel' in line:
                            info["intel_gpu"] = line.strip()
        except:
            pass

        return info

    def benchmark_numpy(self, core_type: str = "auto"):
        """Benchmark NumPy operations."""
        print(f"\n{'='*60}")
        print(f"NumPy Benchmark ({core_type} cores)")
        print(f"{'='*60}")

        # Matrix sizes to test
        sizes = [1000, 2000, 4000]
        operations = ["matmul", "svd", "fft"]

        results = {}

        for size in sizes:
            print(f"\nMatrix size: {size}x{size}")
            results[f"size_{size}"] = {}

            # Generate random matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)

            for op in operations:
                times = []

                for _ in range(5):
                    if op == "matmul":
                        start = time.perf_counter()
                        _ = np.matmul(A, B)
                        elapsed = time.perf_counter() - start
                    elif op == "svd":
                        start = time.perf_counter()
                        _ = np.linalg.svd(A, full_matrices=False)
                        elapsed = time.perf_counter() - start
                    elif op == "fft":
                        start = time.perf_counter()
                        _ = np.fft.fft2(A)
                        elapsed = time.perf_counter() - start

                    times.append(elapsed)

                avg_time = np.mean(times) * 1000  # Convert to ms
                gflops = self._calculate_gflops(op, size, np.mean(times))

                results[f"size_{size}"][op] = {
                    "time_ms": avg_time,
                    "gflops": gflops
                }

                print(f"  {op}: {avg_time:.2f} ms ({gflops:.2f} GFLOPS)")

        self.results["benchmarks"][f"numpy_{core_type}"] = results
        return results

    def benchmark_openvino(self):
        """Benchmark OpenVINO on different devices."""
        if not HAS_OPENVINO:
            print("\nOpenVINO benchmark skipped (not installed)")
            return None

        print(f"\n{'='*60}")
        print("OpenVINO Benchmark")
        print(f"{'='*60}")

        # Create test models of different sizes
        models = {
            "small": self._create_conv_model((1, 3, 224, 224), 32),
            "medium": self._create_conv_model((1, 3, 512, 512), 64),
            "large": self._create_conv_model((1, 3, 1024, 1024), 128),
        }

        results = {}

        for model_name, model_path in models.items():
            print(f"\n{model_name.capitalize()} Model:")

            # Benchmark on all available devices
            bench_results = self.npu_manager.benchmark_model(
                model_path,
                devices=None,  # Test all devices
                num_iterations=50,
                batch_size=1
            )

            results[model_name] = bench_results

            # Print summary
            for device, metrics in bench_results.items():
                if 'error' not in metrics:
                    print(f"  {device}: {metrics['mean_latency_ms']:.2f} ms "
                          f"({metrics['throughput_fps']:.1f} FPS)")
                else:
                    print(f"  {device}: Failed - {metrics['error']}")

            # Cleanup
            Path(model_path).unlink(missing_ok=True)
            Path(model_path.replace('.xml', '.bin')).unlink(missing_ok=True)

        self.results["benchmarks"]["openvino"] = results
        return results

    def benchmark_ai_workloads(self):
        """Benchmark realistic AI workloads."""
        print(f"\n{'='*60}")
        print("AI Workload Benchmark")
        print(f"{'='*60}")

        results = {}

        # 1. Image Classification workload
        print("\n1. Image Classification (MobileNetV2-like)")
        if HAS_OPENVINO:
            model_path = self._create_mobilenet_like_model()
            batch_sizes = [1, 4, 8]

            for batch_size in batch_sizes:
                print(f"\n  Batch size: {batch_size}")
                bench = self.npu_manager.benchmark_model(
                    model_path,
                    devices=["CPU", "GPU", "NPU"],
                    num_iterations=50,
                    batch_size=batch_size
                )

                for device, metrics in bench.items():
                    if 'error' not in metrics:
                        print(f"    {device}: {metrics['throughput_fps']:.1f} images/sec")

                results[f"mobilenet_batch_{batch_size}"] = bench

            Path(model_path).unlink(missing_ok=True)
            Path(model_path.replace('.xml', '.bin')).unlink(missing_ok=True)

        # 2. NLP-like workload (Transformer attention)
        print("\n2. Transformer Attention (BERT-like)")
        if HAS_TORCH and HAS_OPENVINO:
            seq_lengths = [128, 256, 512]

            for seq_len in seq_lengths:
                print(f"\n  Sequence length: {seq_len}")
                model_path = self._create_attention_model(seq_len)

                bench = self.npu_manager.benchmark_model(
                    model_path,
                    devices=["CPU", "NPU"],
                    num_iterations=20,
                    batch_size=1
                )

                for device, metrics in bench.items():
                    if 'error' not in metrics:
                        tokens_per_sec = seq_len * metrics['throughput_fps']
                        print(f"    {device}: {tokens_per_sec:.0f} tokens/sec")

                results[f"attention_seq_{seq_len}"] = bench

                Path(model_path).unlink(missing_ok=True)
                Path(model_path.replace('.xml', '.bin')).unlink(missing_ok=True)

        self.results["benchmarks"]["ai_workloads"] = results
        return results

    def benchmark_memory_bandwidth(self):
        """Benchmark memory bandwidth."""
        print(f"\n{'='*60}")
        print("Memory Bandwidth Benchmark")
        print(f"{'='*60}")

        sizes_mb = [100, 500, 1000, 2000]
        results = {}

        for size_mb in sizes_mb:
            size_bytes = size_mb * 1024 * 1024
            n_elements = size_bytes // 4  # float32

            # Create arrays
            src = np.random.randn(n_elements).astype(np.float32)
            dst = np.empty_like(src)

            # Measure copy bandwidth
            times = []
            for _ in range(10):
                start = time.perf_counter()
                np.copyto(dst, src)
                times.append(time.perf_counter() - start)

            avg_time = np.mean(times)
            bandwidth_gb_s = (size_bytes * 2) / (avg_time * 1024**3)  # Read + Write

            results[f"{size_mb}MB"] = {
                "time_ms": avg_time * 1000,
                "bandwidth_GB/s": bandwidth_gb_s
            }

            print(f"  {size_mb} MB: {bandwidth_gb_s:.2f} GB/s")

        self.results["benchmarks"]["memory_bandwidth"] = results
        return results

    def _calculate_gflops(self, operation: str, size: int, time_sec: float):
        """Calculate GFLOPS for different operations."""
        if operation == "matmul":
            # Matrix multiplication: 2*n^3 operations
            flops = 2 * (size ** 3)
        elif operation == "svd":
            # SVD: approximately 4*n^3 operations
            flops = 4 * (size ** 3)
        elif operation == "fft":
            # 2D FFT: approximately 5*n^2*log2(n) operations
            flops = 5 * (size ** 2) * np.log2(size)
        else:
            flops = 0

        return (flops / time_sec) / 1e9

    def _create_conv_model(self, input_shape, num_filters):
        """Create a simple convolutional model."""
        param = ov.opset10.parameter(input_shape, ov.Type.f32, name="input")

        # Conv -> ReLU -> Conv -> ReLU
        weights1 = ov.opset10.constant(
            np.random.randn(num_filters, input_shape[1], 3, 3).astype(np.float32)
        )
        conv1 = ov.opset10.convolution(param, weights1, [1, 1], [1, 1], [1, 1], [1, 1])
        relu1 = ov.opset10.relu(conv1)

        weights2 = ov.opset10.constant(
            np.random.randn(num_filters, num_filters, 3, 3).astype(np.float32)
        )
        conv2 = ov.opset10.convolution(relu1, weights2, [1, 1], [1, 1], [1, 1], [1, 1])
        relu2 = ov.opset10.relu(conv2)

        model = ov.Model([relu2], [param], "conv_model")

        model_path = Path(f"./temp_conv_model_{num_filters}.xml")
        ov.save_model(model, model_path)

        return str(model_path)

    def _create_mobilenet_like_model(self):
        """Create a MobileNetV2-like model."""
        # Simplified MobileNetV2 with depthwise separable convolutions
        input_shape = [1, 3, 224, 224]
        param = ov.opset10.parameter(input_shape, ov.Type.f32, name="input")

        # Initial conv
        weights = ov.opset10.constant(
            np.random.randn(32, 3, 3, 3).astype(np.float32)
        )
        x = ov.opset10.convolution(param, weights, [2, 2], [1, 1], [1, 1], [1, 1])
        x = ov.opset10.relu(x)

        # Depthwise separable blocks
        for i in range(3):
            # Depthwise conv
            dw_weights = ov.opset10.constant(
                np.random.randn(32, 1, 3, 3).astype(np.float32)
            )
            x = ov.opset10.group_convolution(
                x, dw_weights, [1, 1], [1, 1], [1, 1], [1, 1], 32
            )
            x = ov.opset10.relu(x)

            # Pointwise conv
            pw_weights = ov.opset10.constant(
                np.random.randn(64 if i == 2 else 32, 32, 1, 1).astype(np.float32)
            )
            x = ov.opset10.convolution(x, pw_weights, [1, 1], [0, 0], [0, 0], [1, 1])
            x = ov.opset10.relu(x)

        # Global average pooling
        x = ov.opset10.reduce_mean(x, ov.opset10.constant([2, 3]), keep_dims=True)

        # Final FC
        x = ov.opset10.reshape(x, ov.opset10.constant([1, -1]), special_zero=False)
        fc_weights = ov.opset10.constant(
            np.random.randn(1000, 64).astype(np.float32)
        )
        output = ov.opset10.matmul(x, ov.opset10.transpose(fc_weights, ov.opset10.constant([1, 0])))

        model = ov.Model([output], [param], "mobilenet_like")
        model_path = Path("./temp_mobilenet.xml")
        ov.save_model(model, model_path)

        return str(model_path)

    def _create_attention_model(self, seq_length: int, hidden_dim: int = 768):
        """Create a transformer attention model."""
        # Simplified BERT-like attention
        batch_size = 1
        num_heads = 12
        head_dim = hidden_dim // num_heads

        # Input
        input_shape = [batch_size, seq_length, hidden_dim]
        param = ov.opset10.parameter(input_shape, ov.Type.f32, name="input_ids")

        # Q, K, V projections
        qkv_weight = ov.opset10.constant(
            np.random.randn(hidden_dim, hidden_dim * 3).astype(np.float32)
        )
        qkv = ov.opset10.matmul(param, qkv_weight)

        # Split into Q, K, V
        qkv_splits = ov.opset10.split(qkv, ov.opset10.constant(2), 3)
        q = ov.opset10.gather(qkv_splits, ov.opset10.constant(0), ov.opset10.constant(0))
        k = ov.opset10.gather(qkv_splits, ov.opset10.constant(1), ov.opset10.constant(0))
        v = ov.opset10.gather(qkv_splits, ov.opset10.constant(2), ov.opset10.constant(0))

        # Reshape for multi-head attention
        new_shape = ov.opset10.constant([batch_size, seq_length, num_heads, head_dim])
        q = ov.opset10.reshape(q, new_shape, special_zero=False)
        k = ov.opset10.reshape(k, new_shape, special_zero=False)
        v = ov.opset10.reshape(v, new_shape, special_zero=False)

        # Transpose to [batch, heads, seq, dim]
        perm = ov.opset10.constant([0, 2, 1, 3])
        q = ov.opset10.transpose(q, perm)
        k = ov.opset10.transpose(k, perm)
        v = ov.opset10.transpose(v, perm)

        # Attention scores
        k_t = ov.opset10.transpose(k, ov.opset10.constant([0, 1, 3, 2]))
        scores = ov.opset10.matmul(q, k_t)

        # Scale
        scale = ov.opset10.constant(np.sqrt(head_dim).astype(np.float32))
        scores = ov.opset10.divide(scores, scale)

        # Softmax
        attention_weights = ov.opset10.softmax(scores, 3)

        # Apply attention to values
        context = ov.opset10.matmul(attention_weights, v)

        # Reshape back
        context = ov.opset10.transpose(context, ov.opset10.constant([0, 2, 1, 3]))
        context = ov.opset10.reshape(
            context,
            ov.opset10.constant([batch_size, seq_length, hidden_dim]),
            special_zero=False
        )

        # Output projection
        out_weight = ov.opset10.constant(
            np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
        )
        output = ov.opset10.matmul(context, out_weight)

        model = ov.Model([output], [param], "attention_model")
        model_path = Path(f"./temp_attention_{seq_length}.xml")
        ov.save_model(model, model_path)

        return str(model_path)

    def save_results(self):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {filename}")
        return filename

    def print_summary(self):
        """Print benchmark summary."""
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        print(f"\nSystem Information:")
        print(f"  CPU: {self.system_info.get('cpu_model', 'Unknown')}")
        print(f"  Cores: {self.system_info['cpu_count_physical']} physical, "
              f"{self.system_info['cpu_count']} logical")
        print(f"  Memory: {self.system_info['memory_gb']:.1f} GB")

        if 'intel_gpu' in self.system_info:
            print(f"  GPU: {self.system_info['intel_gpu']}")

        if HAS_OPENVINO and hasattr(self, 'npu_manager'):
            if "NPU" in self.npu_manager.available_devices:
                npu_info = self.npu_manager.available_devices["NPU"]
                print(f"  NPU: Available - "
                      f"Driver: {'Yes' if npu_info.get('driver_loaded') else 'No'}")

        # Performance highlights
        if 'numpy_p' in self.results['benchmarks']:
            numpy_p = self.results['benchmarks']['numpy_p']
            if 'size_2000' in numpy_p:
                matmul_perf = numpy_p['size_2000']['matmul']['gflops']
                print(f"\n  NumPy (P-cores): {matmul_perf:.1f} GFLOPS (2K MatMul)")

        if 'openvino' in self.results['benchmarks']:
            ov_results = self.results['benchmarks']['openvino']
            if 'medium' in ov_results:
                print(f"\n  OpenVINO Performance (512x512 Conv):")
                for device, metrics in ov_results['medium'].items():
                    if 'error' not in metrics:
                        print(f"    {device}: {metrics['throughput_fps']:.1f} FPS")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive AI Stack Benchmark")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark (subset of tests)')
    parser.add_argument('--devices', nargs='+',
                       choices=['CPU', 'GPU', 'NPU', 'ALL'],
                       default=['ALL'],
                       help='Devices to benchmark')
    parser.add_argument('--skip-numpy', action='store_true',
                       help='Skip NumPy benchmarks')
    parser.add_argument('--skip-openvino', action='store_true',
                       help='Skip OpenVINO benchmarks')
    parser.add_argument('--output', '-o', type=str,
                       default='./benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Initialize benchmark suite
    benchmark = AIStackBenchmark(output_dir=args.output)

    # Print system info
    print("AI Stack Benchmark v1.0")
    print(f"System: {benchmark.system_info['platform']}")
    print(f"CPU: {benchmark.system_info.get('cpu_model', 'Unknown')}")

    # Run benchmarks
    if not args.skip_numpy:
        # Check if we're already on specific cores
        current_affinity = psutil.Process().cpu_affinity()
        if len(current_affinity) < psutil.cpu_count():
            # Already constrained, use current setting
            benchmark.benchmark_numpy("current")
        else:
            # Full system, can test P vs E
            benchmark.benchmark_numpy("p")
            if not args.quick:
                benchmark.benchmark_numpy("e")

    if not args.skip_openvino and HAS_OPENVINO:
        benchmark.benchmark_openvino()

        if not args.quick:
            benchmark.benchmark_ai_workloads()

    if not args.quick:
        benchmark.benchmark_memory_bandwidth()

    # Save results
    benchmark.save_results()

    # Print summary
    benchmark.print_summary()


if __name__ == "__main__":
    main()
