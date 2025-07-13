#!/usr/bin/env python3
"""
NEMWAS Framework Validation Script

This script validates the complete NEMWAS installation and reports any issues.
Run this after setup to ensure everything is working correctly.
"""

import sys
import os
import time
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib.util

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'

# Test results storage
test_results = {
    "passed": 0,
    "failed": 0,
    "warnings": 0,
    "errors": []
}


def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'=' * 60}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(60)}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 60}{RESET}\n")


def print_section(text: str):
    """Print a section header"""
    print(f"\n{BOLD}▶ {text}{RESET}")
    print("-" * 50)


def print_success(text: str):
    """Print success message"""
    print(f"{GREEN}✓{RESET} {text}")
    test_results["passed"] += 1


def print_warning(text: str):
    """Print warning message"""
    print(f"{YELLOW}⚠{RESET} {text}")
    test_results["warnings"] += 1


def print_error(text: str):
    """Print error message"""
    print(f"{RED}✗{RESET} {text}")
    test_results["failed"] += 1
    test_results["errors"].append(text)


def check_python_version() -> bool:
    """Check Python version requirements"""
    print_section("Python Version Check")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 9:
        print_success(f"Python {version_str} (requires >= 3.9)")
        return True
    else:
        print_error(f"Python {version_str} (requires >= 3.9)")
        return False


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed"""
    print_section("Dependencies Check")

    required_packages = {
        "openvino": "2024.0.0",
        "transformers": "4.36.0",
        "torch": "2.1.0",
        "fastapi": "0.104.0",
        "pydantic": "2.5.0",
        "numpy": "1.24.0",
        "faiss": "1.7.4",
        "sentence_transformers": "2.2.2",
        "prometheus_client": "0.19.0"
    }

    results = {}

    for package, min_version in required_packages.items():
        try:
            module = importlib.import_module(package.replace("-", "_"))
            version = getattr(module, "__version__", "unknown")

            # Special handling for certain packages
            if package == "faiss":
                import faiss
                version = "installed"

            print_success(f"{package} {version} (requires >= {min_version})")
            results[package] = True

        except ImportError:
            print_error(f"{package} not installed (requires >= {min_version})")
            results[package] = False

    return results


def check_directory_structure() -> bool:
    """Check if all required directories exist"""
    print_section("Directory Structure Check")

    required_dirs = [
        "src",
        "src/core",
        "src/agents",
        "src/capability",
        "src/performance",
        "src/nlp",
        "src/plugins",
        "src/utils",
        "src/api",
        "models",
        "models/cache",
        "models/quantized",
        "models/original",
        "plugins",
        "plugins/builtin",
        "plugins/community",
        "data",
        "data/capabilities",
        "data/metrics",
        "data/embeddings",
        "config",
        "scripts",
        "docker",
        "tests",
        "examples"
    ]

    all_exist = True

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_success(f"{dir_path}/")
        else:
            print_error(f"{dir_path}/ (missing)")
            all_exist = False
            # Try to create missing directory
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f"  {YELLOW}→ Created missing directory{RESET}")
            except Exception as e:
                print(f"  {RED}→ Failed to create: {e}{RESET}")

    return all_exist


def check_core_files() -> bool:
    """Check if all core files exist"""
    print_section("Core Files Check")

    core_files = [
        "main.py",
        "requirements.txt",
        "setup.py",
        "Makefile",
        "README.md",
        "config/default.yaml",
        "src/__init__.py",
        "src/core/__init__.py",
        "src/core/agent.py",
        "src/core/npu_manager.py",
        "src/core/react.py",
        "src/capability/learner.py",
        "src/performance/tracker.py",
        "src/nlp/interface.py",
        "src/plugins/interface.py",
        "src/api/server.py",
        "src/utils/config.py",
        "scripts/setup_npu.sh",
        "scripts/download_models.py",
        "docker/Dockerfile",
        "docker/docker-compose.yml"
    ]

    all_exist = True

    for file_path in core_files:
        path = Path(file_path)
        if path.exists():
            print_success(f"{file_path}")
        else:
            print_error(f"{file_path} (missing)")
            all_exist = False

    return all_exist


def check_models() -> Dict[str, bool]:
    """Check if models are downloaded"""
    print_section("Model Files Check")

    model_files = {
        "tinyllama-1.1b": [
            "models/original/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "models/original/tinyllama-1.1b-chat.xml"
        ],
        "all-minilm-l6": [
            "models/original/pytorch_model.bin",
            "models/original/all-MiniLM-L6-v2.xml"
        ]
    }

    results = {}
    any_model_found = False

    for model_name, possible_files in model_files.items():
        found = False
        for file_path in possible_files:
            if Path(file_path).exists():
                print_success(f"{model_name}: {file_path}")
                found = True
                any_model_found = True
                break

        if not found:
            print_warning(f"{model_name}: Not found (run: python scripts/download_models.py --models {model_name})")

        results[model_name] = found

    if not any_model_found:
        print_error("No models found! Run: python scripts/download_models.py --minimal")

    return results


def check_hardware() -> Dict[str, any]:
    """Check hardware capabilities"""
    print_section("Hardware Check")

    results = {}

    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices

        print_success(f"OpenVINO version: {ov.__version__}")
        print_success(f"Available devices: {devices}")

        results["openvino_version"] = ov.__version__
        results["devices"] = devices

        # Check for NPU
        if "NPU" in devices:
            print_success("NPU detected - hardware acceleration available!")
            results["npu_available"] = True

            # Try to get NPU info
            try:
                npu_name = core.get_property("NPU", "FULL_DEVICE_NAME")
                print(f"  NPU: {npu_name}")
            except:
                pass
        else:
            print_warning("NPU not detected - will use CPU fallback")
            print(f"  {YELLOW}→ For NPU support, run: ./scripts/setup_npu.sh{RESET}")
            results["npu_available"] = False

        # Check CPU info
        try:
            import platform
            print(f"  CPU: {platform.processor()}")
            results["cpu"] = platform.processor()
        except:
            pass

        # Check memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"  Memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
            results["memory_gb"] = mem.total / (1024**3)
        except:
            pass

    except ImportError:
        print_error("OpenVINO not installed")
        results["openvino_available"] = False

    return results


def check_imports() -> bool:
    """Test importing all modules"""
    print_section("Module Import Test")

    modules_to_test = [
        "src",
        "src.core.agent",
        "src.core.npu_manager",
        "src.core.react",
        "src.capability.learner",
        "src.performance.tracker",
        "src.nlp.interface",
        "src.plugins.interface",
        "src.api.server",
        "src.utils.config"
    ]

    all_imported = True

    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print_success(f"import {module_name}")
        except Exception as e:
            print_error(f"import {module_name} - {str(e)}")
            all_imported = False

    return all_imported


async def test_basic_functionality():
    """Test basic agent functionality"""
    print_section("Basic Functionality Test")

    try:
        # Test 1: Create NPU Manager
        from src.core.npu_manager import NPUManager
        npu_manager = NPUManager()
        print_success("NPU Manager initialization")

        # Test 2: Load configuration
        from src.utils.config import load_config
        config = load_config("config/default.yaml")
        print_success("Configuration loading")

        # Test 3: Create agent (if model exists)
        model_path = Path("models/original/tinyllama-1.1b-chat.xml")
        if model_path.exists():
            from src.core.agent import NEMWASAgent, AgentConfig

            agent_config = AgentConfig(
                name="Test-Agent",
                model_path=str(model_path),
                device_preference=["NPU", "GPU", "CPU"]
            )

            agent = NEMWASAgent(agent_config, npu_manager)
            print_success(f"Agent creation (device: {agent.device})")

            # Test 4: Simple calculation
            result = await agent.process("What is 2 + 2?")
            if "4" in result:
                print_success("Agent task execution")
            else:
                print_warning(f"Agent execution returned unexpected result: {result[:50]}...")
        else:
            print_warning("Skipping agent tests - no model found")

        # Test 5: Natural Language Interface
        from src.nlp.interface import NaturalLanguageInterface
        nl_interface = NaturalLanguageInterface()
        intent = nl_interface.parse("create an agent for data analysis")
        print_success(f"NLP parsing (intent: {intent.intent_type.value})")

        # Test 6: Performance Tracker
        from src.performance.tracker import PerformanceTracker
        tracker = PerformanceTracker(enable_prometheus=False)
        print_success("Performance tracker initialization")

        # Test 7: Plugin Registry
        from src.plugins.interface import PluginRegistry
        registry = PluginRegistry()
        plugins = registry.discover_plugins()
        print_success(f"Plugin discovery ({len(plugins)} plugins found)")

    except Exception as e:
        print_error(f"Functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_api_server():
    """Test API server startup"""
    print_section("API Server Test")

    try:
        # Check if port 8080 is available
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 8080))
        sock.close()

        if result == 0:
            print_warning("Port 8080 already in use - skipping API test")
            return

        # Try to create FastAPI app
        from src.api.server import create_app
        from main import NEMWASCore

        # Create minimal core
        core = NEMWASCore("config/default.yaml")
        app = create_app(core)

        print_success("API server creation")

        # Test endpoints exist
        routes = [route.path for route in app.routes]
        important_routes = ["/", "/health", "/status", "/tasks", "/agents"]

        for route in important_routes:
            if route in routes:
                print_success(f"Endpoint: {route}")
            else:
                print_error(f"Endpoint missing: {route}")

    except Exception as e:
        print_error(f"API server test failed: {str(e)}")


def run_performance_benchmark():
    """Run a simple performance benchmark"""
    print_section("Performance Benchmark")

    try:
        import numpy as np
        import time

        # Test 1: NumPy operations (baseline)
        size = 1000000
        data = np.random.rand(size)

        start = time.time()
        result = np.sum(data ** 2)
        numpy_time = time.time() - start

        print_success(f"NumPy benchmark: {numpy_time*1000:.2f}ms for {size} elements")

        # Test 2: Import speed
        start = time.time()
        import src.core.agent
        import_time = time.time() - start

        print_success(f"Module import time: {import_time*1000:.2f}ms")

        # Test 3: Config parsing
        start = time.time()
        from src.utils.config import load_config
        config = load_config("config/default.yaml")
        config_time = time.time() - start

        print_success(f"Config load time: {config_time*1000:.2f}ms")

    except Exception as e:
        print_error(f"Performance benchmark failed: {str(e)}")


def generate_report():
    """Generate validation report"""
    print_header("VALIDATION REPORT")

    total_tests = test_results["passed"] + test_results["failed"]
    success_rate = (test_results["passed"] / total_tests * 100) if total_tests > 0 else 0

    print(f"{BOLD}Summary:{RESET}")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {GREEN}{test_results['passed']}{RESET}")
    print(f"  Failed: {RED}{test_results['failed']}{RESET}")
    print(f"  Warnings: {YELLOW}{test_results['warnings']}{RESET}")
    print(f"  Success Rate: {success_rate:.1f}%")

    if test_results["failed"] > 0:
        print(f"\n{BOLD}Failed Tests:{RESET}")
        for error in test_results["errors"]:
            print(f"  {RED}✗{RESET} {error}")

    # Save report to file
    report_file = Path("validation_report.json")
    with open(report_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": test_results,
            "success_rate": success_rate
        }, f, indent=2)

    print(f"\n{BOLD}Report saved to:{RESET} {report_file}")

    # Final status
    print(f"\n{BOLD}Final Status:{RESET}")
    if test_results["failed"] == 0:
        print(f"{GREEN}✅ NEMWAS framework is properly installed and functional!{RESET}")
        return 0
    else:
        print(f"{RED}❌ NEMWAS framework has issues that need to be resolved.{RESET}")
        print(f"\n{BOLD}Recommended Actions:{RESET}")

        # Provide specific recommendations
        if "OpenVINO not installed" in test_results["errors"]:
            print("  1. Install OpenVINO: pip install openvino>=2024.0.0")

        if any("not installed" in error for error in test_results["errors"]):
            print("  2. Install missing dependencies: pip install -r requirements.txt")

        if any("missing" in error and "models" in error for error in test_results["errors"]):
            print("  3. Download models: python scripts/download_models.py --minimal")

        if any("import" in error for error in test_results["errors"]):
            print("  4. Check PYTHONPATH and ensure you're in the project root directory")

        return 1


async def main():
    """Main validation function"""
    print_header("NEMWAS FRAMEWORK VALIDATION")
    print(f"Validating installation at: {os.getcwd()}")

    # Run all checks
    check_python_version()
    check_dependencies()
    check_directory_structure()
    check_core_files()
    check_models()
    check_hardware()
    check_imports()
    await test_basic_functionality()
    test_api_server()
    run_performance_benchmark()

    # Generate report
    exit_code = generate_report()

    return exit_code


if __name__ == "__main__":
    # Run validation
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
