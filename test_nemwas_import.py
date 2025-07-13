#!/usr/bin/env python3
"""Test NEMWAS imports and basic functionality"""

import sys
from pathlib import Path

def test_imports():
    """Test all NEMWAS imports"""
    print("Testing NEMWAS imports...")
    print("=" * 50)
    
    # Test main package
    try:
        import src
        print(f"✓ NEMWAS v{src.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import src: {e}")
        return False
    
    # Test core modules
    try:
        from src.core import NPUManager, NEMWASAgent, AgentConfig
        print("✓ Core modules imported")
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    # Test other modules
    modules_to_test = [
        ("capability", "CapabilityLearner"),
        ("performance", "PerformanceTracker"),
        ("nlp", "NaturalLanguageInterface"),
        ("plugins", "PluginRegistry"),
        ("utils", "load_config"),
        ("api", "create_app"),
    ]
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(f"src.{module_name}", fromlist=[class_name])
            getattr(module, class_name)
            print(f"✓ {module_name} module imported")
        except (ImportError, AttributeError) as e:
            print(f"✗ Failed to import {module_name}: {e}")
            return False
    
    print("
✅ All imports successful!")
    return True

def test_basic_functionality():
    """Test basic NEMWAS functionality"""
    print("
Testing basic functionality...")
    print("=" * 50)
    
    try:
        from src.core import NPUManager
        
        # Test NPU Manager
        print("Testing NPU Manager...")
        npu_manager = NPUManager()
        print(f"  Available devices: {npu_manager.available_devices}")
        print(f"  Device preference: {npu_manager.device_preference}")
        print("✓ NPU Manager works")
        
    except Exception as e:
        print(f"✗ NPU Manager test failed: {e}")
        return False
    
    try:
        from src.utils import get_default_config
        
        # Test configuration
        print("
Testing configuration...")
        config = get_default_config()
        print(f"  System name: {config['system']['name']}")
        print(f"  Version: {config['system']['version']}")
        print("✓ Configuration works")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    print("
✅ Basic functionality tests passed!")
    return True

if __name__ == "__main__":
    print("🚀 NEMWAS Test Suite")
    print("=" * 50)
    
    # Add src to path if needed
    if not any("nemwas" in str(p) for p in sys.path):
        sys.path.insert(0, str(Path(__file__).parent))
    
    # Run tests
    import_success = test_imports()
    
    if import_success:
        functionality_success = test_basic_functionality()
        
        if functionality_success:
            print("
🎉 All tests passed! NEMWAS is ready to use.")
            sys.exit(0)
    
    print("
❌ Some tests failed. Please check the output above.")
    sys.exit(1)
