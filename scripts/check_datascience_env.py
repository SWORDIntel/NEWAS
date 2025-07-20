#!/usr/bin/env python3
"""Check datascience environment compatibility for NEMWAS."""

import sys
import importlib.util
from packaging import version

REQUIRED_PACKAGES = {
    'openvino': '2024.0.0',
    'transformers': '4.36.0',
    'torch': '2.1.0',
    'numpy': '1.24.0'
}

OPTIONAL_PACKAGES = {
    'openvino_genai': None,
    'optimum': None,
    'jupyter': None,
    'tensorflow': None
}

ADVANCED_FEATURES = [
    'openvino.runtime.Core',
    'openvino.runtime.CompiledModel',
    'transformers.AutoModel',
    'torch.nn.Module'
]

def check_environment():
    """Check if the datascience environment is compatible with NEMWAS."""
    issues = []
    warnings = []
    features = []
    
    print("Checking datascience environment compatibility...")
    print("-" * 60)
    
    # Check required packages
    print("\nRequired packages:")
    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            module = importlib.import_module(package.replace('-', '_'))
            current = getattr(module, '__version__', 'unknown')
            
            if current != 'unknown' and min_version:
                if version.parse(current) >= version.parse(min_version):
                    print(f"  ✓ {package} {current} (>= {min_version})")
                else:
                    issues.append(f"{package} version {current} < required {min_version}")
                    print(f"  ✗ {package} {current} (requires >= {min_version})")
            else:
                print(f"  ✓ {package} {current}")
        except ImportError:
            issues.append(f"{package} not found")
            print(f"  ✗ {package} not found")
    
    # Check optional packages
    print("\nOptional packages:")
    for package, min_version in OPTIONAL_PACKAGES.items():
        try:
            module = importlib.import_module(package.replace('-', '_'))
            current = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package} {current}")
            features.append(package)
        except ImportError:
            print(f"  - {package} not found (optional)")
    
    # Check advanced features
    print("\nAdvanced features:")
    for feature_path in ADVANCED_FEATURES:
        parts = feature_path.split('.')
        try:
            module = importlib.import_module('.'.join(parts[:-1]))
            if hasattr(module, parts[-1]):
                print(f"  ✓ {feature_path}")
            else:
                warnings.append(f"Feature {feature_path} not available")
                print(f"  ✗ {feature_path} not available")
        except ImportError:
            warnings.append(f"Module for {feature_path} not found")
            print(f"  ✗ {feature_path} module not found")
    
    # Check for NPU support
    print("\nHardware acceleration:")
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        print(f"  Available devices: {devices}")
        
        if 'NPU' in devices:
            print("  ✓ NPU support available")
            features.append("NPU acceleration")
        else:
            print("  - NPU not detected")
            
        if 'GPU' in devices:
            print("  ✓ GPU support available")
            features.append("GPU acceleration")
    except Exception as e:
        warnings.append(f"Could not check hardware support: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("✗ Environment compatibility issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nThe datascience environment may not be fully compatible.")
        print("NEMWAS will create a local environment instead.")
        sys.exit(1)
    else:
        print("✓ Datascience environment is fully compatible with NEMWAS")
        
        if features:
            print("\nAvailable features:")
            for feature in features:
                print(f"  - {feature}")
                
        if warnings:
            print("\nWarnings (non-critical):")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("\nThe advanced environment can be used for enhanced performance!")
        sys.exit(0)

if __name__ == "__main__":
    check_environment()