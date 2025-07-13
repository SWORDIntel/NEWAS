import sys
import os
from pathlib import Path

print("Python Path:")
for p in sys.path:
    print(f"  {p}")

print("\nCurrent Directory:", os.getcwd())

print("\nProject Structure:")
for p in Path(".").rglob("__init__.py"):
    print(f"  {p}")

print("\nTrying imports:")
test_imports = [
    "src.core.agent",
    "src.core.npu_manager",
    "core.agent",
    "nem.core.agent",
    "src.nem.core.agent"
]

for imp in test_imports:
    try:
        exec(f"import {imp}")
        print(f"✓ {imp}")
    except ImportError as e:
        print(f"✗ {imp}: {e}")

# Check if __init__.py files exist
print("\nChecking __init__.py files:")
required_inits = [
    "src/__init__.py",
    "src/core/__init__.py",
    "src/capability/__init__.py",
    "src/performance/__init__.py",
    "src/plugins/__init__.py",
    "src/utils/__init__.py",
]

for init_file in required_inits:
    if Path(init_file).exists():
        print(f"✓ {init_file}")
    else:
        print(f"✗ {init_file} - MISSING!")
