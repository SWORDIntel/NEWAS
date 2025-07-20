#!/usr/bin/env python3
"""Install only missing dependencies from requirements.txt."""

import subprocess
import sys
import pkg_resources
from pathlib import Path
from packaging import version
import re

def get_installed_packages():
    """Get dict of installed packages and versions."""
    return {pkg.key: pkg.version for pkg in pkg_resources.working_set}

def parse_requirements(req_file):
    """Parse requirements.txt file."""
    requirements = []
    
    if not Path(req_file).exists():
        print(f"Error: {req_file} not found")
        return requirements
        
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
                
            # Remove inline comments
            if '#' in line:
                line = line.split('#')[0].strip()
                
            # Handle different requirement formats
            if '>=' in line:
                match = re.match(r'([^>=]+)>=(.+)', line)
                if match:
                    pkg_name = match.group(1).strip()
                    min_version = match.group(2).strip()
                    requirements.append((pkg_name, min_version, '>='))
            elif '==' in line:
                match = re.match(r'([^==]+)==(.+)', line)
                if match:
                    pkg_name = match.group(1).strip()
                    exact_version = match.group(2).strip()
                    requirements.append((pkg_name, exact_version, '=='))
            elif '~=' in line:
                match = re.match(r'([^~=]+)~=(.+)', line)
                if match:
                    pkg_name = match.group(1).strip()
                    compatible_version = match.group(2).strip()
                    requirements.append((pkg_name, compatible_version, '~='))
            else:
                # Package without version specifier
                requirements.append((line.strip(), None, None))
                
    return requirements

def normalize_package_name(name):
    """Normalize package name for comparison."""
    return name.lower().replace('-', '_').replace('.', '_')

def check_version_compatibility(installed_ver, required_ver, operator):
    """Check if installed version meets requirement."""
    try:
        installed = version.parse(installed_ver)
        required = version.parse(required_ver)
        
        if operator == '>=':
            return installed >= required
        elif operator == '==':
            return installed == required
        elif operator == '~=':
            # Compatible version (~=1.4.2 means >=1.4.2, <1.5.0)
            major_minor = '.'.join(required_ver.split('.')[:2])
            next_minor = '.'.join([
                required_ver.split('.')[0],
                str(int(required_ver.split('.')[1]) + 1),
                '0'
            ])
            return installed >= required and installed < version.parse(next_minor)
        else:
            return True
    except:
        return False

def main():
    """Install missing dependencies."""
    if len(sys.argv) < 2:
        print("Usage: install_missing_deps.py requirements.txt")
        sys.exit(1)
    
    req_file = sys.argv[1]
    installed = get_installed_packages()
    requirements = parse_requirements(req_file)
    
    if not requirements:
        print("No requirements found.")
        return
    
    missing = []
    upgrades = []
    already_satisfied = []
    
    for pkg_name, req_version, op in requirements:
        pkg_key = normalize_package_name(pkg_name)
        
        # Special handling for package name mappings
        package_mappings = {
            'pillow': 'pil',
            'msgpack_python': 'msgpack',
            'protobuf': 'protobuf',
        }
        
        # Check both normalized and mapped names
        found = False
        for check_key in [pkg_key, package_mappings.get(pkg_key, pkg_key)]:
            if check_key in installed:
                found = True
                installed_version = installed[check_key]
                
                if req_version:
                    if check_version_compatibility(installed_version, req_version, op):
                        already_satisfied.append(f"{pkg_name} ({installed_version})")
                    else:
                        upgrades.append(f"{pkg_name}{op}{req_version}")
                else:
                    already_satisfied.append(f"{pkg_name} ({installed_version})")
                break
        
        if not found:
            if req_version:
                missing.append(f"{pkg_name}{op}{req_version}")
            else:
                missing.append(pkg_name)
    
    # Report status
    print(f"\nDependency check for {req_file}:")
    print("-" * 60)
    
    if already_satisfied:
        print(f"✓ Already satisfied: {len(already_satisfied)} packages")
        if len(already_satisfied) <= 10:
            for pkg in already_satisfied[:10]:
                print(f"  - {pkg}")
        else:
            print(f"  (showing first 10 of {len(already_satisfied)})")
            for pkg in already_satisfied[:10]:
                print(f"  - {pkg}")
    
    # Install missing and upgrades
    to_install = missing + upgrades
    
    if to_install:
        print(f"\n📦 Installing/upgrading {len(to_install)} packages:")
        for pkg in to_install:
            print(f"  - {pkg}")
        
        print("\nInstalling...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + to_install)
            print(f"\n✓ Successfully installed/upgraded {len(to_install)} packages")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Error installing packages: {e}")
            sys.exit(1)
    else:
        print("\n✓ All required packages already installed")

if __name__ == "__main__":
    main()