#!/bin/bash
# Setup NPU support on Debian Linux

set -e

echo "==================================="
echo "NEMWAS NPU Setup Script for Debian"
echo "==================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo "Please do not run this script as root. It will request sudo when needed."
   exit 1
fi

# Detect Linux distribution
if [ -f /etc/debian_version ]; then
    DISTRO="debian"
    VERSION=$(cat /etc/debian_version)
    echo "Detected Debian $VERSION"
else
    echo "This script is designed for Debian Linux."
    echo "Your distribution may require manual setup."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to check Intel CPU
check_intel_cpu() {
    echo "Checking CPU..."
    
    cpu_info=$(lscpu | grep "Model name" || echo "Unknown")
    echo "CPU: $cpu_info"
    
    # Check for Intel Core Ultra (Meteor Lake)
    if lscpu | grep -q "Intel.*Core.*Ultra"; then
        echo "✓ Intel Core Ultra processor detected - NPU should be available"
        return 0
    elif lscpu | grep -q "Intel"; then
        echo "⚠ Intel processor detected, but may not have NPU"
        echo "  NPU is available on Intel Core Ultra (Meteor Lake) and newer"
        return 1
    else
        echo "✗ Non-Intel processor detected - NPU not available"
        return 1
    fi
}

# Function to install Intel Graphics Compute Runtime
install_compute_runtime() {
    echo ""
    echo "Installing Intel Graphics Compute Runtime..."
    
    # Add Intel graphics repository
    wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | sudo apt-key add -
    sudo apt-add-repository "deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main"
    
    # Update and install
    sudo apt-get update
    sudo apt-get install -y \
        intel-opencl-icd \
        intel-level-zero-gpu \
        level-zero \
        intel-media-va-driver-non-free \
        libmfx1 \
        libigdgmm11 \
        libgmmlib11 \
        libigc1 \
        libigc-dev \
        intel-igc-cm \
        libigdfcl1 \
        libigdfcl-dev \
        libigfxcmrt7 \
        libigfxcmrt-dev
        
    echo "✓ Intel Graphics Compute Runtime installed"
}

# Function to install OpenVINO runtime
install_openvino_runtime() {
    echo ""
    echo "Installing OpenVINO runtime..."
    
    # Download and install OpenVINO runtime
    wget https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.0/linux/l_openvino_toolkit_ubuntu22_2024.0.0.14509.34caeefd078_x86_64.tgz
    
    tar -xf l_openvino_toolkit_*.tgz
    cd l_openvino_toolkit_*
    
    sudo ./install_dependencies/install_openvino_dependencies.sh
    
    # Install to /opt/intel/openvino
    sudo mkdir -p /opt/intel
    sudo cp -r . /opt/intel/openvino_2024
    sudo ln -sf /opt/intel/openvino_2024 /opt/intel/openvino
    
    # Setup environment
    echo "source /opt/intel/openvino/setupvars.sh" >> ~/.bashrc
    
    cd ..
    rm -rf l_openvino_toolkit_*
    
    echo "✓ OpenVINO runtime installed"
}

# Function to setup NPU permissions
setup_npu_permissions() {
    echo ""
    echo "Setting up NPU permissions..."
    
    # Add user to required groups
    sudo usermod -a -G video,render $USER
    
    # Create udev rules for NPU access
    cat << 'UDEV_EOF' | sudo tee /etc/udev/rules.d/97-intel-npu.rules
# Intel NPU device rules
SUBSYSTEM=="pci", ATTR{vendor}=="0x8086", ATTR{class}=="0x048000", MODE="0666", GROUP="render"
SUBSYSTEM=="accel", KERNEL=="accel*", MODE="0666", GROUP="render"

# Intel Neural Compute Stick 2 (if using external NPU)
SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666", GROUP="users"
UDEV_EOF

    # Reload udev rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    echo "✓ NPU permissions configured"
    echo "  Note: You may need to log out and back in for group changes to take effect"
}

# Function to test NPU availability
test_npu() {
    echo ""
    echo "Testing NPU availability..."
    
    # Source OpenVINO environment
    if [ -f /opt/intel/openvino/setupvars.sh ]; then
        source /opt/intel/openvino/setupvars.sh
    fi
    
    # Create test script
    cat << 'TEST_EOF' > test_npu.py
import openvino as ov

core = ov.Core()
devices = core.available_devices

print("Available devices:")
for device in devices:
    print(f"  - {device}")
    if device == "NPU":
        print("    ✓ NPU is available!")
        try:
            # Get NPU properties
            npu_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"    NPU Name: {npu_name}")
        except:
            pass

if "NPU" not in devices:
    print("\n⚠ NPU not detected. This could mean:")
    print("  - Your CPU doesn't have an NPU")
    print("  - NPU drivers are not properly installed")
    print("  - You need to reboot after installation")
TEST_EOF

    python3 test_npu.py
    rm test_npu.py
}

# Function to install Python dependencies
install_python_deps() {
    echo ""
    echo "Installing Python dependencies..."
    
    # Ensure pip is installed
    sudo apt-get install -y python3-pip python3-venv
    
    # Install OpenVINO Python packages
    pip3 install --user openvino openvino-dev[tensorflow,pytorch,onnx] openvino-genai
    
    echo "✓ Python dependencies installed"
}

# Main installation flow
main() {
    echo ""
    
    # Check CPU
    check_intel_cpu
    
    # Ask user what to install
    echo ""
    echo "What would you like to install?"
    echo "1) Full NPU support (Intel Graphics Runtime + OpenVINO)"
    echo "2) OpenVINO only (if graphics runtime already installed)"
    echo "3) Just test NPU availability"
    echo "4) Exit"
    
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            install_compute_runtime
            install_openvino_runtime
            setup_npu_permissions
            install_python_deps
            test_npu
            ;;
        2)
            install_openvino_runtime
            setup_npu_permissions
            install_python_deps
            test_npu
            ;;
        3)
            test_npu
            ;;
        4)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
    
    echo ""
    echo "==================================="
    echo "Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Log out and back in for group changes to take effect"
    echo "2. Run 'source /opt/intel/openvino/setupvars.sh' in new terminals"
    echo "3. Test NPU with: python3 -c \"import openvino as ov; print(ov.Core().available_devices)\""
    echo ""
    echo "For Intel Neural Compute Stick 2:"
    echo "  - Plug in the device"
    echo "  - Check with: lsusb | grep 03e7"
    echo "==================================="
}

# Run main function
main
