#!/bin/bash
# NEMWAS Quick Start Script

set -e

echo "🚀 NEMWAS Quick Start"
echo "===================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}Error: Python $required_version or higher is required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create directory structure
echo ""
echo "Creating directory structure..."
make init-dirs
echo -e "${GREEN}✓ Directories created${NC}"

# Download minimal models
echo ""
echo "Downloading models (this may take a few minutes)..."
python scripts/download_models.py --minimal
echo -e "${GREEN}✓ Models downloaded${NC}"

# Check NPU availability
echo ""
echo "Checking hardware acceleration..."
python -c "
import openvino as ov
core = ov.Core()
devices = core.available_devices
print(f'Available devices: {devices}')
if 'NPU' in devices:
    print('${GREEN}✓ NPU detected! Hardware acceleration available${NC}')
else:
    print('${YELLOW}⚠ NPU not detected. Using CPU fallback${NC}')
    print('  Run ./scripts/setup_npu.sh for NPU setup instructions')
"

# Success message
echo ""
echo -e "${GREEN}✅ NEMWAS is ready to use!${NC}"
echo ""
echo "Quick test commands:"
echo "  1. Interactive mode:  python main.py --interactive"
echo "  2. API server:        python main.py"
echo "  3. Single command:    python main.py --command \"What is 2 + 2?\""
echo "  4. Run example:       python examples/simple_example.py"
echo ""
echo "For NPU setup (Intel Core Ultra), run: ./scripts/setup_npu.sh"
echo ""
echo "Happy agent building! 🤖"
