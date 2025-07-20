#!/bin/bash
# NEMWAS Quick Start Script with Advanced Environment Detection

set -e

echo "🚀 NEMWAS Quick Start"
echo "===================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATASCIENCE_ENV_PATH="$HOME/datascience/envs/dsenv"
USE_ADVANCED=false
USE_LOCAL=false
AUTO_DETECT=false
SKIP_PROMPT=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --use-advanced) USE_ADVANCED=true; SKIP_PROMPT=true ;;
        --use-local) USE_LOCAL=true; SKIP_PROMPT=true ;;
        --auto) AUTO_DETECT=true; SKIP_PROMPT=true ;;
        -h|--help) 
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --use-advanced    Force use of advanced datascience environment"
            echo "  --use-local       Force use of local virtual environment"
            echo "  --auto            Auto-detect best environment without prompting"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Function to detect advanced environment
detect_advanced_environment() {
    local env_status="not_found"
    local env_details=""
    
    if [ -d "$DATASCIENCE_ENV_PATH" ] && [ -f "$DATASCIENCE_ENV_PATH/bin/activate" ]; then
        # Check OpenVINO version in advanced environment
        local openvino_version=$(
            source "$DATASCIENCE_ENV_PATH/bin/activate" 2>/dev/null && 
            python -c "import openvino; print(openvino.__version__)" 2>/dev/null || echo "none"
        )
        
        if [ "$openvino_version" != "none" ]; then
            env_status="found"
            env_details="OpenVINO $openvino_version"
            
            # Check for additional features
            if command -v nvidia-smi >/dev/null 2>&1; then
                env_details="$env_details, CUDA available"
            fi
            
            # Check for NPU support
            if [ -e "/dev/accel/accel0" ]; then
                env_details="$env_details, NPU detected"
            fi
        fi
    fi
    
    echo "$env_status|$env_details"
}

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo -e "${RED}Error: Python $required_version or higher is required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $python_version${NC}"

# Detect advanced environment
echo ""
echo "Detecting environments..."
ADVANCED_ENV_INFO=$(detect_advanced_environment)
ADVANCED_ENV_STATUS=$(echo "$ADVANCED_ENV_INFO" | cut -d'|' -f1)
ADVANCED_ENV_DETAILS=$(echo "$ADVANCED_ENV_INFO" | cut -d'|' -f2)

# Determine which environment to use
USE_ENV="local"  # default

if [ "$USE_ADVANCED" = true ]; then
    if [ "$ADVANCED_ENV_STATUS" = "found" ]; then
        USE_ENV="advanced"
    else
        echo -e "${RED}Error: Advanced environment requested but not found${NC}"
        exit 1
    fi
elif [ "$USE_LOCAL" = true ]; then
    USE_ENV="local"
elif [ "$AUTO_DETECT" = true ]; then
    if [ "$ADVANCED_ENV_STATUS" = "found" ]; then
        USE_ENV="advanced"
        echo -e "${BLUE}Auto-detected advanced environment${NC}"
    fi
elif [ "$SKIP_PROMPT" = false ] && [ "$ADVANCED_ENV_STATUS" = "found" ]; then
    # Prompt user for choice
    echo -e "${BLUE}Advanced data science environment detected!${NC}"
    echo "  Location: $DATASCIENCE_ENV_PATH"
    echo "  Features: $ADVANCED_ENV_DETAILS"
    echo ""
    echo "Which environment would you like to use?"
    echo "  1) Advanced datascience environment (recommended if available)"
    echo "  2) Create/use local virtual environment"
    echo "  3) Auto-detect (use advanced if suitable)"
    echo ""
    read -p "Enter choice [1-3] (default: 3): " -n 1 -r env_choice
    echo ""
    
    case "$env_choice" in
        1) USE_ENV="advanced" ;;
        2) USE_ENV="local" ;;
        3|"")
            if [ "$ADVANCED_ENV_STATUS" = "found" ]; then
                USE_ENV="advanced"
            else
                USE_ENV="local"
            fi
            ;;
        *) echo -e "${YELLOW}Invalid choice, using auto-detect${NC}"
           USE_ENV="advanced" ;;
    esac
fi

# Save environment choice
echo "USE_ENV=$USE_ENV" > .env.quickstart

# Setup environment based on choice
if [ "$USE_ENV" = "advanced" ]; then
    echo ""
    echo -e "${BLUE}Using advanced datascience environment${NC}"
    echo "  Features: $ADVANCED_ENV_DETAILS"
    
    # Activate advanced environment
    source "$DATASCIENCE_ENV_PATH/bin/activate"
    echo -e "${GREEN}✓ Advanced environment activated${NC}"
    
    # Export environment variables for advanced features
    export NEMWAS_USE_ADVANCED_ENV=1
    export OV_NPU_PLATFORM=3800
    export OPENVINO_ENABLE_NPU=1
    
else
    # Create/use local virtual environment
    echo ""
    echo "Using local virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${YELLOW}Virtual environment already exists${NC}"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."

if [ "$USE_ENV" = "advanced" ]; then
    # Check for missing dependencies in advanced environment
    echo "Checking dependencies in advanced environment..."
    pip install --quiet --upgrade pip
    
    # Install only missing packages
    if [ -f "scripts/install_missing_deps.py" ]; then
        python scripts/install_missing_deps.py requirements.txt
    else
        # Fallback to regular install
        pip install --quiet -r requirements.txt
    fi
else
    # Regular installation for local environment
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
fi

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

if [ "$USE_ENV" = "advanced" ]; then
    # Advanced environment may have additional optimizations
    python -c "
import openvino as ov
import os

print('Advanced OpenVINO Configuration:')
print(f'  Version: {ov.__version__}')

core = ov.Core()
devices = core.available_devices
print(f'  Available devices: {devices}')

# Check NPU
if 'NPU' in devices:
    print('${GREEN}✓ NPU detected! Hardware acceleration available${NC}')
    npu_props = core.get_property('NPU', 'SUPPORTED_PROPERTIES')
    print(f'  NPU Properties available: {len(npu_props)} properties')
else:
    print('${YELLOW}⚠ NPU not detected. Using CPU fallback${NC}')
    
# Check GPU
if 'GPU' in devices:
    print('${GREEN}✓ GPU detected for additional acceleration${NC}')
    
# Check for P-core/E-core optimization
if os.environ.get('NEMWAS_USE_ADVANCED_ENV'):
    print('${BLUE}✓ P-core/E-core NumPy optimization available${NC}')
    print('  Use np-p for P-core optimized runs')
    print('  Use np-e for E-core optimized runs')
"
else
    # Standard NPU check for local environment
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
fi

# Success message
echo ""
echo -e "${GREEN}✅ NEMWAS is ready to use!${NC}"
echo ""

if [ "$USE_ENV" = "advanced" ]; then
    echo -e "${BLUE}Using Advanced Environment:${NC}"
    echo "  - $ADVANCED_ENV_DETAILS"
    echo "  - Enhanced performance optimizations"
    echo "  - P-core/E-core NumPy switching available"
    echo ""
    echo "Quick test commands (with advanced features):"
    echo "  1. Interactive mode:  python main.py --interactive"
    echo "  2. API server:        python main.py"
    echo "  3. P-core optimized:  np-p main.py --interactive"
    echo "  4. E-core optimized:  np-e main.py --interactive"
    echo "  5. Benchmark:         make benchmark"
else
    echo "Quick test commands:"
    echo "  1. Interactive mode:  python main.py --interactive"
    echo "  2. API server:        python main.py"
    echo "  3. Single command:    python main.py --command \"What is 2 + 2?\""
    echo "  4. Run example:       python examples/simple_example.py"
    echo ""
    echo "For NPU setup (Intel Core Ultra), run: ./scripts/setup_npu.sh"
fi

echo ""
echo "Environment Management:"
echo "  - Switch to advanced: make use-datascience-env"
echo "  - Switch to local:    make use-local-env"
echo "  - Check status:       make env-info"
echo ""
echo "Happy agent building! 🤖"