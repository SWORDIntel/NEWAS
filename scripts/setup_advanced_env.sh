#!/bin/bash
# Setup script for creating an advanced data science environment

set -e

echo "🔧 Advanced Data Science Environment Setup"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DATASCIENCE_DIR="$HOME/datascience"
VENV_NAME="venv"

# Check if environment already exists
if [ -d "$DATASCIENCE_DIR" ]; then
    echo -e "${YELLOW}Warning: $DATASCIENCE_DIR already exists${NC}"
    read -p "Do you want to continue and potentially overwrite? (y/N): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
fi

# Create directory structure
echo "Creating directory structure..."
mkdir -p "$DATASCIENCE_DIR"/{notebooks,data,models,scripts}
echo -e "${GREEN}✓ Directory structure created${NC}"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
cd "$DATASCIENCE_DIR"
python3 -m venv "$VENV_NAME"
echo -e "${GREEN}✓ Virtual environment created${NC}"

# Activate environment
source "$VENV_NAME/bin/activate"

# Install base packages
echo ""
echo "Installing base data science packages..."
pip install --upgrade pip setuptools wheel

# Core data science stack
pip install numpy pandas matplotlib seaborn jupyter scikit-learn

# Deep learning frameworks (optional)
echo ""
read -p "Install deep learning frameworks (PyTorch, TensorFlow)? (y/N): " install_dl
if [[ "$install_dl" =~ ^[Yy]$ ]]; then
    echo "Installing deep learning frameworks..."
    
    # Check for CUDA
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}CUDA detected - installing GPU versions${NC}"
        # PyTorch with CUDA
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        # TensorFlow with CUDA
        pip install tensorflow[and-cuda]
    else
        echo -e "${YELLOW}No CUDA detected - installing CPU versions${NC}"
        pip install torch torchvision torchaudio
        pip install tensorflow
    fi
fi

# Additional tools
echo ""
echo "Installing additional tools..."
pip install ipykernel notebook jupyterlab
pip install plotly dash streamlit
pip install opencv-python pillow
pip install nltk spacy transformers

echo -e "${GREEN}✓ Packages installed${NC}"

# Create example notebook
echo ""
echo "Creating example notebook..."
cat > "$DATASCIENCE_DIR/notebooks/welcome.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Welcome to Your Advanced Data Science Environment\n",
    "\n",
    "This environment is now integrated with NEMWAS!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "print(f\"Python: {sys.version}\")\n",
    "print(f\"NumPy: {np.__version__}\")\n",
    "print(f\"Pandas: {pd.__version__}\")\n",
    "print(f\"PyTorch: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create activation script
echo ""
echo "Creating activation script..."
cat > "$DATASCIENCE_DIR/activate.sh" << EOF
#!/bin/bash
# Quick activation script for data science environment

echo "Activating data science environment..."
source "$DATASCIENCE_DIR/$VENV_NAME/bin/activate"
echo "Environment activated!"
echo "Python: \$(which python)"
echo "Jupyter: \$(which jupyter)"
EOF
chmod +x "$DATASCIENCE_DIR/activate.sh"

# Create README
cat > "$DATASCIENCE_DIR/README.md" << EOF
# Advanced Data Science Environment

This environment is configured to work with NEMWAS and provides:

- Python virtual environment with data science packages
- Jupyter Lab for interactive development
- Deep learning frameworks (if installed)
- Integration with NEMWAS quickstart

## Quick Start

1. Activate the environment:
   \`\`\`bash
   source ~/datascience/activate.sh
   \`\`\`

2. Start Jupyter Lab:
   \`\`\`bash
   jupyter lab
   \`\`\`

3. Use with NEMWAS:
   \`\`\`bash
   cd /path/to/NEMWAS
   ./quickstart.sh --use-advanced
   \`\`\`

## Directory Structure

- \`notebooks/\` - Jupyter notebooks
- \`data/\` - Datasets and data files
- \`models/\` - Trained models and checkpoints
- \`scripts/\` - Python scripts and utilities
- \`venv/\` - Python virtual environment

## Installed Packages

Core: numpy, pandas, matplotlib, seaborn, jupyter, scikit-learn
Optional: pytorch, tensorflow, transformers
Tools: jupyterlab, plotly, streamlit
EOF

# Success message
echo ""
echo -e "${GREEN}✅ Advanced data science environment setup complete!${NC}"
echo ""
echo "Environment location: $DATASCIENCE_DIR"
echo ""
echo "To activate this environment:"
echo "  source $DATASCIENCE_DIR/activate.sh"
echo ""
echo "To use with NEMWAS:"
echo "  cd /path/to/NEMWAS"
echo "  ./quickstart.sh --use-advanced"
echo ""
echo -e "${BLUE}Happy data science! 📊🔬${NC}"