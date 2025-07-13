#!/usr/bin/env python3
"""Setup script for NEMWAS - Neural-Enhanced Multi-Workforce Agent System"""

import os
from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Package metadata
setup(
    name="nemwas",
    version="1.0.0",
    author="NEMWAS Team",
    author_email="team@nemwas.ai",
    description="Neural-Enhanced Multi-Workforce Agent System - NPU-accelerated AI framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/nemwas",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/nemwas/issues",
        "Documentation": "https://nemwas.readthedocs.io",
        "Source Code": "https://github.com/your-org/nemwas",
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_dir={"": "."},
    include_package_data=True,
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
        "npu": [
            "openvino>=2024.0.0",
            "openvino-genai>=2024.3",
            "nncf>=2.7.0",
        ],
        "plugins": [
            "pluggy>=1.3.0",
            "importlib-metadata>=6.8.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "nemwas=main:main",
            "nemwas-server=src.api.server:run_server",
            "nemwas-download-models=scripts.download_models:main",
        ],
        "nemwas.plugins": [
            "web_search=plugins.builtin.web_search_plugin:WebSearchPlugin",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: Intel",
        "Framework :: FastAPI",
        "Natural Language :: English",
    ],
    
    # Keywords
    keywords=[
        "ai",
        "agents",
        "multi-agent",
        "npu",
        "openvino",
        "neural-processing-unit",
        "llm",
        "react-pattern",
        "artificial-intelligence",
        "machine-learning",
    ],
    
    # Package data
    package_data={
        "nemwas": [
            "config/*.yaml",
            "config/*.yml",
            "plugins/builtin/*.py",
            "data/.gitkeep",
            "models/.gitkeep",
        ],
    },
    
    # Data files
    data_files=[
        ("config", ["config/default.yaml"]),
        ("scripts", [
            "scripts/setup_npu.sh",
            "scripts/download_models.py",
        ]),
        ("docker", [
            "docker/Dockerfile",
            "docker/docker-compose.yml",
        ]),
    ],
    
    # Zip safety
    zip_safe=False,
)

# Create necessary directories after installation
def post_install():
    """Create necessary directories after installation"""
    directories = [
        "models/cache",
        "models/quantized", 
        "models/original",
        "data/capabilities",
        "data/metrics",
        "data/embeddings",
        "plugins/builtin",
        "plugins/community",
        "logs",
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            # Create .gitkeep file
            (path / ".gitkeep").touch()

# Note: post_install would need to be called via setup.py hooks
# or as part of the installation process
