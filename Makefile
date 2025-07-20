# NEMWAS Makefile

.PHONY: help install setup-npu download-models test lint format clean run run-interactive docker-build docker-up docker-down

# ========== Environment Detection ==========
# Advanced environment paths
DATASCIENCE_ENV_PATH := $(HOME)/datascience/envs/dsenv
DATASCIENCE_VENV_PATH := $(DATASCIENCE_ENV_PATH)/bin/activate
SYSTEM_PYTHON := python3
PYTHON := python

# Check if advanced datascience environment exists
HAS_DATASCIENCE_ENV := $(shell test -d "$(DATASCIENCE_ENV_PATH)" && echo "yes" || echo "no")
HAS_DATASCIENCE_VENV := $(shell test -f "$(DATASCIENCE_VENV_PATH)" && echo "yes" || echo "no")

# OpenVINO Detection
OPENVINO_ADVANCED_VERSION := 2025.0.0
OPENVINO_MIN_VERSION := 2024.0.0

# Detect OpenVINO in datascience environment
ifeq ($(HAS_DATASCIENCE_VENV),yes)
    DATASCIENCE_OPENVINO_VERSION := $(shell . $(DATASCIENCE_VENV_PATH) && python -c "import openvino; print(openvino.__version__)" 2>/dev/null || echo "none")
    HAS_ADVANCED_OPENVINO := $(shell . $(DATASCIENCE_VENV_PATH) && python -c "import openvino; v = openvino.__version__.split('-')[0]; print('yes' if v >= '$(OPENVINO_MIN_VERSION)' else 'no')" 2>/dev/null || echo "no")
else
    DATASCIENCE_OPENVINO_VERSION := none
    HAS_ADVANCED_OPENVINO := no
endif

# Current environment OpenVINO detection
CURRENT_OPENVINO_VERSION := $(shell $(PYTHON) -c "import openvino; print(openvino.__version__)" 2>/dev/null || echo "none")

# Environment selection logic
ifeq ($(HAS_ADVANCED_OPENVINO),yes)
    USE_DATASCIENCE_ENV := yes
    VENV_ACTIVATE := . $(DATASCIENCE_VENV_PATH) &&
    ENV_PYTHON := $(DATASCIENCE_ENV_PATH)/bin/python
else
    USE_DATASCIENCE_ENV := no
    VENV_ACTIVATE := 
    ENV_PYTHON := $(PYTHON)
endif

# Override with environment variable if set
ifdef NEMWAS_USE_LOCAL_ENV
    USE_DATASCIENCE_ENV := no
    VENV_ACTIVATE := . venv/bin/activate &&
    ENV_PYTHON := venv/bin/python
endif

# Default target
help:
	@echo "NEMWAS - Neural-Enhanced Multi-Workforce Agent System"
	@echo "===================================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make quickstart       - Quick setup with environment detection"
	@echo "  make quickstart-advanced - Quick setup using advanced environment"
	@echo "  make quickstart-local - Quick setup using local environment"
	@echo "  make install          - Install Python dependencies"
	@echo "  make setup-npu        - Setup NPU support (Debian)"
	@echo "  make download-models  - Download required models"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linting"
	@echo "  make format          - Format code"
	@echo "  make run             - Run NEMWAS API server"
	@echo "  make run-interactive - Run NEMWAS in interactive mode"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-up       - Start Docker services"
	@echo "  make docker-down     - Stop Docker services"
	@echo "  make check-env       - Check for advanced environment"
	@echo "  make clean           - Clean temporary files"

# Display environment information
.PHONY: env-info
env-info:
	@echo "=== Environment Detection Results ==="
	@echo "Datascience environment path: $(DATASCIENCE_ENV_PATH)"
	@echo "Datascience environment exists: $(HAS_DATASCIENCE_ENV)"
	@echo "Datascience venv exists: $(HAS_DATASCIENCE_VENV)"
	@echo "Datascience OpenVINO version: $(DATASCIENCE_OPENVINO_VERSION)"
	@echo "Has advanced OpenVINO: $(HAS_ADVANCED_OPENVINO)"
	@echo "Current OpenVINO version: $(CURRENT_OPENVINO_VERSION)"
	@echo "Using datascience environment: $(USE_DATASCIENCE_ENV)"
	@echo "Python executable: $(ENV_PYTHON)"
	@echo ""
	@if [ "$(USE_DATASCIENCE_ENV)" = "yes" ]; then \
		echo "✓ Using advanced datascience environment with OpenVINO $(DATASCIENCE_OPENVINO_VERSION)"; \
	else \
		echo "✗ Using local environment (datascience env not suitable)"; \
	fi

# Install dependencies with environment detection
install: env-info
ifeq ($(USE_DATASCIENCE_ENV),yes)
	@echo "Using datascience environment - checking additional dependencies..."
	@$(VENV_ACTIVATE) $(ENV_PYTHON) -m pip install --upgrade pip
	@# Install only missing dependencies
	@$(VENV_ACTIVATE) $(ENV_PYTHON) scripts/install_missing_deps.py requirements.txt || \
		(echo "Installing all requirements..." && $(VENV_ACTIVATE) $(ENV_PYTHON) -m pip install -r requirements.txt)
else
	@echo "Installing Python dependencies in local environment..."
	@if [ -d "venv" ]; then \
		. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt; \
	else \
		pip install --upgrade pip && pip install -r requirements.txt; \
	fi
endif
	@echo "✓ Dependencies installed"

# Setup NPU support
setup-npu:
	@echo "Setting up NPU support..."
	chmod +x scripts/setup_npu.sh
	./scripts/setup_npu.sh
	@echo "✓ NPU setup complete"

# Download models
download-models:
	@echo "Downloading models..."
	python scripts/download_models.py --minimal --optimize-npu
	@echo "✓ Models downloaded"

# Download all models
download-models-all:
	@echo "Downloading all models..."
	python scripts/download_models.py --models all --optimize-npu
	@echo "✓ All models downloaded"

# Run tests with environment detection
test:
	@echo "Running tests..."
ifeq ($(USE_DATASCIENCE_ENV),yes)
	@$(VENV_ACTIVATE) $(ENV_PYTHON) -m pytest tests/ -v --cov=src --cov-report=html
else
	@if [ -d "venv" ]; then \
		. venv/bin/activate && python -m pytest tests/ -v --cov=src --cov-report=html; \
	else \
		python -m pytest tests/ -v --cov=src --cov-report=html; \
	fi
endif
	@echo "✓ Tests complete"

# Run specific test
test-one:
	@echo "Running test: $(TEST)"
	python -m pytest tests/$(TEST) -v

# Lint code
lint:
	@echo "Running linters..."
	flake8 src/ tests/ --max-line-length=120 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	@echo "✓ Linting complete"

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/ scripts/ *.py
	isort src/ tests/ scripts/ *.py
	@echo "✓ Code formatted"

# Run NEMWAS API server with environment detection
run:
	@echo "Starting NEMWAS API server..."
ifeq ($(USE_DATASCIENCE_ENV),yes)
	@echo "Using advanced OpenVINO $(DATASCIENCE_OPENVINO_VERSION) from datascience environment"
	@$(VENV_ACTIVATE) $(ENV_PYTHON) main.py --config config/default.yaml
else
	@if [ -d "venv" ]; then \
		. venv/bin/activate && python main.py --config config/default.yaml; \
	else \
		python main.py --config config/default.yaml; \
	fi
endif

# Run NEMWAS in interactive mode with environment detection
run-interactive:
	@echo "Starting NEMWAS interactive mode..."
ifeq ($(USE_DATASCIENCE_ENV),yes)
	@echo "Using advanced OpenVINO $(DATASCIENCE_OPENVINO_VERSION) from datascience environment"
	@$(VENV_ACTIVATE) $(ENV_PYTHON) main.py --config config/default.yaml --interactive
else
	@if [ -d "venv" ]; then \
		. venv/bin/activate && python main.py --config config/default.yaml --interactive; \
	else \
		python main.py --config config/default.yaml --interactive; \
	fi
endif

# Run with custom config
run-config:
	@echo "Starting NEMWAS with config: $(CONFIG)"
	python main.py --config $(CONFIG)

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -f docker/Dockerfile -t nemwas:latest .
	@echo "✓ Docker image built"

# Start Docker services
docker-up:
	@echo "Starting Docker services..."
	docker-compose -f docker/docker-compose.yml up -d
	@echo "✓ Services started"
	@echo "  API: http://localhost:8080"
	@echo "  Prometheus: http://localhost:9091"
	@echo "  Grafana: http://localhost:3000"

# Stop Docker services
docker-down:
	@echo "Stopping Docker services..."
	docker-compose -f docker/docker-compose.yml down
	@echo "✓ Services stopped"

# View Docker logs
docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f nemwas

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "✓ Cleanup complete"

# Create necessary directories
init-dirs:
	@echo "Creating directory structure..."
	mkdir -p models/{cache,quantized,original}
	mkdir -p data/{capabilities,metrics,embeddings}
	mkdir -p plugins/{builtin,community}
	mkdir -p logs
	@echo "✓ Directories created"

# Full setup from scratch
setup: init-dirs install download-models
	@echo "✓ NEMWAS setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make setup-npu' if you have Intel NPU hardware"
	@echo "2. Run 'make run' to start the API server"
	@echo "3. Or run 'make run-interactive' for interactive mode"

# Development setup
dev-setup: setup
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "✓ Development environment ready"

# Run benchmarks with environment detection
benchmark:
	@echo "Running benchmarks..."
ifeq ($(USE_DATASCIENCE_ENV),yes)
	@echo "Benchmarking with advanced OpenVINO optimizations..."
	@$(VENV_ACTIVATE) $(ENV_PYTHON) tests/benchmarks/run_benchmarks.py
else
	@if [ -d "venv" ]; then \
		. venv/bin/activate && python tests/benchmarks/run_benchmarks.py; \
	else \
		python tests/benchmarks/run_benchmarks.py; \
	fi
endif
	@echo "✓ Benchmarks complete"

# Generate documentation
docs:
	@echo "Generating documentation..."
	cd docs && make html
	@echo "✓ Documentation generated"
	@echo "View at: docs/_build/html/index.html"

# Check system compatibility
check-system:
	@echo "Checking system compatibility..."
	@python scripts/check_system.py

# Check for advanced environment
check-env:
	@echo "Checking for advanced data science environment..."
	@if [ -d "$$HOME/datascience" ]; then \
		echo "✓ Advanced environment detected at $$HOME/datascience"; \
		if [ -d "$$HOME/datascience/venv" ] || [ -d "$$HOME/datascience/env" ] || [ -d "$$HOME/datascience/.venv" ]; then \
			echo "  - Virtual environment found"; \
		fi; \
		if [ -d "/usr/local/cuda" ] || [ -n "$$CUDA_HOME" ]; then \
			echo "  - CUDA support available"; \
		fi; \
		if command -v jupyter >/dev/null 2>&1; then \
			echo "  - Jupyter available"; \
		fi; \
	else \
		echo "⚠ Advanced environment not found"; \
		echo "  Using local virtual environment"; \
	fi

# Quick start with environment detection
quickstart:
	@chmod +x quickstart.sh
	@./quickstart.sh

# Quick start with advanced environment
quickstart-advanced:
	@chmod +x quickstart.sh
	@./quickstart.sh --use-advanced

# Quick start with local environment
quickstart-local:
	@chmod +x quickstart.sh
	@./quickstart.sh --use-local

# Monitor system
monitor:
	@echo "Starting system monitor..."
	@watch -n 1 'curl -s http://localhost:8080/status | python -m json.tool'

# Quick test command
quick-test:
	@echo "Running quick test..."
	python main.py --command "What is 2 + 2?"

# Version info
version:
	@echo "NEMWAS Version Information:"
	@python -c "import src; print(f'NEMWAS: {src.__version__}')"
	@python -c "import openvino as ov; print(f'OpenVINO: {ov.__version__}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt
	pip freeze > requirements.lock
	@echo "✓ Dependencies updated"

# Switch to datascience environment
.PHONY: use-datascience-env
use-datascience-env:
	@if [ "$(HAS_ADVANCED_OPENVINO)" = "yes" ]; then \
		echo "export NEMWAS_USE_DATASCIENCE_ENV=1" > .env.local; \
		echo "✓ Configured to use datascience environment"; \
		echo "  OpenVINO version: $(DATASCIENCE_OPENVINO_VERSION)"; \
		echo "  Run 'make env-info' to verify"; \
	else \
		echo "✗ Cannot use datascience environment:"; \
		if [ "$(HAS_DATASCIENCE_VENV)" = "no" ]; then \
			echo "  - Environment not found at $(DATASCIENCE_ENV_PATH)"; \
		else \
			echo "  - OpenVINO version $(DATASCIENCE_OPENVINO_VERSION) is not advanced enough"; \
			echo "  - Required: >= $(OPENVINO_MIN_VERSION)"; \
		fi; \
	fi

# Switch to local environment
.PHONY: use-local-env
use-local-env:
	@rm -f .env.local
	@echo "export NEMWAS_USE_LOCAL_ENV=1" > .env.local
	@echo "✓ Configured to use local environment"
	@if [ ! -d "venv" ]; then \
		echo "  Run 'make setup' to create local environment"; \
	fi

# Setup environment conditionally
.PHONY: setup-env
setup-env:
ifeq ($(USE_DATASCIENCE_ENV),yes)
	@echo "Using existing datascience environment at $(DATASCIENCE_ENV_PATH)"
	@echo "Checking compatibility..."
	@$(VENV_ACTIVATE) $(ENV_PYTHON) scripts/check_datascience_env.py || \
		(echo "Compatibility check failed, creating local environment" && \
		$(SYSTEM_PYTHON) -m venv venv && \
		echo "✓ Local virtual environment created as fallback")
else
	@echo "Creating new virtual environment..."
	@if [ ! -d "venv" ]; then \
		$(SYSTEM_PYTHON) -m venv venv; \
		echo "✓ Virtual environment created"; \
	else \
		echo "✓ Virtual environment already exists"; \
	fi
endif
