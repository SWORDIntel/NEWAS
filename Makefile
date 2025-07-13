# NEMWAS Makefile

.PHONY: help install setup-npu download-models test lint format clean run run-interactive docker-build docker-up docker-down

# Default target
help:
	@echo "NEMWAS - Neural-Enhanced Multi-Workforce Agent System"
	@echo "===================================================="
	@echo ""
	@echo "Available targets:"
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
	@echo "  make clean           - Clean temporary files"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install --upgrade pip
	pip install -r requirements.txt
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

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --cov=src --cov-report=html
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

# Run NEMWAS API server
run:
	@echo "Starting NEMWAS API server..."
	python main.py --config config/default.yaml

# Run NEMWAS in interactive mode
run-interactive:
	@echo "Starting NEMWAS interactive mode..."
	python main.py --config config/default.yaml --interactive

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

# Run benchmarks
benchmark:
	@echo "Running benchmarks..."
	python tests/benchmarks/run_benchmarks.py
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
