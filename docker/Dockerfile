# NEMWAS Docker Image - Debian-based with NPU support
FROM debian:bookworm-slim

# Build arguments
ARG OPENVINO_VERSION=2024.0.0
ARG PYTHON_VERSION=3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # OpenVINO dependencies
    libgomp1 \
    libtbb-dev \
    libpugixml-dev \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # NPU support
    pciutils \
    usbutils \
    # System monitoring
    htop \
    iotop \
    sysstat \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install OpenVINO
RUN pip install --no-cache-dir openvino==${OPENVINO_VERSION} \
    openvino-dev[tensorflow,pytorch,onnx]==${OPENVINO_VERSION} \
    openvino-genai

# Create nemwas user
RUN useradd -m -s /bin/bash nemwas \
    && usermod -aG video,render nemwas

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p \
    /app/models/cache \
    /app/models/quantized \
    /app/models/original \
    /app/data/capabilities \
    /app/data/metrics \
    /app/data/embeddings \
    /app/plugins/builtin \
    /app/plugins/community \
    && chown -R nemwas:nemwas /app

# Download default models (can be overridden with volume mount)
RUN python scripts/download_models.py --minimal

# Setup NPU permissions (will be configured at runtime)
COPY scripts/setup_npu_permissions.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/setup_npu_permissions.sh

# Expose ports
EXPOSE 8080  # API
EXPOSE 9090  # Metrics
EXPOSE 8081  # Web UI

# Switch to nemwas user
USER nemwas

# Environment variables
ENV PYTHONPATH=/app
ENV OPENVINO_LOG_LEVEL=WARNING
ENV OMP_NUM_THREADS=4
ENV TBB_NUM_THREADS=4

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Default command
CMD ["python", "main.py", "--config", "config/default.yaml"]
