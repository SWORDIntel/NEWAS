#!/bin/bash
# Start NEMWAS Docker Stack

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Starting NEMWAS Docker Stack"
echo "=============================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo "Please start Docker daemon"
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p grafana/dashboards grafana/datasources
mkdir -p ../models/original ../models/cache ../models/quantized
mkdir -p ../data/capabilities ../data/metrics ../data/embeddings
mkdir -p ../plugins/community

# Copy configuration files if they don't exist
if [ ! -f "prometheus.yml" ]; then
    echo "Creating prometheus.yml..."
    cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'nemwas'
    static_configs:
      - targets: ['nemwas:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF
fi

# Check if models exist
if [ ! -f "../models/original/tinyllama-1.1b-chat.xml" ]; then
    echo -e "${YELLOW}Warning: Models not found${NC}"
    echo "Would you like to download models now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cd ..
        python scripts/download_models.py --minimal
        cd docker
    else
        echo "You can download models later with: python scripts/download_models.py"
    fi
fi

# Build images
echo ""
echo "Building Docker images..."
docker-compose build

# Start services
echo ""
echo "Starting services..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."

# Check NEMWAS
if curl -f http://localhost:8080/health &> /dev/null; then
    echo -e "${GREEN}✓ NEMWAS API is healthy${NC}"
else
    echo -e "${RED}✗ NEMWAS API is not responding${NC}"
fi

# Check Prometheus
if curl -f http://localhost:9091/-/healthy &> /dev/null; then
    echo -e "${GREEN}✓ Prometheus is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Prometheus is not responding${NC}"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health &> /dev/null; then
    echo -e "${GREEN}✓ Grafana is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Grafana is not responding${NC}"
fi

# Check Redis
if docker exec nemwas-redis redis-cli ping &> /dev/null; then
    echo -e "${GREEN}✓ Redis is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Redis is not responding${NC}"
fi

# Display access information
echo ""
echo "====================================="
echo -e "${GREEN}NEMWAS Stack is running!${NC}"
echo "====================================="
echo ""
echo "Access points:"
echo "  • NEMWAS API:    http://localhost:8080"
echo "  • API Docs:      http://localhost:8080/docs"
echo "  • Prometheus:    http://localhost:9091"
echo "  • Grafana:       http://localhost:3000 (admin/admin)"
echo "  • Redis:         localhost:6379"
echo ""
echo "Useful commands:"
echo "  • View logs:     docker-compose logs -f nemwas"
echo "  • Stop stack:    docker-compose down"
echo "  • View metrics:  curl http://localhost:8080/metrics"
echo "  • Test API:      curl http://localhost:8080/status"
echo ""
echo "To test NEMWAS:"
echo "  curl -X POST http://localhost:8080/tasks \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"What is 2 + 2?\"}'"
echo ""
