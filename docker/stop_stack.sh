#!/bin/bash
# Stop NEMWAS Docker Stack

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🛑 Stopping NEMWAS Docker Stack"
echo "=============================="

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Ask for confirmation
echo -e "${YELLOW}This will stop all NEMWAS services.${NC}"
read -p "Continue? (y/n) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Stop services
echo ""
echo "Stopping services..."
docker-compose down

# Ask about volumes
echo ""
echo -e "${YELLOW}Do you want to remove data volumes as well?${NC}"
echo "This will delete all models, metrics, and application data!"
read -p "Remove volumes? (y/n) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing volumes..."
    docker-compose down -v
    echo -e "${GREEN}✓ All volumes removed${NC}"
else
    echo "Volumes preserved. Data will persist for next run."
fi

# Clean up unused resources
echo ""
echo "Cleaning up unused Docker resources..."
docker system prune -f

echo ""
echo -e "${GREEN}✓ NEMWAS stack stopped successfully${NC}"
echo ""
echo "To restart the stack, run: ./start-stack.sh"
