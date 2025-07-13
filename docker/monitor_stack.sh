#!/bin/bash
# Monitor NEMWAS Docker Stack

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

show_menu() {
    echo ""
    echo -e "${BLUE}NEMWAS Docker Stack Monitor${NC}"
    echo "============================"
    echo "1. View all logs (follow)"
    echo "2. View NEMWAS logs only"
    echo "3. Show container status"
    echo "4. Show resource usage"
    echo "5. Test API health"
    echo "6. Show metrics sample"
    echo "7. Execute command in NEMWAS container"
    echo "8. Show NPU status"
    echo "9. Back to menu / Refresh"
    echo "0. Exit"
    echo ""
}

check_api_health() {
    echo -e "${BLUE}Checking API Health...${NC}"
    echo ""

    # NEMWAS API
    if curl -s -f http://localhost:8080/health > /dev/null; then
        echo -e "${GREEN}✓ NEMWAS API${NC}"
        health=$(curl -s http://localhost:8080/health | python3 -m json.tool 2>/dev/null || echo "Could not parse JSON")
        echo "$health" | head -20
    else
        echo -e "${RED}✗ NEMWAS API is not responding${NC}"
    fi

    echo ""

    # System Status
    echo -e "${BLUE}System Status:${NC}"
    status=$(curl -s http://localhost:8080/status | python3 -m json.tool 2>/dev/null || echo "Could not get status")
    echo "$status" | head -30
}

show_metrics_sample() {
    echo -e "${BLUE}Sample Metrics:${NC}"
    echo ""

    metrics=$(curl -s http://localhost:8080/metrics | grep -E "^nemwas_" | head -20)
    if [ -n "$metrics" ]; then
        echo "$metrics"
    else
        echo "No NEMWAS metrics available yet"
    fi
}

show_npu_status() {
    echo -e "${BLUE}NPU Status in Container:${NC}"
    echo ""

    docker exec nemwas-core python3 -c "
import openvino as ov
core = ov.Core()
devices = core.available_devices
print(f'Available devices: {devices}')
if 'NPU' in devices:
    print('✓ NPU is available!')
    try:
        print(f'NPU Name: {core.get_property(\"NPU\", \"FULL_DEVICE_NAME\")}')
    except:
        pass
else:
    print('✗ NPU not detected')
    print('  This is normal if:')
    print('  - Host system doesn\\'t have Intel NPU')
    print('  - NPU drivers not installed on host')
    print('  - Container needs restart after driver install')
"
}

# Main loop
while true; do
    show_menu
    read -p "Select option: " choice

    case $choice in
        1)
            echo -e "${BLUE}Following all logs (Ctrl+C to stop)...${NC}"
            docker-compose logs -f
            ;;
        2)
            echo -e "${BLUE}Following NEMWAS logs (Ctrl+C to stop)...${NC}"
            docker-compose logs -f nemwas
            ;;
        3)
            echo -e "${BLUE}Container Status:${NC}"
            docker-compose ps
            ;;
        4)
            echo -e "${BLUE}Resource Usage:${NC}"
            docker stats --no-stream $(docker-compose ps -q)
            ;;
        5)
            check_api_health
            read -p "Press Enter to continue..."
            ;;
        6)
            show_metrics_sample
            read -p "Press Enter to continue..."
            ;;
        7)
            echo -e "${BLUE}Enter command to execute in NEMWAS container:${NC}"
            read -p "> " cmd
            docker exec -it nemwas-core $cmd
            read -p "Press Enter to continue..."
            ;;
        8)
            show_npu_status
            read -p "Press Enter to continue..."
            ;;
        9)
            clear
            continue
            ;;
        0)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            sleep 1
            ;;
    esac
done
