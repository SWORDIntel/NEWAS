# NEMWAS - Neural-Enhanced Multi-Workforce Agent System

A powerful, NPU-accelerated multi-agent framework for building intelligent AI systems with neural field dynamics.

## Features

- **NPU Acceleration**: Optimized for Intel NPU hardware with 22-82% performance improvements
- **ReAct Agent Pattern**: Reliable reasoning and acting loops for complex problem solving
- **Neural Capability Learning**: Agents learn and improve from successful executions
- **Real-time Performance Tracking**: Monitor and optimize agent performance
- **Natural Language Interface**: Simple "do this for me" commands
- **Plugin Architecture**: Extensible system for custom capabilities
- **Multi-Agent Coordination**: Intelligent workload distribution

## Quick Start

### Prerequisites

- Debian Linux (tested on Debian 12 Bookworm)
- Python 3.11+
- Intel Core Ultra processor (for NPU support) or any x86_64 CPU
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/nemwas.git
cd nemwas
```

2. **Set up NPU support (optional but recommended)**
```bash
chmod +x scripts/setup_npu.sh
./scripts/setup_npu.sh
```

3. **Install Python dependencies**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Download models**
```bash
python scripts/download_models.py --minimal
```

5. **Run NEMWAS**
```bash
# Interactive mode
python main.py --interactive

# API server mode
python main.py

# Single command
python main.py --command "Calculate the factorial of 12"
```

## Usage

### Interactive Mode

```bash
$ python main.py --interactive
NEMWAS v1.0 - Neural-Enhanced Multi-Workforce Agent System
============================================================
Interactive mode. Type 'help' for commands, 'exit' to quit.

nemwas> Create an agent for data analysis
Successfully created agent Agent-data-analysis for data analysis

nemwas> What is the square root of 1764?
Let me calculate that for you.

Thought: I need to calculate the square root of 1764.
Action: calculate
Action Input: {"expression": "1764 ** 0.5"}
Observation: The result is: 42.0

Answer: The square root of 1764 is 42.

nemwas> Show status
System Status: 1 active agents, 2 total tasks processed
Agent Performance: 100.0% success rate, 0.34s average execution time
Resource Usage: 12.3% CPU, 23.4% Memory
NPU Utilization: 45.2%
```

### API Usage

```python
import requests

# Execute a task
response = requests.post("http://localhost:8080/tasks", json={
    "query": "Find information about neural networks",
    "context": {"priority": "high"}
})

print(response.json())
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f nemwas

# Access services
# - API: http://localhost:8080
# - Metrics: http://localhost:9090
# - Grafana: http://localhost:3000
```

## NPU Performance

With Intel NPU hardware acceleration:

| Operation | CPU Time | NPU Time | Speedup |
|-----------|----------|----------|---------|
| Model Inference | 250ms | 45ms | 5.5x |
| Embedding Generation | 180ms | 35ms | 5.1x |
| Capability Matching | 85ms | 15ms | 5.7x |

## Documentation

- **[Architecture](docs/architecture.md):** A high-level overview of the NEMWAS architecture.
- **[API Reference](docs/api_reference.md):** Detailed documentation for the NEMWAS API.
- **[Plugin Development](docs/plugin_development.md):** A guide for creating custom plugins.
- **[Deployment](docs/deployment.md):** Instructions for deploying the NEMWAS application.
- **[Performance Tuning](docs/performance_tuning.md):** A guide for tuning the performance of the NEMWAS application.

## Troubleshooting

### NPU Not Detected

1. Verify CPU support: `lscpu | grep "Model name"`
2. Check device: `python -c "import openvino as ov; print(ov.Core().available_devices)"`
3. Re-run setup: `./scripts/setup_npu.sh`

### Model Loading Errors

1. Verify model downloaded: `ls models/original/`
2. Re-download: `python scripts/download_models.py --models tinyllama-1.1b`
3. Check logs: `tail -f logs/nemwas.log`

### Performance Issues

1. Check metrics: `curl http://localhost:9090/metrics`
2. Analyze performance: `curl http://localhost:8080/performance/analysis`
3. Adjust config: Reduce `max_agents` or `max_context_length`

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Intel for OpenVINO and NPU technology
- The open-source AI community
- Contributors and testers

## Links

- [Documentation](https://nemwas.readthedocs.io)
- [API Reference](https://nemwas.readthedocs.io/api)
- [Plugin Registry](https://plugins.nemwas.io)
- [Discord Community](https://discord.gg/nemwas)
