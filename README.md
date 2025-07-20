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

## Command-Line Interface

### Basic Usage

```bash
# Start NEMWAS API server (default mode)
python main.py

# Start with custom configuration
python main.py --config config/production.yaml

# Interactive mode
python main.py --interactive

# Execute single command
python main.py --command "Calculate the factorial of 12"
```

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--config` | `-c` | Path to configuration file | `config/default.yaml` |
| `--interactive` | `-i` | Start in interactive mode | False |
| `--command` | | Execute single command and exit | None |

### Model Download Script

Download and optimize models for NPU deployment:

```bash
# Download minimal set for quick start
python scripts/download_models.py --minimal

# Download specific models
python scripts/download_models.py --models tinyllama-1.1b mistral-7b

# Download and optimize for NPU
python scripts/download_models.py --optimize-npu

# List available models
python scripts/download_models.py --list

# Specify output directory
python scripts/download_models.py --output-dir /path/to/models
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

## Model Configuration

### Supported Models

NEMWAS supports various models optimized for NPU deployment:

| Model | Size | Purpose | NPU Optimized | FP16 Support |
|-------|------|---------|---------------|--------------|
| TinyLlama-1.1B | 1.1B | Default LLM, chat | ✅ | ✅ |
| Mistral-7B | 7B | Advanced reasoning | ✅ | ✅ |
| CodeBERT-base | 125M | Code understanding | ✅ | ✅ |
| all-MiniLM-L6-v2 | 22M | Embeddings | ✅ | ✅ |

### Model Configuration Options

Configure models in `config/default.yaml`:

```yaml
models:
  # Default model selection
  default_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  default_model_path: "./models/original/tinyllama-1.1b-chat.xml"
  
  # Model cache directory
  model_cache_dir: "./models/cache"
  
  # Quantization settings for NPU optimization
  quantization_preset: "mixed"  # Options: performance, mixed, accuracy
```

### Quantization Presets

| Preset | Description | Use Case | Performance | Accuracy |
|--------|-------------|----------|-------------|----------|
| `performance` | Maximum speed optimization | Production, real-time | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `mixed` | Balanced optimization | Default, general use | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| `accuracy` | Preserve model accuracy | Research, critical tasks | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### FP16 Format Handling

NEMWAS automatically handles FP16 (half-precision) format conversion for optimal NPU performance:

1. **Automatic Conversion**: Models are converted to FP16 during OpenVINO optimization
2. **Memory Efficiency**: FP16 reduces model size by ~50% with minimal accuracy loss
3. **NPU Acceleration**: Intel NPU hardware has native FP16 support for faster inference

```bash
# Models are automatically converted to FP16 during download
python scripts/download_models.py --optimize-npu

# The conversion process includes:
# 1. Download original model (FP32)
# 2. Convert to OpenVINO IR format
# 3. Apply FP16 compression (--compress_to_fp16 flag)
# 4. Apply NPU-specific optimizations
```

### Model Directory Structure

```
models/
├── original/          # Original downloaded models
│   ├── tinyllama-1.1b-chat.bin
│   └── mistral-7b.safetensors
├── openvino/         # OpenVINO IR format (FP16)
│   ├── tinyllama-1.1b-chat.xml
│   └── tinyllama-1.1b-chat.bin
├── cache/            # NPU-optimized cache
│   └── npu_compiled_models/
└── quantized/        # Additional quantization
    └── int8_models/
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

## Advanced Usage

### Using Different Models

```python
# Configure specific model for an agent
from src.core.agent import AgentConfig, NEMWASAgent
from src.core.npu_manager import NPUManager

# For code analysis tasks
code_agent_config = AgentConfig(
    name="Code-Expert",
    model_path="./models/openvino/codebert-base.xml",
    device_preference=["NPU", "CPU"],
    quantization_preset="accuracy"  # Prioritize accuracy for code
)

# For fast responses
speed_agent_config = AgentConfig(
    name="Speed-Agent", 
    model_path="./models/openvino/tinyllama-1.1b.xml",
    device_preference=["NPU"],
    quantization_preset="performance"  # Maximum speed
)

# For complex reasoning
reasoning_agent_config = AgentConfig(
    name="Reasoning-Expert",
    model_path="./models/openvino/mistral-7b.xml", 
    device_preference=["NPU", "GPU", "CPU"],
    quantization_preset="mixed"  # Balance speed and accuracy
)
```

### Multi-Agent Workflows

```python
# Create specialized agents for complex tasks
async def analyze_codebase(project_path):
    # Code analysis agent
    code_agent = await core.create_agent(
        "analyze code structure and patterns",
        model="codebert-base"
    )
    
    # Documentation agent
    doc_agent = await core.create_agent(
        "generate documentation",
        model="tinyllama-1.1b"
    )
    
    # Security agent
    security_agent = await core.create_agent(
        "scan for security vulnerabilities",
        model="mistral-7b"
    )
    
    # Coordinate agents
    results = await core.coordinate_agents([
        (code_agent, "Analyze the codebase architecture"),
        (security_agent, "Identify security issues"),
        (doc_agent, "Generate API documentation")
    ])
    
    return results
```

### Custom Plugin Development

```python
# Create a custom plugin for specific tasks
from src.plugins.interface import ToolPlugin

class DataAnalysisPlugin(ToolPlugin):
    def __init__(self):
        super().__init__(
            name="data_analysis",
            version="1.0.0",
            description="Advanced data analysis tools"
        )
    
    def get_tool_definition(self):
        return {
            "name": "analyze_dataset",
            "description": "Analyze dataset with NPU acceleration",
            "function": self.analyze_dataset,
            "parameters": {
                "dataset_path": "Path to dataset",
                "analysis_type": "Type of analysis"
            }
        }
    
    async def analyze_dataset(self, dataset_path, analysis_type):
        # Use NPU-accelerated operations
        if self.npu_manager.has_npu():
            # Fast NPU path
            return await self._npu_analysis(dataset_path, analysis_type)
        else:
            # CPU fallback
            return await self._cpu_analysis(dataset_path, analysis_type)
```

### Performance Optimization Tips

1. **Model Selection**
   - Use TinyLlama for general tasks (fastest)
   - Use Mistral-7B for complex reasoning
   - Use CodeBERT for code-specific tasks

2. **Quantization Settings**
   ```yaml
   # For real-time applications
   quantization_preset: "performance"
   compilation_mode: "LATENCY"
   
   # For batch processing
   quantization_preset: "mixed"
   compilation_mode: "THROUGHPUT"
   ```

3. **Memory Management**
   ```yaml
   # Adjust based on available RAM
   npu:
     max_memory_mb: 4096  # Increase for larger models
   agents:
     max_context_length: 2048  # Reduce for memory constraints
   ```

4. **Device Optimization**
   ```python
   # Force NPU usage for maximum performance
   config = AgentConfig(
       device_preference=["NPU"],  # No fallback
       turbo_mode=True
   )
   ```

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
