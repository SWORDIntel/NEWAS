# Performance Tuning Guide

This guide provides instructions for tuning the performance of the NEMWAS application.

## NPU Acceleration

NEMWAS is optimized for Intel NPU hardware. If you have a compatible processor, you can enable NPU acceleration to significantly improve performance.

### Enabling NPU Acceleration

To enable NPU acceleration, make sure you have run the NPU setup script:

```bash
./scripts/setup_npu.sh
```

You can also specify the device preference for each agent in the `config/default.yaml` file:

```yaml
agent:
  device_preference: ["NPU", "GPU", "CPU"]
```

## Model Selection

The model you choose can have a significant impact on performance. Smaller models are faster but less accurate, while larger models are more accurate but slower.

### Changing the Default Model

You can change the default model in the `config/default.yaml` file:

```yaml
models:
  default_model_path: "./models/mistral-7b.xml"
```

## Configuration Options

The following configuration options in `config/default.yaml` can be adjusted to tune performance:

- `max_agents`: The maximum number of agents that can be active at the same time.
- `max_context_length`: The maximum context length for the language model.
- `temperature`: The temperature for the language model.

## Performance Analysis API

The `/metrics/performance/analysis` API endpoint can be used to analyze performance trends and identify bottlenecks.

### Example Request

```bash
curl http://localhost:8080/metrics/performance/analysis
```

### Example Response

```json
{
  "trend": "stable",
  "current_avg_time": 0.34,
  "historical_avg_time": 0.35,
  "success_rate": 1,
  "device_performance": {
    "CPU": 0.34
  },
  "optimal_device": "CPU",
  "recommendations": [],
  "bottlenecks": []
}
```
