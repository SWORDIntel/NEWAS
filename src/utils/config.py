"""Configuration management for NEMWAS"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""

    path = Path(config_path)

    if not path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return get_default_config()

    try:
        with open(path, 'r') as f:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                config = yaml.safe_load(f)
            elif path.suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        # Merge with defaults
        default_config = get_default_config()
        merged_config = merge_configs(default_config, config)

        # Apply environment overrides
        merged_config = apply_env_overrides(merged_config)

        # Validate configuration
        validate_config(merged_config)

        logger.info(f"Configuration loaded from {config_path}")
        return merged_config

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""

    return {
        # System configuration
        "system": {
            "name": "NEMWAS",
            "version": "1.0.0",
            "log_level": "INFO"
        },

        # Model configuration
        "models": {
            "default_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "default_model_path": "./models/original/tinyllama-1.1b-chat.xml",
            "model_cache_dir": "./models/cache",
            "quantization_preset": "mixed"
        },

        # NPU configuration
        "npu": {
            "device_priority": ["NPU", "GPU", "CPU"],
            "cache_dir": "./models/cache",
            "enable_profiling": False,
            "compilation_mode": "THROUGHPUT",
            "turbo_mode": True
        },

        # Agent configuration
        "agents": {
            "max_agents": 10,
            "max_context_length": 4096,
            "default_temperature": 0.7,
            "max_iterations": 5,
            "enable_learning": True,
            "enable_performance_tracking": True
        },

        # Performance tracking
        "performance": {
            "metrics_dir": "./data/metrics",
            "enable_prometheus": True,
            "prometheus_port": 9090,
            "history_size": 1000,
            "export_interval": 3600  # seconds
        },

        # Plugin configuration
        "plugins": {
            "plugin_dirs": ["./plugins/builtin", "./plugins/community"],
            "auto_load": [],
            "enable_hot_reload": True
        },

        # API configuration
        "api": {
            "host": "0.0.0.0",
            "port": 8080,
            "enable_cors": True,
            "enable_docs": True,
            "max_request_size": 10485760  # 10MB
        },

        # Storage configuration
        "storage": {
            "data_dir": "./data",
            "capability_dir": "./data/capabilities",
            "embedding_dir": "./data/embeddings",
            "metrics_dir": "./data/metrics"
        },

        # Natural Language Interface
        "nlp": {
            "embedding_model": "all-MiniLM-L6-v2",
            "enable_completions": True,
            "max_completions": 5
        },

        # Resource limits
        "resources": {
            "max_memory_gb": 8,
            "max_cpu_threads": 4,
            "enable_memory_monitoring": True
        }
    }


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries"""

    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration"""

    # Environment variable mapping
    env_mapping = {
        "NEMWAS_LOG_LEVEL": ("system", "log_level"),
        "NEMWAS_MODEL_PATH": ("models", "default_model_path"),
        "NEMWAS_API_HOST": ("api", "host"),
        "NEMWAS_API_PORT": ("api", "port"),
        "NEMWAS_MAX_AGENTS": ("agents", "max_agents"),
        "NEMWAS_DEVICE_PRIORITY": ("npu", "device_priority"),
        "NEMWAS_ENABLE_NPU_PROFILING": ("npu", "enable_profiling"),
        "NEMWAS_PROMETHEUS_PORT": ("performance", "prometheus_port"),
    }

    for env_var, config_path in env_mapping.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the correct position in config
            current = config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value with type conversion
            final_key = config_path[-1]

            # Type conversion based on existing type
            if final_key in current:
                existing_value = current[final_key]
                if isinstance(existing_value, bool):
                    value = value.lower() in ['true', '1', 'yes', 'on']
                elif isinstance(existing_value, int):
                    value = int(value)
                elif isinstance(existing_value, float):
                    value = float(value)
                elif isinstance(existing_value, list):
                    value = value.split(',')

            current[final_key] = value
            logger.info(f"Applied environment override: {env_var} = {value}")

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""

    errors = []

    # Check required paths
    required_paths = [
        config.get('models', {}).get('default_model_path'),
        config.get('storage', {}).get('data_dir'),
    ]

    for path in required_paths:
        if path and not Path(path).parent.exists():
            errors.append(f"Parent directory does not exist: {path}")

    # Check numeric ranges
    if config.get('agents', {}).get('max_agents', 0) < 1:
        errors.append("max_agents must be at least 1")

    if config.get('agents', {}).get('max_context_length', 0) < 512:
        errors.append("max_context_length must be at least 512")

    if config.get('api', {}).get('port', 0) < 1 or config.get('api', {}).get('port', 0) > 65535:
        errors.append("API port must be between 1 and 65535")

    # Check device priority
    device_priority = config.get('npu', {}).get('device_priority', [])
    valid_devices = ['NPU', 'GPU', 'CPU', 'MYRIAD']
    for device in device_priority:
        if device not in valid_devices:
            errors.append(f"Invalid device in device_priority: {device}")

    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    return True


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to file"""

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(path, 'w') as f:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif path.suffix == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        logger.info(f"Configuration saved to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save config to {filepath}: {e}")
        raise


class Config:
    """Configuration wrapper with attribute access"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Configuration has no attribute '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self._config.copy()


# Create __init__.py files for package structure
def ensure_package_structure():
    """Ensure all package directories have __init__.py files"""

    src_path = Path(__file__).parent.parent

    # All subdirectories that should be packages
    package_dirs = [
        src_path,
        src_path / "core",
        src_path / "agents",
        src_path / "capability",
        src_path / "performance",
        src_path / "nlp",
        src_path / "plugins",
        src_path / "utils",
        src_path / "api"
    ]

    for package_dir in package_dirs:
        init_file = package_dir / "__init__.py"
        if not init_file.exists() and package_dir.exists():
            init_file.touch()
            logger.debug(f"Created {init_file}")


# Run package structure check on import
ensure_package_structure()
