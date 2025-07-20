"""Configuration utilities for NEMWAS"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for NEMWAS"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else Path("config/default.yaml")
        self.config = {}
        self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            self.config = self._get_default_config()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from: {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "system": {
                "name": "NEMWAS",
                "version": "1.0.0",
                "log_level": "INFO"
            },
            "models": {
                "default_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "default_model_path": "./models/openvino/tinyllama-1.1b-chat.xml",
                "model_cache_dir": "./models/cache",
                "quantization_preset": "mixed"
            },
            "npu": {
                "device_priority": ["NPU", "GPU", "CPU"],
                "cache_dir": "./models/cache",
                "enable_profiling": False,
                "compilation_mode": "LATENCY",
                "turbo_mode": True,
                "max_memory_mb": 2048
            },
            "agents": {
                "max_agents": 10,
                "max_context_length": 4096,
                "default_temperature": 0.7,
                "max_new_tokens": 512,
                "max_iterations": 5,
                "enable_learning": True,
                "enable_performance_tracking": True
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080,
                "enable_cors": True,
                "enable_docs": True
            },
            "storage": {
                "data_dir": "./data",
                "capability_dir": "./data/capabilities",
                "embedding_dir": "./data/embeddings",
                "metrics_dir": "./data/metrics"
            }
        }
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Example: NEMWAS_API_PORT=8081 overrides api.port
        for key, value in os.environ.items():
            if key.startswith("NEMWAS_"):
                config_key = key[7:].lower().replace("_", ".")
                self._set_nested(self.config, config_key, value)
    
    def _set_nested(self, d: Dict[str, Any], key: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        
        # Try to convert value to appropriate type
        if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif "." in value and all(part.isdigit() for part in value.split(".", 1)):
            value = float(value)
        
        d[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split(".")
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        self._set_nested(self.config, key, value)
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Configuration saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style setting"""
        self.set(key, value)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file"""
    config = Config(config_path)
    return config.to_dict()