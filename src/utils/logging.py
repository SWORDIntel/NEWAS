"""Logging utilities for NEMWAS"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import structlog

def setup_logging(config_path: str = None, log_level: str = "INFO"):
    """Setup structured logging for NEMWAS"""
    
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)
    else:
        # Default configuration
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/nemwas.log", mode="a")
            ]
        )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return structlog.get_logger(name)
