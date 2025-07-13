"""
NEMWAS - Neural-Enhanced Multi-Workforce Agent System

A powerful, NPU-accelerated multi-agent framework for building intelligent AI systems.
"""

__version__ = "1.0.0"
__author__ = "NEMWAS Team"
__license__ = "MIT"

# Core imports
from .core.agent import NEMWASAgent, AgentConfig, AgentContext
from .core.npu_manager import NPUManager
from .core.react import ReActExecutor, Tool, ReActResult

# Component imports
from .capability.learner import CapabilityLearner, Capability
from .performance.tracker import PerformanceTracker, TaskMetrics, AgentMetrics
from .nlp.interface import NaturalLanguageInterface, IntentType, ParsedIntent
from .plugins.interface import NEMWASPlugin, ToolPlugin, CapabilityPlugin, PluginRegistry

# Utility imports
from .utils.config import load_config, Config

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    
    # Core classes
    "NEMWASAgent",
    "AgentConfig", 
    "AgentContext",
    "NPUManager",
    "ReActExecutor",
    "Tool",
    "ReActResult",
    
    # Components
    "CapabilityLearner",
    "Capability",
    "PerformanceTracker",
    "TaskMetrics",
    "AgentMetrics",
    "NaturalLanguageInterface",
    "IntentType",
    "ParsedIntent",
    
    # Plugin system
    "NEMWASPlugin",
    "ToolPlugin",
    "CapabilityPlugin",
    "PluginRegistry",
    
    # Utilities
    "load_config",
    "Config",
]

# Package initialization
import logging

# Set up package logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Check for required dependencies
try:
    import openvino as ov
    logger.debug(f"OpenVINO version: {ov.__version__}")
except ImportError:
    logger.warning("OpenVINO not installed. NPU acceleration will not be available.")

# Print banner when imported directly
if __name__ == "__main__":
    print(f"NEMWAS v{__version__}")
    print("Neural-Enhanced Multi-Workforce Agent System")
    print("=" * 50)
