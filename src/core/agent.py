"""NEMWAS Agent implementation with NPU acceleration"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json

from ..agents.base_agent import BaseAgent
from .react import ReActLoop, Tool
from ..capability.learner import CapabilityLearner
from ..performance.tracker import PerformanceTracker

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for NEMWAS Agent"""
    name: str
    model_path: str = "./models/openvino/tinyllama-1.1b-chat.xml"
    device_preference: List[str] = field(default_factory=lambda: ["NPU", "GPU", "CPU"])
    max_iterations: int = 5
    enable_learning: bool = True
    enable_performance_tracking: bool = True
    temperature: float = 0.7
    max_new_tokens: int = 512
    quantization_preset: str = "mixed"  # performance, mixed, accuracy
    turbo_mode: bool = True


@dataclass
class AgentContext:
    """Context for agent execution"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_to_history(self, entry: Dict[str, Any]):
        """Add entry to context history"""
        self.history.append(entry)
    
    def export(self) -> Dict[str, Any]:
        """Export context as dictionary"""
        return {
            "task_id": self.task_id,
            "metadata": self.metadata,
            "history": self.history,
            "performance_metrics": self.performance_metrics
        }


class NEMWASAgent(BaseAgent):
    """Neural-Enhanced Multi-Workforce Agent with NPU acceleration"""
    
    def __init__(self, config: AgentConfig, npu_manager=None):
        super().__init__(config.name, specialties=["general", "reasoning", "task_execution"])
        self.config = config
        self.npu_manager = npu_manager
        self.agent_id = f"Agent-{config.name}-{uuid.uuid4().hex[:8]}"
        
        # Initialize components
        self.react_loop = None
        self.capability_learner = None
        self.performance_tracker = None
        self.tools = {}
        self.context = AgentContext()
        
        # Initialize based on config
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize agent components"""
        try:
            # Initialize ReAct loop
            self.react_loop = ReActLoop(
                model_path=self.config.model_path,
                npu_manager=self.npu_manager,
                device_preference=self.config.device_preference,
                max_iterations=self.config.max_iterations
            )
            
            # Initialize capability learner if enabled
            if self.config.enable_learning:
                self.capability_learner = CapabilityLearner(
                    npu_manager=self.npu_manager,
                    storage_path=Path("./data/capabilities")
                )
            
            # Initialize performance tracker if enabled
            if self.config.enable_performance_tracking:
                self.performance_tracker = PerformanceTracker(
                    metrics_dir=Path("./data/metrics")
                )
            
            logger.info(f"Initialized agent: {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent components: {e}")
            raise
    
    def register_tool(self, tool: Tool):
        """Register a tool with the agent"""
        self.tools[tool.name] = tool
        if self.react_loop:
            self.react_loop.register_tool(tool)
        logger.info(f"Registered tool: {tool.name} for agent {self.agent_id}")
    
    async def process_task(self, task: Union[str, Dict[str, Any]]) -> Any:
        """Process a task using ReAct pattern"""
        try:
            # Convert string task to dict format
            if isinstance(task, str):
                task = {"query": task}
            
            # Update context
            self.context.task_id = str(uuid.uuid4())
            self.context.add_to_history({
                "type": "task_start",
                "task": task,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Track performance if enabled
            if self.performance_tracker:
                await self.performance_tracker.start_task(self.context.task_id)
            
            # Execute task using ReAct loop
            result = await self.react_loop.run(task.get("query", ""))
            
            # Learn from successful execution if enabled
            if self.config.enable_learning and self.capability_learner:
                await self._learn_from_execution(task, result)
            
            # Track completion if enabled
            if self.performance_tracker:
                await self.performance_tracker.end_task(
                    self.context.task_id,
                    success=True,
                    result=result
                )
            
            # Update context
            self.context.add_to_history({
                "type": "task_complete",
                "result": result,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            
            # Track failure if enabled
            if self.performance_tracker:
                await self.performance_tracker.end_task(
                    self.context.task_id,
                    success=False,
                    error=str(e)
                )
            
            # Update context
            self.context.add_to_history({
                "type": "task_error",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            })
            
            raise
    
    async def _learn_from_execution(self, task: Dict[str, Any], result: Any):
        """Learn from successful task execution"""
        try:
            # Extract execution trace from ReAct loop
            execution_trace = self.react_loop.get_execution_trace()
            
            # Create capability from trace
            capability = {
                "task": task.get("query", ""),
                "result": result,
                "trace": execution_trace,
                "tools_used": list(execution_trace.get("tools_used", [])),
                "performance": self.context.performance_metrics
            }
            
            # Store capability
            await self.capability_learner.learn_capability(capability)
            
        except Exception as e:
            logger.warning(f"Failed to learn from execution: {e}")
    
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle specific task type"""
        # NEMWAS agents are general purpose
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        status = {
            "agent_id": self.agent_id,
            "name": self.config.name,
            "active": self.active,
            "device": self.react_loop.device if self.react_loop else "Not initialized",
            "tools": list(self.tools.keys()),
            "tasks_processed": len([h for h in self.context.history if h["type"] == "task_complete"])
        }
        
        # Add performance metrics if available
        if self.performance_tracker:
            metrics = self.performance_tracker.get_agent_metrics(self.agent_id)
            status["performance"] = metrics
        
        return status
    
    def export_context(self) -> Dict[str, Any]:
        """Export agent context"""
        return {
            "agent_id": self.agent_id,
            "config": {
                "name": self.config.name,
                "model_path": self.config.model_path,
                "device_preference": self.config.device_preference
            },
            "context": self.context.export(),
            "status": self.get_status()
        }
    
    def shutdown(self):
        """Gracefully shutdown agent"""
        logger.info(f"Shutting down agent: {self.agent_id}")
        
        # Export context before shutdown
        context_export = self.export_context()
        
        # Save to disk
        export_path = Path(f"./data/agent_exports/{self.agent_id}.json")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(context_export, f, indent=2)
        
        # Call parent shutdown
        super().shutdown()
        
        logger.info(f"Agent {self.agent_id} shutdown complete")