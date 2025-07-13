"""Main NEMWAS Agent Implementation"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import uuid

import openvino_genai as ov_genai
from pydantic import BaseModel, Field

from .npu_manager import NPUManager
from .react import ReActExecutor, Tool, ReActResult
from ..capability.learner import CapabilityLearner
from ..performance.tracker import PerformanceTracker
from ..utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent execution"""
    agent_id: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_task: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentConfig(BaseModel):
    """Agent configuration"""
    name: str = "NEMWAS-Agent"
    model_path: str
    device_preference: List[str] = ["NPU", "GPU", "CPU"]
    max_context_length: int = 4096
    temperature: float = 0.7
    max_iterations: int = 5
    enable_learning: bool = True
    enable_performance_tracking: bool = True


class NEMWASAgent:
    """Neural-Enhanced Multi-Workforce Agent System"""

    def __init__(self, config: AgentConfig, npu_manager: Optional[NPUManager] = None):
        self.config = config
        self.agent_id = str(uuid.uuid4())
        self.context = AgentContext(agent_id=self.agent_id)

        # Initialize NPU manager
        self.npu_manager = npu_manager or NPUManager()

        # Initialize LLM pipeline with NPU optimization
        self._initialize_llm()

        # Initialize components
        self.react_executor = ReActExecutor(self)
        self.capability_learner = CapabilityLearner() if config.enable_learning else None
        self.performance_tracker = PerformanceTracker() if config.enable_performance_tracking else None

        # Tool registry
        self.tools: Dict[str, Tool] = {}
        self._register_default_tools()

        logger.info(f"NEMWAS Agent {self.agent_id} initialized on device: {self.device}")

    def _initialize_llm(self):
        """Initialize LLM with NPU optimization"""
        try:
            # Check if model needs optimization
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Optimize for NPU if available and not already optimized
            if self.npu_manager.available_devices[0] == "NPU" and not str(model_path).endswith("_npu_optimized.xml"):
                logger.info("Optimizing model for NPU...")
                optimized_path = self.npu_manager.optimize_model_for_npu(
                    str(model_path),
                    model_type="llm",
                    quantization_preset="mixed"
                )
                model_path = optimized_path

            # Select best device
            self.device = self.npu_manager.select_optimal_device(str(model_path))

            # Create LLM pipeline
            logger.info(f"Loading LLM pipeline on {self.device}...")
            self.llm_pipeline = ov_genai.LLMPipeline(str(model_path), self.device)

            # Configure generation parameters
            self.generation_config = ov_genai.GenerationConfig()
            self.generation_config.max_new_tokens = 512
            self.generation_config.temperature = self.config.temperature
            self.generation_config.do_sample = True
            self.generation_config.top_p = 0.9

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    def _register_default_tools(self):
        """Register default agent tools"""

        # Calculator tool
        self.register_tool(Tool(
            name="calculate",
            description="Perform mathematical calculations",
            function=self._calculate,
            parameters={"expression": "str"}
        ))

        # Memory tool
        self.register_tool(Tool(
            name="remember",
            description="Store information in agent memory",
            function=self._remember,
            parameters={"key": "str", "value": "str"}
        ))

        # Recall tool
        self.register_tool(Tool(
            name="recall",
            description="Retrieve information from agent memory",
            function=self._recall,
            parameters={"key": "str"}
        ))

    def register_tool(self, tool: Tool):
        """Register a new tool for the agent"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    async def process(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Process a query using ReAct pattern with NPU acceleration"""

        start_time = time.time()

        # Update context
        self.context.current_task = query
        self.context.conversation_history.append({"role": "user", "content": query})

        try:
            # Track performance
            if self.performance_tracker:
                self.performance_tracker.start_task(self.agent_id, query)

            # Execute ReAct loop
            result = await self.react_executor.execute(
                query=query,
                context=context,
                max_iterations=self.config.max_iterations
            )

            # Learn from execution if enabled
            if self.capability_learner and result.success:
                await self._learn_from_execution(query, result)

            # Update conversation history
            self.context.conversation_history.append({
                "role": "assistant",
                "content": result.final_answer
            })

            # Track performance metrics
            if self.performance_tracker:
                execution_time = time.time() - start_time
                self.performance_tracker.end_task(
                    self.agent_id,
                    success=result.success,
                    execution_time=execution_time,
                    device_used=self.device,
                    iterations=len(result.steps)
                )

            return result.final_answer

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            self.context.conversation_history.append({
                "role": "assistant",
                "content": error_msg
            })
            return error_msg

    def generate(self, prompt: str) -> str:
        """Generate text using NPU-accelerated LLM"""

        # Add conversation context if available
        if self.context.conversation_history:
            context_prompt = self._build_context_prompt(prompt)
        else:
            context_prompt = prompt

        # Generate with NPU acceleration
        try:
            response = self.llm_pipeline.generate(context_prompt, self.generation_config)
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Fallback to simpler generation
            return "I apologize, but I'm having trouble generating a response right now."

    def _build_context_prompt(self, prompt: str) -> str:
        """Build prompt with conversation context"""

        # Limit context to fit within max length
        context_parts = []
        total_length = len(prompt)

        for msg in reversed(self.context.conversation_history[-10:]):  # Last 10 messages
            msg_text = f"{msg['role']}: {msg['content']}\n"
            if total_length + len(msg_text) < self.config.max_context_length - 500:
                context_parts.insert(0, msg_text)
                total_length += len(msg_text)
            else:
                break

        if context_parts:
            context = "".join(context_parts)
            return f"Previous conversation:\n{context}\n\nCurrent request: {prompt}"

        return prompt

    async def _learn_from_execution(self, query: str, result: ReActResult):
        """Learn from successful execution"""

        if not self.capability_learner:
            return

        # Extract successful patterns
        successful_tools = []
        for step in result.steps:
            if step.get("tool_name") and step.get("success"):
                successful_tools.append({
                    "tool": step["tool_name"],
                    "context": step.get("thought", ""),
                    "result": step.get("observation", "")
                })

        if successful_tools:
            # Create capability entry
            capability = {
                "query_pattern": query,
                "successful_tools": successful_tools,
                "execution_time": result.execution_time,
                "timestamp": time.time()
            }

            # Learn capability
            await self.capability_learner.learn(capability)

    # Default tool implementations
    def _calculate(self, expression: str) -> str:
        """Simple calculator tool"""
        try:
            # Safe evaluation of mathematical expressions
            import ast
            import operator

            # Allowed operators
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg
            }

            def eval_expr(node):
                if isinstance(node, ast.Num):
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](eval_expr(node.operand))
                else:
                    raise TypeError(f"Unsupported type {type(node)}")

            node = ast.parse(expression, mode='eval')
            result = eval_expr(node.body)
            return f"The result is: {result}"

        except Exception as e:
            return f"Error calculating expression: {str(e)}"

    def _remember(self, key: str, value: str) -> str:
        """Store information in agent memory"""
        if "memory" not in self.context.metadata:
            self.context.metadata["memory"] = {}

        self.context.metadata["memory"][key] = value
        return f"Stored '{key}' in memory"

    def _recall(self, key: str) -> str:
        """Retrieve information from agent memory"""
        memory = self.context.metadata.get("memory", {})

        if key in memory:
            return f"Retrieved from memory: {memory[key]}"
        else:
            return f"No memory found for key '{key}'"

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        metrics = {
            "agent_id": self.agent_id,
            "device": self.device,
            "total_queries": len(self.context.conversation_history) // 2,
            "capabilities": len(self.context.capabilities)
        }

        if self.performance_tracker:
            metrics.update(self.performance_tracker.get_agent_metrics(self.agent_id))

        # Add NPU metrics
        if self.device == "NPU":
            metrics["npu_metrics"] = self.npu_manager.get_device_metrics("NPU")

        return metrics

    def export_context(self) -> Dict[str, Any]:
        """Export agent context for persistence"""
        return {
            "agent_id": self.agent_id,
            "conversation_history": self.context.conversation_history,
            "capabilities": self.context.capabilities,
            "metadata": self.context.metadata,
            "performance_metrics": self.context.performance_metrics
        }

    def import_context(self, context_data: Dict[str, Any]):
        """Import agent context from saved data"""
        self.context.conversation_history = context_data.get("conversation_history", [])
        self.context.capabilities = context_data.get("capabilities", [])
        self.context.metadata = context_data.get("metadata", {})
        self.context.performance_metrics = context_data.get("performance_metrics", {})
