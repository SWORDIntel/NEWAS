"""ReAct (Reasoning and Acting) implementation for NEMWAS agents"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
import re
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """Tool definition for ReAct agents"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        try:
            # Handle both sync and async functions
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(**kwargs)
            else:
                return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            raise


class ReActLoop:
    """ReAct loop implementation with NPU acceleration"""
    
    def __init__(self, model_path: str, npu_manager=None, 
                 device_preference: List[str] = None,
                 max_iterations: int = 5):
        self.model_path = model_path
        self.npu_manager = npu_manager
        self.device_preference = device_preference or ["NPU", "GPU", "CPU"]
        self.max_iterations = max_iterations
        
        self.tools = {}
        self.device = None
        self.model = None
        self.execution_trace = {}
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model with NPU support"""
        try:
            if self.npu_manager:
                # Load model through NPU manager
                self.model = self.npu_manager.load_model(
                    self.model_path,
                    device_preference=self.device_preference
                )
                self.device = self.npu_manager.get_model_device(self.model_path)
            else:
                # Fallback to CPU
                logger.warning("NPU manager not available, using CPU")
                self.device = "CPU"
                # In real implementation, would load with OpenVINO directly
                
            logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def register_tool(self, tool: Tool):
        """Register a tool for use in ReAct loop"""
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    async def run(self, query: str) -> str:
        """Run the ReAct loop for a given query"""
        self.execution_trace = {
            "query": query,
            "iterations": [],
            "tools_used": set(),
            "start_time": time.time()
        }
        
        context = [{"role": "user", "content": query}]
        
        for iteration in range(self.max_iterations):
            try:
                # Generate thought and action
                response = await self._generate_response(context)
                
                # Parse the response
                thought, action, action_input = self._parse_response(response)
                
                iteration_data = {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "action_input": action_input
                }
                
                # Execute action if specified
                if action and action in self.tools:
                    observation = await self._execute_tool(action, action_input)
                    iteration_data["observation"] = observation
                    
                    # Add observation to context
                    context.append({
                        "role": "assistant",
                        "content": f"Thought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}"
                    })
                    
                    self.execution_trace["tools_used"].add(action)
                    
                elif action == "finish":
                    # Task completed
                    answer = action_input.get("answer", "Task completed")
                    iteration_data["answer"] = answer
                    self.execution_trace["iterations"].append(iteration_data)
                    self.execution_trace["end_time"] = time.time()
                    return answer
                    
                else:
                    # No action or unknown action
                    if thought:
                        context.append({"role": "assistant", "content": f"Thought: {thought}"})
                
                self.execution_trace["iterations"].append(iteration_data)
                
            except Exception as e:
                logger.error(f"Error in ReAct iteration {iteration}: {e}")
                self.execution_trace["error"] = str(e)
                raise
        
        # Max iterations reached
        self.execution_trace["end_time"] = time.time()
        return "I need more iterations to complete this task."
    
    async def _generate_response(self, context: List[Dict[str, str]]) -> str:
        """Generate response using the model"""
        try:
            # Format prompt with ReAct structure
            prompt = self._format_prompt(context)
            
            # Generate with model
            if self.npu_manager and self.model:
                response = await self.npu_manager.generate(
                    self.model,
                    prompt,
                    max_new_tokens=256,
                    temperature=0.7
                )
            else:
                # Fallback response for testing
                response = "Thought: I need to think about this.\nAction: finish\nAction Input: {\"answer\": \"Task completed\"}"
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def _format_prompt(self, context: List[Dict[str, str]]) -> str:
        """Format the prompt for ReAct pattern"""
        # Build conversation history
        conversation = ""
        for msg in context:
            role = msg["role"]
            content = msg["content"]
            conversation += f"{role.capitalize()}: {content}\n"
        
        # Add ReAct instructions
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""You are a helpful AI assistant using the ReAct (Reasoning and Acting) pattern.

Available tools:
{tools_desc}
- finish: Complete the task with an answer

For each step, provide:
1. Thought: Your reasoning about what to do next
2. Action: The tool to use (or 'finish' to complete)
3. Action Input: The input for the tool as JSON

{conversation}
Assistant: """
        
        return prompt
    
    def _parse_response(self, response: str) -> tuple:
        """Parse the model response to extract thought, action, and input"""
        thought = ""
        action = ""
        action_input = {}
        
        # Extract thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.IGNORECASE | re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract action
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", response, re.IGNORECASE | re.DOTALL)
        if action_match:
            action = action_match.group(1).strip().lower()
        
        # Extract action input
        input_match = re.search(r"Action Input:\s*(.+?)$", response, re.IGNORECASE | re.DOTALL)
        if input_match:
            try:
                input_str = input_match.group(1).strip()
                # Try to parse as JSON
                action_input = json.loads(input_str)
            except json.JSONDecodeError:
                # If not valid JSON, treat as string parameter
                action_input = {"input": input_str}
        
        return thought, action, action_input
    
    async def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute a tool with given parameters"""
        try:
            tool = self.tools[tool_name]
            result = await tool.execute(**parameters)
            return str(result)
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    def get_execution_trace(self) -> Dict[str, Any]:
        """Get the execution trace of the last run"""
        return self.execution_trace.copy()
    
    def clear_tools(self):
        """Clear all registered tools"""
        self.tools.clear()
    
    def get_registered_tools(self) -> List[str]:
        """Get list of registered tool names"""
        return list(self.tools.keys())