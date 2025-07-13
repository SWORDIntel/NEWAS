"""ReAct (Reasoning and Acting) Pattern Implementation"""

import re
import json
import logging
import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions in ReAct pattern"""
    THINK = "think"
    ACT = "act"
    OBSERVE = "observe"
    ANSWER = "answer"


@dataclass
class Tool:
    """Tool definition for agent use"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, str]  # parameter_name -> type_description

    async def execute(self, **kwargs) -> str:
        """Execute tool with given parameters"""
        try:
            if asyncio.iscoroutinefunction(self.function):
                result = await self.function(**kwargs)
            else:
                result = self.function(**kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}")
            return f"Error: {str(e)}"


@dataclass
class ReActStep:
    """Single step in ReAct execution"""
    step_number: int
    action_type: ActionType
    content: str
    tool_name: Optional[str] = None
    tool_params: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class ReActResult:
    """Result of ReAct execution"""
    success: bool
    final_answer: str
    steps: List[Dict[str, Any]]
    execution_time: float
    iterations_used: int


class ReActExecutor:
    """Executes ReAct pattern for problem solving"""

    # ReAct prompt template
    REACT_TEMPLATE = """You are an AI assistant that solves problems step by step using the ReAct (Reasoning and Acting) approach.

Available tools:
{tools_description}

Format your response EXACTLY as follows:
Thought: [Your reasoning about what to do next]
Action: [tool_name]
Action Input: {{"param1": "value1", "param2": "value2"}}

After receiving an observation, continue with:
Thought: [Reflection on the observation]
...

When you have the final answer:
Thought: [Final reasoning]
Answer: [Your final answer to the user]

Question: {query}

Let's solve this step by step."""

    def __init__(self, agent):
        self.agent = agent
        self.tools = agent.tools

    async def execute(self,
                     query: str,
                     context: Optional[Dict[str, Any]] = None,
                     max_iterations: int = 5) -> ReActResult:
        """Execute ReAct loop to solve a problem"""

        start_time = time.time()
        steps = []
        iteration = 0

        # Build tools description
        tools_desc = self._build_tools_description()

        # Initial prompt
        prompt = self.REACT_TEMPLATE.format(
            tools_description=tools_desc,
            query=query
        )

        # Add context if provided
        if context:
            prompt += f"\n\nAdditional context: {json.dumps(context)}"

        # Conversation history for multi-turn reasoning
        conversation = prompt

        while iteration < max_iterations:
            iteration += 1

            try:
                # Generate next action
                response = self.agent.generate(conversation)

                # Parse response
                action_type, content, tool_info = self._parse_response(response)

                # Create step record
                step = ReActStep(
                    step_number=iteration,
                    action_type=action_type,
                    content=content,
                    timestamp=time.time()
                )

                # Handle different action types
                if action_type == ActionType.ANSWER:
                    # Final answer reached
                    return ReActResult(
                        success=True,
                        final_answer=content,
                        steps=self._steps_to_dict(steps + [step]),
                        execution_time=time.time() - start_time,
                        iterations_used=iteration
                    )

                elif action_type == ActionType.ACT and tool_info:
                    # Execute tool
                    tool_name, tool_params = tool_info
                    step.tool_name = tool_name
                    step.tool_params = tool_params

                    if tool_name in self.tools:
                        observation = await self.tools[tool_name].execute(**tool_params)
                        step.observation = observation

                        # Add observation to conversation
                        conversation += f"\n{response}\nObservation: {observation}\n"
                    else:
                        # Tool not found
                        observation = f"Error: Tool '{tool_name}' not found"
                        step.observation = observation
                        conversation += f"\n{response}\nObservation: {observation}\n"

                else:
                    # Just thinking, continue conversation
                    conversation += f"\n{response}\n"

                steps.append(step)

            except Exception as e:
                logger.error(f"ReAct execution error at iteration {iteration}: {e}")
                # Try to recover or fail gracefully
                return ReActResult(
                    success=False,
                    final_answer=f"I encountered an error while solving this problem: {str(e)}",
                    steps=self._steps_to_dict(steps),
                    execution_time=time.time() - start_time,
                    iterations_used=iteration
                )

        # Max iterations reached
        return ReActResult(
            success=False,
            final_answer="I couldn't complete the task within the allowed iterations. Here's what I found so far: " +
                         self._summarize_steps(steps),
            steps=self._steps_to_dict(steps),
            execution_time=time.time() - start_time,
            iterations_used=iteration
        )

    def _build_tools_description(self) -> str:
        """Build formatted description of available tools"""
        descriptions = []

        for name, tool in self.tools.items():
            params_str = ", ".join([f"{p}: {t}" for p, t in tool.parameters.items()])
            descriptions.append(f"- {name}({params_str}): {tool.description}")

        return "\n".join(descriptions)

    def _parse_response(self, response: str) -> Tuple[ActionType, str, Optional[Tuple[str, Dict]]]:
        """Parse model response to extract action type and content"""

        # Clean response
        response = response.strip()

        # Check for final answer
        answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return ActionType.ANSWER, answer_match.group(1).strip(), None

        # Check for action
        action_match = re.search(r'Action:\s*(\w+)', response, re.IGNORECASE)
        if action_match:
            tool_name = action_match.group(1).strip()

            # Extract action input
            input_match = re.search(r'Action Input:\s*({.+?})', response, re.IGNORECASE | re.DOTALL)
            if input_match:
                try:
                    tool_params = json.loads(input_match.group(1))
                    return ActionType.ACT, response, (tool_name, tool_params)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse action input: {input_match.group(1)}")
                    # Try to extract parameters manually
                    tool_params = self._extract_params_fallback(input_match.group(1))
                    if tool_params:
                        return ActionType.ACT, response, (tool_name, tool_params)

        # Check for thought
        thought_match = re.search(r'Thought:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if thought_match:
            return ActionType.THINK, thought_match.group(1).strip(), None

        # Default to thinking if no clear pattern
        return ActionType.THINK, response, None

    def _extract_params_fallback(self, params_str: str) -> Optional[Dict[str, Any]]:
        """Fallback parameter extraction for malformed JSON"""
        try:
            # Try to fix common JSON errors
            params_str = params_str.strip()

            # Add quotes to unquoted keys
            params_str = re.sub(r'(\w+):', r'"\1":', params_str)

            # Try parsing again
            return json.loads(params_str)
        except:
            # Manual extraction as last resort
            params = {}
            matches = re.findall(r'["\']?(\w+)["\']?\s*:\s*["\']?([^,"\']+)["\']?', params_str)
            for key, value in matches:
                # Try to infer type
                if value.lower() in ['true', 'false']:
                    params[key] = value.lower() == 'true'
                elif value.replace('.', '').replace('-', '').isdigit():
                    params[key] = float(value) if '.' in value else int(value)
                else:
                    params[key] = value.strip()

            return params if params else None

    def _steps_to_dict(self, steps: List[ReActStep]) -> List[Dict[str, Any]]:
        """Convert ReActStep objects to dictionaries"""
        return [
            {
                "step_number": step.step_number,
                "action_type": step.action_type.value,
                "content": step.content,
                "tool_name": step.tool_name,
                "tool_params": step.tool_params,
                "observation": step.observation,
                "timestamp": step.timestamp
            }
            for step in steps
        ]

    def _summarize_steps(self, steps: List[ReActStep]) -> str:
        """Summarize execution steps for partial results"""
        summary = []

        for step in steps:
            if step.action_type == ActionType.ACT and step.observation:
                summary.append(f"- Used {step.tool_name}: {step.observation}")
            elif step.action_type == ActionType.THINK:
                summary.append(f"- Reasoning: {step.content[:100]}...")

        return "\n".join(summary) if summary else "No significant progress made."
