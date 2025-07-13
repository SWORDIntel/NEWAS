"""Multi-agent coordinator for NEMWAS"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict
import time

from .base_agent import BaseAgent, AgentMessage

logger = logging.getLogger(__name__)

class AgentCoordinator:
    """Coordinates multiple agents and routes tasks"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = asyncio.Queue()
        self.result_cache = {}
        self.agent_metrics = defaultdict(lambda: {"tasks": 0, "successes": 0})
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent {agent.name} ({agent.agent_id})")
        
    def unregister_agent(self, agent_id: str):
        """Remove an agent from the coordinator"""
        if agent_id in self.agents:
            agent = self.agents.pop(agent_id)
            agent.shutdown()
            logger.info(f"Unregistered agent {agent.name}")
            
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task for processing"""
        task_id = str(time.time())
        task["task_id"] = task_id
        await self.task_queue.put(task)
        return task_id
        
    async def get_result(self, task_id: str, timeout: float = 30) -> Any:
        """Get result for a task"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if task_id in self.result_cache:
                return self.result_cache.pop(task_id)
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Task {task_id} timed out")
        
    def find_best_agent(self, task: Dict[str, Any]) -> Optional[BaseAgent]:
        """Find the best agent for a task"""
        task_type = task.get("type", "general")
        
        # Find agents that can handle this task
        capable_agents = [
            agent for agent in self.agents.values()
            if agent.can_handle(task_type) and agent.active
        ]
        
        if not capable_agents:
            return None
            
        # Select agent with best performance
        best_agent = max(
            capable_agents,
            key=lambda a: self._get_agent_score(a.agent_id)
        )
        
        return best_agent
        
    def _get_agent_score(self, agent_id: str) -> float:
        """Calculate agent performance score"""
        metrics = self.agent_metrics[agent_id]
        if metrics["tasks"] == 0:
            return 0.5
        return metrics["successes"] / metrics["tasks"]
