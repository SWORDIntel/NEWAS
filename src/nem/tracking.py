import time
from typing import Dict, Any, List

class AgentTracker:
    """
    A class to track agent status, performance, and other metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_data: Dict[str, Dict[str, Any]] = {}

    def track_agent_creation(self, agent_id: str, agent_type: str) -> None:
        """
        Tracks the creation of an agent.
        """
        self.agent_data[agent_id] = {
            "type": agent_type,
            "created_at": time.time(),
            "status": "idle",
            "tasks_processed": 0,
            "success_rate": 1.0,
            "average_execution_time": 0.0,
        }

    def track_agent_status(self, agent_id: str, status: str) -> None:
        """
        Tracks the status of an agent.
        """
        if agent_id in self.agent_data:
            self.agent_data[agent_id]["status"] = status

    def track_task_execution(
        self, agent_id: str, success: bool, execution_time: float
    ) -> None:
        """
        Tracks the execution of a task by an agent.
        """
        if agent_id in self.agent_data:
            agent = self.agent_data[agent_id]
            agent["tasks_processed"] += 1
            agent["success_rate"] = (
                (agent["success_rate"] * (agent["tasks_processed"] - 1)) + int(success)
            ) / agent["tasks_processed"]
            agent["average_execution_time"] = (
                (agent["average_execution_time"] * (agent["tasks_processed"] - 1))
                + execution_time
            ) / agent["tasks_processed"]

    def get_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """
        Returns the data for a specific agent.
        """
        return self.agent_data.get(agent_id, {})

    def get_all_agent_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns the data for all agents.
        """
        return self.agent_data
