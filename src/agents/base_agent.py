"""Base agent interface for NEMWAS"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import uuid

@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender_id: str
    recipient_id: str
    content: Any
    message_type: str = "task"
    metadata: Dict[str, Any] = None

class BaseAgent(ABC):
    """Abstract base class for all NEMWAS agents"""
    
    def __init__(self, name: str, specialties: List[str] = None):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.specialties = specialties or []
        self.active = True
        
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Any:
        """Process a task and return result"""
        pass
    
    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle specific task type"""
        pass
    
    async def send_message(self, recipient_id: str, content: Any) -> None:
        """Send message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            content=content
        )
        # Implementation would use message broker
        
    def shutdown(self):
        """Gracefully shutdown agent"""
        self.active = False
