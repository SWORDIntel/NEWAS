import time
from typing import Dict, Any, List

class PerformanceTracker:
    """
    A class to track the performance of the system.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_data: List[Dict[str, Any]] = []

    def track(self, event_name: str, data: Dict[str, Any]) -> None:
        """
        Tracks a performance event.
        """
        event = {
            "event_name": event_name,
            "timestamp": time.time(),
            "data": data,
        }
        self.performance_data.append(event)

    def get_performance_data(self) -> List[Dict[str, Any]]:
        """
        Returns the performance data.
        """
        return self.performance_data
