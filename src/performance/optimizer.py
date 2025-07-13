import time
from typing import Dict, Any, Callable

class PerformanceOptimizer:
    """
    A class to optimize the performance of the system.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimizer_settings = self.config.get("performance_optimizer", {})
        self.enabled = self.optimizer_settings.get("enabled", False)

    def optimize(self, func: Callable, *args, **kwargs) -> Any:
        """
        A decorator to optimize the performance of a function.
        """
        if not self.enabled:
            return func(*args, **kwargs)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        if execution_time > self.optimizer_settings.get("slow_execution_threshold", 1.0):
            print(f"Warning: {func.__name__} took {execution_time:.2f}s to execute.")

        return result
