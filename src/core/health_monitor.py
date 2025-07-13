from typing import Dict, List

class AgentHealthMonitor:
    def __init__(self, threshold_config: Dict):
        self.thresholds = threshold_config
        self.alerts = []

    def check_agent_health(self, agent_id: str, metrics: Dict) -> List[str]:
        """Check agent health against thresholds"""
        issues = []

        if metrics.get('error_rate', 0) > self.thresholds['max_error_rate']:
            issues.append(f"High error rate: {metrics['error_rate']}")

        if metrics.get('avg_execution_time', 0) > self.thresholds['max_exec_time']:
            issues.append(f"Slow execution: {metrics['avg_execution_time']}s")

        if metrics.get('memory_usage', 0) > self.thresholds['max_memory']:
            issues.append(f"High memory: {metrics['memory_usage']}MB")

        return issues
