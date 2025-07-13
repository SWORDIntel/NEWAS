import json
from typing import Dict, Any

class MemoryPersistence:
    """
    A class to save and load agent memory to and from disk.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.persistence_settings = self.config.get("memory_persistence", {})
        self.enabled = self.persistence_settings.get("enabled", False)
        self.path = self.persistence_settings.get("path", "data/memory.json")

    def save(self, memory: Dict[str, Any]) -> None:
        """
        Saves the memory to disk.
        """
        if not self.enabled:
            return

        with open(self.path, "w") as f:
            json.dump(memory, f)

    def load(self) -> Dict[str, Any]:
        """
        Loads the memory from disk.
        """
        if not self.enabled:
            return {}

        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
