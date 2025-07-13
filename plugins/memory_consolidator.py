from typing import Dict, Any
from src.plugins.interface import ToolPlugin, PluginMetadata

class MemoryConsolidatorPlugin(ToolPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="memory_consolidator",
            version="1.0.0",
            author="Your Name",
            description="Compresses agent memories and enables cross-session learning.",
            npu_compatible=True
        )

    def get_tool_definition(self):
        return {
            'name': 'memory_consolidator',
            'description': 'Compresses agent memories and enables cross-session learning.',
            'function': self.execute,
            'parameters': {
                "agent_id": {
                    "type": "string",
                    "description": "The ID of the agent to consolidate memories for. If not provided, consolidates memories across all agents."
                }
            }
        }

    def execute(self, agent_id: str = None) -> str:
        """Consolidate memories across agents or for specific agent"""

        if agent_id:
            # Consolidate single agent
            agent = self.registry.core.agents.get(agent_id)
            if agent and hasattr(agent, 'memory_persistence'):
                memories = agent.memory_persistence.load()
                consolidated = self._consolidate_memories(memories)
                agent.memory_persistence.save(consolidated)
                return f"Consolidated memories for agent {agent_id}"
        else:
            # Consolidate across all agents
            all_memories = {}
            for aid, agent in self.registry.core.agents.items():
                if hasattr(agent, 'memory_persistence'):
                    all_memories[aid] = agent.memory_persistence.load()

            # Find common patterns
            common_patterns = self._extract_common_patterns(all_memories)

            # Share learnings across agents
            for agent in self.registry.core.agents.values():
                if hasattr(agent, 'capability_learner'):
                    for pattern in common_patterns:
                        agent.capability_learner.learn(pattern)

            return f"Consolidated memories across {len(all_memories)} agents"

    def _consolidate_memories(self, memories: Dict) -> Dict:
        """Compress and organize memories"""
        # Implement memory compression logic
        # Remove duplicates, summarize patterns, etc.
        return memories # Placeholder

    def _extract_common_patterns(self, all_memories: Dict[str, Any]) -> list:
        """Extract common patterns from all agent memories"""
        # Implement logic to find common successful patterns
        return [] # Placeholder

plugin_class = MemoryConsolidatorPlugin
