"""Natural Language Interface for NEMWAS"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Types of user intents"""
    CREATE_AGENT = "create_agent"
    EXECUTE_TASK = "execute_task"
    QUERY_STATUS = "query_status"
    ANALYZE_PERFORMANCE = "analyze_performance"
    MANAGE_CAPABILITIES = "manage_capabilities"
    CONFIGURE_SYSTEM = "configure_system"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class ParsedIntent:
    """Parsed user intent"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    original_text: str


class NaturalLanguageInterface:
    """Natural language processing for user interactions"""

    def __init__(self):
        # Intent patterns
        self.intent_patterns = {
            IntentType.CREATE_AGENT: [
                r"create\s+(?:an?\s+)?agent",
                r"spawn\s+(?:an?\s+)?agent",
                r"start\s+(?:an?\s+)?(?:new\s+)?agent",
                r"initialize\s+(?:an?\s+)?agent",
                r"build\s+(?:an?\s+)?agent\s+(?:for|to)",
            ],
            IntentType.EXECUTE_TASK: [
                r"(?:please\s+)?(?:can\s+you\s+)?(.+)",
                r"i\s+need\s+(?:you\s+)?to\s+(.+)",
                r"help\s+me\s+(.+)",
                r"execute\s+(.+)",
                r"run\s+(.+)",
                r"do\s+(.+)",
            ],
            IntentType.QUERY_STATUS: [
                r"(?:what(?:'s|is)\s+)?(?:the\s+)?status",
                r"how\s+(?:is|are)\s+(?:the\s+)?agent",
                r"show\s+(?:me\s+)?metrics",
                r"(?:get|show)\s+performance",
                r"list\s+agents?",
            ],
            IntentType.ANALYZE_PERFORMANCE: [
                r"analyze\s+performance",
                r"(?:show|what)\s+(?:is|are)\s+(?:the\s+)?trend",
                r"performance\s+report",
                r"optimization\s+recommend",
                r"benchmark",
            ],
            IntentType.MANAGE_CAPABILITIES: [
                r"(?:list|show)\s+capabilities",
                r"what\s+can\s+(?:the\s+)?agent\s+do",
                r"add\s+capability",
                r"learn\s+(?:to|how)",
                r"forget\s+capability",
            ],
            IntentType.CONFIGURE_SYSTEM: [
                r"(?:set|change|update)\s+config",
                r"configure\s+(.+)",
                r"(?:enable|disable)\s+(.+)",
                r"use\s+(?:npu|gpu|cpu)",
                r"(?:set|change)\s+model",
            ],
            IntentType.HELP: [
                r"help",
                r"what\s+can\s+you\s+do",
                r"how\s+(?:do\s+i|to)",
                r"show\s+commands",
                r"documentation",
            ],
        }

        # Entity extraction patterns
        self.entity_patterns = {
            'agent_name': r"agent\s+(?:named?|called)\s+(\w+)",
            'task_description': r"(?:to|for)\s+(.+?)(?:\.|$)",
            'model_name': r"model\s+(\S+)",
            'device': r"(?:npu|gpu|cpu|myriad)",
            'number': r"\d+",
            'percentage': r"\d+(?:\.\d+)?%",
            'time_period': r"\d+\s*(?:second|minute|hour|day)s?",
        }

        logger.info("Natural Language Interface initialized")

    def parse(self, text: str) -> ParsedIntent:
        """Parse user input to extract intent and entities"""

        text = text.lower().strip()

        # Try to match intents
        best_match = None
        best_confidence = 0.0

        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Calculate confidence based on match quality
                    confidence = self._calculate_confidence(match, text)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = (intent_type, match)

        if best_match:
            intent_type, match = best_match
            entities = self._extract_entities(text, intent_type, match)
        else:
            # Default to task execution if no specific intent matched
            intent_type = IntentType.EXECUTE_TASK
            entities = {'task': text}
            best_confidence = 0.5

        return ParsedIntent(
            intent_type=intent_type,
            confidence=best_confidence,
            entities=entities,
            original_text=text
        )

    def _calculate_confidence(self, match: re.Match, text: str) -> float:
        """Calculate confidence score for intent match"""

        # Base confidence on match coverage
        match_length = len(match.group(0))
        text_length = len(text)
        coverage = match_length / text_length

        # Boost confidence if match is at the beginning
        position_boost = 0.2 if match.start() == 0 else 0.0

        # Calculate final confidence
        confidence = min(0.9, coverage + position_boost)

        return confidence

    def _extract_entities(self, text: str, intent_type: IntentType, match: re.Match) -> Dict[str, Any]:
        """Extract entities based on intent type"""

        entities = {}

        # Extract common entities
        for entity_name, pattern in self.entity_patterns.items():
            entity_match = re.search(pattern, text, re.IGNORECASE)
            if entity_match:
                entities[entity_name] = entity_match.group(1) if entity_match.groups() else entity_match.group(0)

        # Intent-specific extraction
        if intent_type == IntentType.CREATE_AGENT:
            # Extract agent configuration
            if 'for' in text:
                task_match = re.search(r'for\s+(.+?)(?:\.|$)', text)
                if task_match:
                    entities['purpose'] = task_match.group(1)

            # Extract model preference
            if 'with' in text:
                config_match = re.search(r'with\s+(.+?)(?:\.|$)', text)
                if config_match:
                    entities['configuration'] = config_match.group(1)

        elif intent_type == IntentType.EXECUTE_TASK:
            # Extract task details
            if match.groups():
                entities['task'] = match.group(1)
            else:
                entities['task'] = text

            # Extract urgency
            if any(word in text for word in ['urgent', 'asap', 'immediately', 'now']):
                entities['priority'] = 'high'
            elif any(word in text for word in ['when you can', 'later', 'eventually']):
                entities['priority'] = 'low'
            else:
                entities['priority'] = 'normal'

        elif intent_type == IntentType.QUERY_STATUS:
            # Extract specific agent or metric
            if 'agent' in text:
                agent_match = re.search(r'agent\s+(\w+)', text)
                if agent_match:
                    entities['agent_id'] = agent_match.group(1)

            if any(word in text for word in ['metric', 'performance', 'stats']):
                entities['query_type'] = 'metrics'
            else:
                entities['query_type'] = 'status'

        elif intent_type == IntentType.ANALYZE_PERFORMANCE:
            # Extract time period
            time_match = re.search(r'(?:last|past)\s+(\d+\s*(?:hour|day|week)s?)', text)
            if time_match:
                entities['time_period'] = time_match.group(1)

            # Extract specific metrics
            metrics = []
            if 'latency' in text:
                metrics.append('latency')
            if 'throughput' in text:
                metrics.append('throughput')
            if 'accuracy' in text or 'success' in text:
                metrics.append('success_rate')

            if metrics:
                entities['metrics'] = metrics

        return entities

    def generate_response(self, intent: ParsedIntent, result: Any) -> str:
        """Generate natural language response based on intent and result"""

        if intent.intent_type == IntentType.CREATE_AGENT:
            if isinstance(result, dict) and result.get('success'):
                agent_id = result.get('agent_id', 'unknown')
                return f"I've successfully created a new agent (ID: {agent_id}) for you. The agent is now ready to handle tasks."
            else:
                return "I encountered an issue while creating the agent. Please check the configuration and try again."

        elif intent.intent_type == IntentType.EXECUTE_TASK:
            if isinstance(result, str):
                return result  # Agent already provided natural response
            elif isinstance(result, dict):
                if result.get('success'):
                    return f"Task completed successfully. {result.get('message', '')}"
                else:
                    return f"I couldn't complete the task. {result.get('error', 'Unknown error occurred.')}"

        elif intent.intent_type == IntentType.QUERY_STATUS:
            if isinstance(result, dict):
                return self._format_status_response(result)
            else:
                return "Status information is currently unavailable."

        elif intent.intent_type == IntentType.ANALYZE_PERFORMANCE:
            if isinstance(result, dict):
                return self._format_performance_response(result)
            else:
                return "Performance analysis is not available at this time."

        elif intent.intent_type == IntentType.MANAGE_CAPABILITIES:
            if isinstance(result, list):
                return self._format_capabilities_response(result)
            else:
                return str(result)

        elif intent.intent_type == IntentType.CONFIGURE_SYSTEM:
            if result.get('success'):
                return f"Configuration updated successfully. {result.get('message', '')}"
            else:
                return f"Failed to update configuration: {result.get('error', 'Unknown error')}"

        elif intent.intent_type == IntentType.HELP:
            return self._generate_help_text()

        else:
            return "I processed your request, but I'm not sure how to format the response."

    def _format_status_response(self, status: Dict[str, Any]) -> str:
        """Format status information as natural language"""

        response_parts = []

        if 'system' in status:
            sys = status['system']
            response_parts.append(
                f"System Status: {sys.get('active_agents', 0)} active agents, "
                f"{sys.get('total_tasks', 0)} total tasks processed"
            )

        if 'agent' in status:
            agent = status['agent']
            response_parts.append(
                f"Agent Performance: {agent.get('success_rate', 0):.1%} success rate, "
                f"{agent.get('avg_execution_time', 0):.2f}s average execution time"
            )

        if 'resources' in status:
            res = status['resources']
            response_parts.append(
                f"Resource Usage: {res.get('cpu_percent', 0):.1f}% CPU, "
                f"{res.get('memory_percent', 0):.1f}% Memory"
            )

            if 'npu_utilization' in res and res['npu_utilization'] > 0:
                response_parts.append(f"NPU Utilization: {res['npu_utilization']:.1f}%")

        return "\n".join(response_parts) if response_parts else "No status information available."

    def _format_performance_response(self, analysis: Dict[str, Any]) -> str:
        """Format performance analysis as natural language"""

        response_parts = []

        # Trend summary
        trend = analysis.get('trend', 'unknown')
        if trend == 'improving':
            response_parts.append("Good news! Performance has been improving.")
        elif trend == 'degrading':
            response_parts.append("Performance has been degrading and may need attention.")
        else:
            response_parts.append("Performance has been stable.")

        # Key metrics
        if 'current_avg_time' in analysis:
            response_parts.append(
                f"Current average execution time: {analysis['current_avg_time']:.2f}s"
            )

        if 'success_rate' in analysis:
            response_parts.append(
                f"Success rate: {analysis['success_rate']:.1%}"
            )

        # Device performance
        if 'optimal_device' in analysis and analysis['optimal_device']:
            response_parts.append(
                f"Optimal device: {analysis['optimal_device']}"
            )

        # Recommendations
        if 'recommendations' in analysis and analysis['recommendations']:
            response_parts.append("\nRecommendations:")
            for rec in analysis['recommendations']:
                response_parts.append(f"• {rec}")

        return "\n".join(response_parts)

    def _format_capabilities_response(self, capabilities: List[Dict[str, Any]]) -> str:
        """Format capabilities list as natural language"""

        if not capabilities:
            return "No capabilities have been learned yet."

        response_parts = ["Here are the available capabilities:"]

        for i, cap in enumerate(capabilities, 1):
            response_parts.append(
                f"{i}. {cap['name']} - {cap.get('description', 'No description')}"
                f" (Success rate: {cap.get('success_rate', 0):.1%})"
            )

        return "\n".join(response_parts)

    def _generate_help_text(self) -> str:
        """Generate help text for users"""

        return """I'm NEMWAS, your Neural-Enhanced Multi-Workforce Agent System. Here's what I can do:

**Agent Management:**
• Create agents: "Create an agent for data analysis"
• Execute tasks: "Analyze this dataset" or "Calculate the sum of..."
• Check status: "Show agent status" or "What's the current performance?"

**Performance Analysis:**
• View metrics: "Show performance metrics"
• Analyze trends: "Analyze performance over the last hour"
• Get recommendations: "How can I optimize performance?"

**Capabilities:**
• List capabilities: "What can the agents do?"
• View learned skills: "Show capabilities"

**Configuration:**
• Change settings: "Use NPU for inference"
• Update configuration: "Set model to Mistral-7B"

**Tips:**
• I learn from successful task completions to improve over time
• NPU acceleration is used automatically when available
• You can be conversational - I'll understand your intent

What would you like to do?"""

    def suggest_completions(self, partial_text: str) -> List[str]:
        """Suggest command completions for partial input"""

        suggestions = []
        partial_lower = partial_text.lower().strip()

        # Common command starters
        command_templates = [
            "Create an agent for ",
            "Show agent status",
            "Analyze performance",
            "List capabilities",
            "Help me with ",
            "Execute task: ",
            "Configure ",
            "Enable NPU acceleration",
            "Show metrics for ",
        ]

        # Find matching templates
        for template in command_templates:
            if template.lower().startswith(partial_lower):
                suggestions.append(template)
            elif partial_lower and partial_lower in template.lower():
                suggestions.append(template)

        # Limit suggestions
        return suggestions[:5]
