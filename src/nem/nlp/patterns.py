"""NLP patterns and utilities for NEMWAS"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Types of NLP patterns"""
    INTENT = "intent"
    ENTITY = "entity"
    MODIFIER = "modifier"
    TEMPORAL = "temporal"
    QUANTITY = "quantity"


@dataclass
class Pattern:
    """NLP pattern definition"""
    name: str
    type: PatternType
    regex: str
    examples: List[str]
    priority: int = 0


# Intent recognition patterns
INTENT_PATTERNS = {
    "create_agent": [
        Pattern(
            name="create_agent_explicit",
            type=PatternType.INTENT,
            regex=r"\b(?:create|spawn|start|initialize|build|make)\s+(?:an?\s+)?(?:new\s+)?agent\b",
            examples=["create an agent", "spawn new agent", "build agent"],
            priority=10
        ),
        Pattern(
            name="create_agent_purpose",
            type=PatternType.INTENT,
            regex=r"\b(?:i\s+)?need\s+an?\s+agent\s+(?:for|to|that)\b",
            examples=["I need an agent for data analysis", "need an agent to help"],
            priority=8
        ),
    ],

    "execute_task": [
        Pattern(
            name="execute_explicit",
            type=PatternType.INTENT,
            regex=r"\b(?:execute|run|perform|do)\s+(?:the\s+)?(?:following\s+)?(?:task|command|action)\b",
            examples=["execute task", "run the following command", "perform action"],
            priority=9
        ),
        Pattern(
            name="execute_implicit",
            type=PatternType.INTENT,
            regex=r"^(?:please\s+)?(?:can\s+you\s+)?(?:help\s+me\s+)?(.+?)(?:\?)?$",
            examples=["calculate 2+2", "please help me analyze this", "can you search for"],
            priority=5
        ),
    ],

    "query_status": [
        Pattern(
            name="status_general",
            type=PatternType.INTENT,
            regex=r"\b(?:what(?:'s|is)\s+)?(?:the\s+)?(?:current\s+)?status\b",
            examples=["what's the status", "current status", "status"],
            priority=10
        ),
        Pattern(
            name="status_agent",
            type=PatternType.INTENT,
            regex=r"\b(?:how\s+(?:is|are)|check)\s+(?:the\s+)?agent(?:s)?\b",
            examples=["how are the agents", "check agent", "how is agent performing"],
            priority=9
        ),
        Pattern(
            name="status_metrics",
            type=PatternType.INTENT,
            regex=r"\b(?:show|get|display)\s+(?:me\s+)?(?:the\s+)?(?:performance\s+)?metrics\b",
            examples=["show metrics", "get performance metrics", "display the metrics"],
            priority=9
        ),
    ],

    "analyze_performance": [
        Pattern(
            name="analyze_explicit",
            type=PatternType.INTENT,
            regex=r"\b(?:analyze|analyse|review|evaluate)\s+(?:the\s+)?performance\b",
            examples=["analyze performance", "review the performance", "evaluate performance"],
            priority=10
        ),
        Pattern(
            name="analyze_trends",
            type=PatternType.INTENT,
            regex=r"\b(?:show|what)\s+(?:are\s+)?(?:the\s+)?(?:performance\s+)?trends?\b",
            examples=["show trends", "what are the performance trends", "show me trends"],
            priority=8
        ),
    ],

    "manage_capabilities": [
        Pattern(
            name="list_capabilities",
            type=PatternType.INTENT,
            regex=r"\b(?:list|show|display|what\s+are)\s+(?:the\s+)?(?:available\s+)?capabilities\b",
            examples=["list capabilities", "what are the capabilities", "show available capabilities"],
            priority=10
        ),
        Pattern(
            name="what_can_do",
            type=PatternType.INTENT,
            regex=r"\bwhat\s+can\s+(?:the\s+)?(?:agent|you|it)\s+do\b",
            examples=["what can the agent do", "what can you do", "what can it do"],
            priority=9
        ),
    ],

    "configure_system": [
        Pattern(
            name="config_set",
            type=PatternType.INTENT,
            regex=r"\b(?:set|change|update|modify)\s+(?:the\s+)?(?:config|configuration|setting)\b",
            examples=["set configuration", "change the config", "update settings"],
            priority=10
        ),
        Pattern(
            name="config_device",
            type=PatternType.INTENT,
            regex=r"\b(?:use|switch\s+to|enable)\s+(?:the\s+)?(?:npu|gpu|cpu)\b",
            examples=["use NPU", "switch to GPU", "enable CPU"],
            priority=9
        ),
    ],
}


# Entity extraction patterns
ENTITY_PATTERNS = {
    "agent_reference": [
        Pattern(
            name="agent_by_name",
            type=PatternType.ENTITY,
            regex=r"\bagent\s+(?:named?|called)\s+([a-zA-Z0-9_-]+)\b",
            examples=["agent named DataAnalyzer", "agent called test-1"],
            priority=10
        ),
        Pattern(
            name="agent_by_id",
            type=PatternType.ENTITY,
            regex=r"\bagent\s+(?:id\s+)?([a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})\b",
            examples=["agent 12345678-1234-1234-1234-123456789012", "agent id abc123"],
            priority=10
        ),
    ],

    "temporal": [
        Pattern(
            name="time_relative",
            type=PatternType.TEMPORAL,
            regex=r"\b(?:last|past|previous|next)\s+(\d+)\s*(second|minute|hour|day|week|month)s?\b",
            examples=["last 5 minutes", "past 24 hours", "previous week"],
            priority=8
        ),
        Pattern(
            name="time_specific",
            type=PatternType.TEMPORAL,
            regex=r"\b(?:since|from|after|before)\s+(\d{1,2}:\d{2}(?:\s*[ap]m)?|\d{4}-\d{2}-\d{2})\b",
            examples=["since 3:00pm", "from 2024-01-01", "after 14:30"],
            priority=8
        ),
    ],

    "quantity": [
        Pattern(
            name="number_simple",
            type=PatternType.QUANTITY,
            regex=r"\b(\d+(?:\.\d+)?)\b",
            examples=["10", "3.14", "1000"],
            priority=5
        ),
        Pattern(
            name="number_percentage",
            type=PatternType.QUANTITY,
            regex=r"\b(\d+(?:\.\d+)?)\s*%\b",
            examples=["50%", "99.9%", "100 %"],
            priority=7
        ),
        Pattern(
            name="number_range",
            type=PatternType.QUANTITY,
            regex=r"\b(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\b",
            examples=["10 to 20", "1-100", "0.5 - 1.0"],
            priority=6
        ),
    ],

    "device": [
        Pattern(
            name="device_type",
            type=PatternType.ENTITY,
            regex=r"\b(npu|gpu|cpu|myriad)\b",
            examples=["NPU", "GPU", "CPU", "MYRIAD"],
            priority=10
        ),
    ],

    "model": [
        Pattern(
            name="model_name",
            type=PatternType.ENTITY,
            regex=r"\b(?:model|llm)\s+([a-zA-Z0-9_/-]+(?:-\d+[bBmM])?)\b",
            examples=["model llama-7b", "llm mistral-7B", "model gpt-3.5"],
            priority=9
        ),
    ],
}


# Modifier patterns
MODIFIER_PATTERNS = {
    "urgency": [
        Pattern(
            name="urgent_high",
            type=PatternType.MODIFIER,
            regex=r"\b(?:urgent|asap|immediately|right\s+now|emergency)\b",
            examples=["urgent", "need this asap", "immediately"],
            priority=10
        ),
        Pattern(
            name="urgent_low",
            type=PatternType.MODIFIER,
            regex=r"\b(?:when\s+you\s+can|no\s+rush|low\s+priority|later|eventually)\b",
            examples=["when you can", "no rush", "low priority"],
            priority=8
        ),
    ],

    "politeness": [
        Pattern(
            name="polite_request",
            type=PatternType.MODIFIER,
            regex=r"\b(?:please|kindly|would\s+you|could\s+you|if\s+possible)\b",
            examples=["please", "could you kindly", "would you mind"],
            priority=5
        ),
    ],

    "certainty": [
        Pattern(
            name="uncertain",
            type=PatternType.MODIFIER,
            regex=r"\b(?:maybe|perhaps|possibly|might|probably|i\s+think)\b",
            examples=["maybe", "I think", "possibly"],
            priority=6
        ),
        Pattern(
            name="certain",
            type=PatternType.MODIFIER,
            regex=r"\b(?:definitely|certainly|surely|must|need\s+to|have\s+to)\b",
            examples=["definitely", "must", "need to"],
            priority=7
        ),
    ],
}


# Common task patterns
TASK_PATTERNS = {
    "calculation": [
        r"(?:calculate|compute|what\s+is|solve)\s+(.+)",
        r"(\d+)\s*([+\-*/^])\s*(\d+)",
        r"(?:sum|product|difference|quotient)\s+of\s+(.+)",
    ],

    "search": [
        r"(?:search|find|look\s+up|look\s+for)\s+(.+)",
        r"(?:information|info|data)\s+(?:about|on|regarding)\s+(.+)",
        r"what\s+(?:is|are)\s+(.+)",
    ],

    "analysis": [
        r"(?:analyze|analyse|examine|evaluate|assess)\s+(.+)",
        r"(?:review|inspect|investigate)\s+(.+)",
        r"(?:breakdown|break\s+down)\s+(.+)",
    ],

    "generation": [
        r"(?:create|generate|make|build|write)\s+(?:a\s+|an\s+)?(.+)",
        r"(?:draft|compose|prepare)\s+(?:a\s+|an\s+)?(.+)",
    ],
}


class PatternMatcher:
    """Utility class for pattern matching"""

    def __init__(self):
        self.compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, List[Tuple[Pattern, re.Pattern]]]:
        """Compile all patterns for efficiency"""
        compiled = {}

        # Compile intent patterns
        for intent, patterns in INTENT_PATTERNS.items():
            compiled[intent] = [
                (p, re.compile(p.regex, re.IGNORECASE))
                for p in patterns
            ]

        # Compile entity patterns
        for entity_type, patterns in ENTITY_PATTERNS.items():
            compiled[f"entity_{entity_type}"] = [
                (p, re.compile(p.regex, re.IGNORECASE))
                for p in patterns
            ]

        # Compile modifier patterns
        for modifier_type, patterns in MODIFIER_PATTERNS.items():
            compiled[f"modifier_{modifier_type}"] = [
                (p, re.compile(p.regex, re.IGNORECASE))
                for p in patterns
            ]

        return compiled

    def match_intent(self, text: str) -> List[Tuple[str, Pattern, re.Match]]:
        """Match text against intent patterns"""
        matches = []

        for intent, pattern_list in self.compiled_patterns.items():
            if not intent.startswith("entity_") and not intent.startswith("modifier_"):
                for pattern, compiled_regex in pattern_list:
                    match = compiled_regex.search(text)
                    if match:
                        matches.append((intent, pattern, match))

        # Sort by priority (highest first)
        matches.sort(key=lambda x: x[1].priority, reverse=True)
        return matches

    def extract_entities(self, text: str) -> Dict[str, List[Tuple[Pattern, str]]]:
        """Extract entities from text"""
        entities = {}

        for key, pattern_list in self.compiled_patterns.items():
            if key.startswith("entity_"):
                entity_type = key[7:]  # Remove "entity_" prefix

                for pattern, compiled_regex in pattern_list:
                    matches = compiled_regex.findall(text)
                    if matches:
                        if entity_type not in entities:
                            entities[entity_type] = []

                        # Handle both simple matches and group matches
                        for match in matches:
                            if isinstance(match, tuple):
                                entities[entity_type].append((pattern, match[0]))
                            else:
                                entities[entity_type].append((pattern, match))

        return entities

    def extract_modifiers(self, text: str) -> Dict[str, List[Pattern]]:
        """Extract modifiers from text"""
        modifiers = {}

        for key, pattern_list in self.compiled_patterns.items():
            if key.startswith("modifier_"):
                modifier_type = key[9:]  # Remove "modifier_" prefix

                for pattern, compiled_regex in pattern_list:
                    if compiled_regex.search(text):
                        if modifier_type not in modifiers:
                            modifiers[modifier_type] = []
                        modifiers[modifier_type].append(pattern)

        return modifiers

    def classify_task(self, text: str) -> Optional[str]:
        """Classify the type of task"""
        for task_type, patterns in TASK_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return task_type
        return None


# Utility functions
def normalize_text(text: str) -> str:
    """Normalize text for better pattern matching"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Expand contractions
    contractions = {
        "what's": "what is",
        "it's": "it is",
        "can't": "cannot",
        "won't": "will not",
        "i'm": "i am",
        "you're": "you are",
        "don't": "do not",
        "doesn't": "does not",
    }

    for contraction, expansion in contractions.items():
        text = re.sub(rf"\b{contraction}\b", expansion, text, flags=re.IGNORECASE)

    return text


def extract_quoted_strings(text: str) -> List[str]:
    """Extract quoted strings from text"""
    # Match both single and double quotes
    pattern = r'["\']([^"\']+)["\']'
    return re.findall(pattern, text)


def extract_code_blocks(text: str) -> List[str]:
    """Extract code blocks from text"""
    # Match code blocks with backticks
    pattern = r'```(?:[a-zA-Z]+\n)?(.*?)```'
    blocks = re.findall(pattern, text, re.DOTALL)

    # Also match inline code
    inline_pattern = r'`([^`]+)`'
    inline_code = re.findall(inline_pattern, text)

    return blocks + inline_code


def is_question(text: str) -> bool:
    """Check if text is a question"""
    text = text.strip()

    # Check for question mark
    if text.endswith('?'):
        return True

    # Check for question words
    question_words = ['what', 'where', 'when', 'who', 'why', 'how', 'which', 'whose', 'whom']
    first_word = text.split()[0].lower() if text.split() else ""

    return first_word in question_words


def split_compound_request(text: str) -> List[str]:
    """Split compound requests into individual tasks"""
    # Split on conjunctions and punctuation
    delimiters = [' and ', ' then ', '. ', ', and ', ', then ', '; ']

    parts = [text]
    for delimiter in delimiters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(delimiter))
        parts = new_parts

    # Clean and filter parts
    return [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
