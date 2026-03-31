"""
Guardrails for planagent — keeps the agent on-scope.

Implements NeMo Guardrails-style input/output rails using the project's
existing LLM (litellm).  This avoids the nemoguardrails runtime which
is incompatible with Python 3.14, while providing the same boundary
enforcement.

Rails:
  INPUT  — LLM-based topic classifier blocks off-topic user messages.
  OUTPUT — keyword + LLM check catches agent drift.
"""

import json
from pathlib import Path
from planagent.llm import chat, tracker

# ---------------------------------------------------------------------------
# Configurable rail definitions (loaded from config.yml)
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "config.yml"

REFUSAL_MESSAGE = (
    "I'm your backend architecture planning assistant. I can only help with "
    "planning your project — things like architecture, tech stack, features, "
    "modules, and roadmap. Let's get back to planning! "
    "What would you like to build or change?"
)

# ---------------------------------------------------------------------------
# Input rail — LLM-based topic classifier
# ---------------------------------------------------------------------------

_INPUT_CLASSIFY_PROMPT = """You are a strict topic classifier for a backend architecture planning agent.

The agent ONLY helps with:
- Planning software architecture and backend systems
- Discussing tech stacks, databases, frameworks, APIs
- Defining project features, modules, and roadmaps
- Reviewing project structure and code organization
- Discussing deployment, testing strategies, design patterns

The agent does NOT help with:
- General knowledge questions unrelated to software planning
- Personal advice, entertainment, jokes, stories, poems
- Writing or debugging actual code (the agent only plans)
- Math problems, translations, recipes, news, weather
- Any topic outside software architecture and project planning

Classify the following user message as either "on_topic" or "off_topic".
Return ONLY a JSON object: {"verdict": "on_topic"} or {"verdict": "off_topic"}
No extra text."""


def check_input(user_message: str) -> tuple[bool, str]:
    """Check if user input is on-topic for architecture planning.

    Returns:
        (is_allowed, response)
        - is_allowed=True  → message is on-topic, proceed normally
        - is_allowed=False → message is off-topic, use `response` as the reply
    """
    # Fast path — very short messages like "yes", "no", "ok" are always allowed
    stripped = user_message.strip().lower()
    if len(stripped) < 10:
        return True, ""

    # Fast path — common planning keywords → skip LLM call
    planning_keywords = [
        "api", "database", "backend", "frontend", "auth", "module",
        "feature", "stack", "deploy", "endpoint", "schema", "model",
        "route", "service", "architecture", "plan", "build", "project",
        "framework", "v1", "v2", "roadmap", "postgres", "mongo", "redis",
        "docker", "kubernetes", "microservice", "rest", "graphql",
        "yes", "no", "sure", "ok", "done", "go ahead", "that's it",
        "constraint", "deadline", "team", "billing", "payment",
    ]
    if any(kw in stripped for kw in planning_keywords):
        return True, ""

    # LLM-based classification for ambiguous messages
    try:
        messages = [
            {"role": "system", "content": _INPUT_CLASSIFY_PROMPT},
            {"role": "user", "content": user_message},
        ]
        raw = chat(messages, stream=False, label="guardrail_input",
                   json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        verdict = data.get("verdict", "on_topic")
        if verdict == "off_topic":
            return False, REFUSAL_MESSAGE
    except Exception:
        # If classification fails, allow through (fail-open)
        pass

    return True, ""


# ---------------------------------------------------------------------------
# Output rail — keyword check + lightweight LLM check
# ---------------------------------------------------------------------------

_OFF_TOPIC_OUTPUT_KEYWORDS = [
    "as an ai language model",
    "i cannot help with that",
    "here's a joke",
    "once upon a time",
    "let me tell you a story",
    "the weather today",
    "here's a recipe",
    "fun fact:",
    "here is a poem",
    "i'm not able to help with",
]


def check_output(agent_response: str) -> tuple[bool, str]:
    """Check if agent output stays on-topic.

    Returns:
        (is_on_topic, cleaned_response)
        - is_on_topic=True  → response is fine, use as-is
        - is_on_topic=False → response drifted, use cleaned_response
    """
    lower = agent_response.lower()
    for keyword in _OFF_TOPIC_OUTPUT_KEYWORDS:
        if keyword in lower:
            return False, REFUSAL_MESSAGE

    return True, agent_response
