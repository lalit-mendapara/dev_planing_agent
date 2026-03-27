import json
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from planagent.llm import chat

console = Console()

# ---------------------------------------------------------------------------
# Pydantic models for plan validation
# ---------------------------------------------------------------------------

class TechStack(BaseModel):
    language: str = ""
    framework: str = ""
    database: str = ""
    auth: str = ""
    cache: str = ""
    queue: str = ""
    other_tools: list[str] = Field(default_factory=list)


class Module(BaseModel):
    name: str
    description: str = ""
    entities: list[str] = Field(default_factory=list)


class ApiEndpoint(BaseModel):
    method: str = "GET"
    path: str = "/"
    description: str = ""
    auth_required: bool = False


class ArchitecturePlan(BaseModel):
    project_name: str = "Untitled"
    description: str = ""
    tech_stack: TechStack = Field(default_factory=TechStack)
    modules_v1: list[Module] = Field(default_factory=list)
    modules_v2: list[Module] = Field(default_factory=list)
    api_endpoints: list[ApiEndpoint] = Field(default_factory=list)
    folder_structure: list[str] = Field(default_factory=list)
    design_patterns: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Plan prompt — now uses STRUCTURED STATE instead of raw conversation
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are a backend architecture planner. Based on the structured project requirements below,
generate a complete backend architecture plan. Return ONLY valid JSON — no markdown fences, no explanation.

Required JSON structure:
{
  "project_name": "...",
  "description": "...",
  "tech_stack": {
    "language": "...", "framework": "...", "database": "...",
    "auth": "...", "cache": "...", "queue": "...", "other_tools": [...]
  },
  "modules_v1": [{"name": "...", "description": "...", "entities": ["..."]}],
  "modules_v2": [{"name": "...", "description": "...", "entities": ["..."]}],
  "api_endpoints": [{"method": "...", "path": "...", "description": "...", "auth_required": false}],
  "folder_structure": ["/app", "/app/users", "..."],
  "design_patterns": ["Repository pattern", "Service layer"]
}

PROJECT REQUIREMENTS:
{requirements}"""


MAX_RETRIES = 2


def _build_requirements(state: dict) -> str:
    """Build a compact requirements text from structured state — NOT raw conversation."""
    parts = []

    if state.get("project_goal"):
        parts.append(f"Goal: {state['project_goal']}")
    if state.get("user_types"):
        parts.append(f"User types: {', '.join(state['user_types'])}")
    if state.get("features_v1"):
        parts.append(f"V1 features: {', '.join(state['features_v1'])}")
    if state.get("features_v2"):
        parts.append(f"V2 features (deferred): {', '.join(state['features_v2'])}")

    ts = state.get("tech_stack", {})
    if ts:
        if isinstance(ts, dict):
            stack_parts = [f"{k}: {v}" for k, v in ts.items() if v]
            parts.append(f"Tech stack: {', '.join(stack_parts)}")
        else:
            parts.append(f"Tech stack: {ts}")

    if state.get("constraints"):
        parts.append(f"Constraints: {', '.join(state['constraints'])}")
    if state.get("gaps_confirmed"):
        parts.append(f"Additional requirements (confirmed gaps): {', '.join(state['gaps_confirmed'])}")

    # Include project scan context if available
    tier1 = state.get("context_tier1")
    if tier1:
        parts.append(f"\nExisting project context:\n{tier1}")

    return "\n".join(parts) if parts else "No structured requirements collected."


def generate_plan(state: dict) -> dict:
    """Generate architecture plan from structured state with validation + retry."""
    requirements = _build_requirements(state)
    prompt = PLAN_PROMPT.replace("{requirements}", requirements)

    messages = [
        {"role": "system", "content": prompt},
    ]

    console.print("[dim]Generating architecture plan...[/dim]")

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = chat(messages, stream=False, label="plan_generation",
                            json_mode=True)

            # Clean response
            raw = response.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(raw)

            # Validate with pydantic
            plan = ArchitecturePlan(**data)
            state["proposal"] = plan.model_dump()
            return state

        except (json.JSONDecodeError, ValidationError) as e:
            if attempt < MAX_RETRIES:
                console.print(f"[dim]Plan validation failed (attempt {attempt + 1}), retrying...[/dim]")
                # Add error feedback for retry
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Your previous response had an error: {str(e)[:200]}. "
                                                 "Please fix it and return valid JSON."},
                ]
            else:
                # Last resort: try to salvage what we can
                console.print("[yellow]Warning: Could not fully validate plan, using raw output.[/yellow]")
                if isinstance(e, json.JSONDecodeError):
                    state["proposal"] = {"raw": response, "project_name": state.get("project_goal", "Project")}
                else:
                    # Pydantic failed but JSON parsed — use the raw dict with defaults
                    state["proposal"] = data

    return state


def display_proposal(state: dict) -> None:
    """Shows the plan to the developer before writing files."""
    plan = state["proposal"]
    console.print("\n[bold purple]" + "_" * 50 + "[/bold purple]")
    console.print(f"[bold]Project:[/bold] {plan.get('project_name', 'N/A')}")
    console.print(f"[bold]Description:[/bold] {plan.get('description', '')}\n")

    stack = plan.get("tech_stack", {})
    console.print("[bold]Tech Stack:[/bold]")
    for k, v in stack.items():
        if v:
            console.print(f"  {k.ljust(12)}: {v}")

    v1 = plan.get("modules_v1", [])
    if v1:
        console.print(f"\n[bold]V1 modules ({len(v1)}):[/bold]")
        for m in v1:
            name = m["name"] if isinstance(m, dict) else m.name
            desc = m.get("description", "") if isinstance(m, dict) else m.description
            console.print(f"  - {name}: {desc}")

    v2 = plan.get("modules_v2", [])
    if v2:
        console.print(f"\n[bold]V2 modules ({len(v2)}):[/bold]")
        for m in v2:
            name = m["name"] if isinstance(m, dict) else m.name
            desc = m.get("description", "") if isinstance(m, dict) else m.description
            console.print(f"  - {name}: {desc}")

    eps = plan.get("api_endpoints", [])
    console.print(f"\n[bold]API endpoints planned:[/bold] {len(eps)}")

    patterns = plan.get("design_patterns", [])
    if patterns:
        console.print(f"[bold]Design patterns:[/bold] {', '.join(patterns)}")

    folders = plan.get("folder_structure", [])
    if folders:
        console.print(f"[bold]Folder structure:[/bold] {len(folders)} directories")

    console.print("[bold purple]" + "_" * 50 + "[/bold purple]\n")