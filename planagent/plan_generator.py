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


class ExistingFeature(BaseModel):
    name: str
    status: str = "implemented"
    location: str = ""
    details: str = ""


class ArchitecturePlan(BaseModel):
    project_name: str = "Untitled"
    description: str = ""
    existing_features: list[ExistingFeature] = Field(default_factory=list)
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

CRITICAL — EXISTING FUNCTIONALITY:
If "EXISTING FEATURES (ALREADY IMPLEMENTED)" is listed below, these features ALREADY EXIST in the codebase.
You MUST:
1. List them in the "existing_features" array (so the developer knows what's already built)
2. Do NOT create duplicate modules, endpoints, or entities for features that already exist
3. Only plan NEW functionality that doesn't overlap with existing features
4. If a requested feature partially overlaps (e.g., user wants "social login" but basic "login" exists),
   mention the existing part in existing_features and only plan the NEW delta

Required JSON structure:
{
  "project_name": "...",
  "description": "...",
  "existing_features": [{"name": "...", "status": "implemented", "location": "file/path.py::function_or_class", "details": "brief description of what exists"}],
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

REMEMBER: modules_v1, modules_v2, and api_endpoints should contain ONLY NEW things to build.
existing_features should contain everything the scan detected as already implemented.

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

    # Existing features with locations — critical for avoiding duplication
    existing_ctx = _build_existing_features_context(state)
    if existing_ctx:
        parts.append(existing_ctx)

    # RAG: inject architecture knowledge for plan generation
    rag_chunks = state.get("rag_context", [])
    if rag_chunks:
        try:
            from planagent.knowledge.retriever import format_chunks_for_prompt
            rag_text = format_chunks_for_prompt(rag_chunks, max_chars=1500)
            if rag_text:
                parts.append(f"\nArchitecture reference material:\n{rag_text}")
        except Exception:
            pass

    # Full conversation context from ConversationStore — gives the LLM
    # complete context at plan generation time (not sent during conversation)
    store = state.get("_conversation_store")
    if store is not None:
        full_convo = store.get_full_context_for_plan()
        if full_convo:
            parts.append(f"\nFull conversation transcript:\n{full_convo}")

    return "\n".join(parts) if parts else "No structured requirements collected."


def _build_existing_features_context(state: dict) -> str:
    """Build a detailed summary of existing functionality with file locations.
    This is injected into the plan prompt so the LLM avoids duplicating features.
    Covers ALL types: auth, payments, models, routes, background tasks, etc."""
    discovered = []

    # Primary source: discovered_features from context_reader scan
    ex = state.get("existing_summary", {})
    if ex:
        discovered = ex.get("discovered_features", [])

    # Fallback: check context_index directly
    if not discovered:
        idx = state.get("context_index", {})
        if idx:
            discovered = idx.get("discovered_features", [])

    if not discovered:
        return ""

    # Only include medium/high confidence features (low = too speculative)
    relevant = [f for f in discovered if f.get("confidence") in ("high", "medium")]
    if not relevant:
        return ""

    lines = ["\n--- EXISTING FEATURES (ALREADY IMPLEMENTED) ---"]
    lines.append("These features are ALREADY in the codebase. Do NOT duplicate them.")

    for feat in relevant:
        name = feat["name"]
        confidence = feat["confidence"]
        evidence = feat.get("evidence", [])
        locations = feat.get("locations", [])

        header = f"\n[{confidence.upper()}] {name}"
        lines.append(header)

        # Show structured locations with file paths
        if locations:
            for loc in locations[:5]:
                loc_type = loc.get("type", "")
                if loc_type == "route":
                    fn = loc.get("function", "")
                    path = loc.get("path", "")
                    file = loc.get("file", "")
                    line_num = loc.get("line", 0)
                    ref = f"  - Route: {path}"
                    if file:
                        ref += f" -> {file}"
                        if fn:
                            ref += f"::{fn}"
                        if line_num:
                            ref += f" (line {line_num})"
                    lines.append(ref)
                elif loc_type == "model":
                    file = loc.get("file", "")
                    model_name = loc.get("name", "")
                    fields = loc.get("fields", [])
                    line_num = loc.get("line", 0)
                    ref = f"  - Model: {model_name}"
                    if fields:
                        ref += f"({', '.join(fields[:6])})"
                    if file:
                        ref += f" in {file}"
                    if line_num:
                        ref += f" (line {line_num})"
                    lines.append(ref)
                elif loc_type == "function":
                    file = loc.get("file", "")
                    fn_name = loc.get("name", "")
                    line_num = loc.get("line", 0)
                    ref = f"  - Function: {fn_name}"
                    if file:
                        ref += f" in {file}"
                    if line_num:
                        ref += f" (line {line_num})"
                    lines.append(ref)
                elif loc_type == "folder":
                    lines.append(f"  - Module: {loc.get('path', '')}")
                elif loc_type == "import":
                    lines.append(f"  - Package: {loc.get('package', '')}")
                elif loc_type == "env":
                    lines.append(f"  - Config: {loc.get('key', '')}")
                elif loc_type in ("decorator", "enum"):
                    file = loc.get("file", "")
                    dec_name = loc.get("name", "")
                    ref = f"  - {loc_type.title()}: {dec_name}"
                    if file:
                        ref += f" in {file}"
                    lines.append(ref)
        elif evidence:
            for ev in evidence[:3]:
                lines.append(f"  - {ev}")

    lines.append("--- END EXISTING FEATURES ---")
    return "\n".join(lines)


def _inject_existing_features_from_scan(state: dict) -> list[dict]:
    """Fallback: build existing_features list from scan data when LLM omits them.
    Ensures the developer always sees what's already implemented."""
    discovered = []
    ex = state.get("existing_summary", {})
    if ex:
        discovered = ex.get("discovered_features", [])
    if not discovered:
        idx = state.get("context_index", {})
        if idx:
            discovered = idx.get("discovered_features", [])

    result = []
    for feat in discovered:
        if feat.get("confidence") not in ("high", "medium"):
            continue
        entry = {"name": feat["name"], "status": "implemented", "location": "", "details": ""}

        # Build location string from structured locations
        locations = feat.get("locations", [])
        loc_parts = []
        detail_parts = []
        for loc in locations[:3]:
            loc_type = loc.get("type", "")
            if loc_type == "route":
                file = loc.get("file", "")
                fn = loc.get("function", "")
                path = loc.get("path", "")
                if file and fn:
                    loc_parts.append(f"{file}::{fn}")
                elif file:
                    loc_parts.append(file)
                if path:
                    detail_parts.append(f"route {path}")
            elif loc_type == "model":
                file = loc.get("file", "")
                name = loc.get("name", "")
                if file:
                    loc_parts.append(f"{file}::{name}")
                fields = loc.get("fields", [])
                if fields:
                    detail_parts.append(f"model {name}({', '.join(fields[:4])})")
            elif loc_type == "function":
                file = loc.get("file", "")
                name = loc.get("name", "")
                if file:
                    loc_parts.append(f"{file}::{name}")
            elif loc_type == "folder":
                loc_parts.append(loc.get("path", ""))
            elif loc_type == "import":
                detail_parts.append(f"uses {loc.get('package', '')}")

        if loc_parts:
            entry["location"] = ", ".join(dict.fromkeys(loc_parts))  # dedupe, preserve order
        if detail_parts:
            entry["details"] = "; ".join(dict.fromkeys(detail_parts))
        else:
            # Fallback to evidence strings
            evidence = feat.get("evidence", [])
            if evidence:
                entry["details"] = "; ".join(evidence[:2])

        result.append(entry)

    return result


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
            proposal = plan.model_dump()

            # Ensure existing features are always populated from scan
            # even if LLM didn't return them (fallback injection)
            if not proposal.get("existing_features"):
                proposal["existing_features"] = _inject_existing_features_from_scan(state)

            state["proposal"] = proposal
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

    # Existing features (already implemented)
    existing = plan.get("existing_features", [])
    if existing:
        console.print(f"\n[bold green]Existing features ({len(existing)}):[/bold green]")
        for ef in existing:
            name = ef["name"] if isinstance(ef, dict) else ef.name
            loc = ef.get("location", "") if isinstance(ef, dict) else ef.location
            details = ef.get("details", "") if isinstance(ef, dict) else ef.details
            line = f"  [green]✓[/green] {name}"
            if loc:
                line += f" [dim]({loc})[/dim]"
            if details:
                line += f" — {details}"
            console.print(line)

    console.print("[bold purple]" + "_" * 50 + "[/bold purple]\n")