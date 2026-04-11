import json
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()


def write_all_outputs(state: dict) -> dict:
    root = Path(state["project_root"])
    out = root / ".planagent"
    out.mkdir(exist_ok=True)
    plan = state["proposal"]
    written = []

    # Auto-snapshot existing plan before overwriting
    from planagent.plan_manager import snapshot_current
    existing_ctx = out / "context.json"
    if existing_ctx.exists():
        msg = state.get("_snapshot_message", "Auto-snapshot before plan update")
        version = snapshot_current(state["project_root"], message=msg)
        if version:
            state["plan_version"] = version
            console.print(f"   [dim]📸 Saved v{version} snapshot[/dim]")

    files = {
        "plan.md": _plan_md(state),
        "architecture.md": _arch_md(plan),
        "api_contracts.md": _api_md(plan),
        "roadmap.md": _roadmap_md(plan),
    }

    # Generate existing_features.md when existing features are detected
    existing = plan.get("existing_features", [])
    if existing:
        files["existing_features.md"] = _existing_features_md(plan, state)

    for name, content in files.items():
        p = out / name
        p.write_text(content, encoding="utf-8")
        written.append(str(p))
        console.print(f"   [green]✅[/green] Wrote {name}")

    # context.json - machine-readable, read by agent 2
    ctx = {
        **plan,
        "scenario": state["scenario"],
        "generated_by": "plan-architect-agent",
        "generated_at": datetime.now().isoformat(),
        "next_agent": "schema-db-agent",
    }
    p = out / "context.json"
    p.write_text(json.dumps(ctx, indent=2), encoding="utf-8")
    written.append(str(p))
    console.print(f"   [green]✅[/green] context.json [dim] (bridge to agent 2) [/dim]")

    state["files_written"] = written
    return state


# ---------------------------------------------------------------------------
# Token report — always written, detailed breakdown
# ---------------------------------------------------------------------------

def write_token_report(state: dict) -> str:
    """Write token usage report to .planagent/token_report.json.
    Returns the path to the report file."""
    root = Path(state["project_root"])
    out = root / ".planagent"
    out.mkdir(exist_ok=True)

    usage = state.get("token_usage", {})
    calls = usage.get("calls", [])

    # Group calls by category
    categories = {}
    for call in calls:
        label = call.get("label", "unknown")
        if "conversation_turn" in label:
            cat = "chat"
        elif "state_extraction" in label:
            cat = "extract"
        elif "conversation_summary" in label:
            cat = "summary"
        elif "plan_generation" in label:
            cat = "plan"
        elif "_cached" in label:
            cat = "cached"
        else:
            cat = "other"

        if cat not in categories:
            categories[cat] = {"count": 0, "input": 0, "output": 0, "total": 0}
        categories[cat]["count"] += 1
        categories[cat]["input"] += call.get("input_tokens", 0)
        categories[cat]["output"] += call.get("output_tokens", 0)
        categories[cat]["total"] += call.get("total_tokens", 0)

    report = {
        "generated_at": datetime.now().isoformat(),
        "cache_hit": state.get("cache_hit", False),
        "scenario": state.get("scenario", "unknown"),
        "totals": {
            "total_calls": len(calls),
            "total_input_tokens": usage.get("total_input_tokens", 0),
            "total_output_tokens": usage.get("total_output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "by_category": categories,
        "per_call_detail": calls,
    }

    report_path = out / "token_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")

    console.print(f"   [green]✅[/green] token_report.json")
    return str(report_path)


# ---------------------------------------------------------------------------
# Markdown generators (unchanged)
# ---------------------------------------------------------------------------

def _plan_md(state):
    plan = state["proposal"]
    lines = [f"# {plan.get('project_name','project')} - plan\n",
             f"{plan.get('description','')}\n"]

    # Existing features notice
    existing = plan.get("existing_features", [])
    if existing:
        lines.append("## Already Implemented\n")
        lines.append(f"> {len(existing)} features are already built in the existing codebase.")
        lines.append("> See [existing_features.md](existing_features.md) for full details with file locations.\n")
        for ef in existing:
            name = ef.get("name", "")
            loc = ef.get("location", "")
            entry = f"- **{name}**"
            if loc:
                entry += f" — `{loc}`"
            lines.append(entry)
        lines.append("")

    lines.append("## V1 Features (New)\n")
    for m in plan.get("modules_v1", []):
        lines.append(f"- **{m['name']}**: {m.get('description','')}")
    lines += ["\n ## V2 Features  (Deffered) \n"]
    for m in plan.get("modules_v2",[]):
        lines.append(f"- **{m['name']}**: {m.get('description','')}")

    return "\n".join(lines)

def _arch_md(plan):
    stack = plan.get("tech_stack",{})
    patterns = plan.get("design_patterns",[])
    folders = plan.get("folder_structure",[])

    lines = ["# Architecture\n","## tech stack \n"]

    for k,v in stack.items():
        if v: 
            lines.append(f"- **{k}**: {v}")
    lines += ["\n ## Design Patterns\n"]
    for p in patterns: 
        lines.append(f"- {p}")
    lines += ['\n ## folder Structure\n ```']
    lines += folders
    lines.append('```')

    return "\n".join(lines)


def _api_md(plan):
    eps = plan.get("api_endpoints",[])
    lines = ["# API contracts\n",
             "| Method | Path | Description | Auth required |",
             "|--------|------|-------------|---------------|"]
    for ep in eps:
        auth = "Yes" if ep.get("auth_required") else "No"
        lines.append(
            f"| {ep.get('method','')} | `{ep.get('path','')}` |"
            f" {ep.get('description','')} | {auth} |"
        )
    
    return "\n".join(lines)

def _existing_features_md(plan: dict, state: dict) -> str:
    """Generate existing_features.md — documents what's already implemented with locations.
    This covers ALL types of existing functionality: auth, payments, models, routes,
    background tasks, file uploads, search, notifications, etc."""
    existing = plan.get("existing_features", [])
    if not existing:
        return ""

    lines = [
        "# Existing Features\n",
        "This document lists features and functionality **already implemented** in your project.",
        "These were auto-detected by scanning your codebase.\n",
        "> **Do not rebuild these.** Integrate with them when building new features.\n",
    ]

    # Group features by confidence/category for readability
    for ef in existing:
        name = ef.get("name", "Unknown")
        status = ef.get("status", "implemented")
        location = ef.get("location", "")
        details = ef.get("details", "")

        lines.append(f"## {name}\n")
        lines.append(f"- **Status**: {status}")
        if location:
            lines.append(f"- **Location**: `{location}`")
        if details:
            lines.append(f"- **Details**: {details}")
        lines.append("")

    # Enrich with raw scan data (routes, models, functions) if available
    idx = state.get("context_index", {})
    discovered = []
    ex = state.get("existing_summary", {})
    if ex:
        discovered = ex.get("discovered_features", [])
    if not discovered and idx:
        discovered = idx.get("discovered_features", [])

    # Build a lookup of feature names already shown
    shown_names = {ef.get("name", "").lower() for ef in existing}

    # Add detailed location appendix from scan data
    has_appendix = False
    for feat in discovered:
        if feat.get("confidence") not in ("high", "medium"):
            continue
        locations = feat.get("locations", [])
        if not locations:
            continue
        feat_name = feat.get("name", "")
        if feat_name.lower() not in shown_names:
            continue

        if not has_appendix:
            lines.append("---\n")
            lines.append("# Detailed File Locations\n")
            lines.append("Precise file paths and line numbers for each existing feature.\n")
            has_appendix = True

        lines.append(f"### {feat_name}\n")
        lines.append("| Type | Location | Details |")
        lines.append("|------|----------|---------|")

        for loc in locations[:8]:
            loc_type = loc.get("type", "")
            file = loc.get("file", "")
            line_num = loc.get("line", 0)

            if loc_type == "route":
                path = loc.get("path", "")
                fn = loc.get("function", "")
                loc_str = f"`{file}`" if file else ""
                if fn:
                    loc_str += f" :: `{fn}()`"
                if line_num:
                    loc_str += f" (line {line_num})"
                lines.append(f"| Route | {loc_str} | `{path}` |")

            elif loc_type == "model":
                model_name = loc.get("name", "")
                fields = loc.get("fields", [])
                loc_str = f"`{file}`" if file else ""
                if line_num:
                    loc_str += f" (line {line_num})"
                detail = f"`{model_name}`"
                if fields:
                    detail += f" — fields: {', '.join(f'`{f}`' for f in fields[:6])}"
                lines.append(f"| Model | {loc_str} | {detail} |")

            elif loc_type == "function":
                fn_name = loc.get("name", "")
                loc_str = f"`{file}`" if file else ""
                if line_num:
                    loc_str += f" (line {line_num})"
                lines.append(f"| Function | {loc_str} | `{fn_name}()` |")

            elif loc_type == "folder":
                path = loc.get("path", "")
                lines.append(f"| Module | `{path}` | directory |")

            elif loc_type == "import":
                pkg = loc.get("package", "")
                lines.append(f"| Package | — | `{pkg}` |")

            elif loc_type == "env":
                key = loc.get("key", "")
                lines.append(f"| Config | `.env` | `{key}` |")

            elif loc_type in ("decorator", "enum"):
                dec_name = loc.get("name", "")
                loc_str = f"`{file}`" if file else ""
                lines.append(f"| {loc_type.title()} | {loc_str} | `{dec_name}` |")

        lines.append("")

    lines.append("---\n")
    lines.append(f"*Generated by PlanAgent on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)


def _roadmap_md(plan):
    lines = ["# Roadmap\n"]

    # Already implemented (no work needed)
    existing = plan.get("existing_features", [])
    if existing:
        lines.append("## Already Implemented (No Work Needed)\n")
        for ef in existing:
            lines.append(f"- [x] {ef.get('name', '')}")
        lines.append("")

    lines.append("## V1 - Launch \n")
    for m in plan.get("modules_v1",[]):
        lines.append(f"- [ ] {m['name']}")
    lines += ["\n ## V2 - After Launch\n"]
    for m in plan.get("modules_v2",[]):
        lines.append(f"- [ ] {m['name']}")
    return "\n".join(lines)