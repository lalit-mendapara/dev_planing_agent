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

    files = {
        "plan.md": _plan_md(state),
        "architecture.md": _arch_md(plan),
        "api_contracts.md": _api_md(plan),
        "roadmap.md": _roadmap_md(plan),
    }

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
             f"{plan.get('description','')}\n",
             "## V1 Features\n"]
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

def _roadmap_md(plan):
    lines = ["# Roadmap\n","## V1 - Launch \n"]
    for m in plan.get("modules_v1",[]):
        lines.append(f"- [ ] {m['name']}")
    lines += ["\n ## V2 - After Launch\n"]
    for m in plan.get("modules_v2",[]):
        lines.append(f"- [ ] {m['name']}")
    return "\n".join(lines)