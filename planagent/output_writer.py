import json
from pathlib import Path
from datetime import datetime
from rich.console import Console

console = Console()

def write_all_outputs(state:dict) -> dict:
    root = Path(state["project_root"])
    out = root / ".planagent"
    out.mkdir(exist_ok=True)
    plan = state["proposal"]
    written = []

    files = {
        "plan.md": _plan_md(state),
        "architecture.md": _arch_md(plan),
        "api_contracts.md":_api_md(plan),
        "roadmap.md": _roadmap_md(plan),
    }

    for name,content in files.items():
        p = out / name
        p.write_text(content, encoding="utf-8")
        written.append(str(p))
        console.print(f"   [green]✅[/green] Wrote {name}")

    # context.json - machine-readable, read by agent 2
    ctx = {
        **plan,
        "scenario": state["scenario"],
        "generated_by":"plan-architect-agent",
        "generated_at": datetime.now().isoformat(),
        "next_agent":"schema-db-agent",
    }
    p=out / "context.json"
    p.write_text(json.dumps(ctx, indent=2), encoding="utf-8")
    written.append(str(p))
    console.print(f"   [green]✅[/green] context.json [dim] (bridge top agent 2) [/dim]")

    state["files_written"] = written
    return state


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


    
    
    
    
    