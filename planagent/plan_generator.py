import json,re
from rich.console import Console
from planagent.llm import chat

console = Console()

PLAN_PROMPT = """
based on the conversation below,generate a backend architecture plan. Return ONLY valid JSON -
no markdown fences, no explanation,just JSON.

Required structure:
{
"project_name":"...",
"description": "...",
"tech_stack": {
    "language": "...",
    "framework": "...",
    "database": "...",
    "auth": "...",
    "cache": "...",
    "queue": "...",
    "other_tools": [...]
},
"modules_v1":[
{
    "name": "...",
    "description": "...",
    "entities":["..."]
}
],
"modules_v2":[
{
    "name": "...",
    "description": "...",
    "entities":["..."]
}
],
"api_endpoints":[
    {
        "method": "...",
        "path": "...",
        "description": "...",
        "auth_required": false
    }
],
"folder_structure":["/app","app/users","..."],
"design_patterns":["Repository pattern","Service layer"]
}

Conversation:
{history}
"""

def generate_plan(state:dict) -> dict:
    history_text = "\n".join([
        f"{m['role'].upper()}:{m['content']}"
        for m in state["conversation_history"]
    ])
    prompt = PLAN_PROMPT.replace("{history}",history_text)
    messages = [
        {"role": "system", "content": prompt}
    ]
    
    console.print("[dim] Generating architecture plan...[/dim]")
    response = chat(messages,stream=False)

    # Extract Json from response
    match  = re.search(r'\{.*\}',response, re.DOTALL)
    if match:
        try:
            state["proposal"] = json.loads(match.group())
        except json.JSONDecodeError:
            state["proposal"] = {"raw":response}
    else:
        state["proposal"] = {"raw":response}
    return state

def display_proposal(state:dict) -> None:
    """shows the plan to the developer before writing files."""
    plan = state["proposal"]
    console.print("\n[bold purple]" + "_" * 50 + "[/bold purple]")
    console.print(f"[bold]Project:[/bold] {plan.get('project_name','N/A')}")
    console.print(f"[bold]Description:[/bold] {plan.get('description','')}\n")

    stack = plan.get("tech_stack",{})
    console.print("[bold]Tech Stack:[/bold]")
    for k,v in stack.items():
        if v: console.print(f" {k.ljust(12)}: {v}")

    v1 = plan.get("modules_v1",[])
    if v1:
        console.print(f"\n[bold]V1 modules ({len(v1)}) : [/bold]")
        for m in v1:
            console.print(f" - {m['name']}: {m.get('description','')}")

    v2 = plan.get("modules_v2",[])
    if v2:
        console.print(f"\n[bold]V2 modules ({len(v2)}) : [/bold]")
        for m in v2:
            console.print(f" - {m['name']}: {m.get('description','')}")

    eps = plan.get("api_endpoints",[])
    console.print(f"\n[bold]API endpoints planned:[/bold] {len(eps)}")
    console.print("[bold purple]" + "_" * 50 + "[/bold purple]\n")


    


    
    