import typer 
from pathlib import Path
from rich.console import Console
from planagent.state import create_initial_state
from planagent.context_reader import read_context
from planagent.conversation_manager import run_conversation
from planagent.plan_generator import generate_plan,display_proposal
from planagent.output_writer import write_all_outputs
app = typer.Typer()
console = Console()

@app.command()
def plan(path:str = typer.Argument(".", help ="path to your project folder")):
    """Plan & Architect Agent — designs your backend through conversation."""

    console.print("\n [bold purple] Plan & Architect Agent v1.0[/bold purple]")
    console.print("[dim]" + "-" * 50 + "[/dim]")
    
    #Phase 0: Context detection
    state = create_initial_state()
    abs_path = str(Path(path).resolve())
    console.print(f"[dim]Scanning {abs_path}...[/dim]")
    state = read_context(abs_path,state)

    if state["scenario"] == "empty":
        console.print(f"[dim]No existing files found. Starting Fresh.[/dim]")
    else:
        s = state["existing_summary"]
        console.print(f"[dim]Found:{s['language']} / {s['framework']} - {s['file_count']} files[/dim]")
        console.print("[dim]" + "-" * 50 + "[/dim] \n")

    #Phase 1: Conversation
    state = run_conversation(state)
    if not state["conversation_complete"]:
        console.print("[yellow]Session ended before completion.[/yellow]")
        raise typer.Exit()

    #Phase 2: Genarate and review plan
    state = generate_plan(state)
    display_proposal(state)

    # Approval loop
    while True:
        if typer.confirm("Ready to write the files?"):
            break
        change = console.input("what do you want to change?")
        state["conversation_history"].append({"role":"user", "content":change})
        state = generate_plan(state)
        display_proposal(state)

    # Phase 3: Write files
    console.print("[dim]Writing files...[/dim]")
    state = write_all_outputs(state)

    console.print(f"\n[bold green] All done. [/bold green]")

if __name__ == "__main__":
    app()
    
    
