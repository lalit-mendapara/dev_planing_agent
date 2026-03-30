import typer
from pathlib import Path
from rich.prompt import Prompt
from planagent.state import create_initial_state
from planagent.context_reader import read_context
from planagent.conversation_manager import run_conversation, prefill_state_from_scan
from planagent.plan_generator import generate_plan
from planagent.output_writer import write_all_outputs, write_token_report
from planagent.ui import console, render_welcome, render_proposal

app = typer.Typer()


@app.command()
def plan(path: str = typer.Argument(".", help="path to your project folder")):
    """Plan & Architect Agent — designs your backend through conversation."""

    # Phase 0: Context detection (with caching)
    state = create_initial_state()
    abs_path = str(Path(path).resolve())
    console.print(f"[dim]Scanning {abs_path}...[/dim]")
    state = read_context(abs_path, state)

    # Welcome banner with project info
    summary = state.get("existing_summary")
    if summary:
        summary["_cache_hit"] = state.get("cache_hit", False)
    render_welcome(scenario=state["scenario"], summary=summary)

    # Pre-populate state from scan (tech stack, user types, gaps)
    state = prefill_state_from_scan(state)

    # Phase 1: Conversation (sliding window + incremental extraction)
    # Token usage is shown live after every turn
    state = run_conversation(state)
    if not state["conversation_complete"]:
        console.print("[yellow]Session ended before completion.[/yellow]")
        raise typer.Exit()

    # Phase 2: Generate and review plan (structured state + pydantic validation)
    state = generate_plan(state)
    render_proposal(state["proposal"])

    # Approval loop
    while True:
        approve = Prompt.ask(
            "[bold]Ready to write the files?[/bold]",
            choices=["yes", "no"],
            default="yes",
        )
        if approve == "yes":
            break
        change = Prompt.ask("[user]What do you want to change?[/user]")
        state["conversation_history"].append({"role": "user", "content": change})
        state = generate_plan(state)
        render_proposal(state["proposal"])

    # Phase 3: Write files + token report
    console.print("[dim]Writing files...[/dim]")
    state = write_all_outputs(state)
    write_token_report(state)

    console.print("\n[bold green]✅ All done.[/bold green]\n")


if __name__ == "__main__":
    app()
