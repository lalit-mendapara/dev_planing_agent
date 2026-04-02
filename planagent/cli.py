import json
import typer
from pathlib import Path
from rich.prompt import Prompt
from planagent.state import create_initial_state
from planagent.context_reader import read_context
from planagent.conversation_manager import run_conversation, prefill_state_from_scan
from planagent.plan_generator import generate_plan
from planagent.output_writer import write_all_outputs, write_token_report
from planagent.ui import (
    console, render_welcome, render_proposal,
    render_version_history, render_diff, render_edit_result,
    render_revision_welcome, EDIT_SECTIONS,
)
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(help="Plan & Architect Agent — 10x faster project planning.")


# ---------------------------------------------------------------------------
# planagent plan  (original command — unchanged flow)
# ---------------------------------------------------------------------------

@app.command()
def plan(path: str = typer.Argument(".", help="path to your project folder")):
    """Generate a new architecture plan through conversation."""

    abs_path = str(Path(path).resolve())
    planagent_dir = Path(abs_path) / ".planagent"

    # Check if a plan already exists
    if planagent_dir.exists() and (planagent_dir / "context.json").exists():
        console.print(Panel(
            Markdown(
                "A plan already exists for this project.\n\n"
                "To modify it, use one of these commands:\n\n"
                "- **planagent revise** — update your existing plan via conversation\n"
                "- **planagent edit <section>** — edit a specific section (low token)\n"
                "- **planagent show** — view the current plan\n\n"
                "To start fresh, delete the `.planagent/` folder first:\n\n"
                "```\nrm -rf .planagent\nplanagent plan\n```"
            ),
            title="[warning]Plan Already Exists[/warning]",
            title_align="left",
            border_style="yellow",
            padding=(1, 2),
        ))
        raise typer.Exit()

    # Phase 0: Context detection (with caching)
    state = create_initial_state()
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


# ---------------------------------------------------------------------------
# planagent revise  — load existing plan and continue conversation
# ---------------------------------------------------------------------------

@app.command()
def revise(path: str = typer.Argument(".", help="path to your project folder")):
    """Resume planning from an existing .planagent/ plan. Add features, change stack, refine architecture."""

    from planagent.plan_manager import (
        load_existing_plan, reconstruct_state_from_plan,
        build_revision_context, apply_plan_update,
    )
    from planagent.conversation_manager import REVISION_SYSTEM_ADDENDUM

    abs_path = str(Path(path).resolve())
    existing_plan = load_existing_plan(abs_path)

    if not existing_plan:
        console.print("[red]No existing plan found in .planagent/context.json[/red]")
        console.print("[dim]Run 'planagent plan' first to generate one.[/dim]")
        raise typer.Exit(1)

    # Show current plan summary
    render_revision_welcome(existing_plan)

    # Rebuild state from existing plan
    state = reconstruct_state_from_plan(abs_path, existing_plan)

    # Re-scan project for fresh context (uses cache if unchanged)
    console.print(f"[dim]Scanning {abs_path}...[/dim]")
    state = read_context(abs_path, state)
    state = prefill_state_from_scan(state)

    # Inject revision context into conversation
    revision_ctx = build_revision_context(existing_plan)
    state["conversation_history"].append({
        "role": "system",
        "content": REVISION_SYSTEM_ADDENDUM.replace("{plan_summary}", revision_ctx),
    })

    # Phase 1: Conversation (revision mode)
    state["conversation_complete"] = False
    state = run_conversation(state)
    if not state["conversation_complete"]:
        console.print("[yellow]Session ended before completion.[/yellow]")
        raise typer.Exit()

    # Phase 2: Regenerate plan with revisions
    state = generate_plan(state)
    render_proposal(state["proposal"])

    # Approval loop
    while True:
        approve = Prompt.ask(
            "[bold]Ready to write the updated files?[/bold]",
            choices=["yes", "no"],
            default="yes",
        )
        if approve == "yes":
            break
        change = Prompt.ask("[user]What do you want to change?[/user]")
        state["conversation_history"].append({"role": "user", "content": change})
        state = generate_plan(state)
        render_proposal(state["proposal"])

    # Phase 3: Write (auto-snapshots old version)
    console.print("[dim]Writing updated files...[/dim]")
    state["_snapshot_message"] = "Before revision update"
    state = write_all_outputs(state)
    write_token_report(state)

    console.print("\n[bold green]✅ Plan revised successfully.[/bold green]\n")


# ---------------------------------------------------------------------------
# planagent edit  — targeted section editing (lowest token usage)
# ---------------------------------------------------------------------------

@app.command()
def edit(
    section: str = typer.Argument(
        ...,
        help="Section to edit: modules, modules_v2, api, stack, folders, patterns, description",
    ),
    path: str = typer.Option(".", help="path to your project folder"),
):
    """Edit a specific plan section without full regeneration. Ultra-low token usage."""

    from planagent.plan_manager import (
        load_existing_plan, edit_plan_section, apply_plan_update,
    )
    from planagent.llm import tracker

    abs_path = str(Path(path).resolve())

    if section not in EDIT_SECTIONS:
        console.print(f"[red]Unknown section '{section}'.[/red]")
        console.print(f"[dim]Available: {', '.join(EDIT_SECTIONS.keys())}[/dim]")
        raise typer.Exit(1)

    existing_plan = load_existing_plan(abs_path)
    if not existing_plan:
        console.print("[red]No existing plan found. Run 'planagent plan' first.[/red]")
        raise typer.Exit(1)

    section_key, section_label = EDIT_SECTIONS[section]

    # Show current section content
    current = existing_plan.get(section_key, {})
    console.print(Panel(
        f"```json\n{json.dumps(current, indent=2, default=str)}\n```" if not isinstance(current, str)
        else current,
        title=f"[dim]Current {section_label}[/dim]",
        border_style="dim",
        padding=(1, 2),
    ))

    # Get edit request
    edit_request = Prompt.ask(f"[user]What do you want to change in {section_label}?[/user]")
    if not edit_request.strip():
        console.print("[dim]No changes requested.[/dim]")
        raise typer.Exit()

    # Targeted LLM edit — sends ONLY the section, not the whole plan
    console.print("[dim]Applying targeted edit...[/dim]")
    updated_plan = edit_plan_section(abs_path, section_key, edit_request)

    if updated_plan:
        render_proposal(updated_plan)
        approve = Prompt.ask(
            "[bold]Apply this change?[/bold]",
            choices=["yes", "no"],
            default="yes",
        )
        if approve == "yes":
            apply_plan_update(abs_path, updated_plan)
            render_edit_result(section_label, True, tracker.total)
        else:
            console.print("[dim]Edit discarded.[/dim]")
    else:
        render_edit_result(section_label, False)


# ---------------------------------------------------------------------------
# planagent history  — show version history
# ---------------------------------------------------------------------------

@app.command()
def history(path: str = typer.Argument(".", help="path to your project folder")):
    """Show plan version history."""

    from planagent.plan_manager import list_versions

    abs_path = str(Path(path).resolve())
    versions = list_versions(abs_path)
    render_version_history(versions)


# ---------------------------------------------------------------------------
# planagent diff  — compare plan versions
# ---------------------------------------------------------------------------

@app.command()
def diff(
    v1: int = typer.Argument(..., help="First version number"),
    v2: int = typer.Argument(0, help="Second version (0 = current)"),
    file: str = typer.Option("plan.md", help="File to diff"),
    path: str = typer.Option(".", help="path to your project folder"),
):
    """Compare two plan versions (or a version vs current)."""

    from planagent.plan_manager import diff_versions, diff_current_vs_version

    abs_path = str(Path(path).resolve())

    if v2 == 0:
        result = diff_current_vs_version(abs_path, v1, file)
        title = f"v{v1} → current ({file})"
    else:
        result = diff_versions(abs_path, v1, v2, file)
        title = f"v{v1} → v{v2} ({file})"

    render_diff(result, title=title)


# ---------------------------------------------------------------------------
# planagent rollback  — restore a previous version
# ---------------------------------------------------------------------------

@app.command()
def rollback(
    version: int = typer.Argument(..., help="Version number to restore"),
    path: str = typer.Option(".", help="path to your project folder"),
):
    """Rollback plan to a previous version (current is auto-snapshotted first)."""

    from planagent.plan_manager import rollback_to_version, load_existing_plan

    abs_path = str(Path(path).resolve())

    approve = Prompt.ask(
        f"[bold]Rollback to v{version}? Current plan will be snapshotted first.[/bold]",
        choices=["yes", "no"],
        default="yes",
    )
    if approve != "yes":
        console.print("[dim]Rollback cancelled.[/dim]")
        raise typer.Exit()

    success = rollback_to_version(abs_path, version)
    if success:
        console.print(f"\n[bold green]✅ Restored plan from v{version}.[/bold green]")
        plan = load_existing_plan(abs_path)
        if plan:
            render_proposal(plan)
    else:
        console.print(f"[red]Version v{version} not found.[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# planagent show  — display current plan
# ---------------------------------------------------------------------------

@app.command()
def show(path: str = typer.Argument(".", help="path to your project folder")):
    """Display the current architecture plan."""

    from planagent.plan_manager import load_existing_plan

    abs_path = str(Path(path).resolve())
    plan = load_existing_plan(abs_path)
    if not plan:
        console.print("[red]No plan found. Run 'planagent plan' first.[/red]")
        raise typer.Exit(1)
    render_proposal(plan)


if __name__ == "__main__":
    app()
