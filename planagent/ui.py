"""
Claude Code-style terminal UI for planagent.
Renders Markdown panels, multiple-choice prompts, token bars, and banners.
"""

import re
import json
from datetime import datetime
import questionary
from questionary import Style as QStyle
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich.prompt import Prompt

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = Theme({
    "agent": "bold cyan",
    "user": "bold purple",
    "option": "bold white",
    "option.key": "bold cyan",
    "option.desc": "white",
    "dim": "dim",
    "success": "bold green",
    "warning": "bold yellow",
    "token.input": "cyan",
    "token.output": "green",
    "token.total": "bold yellow",
})

console = Console(theme=THEME)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIONS_PATTERN = re.compile(
    r"```options\s*(.*?)\s*```", re.DOTALL
)
OPTIONS_MULTI_PATTERN = re.compile(
    r"```options-multi\s*(.*?)\s*```", re.DOTALL
)

BACK_COMMAND = "/back"
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)

BANNER = r"""
 ╔═══════════════════════════════════════════════════╗
 ║        Plan & Architect Agent  v3.0               ║
 ║        ─────────────────────────────              ║
 ║        Interactive Planning Mode                  ║
 ╚═══════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Parse options from LLM response
# ---------------------------------------------------------------------------

def parse_agent_response(raw: str) -> tuple[str, list[str], bool]:
    """Split LLM output into (message_text, options_list, multi_select).

    The LLM is instructed to embed options like:
        ```options
        ["Option A description", "Option B description", "Option C description"]
        ```
    Or for multi-select (user can pick more than one):
        ```options-multi
        ["Feature A", "Feature B", "Feature C", "Feature D"]
        ```

    Returns:
        message_text: The response with the options block stripped out.
        options: List of option strings (may be empty).
        multi_select: True if options-multi block was used.
    """
    # Check multi-select first (more specific pattern)
    multi_match = OPTIONS_MULTI_PATTERN.search(raw)
    if multi_match:
        options_raw = multi_match.group(1).strip()
        message_text = OPTIONS_MULTI_PATTERN.sub("", raw).strip()
        try:
            options = json.loads(options_raw)
            if isinstance(options, list) and all(isinstance(o, str) for o in options):
                return message_text, options[:6], True  # cap at 6 for multi
        except (json.JSONDecodeError, TypeError):
            pass
        return raw.strip(), [], False

    # Single-select
    match = OPTIONS_PATTERN.search(raw)
    if not match:
        return raw.strip(), [], False

    options_raw = match.group(1).strip()
    message_text = OPTIONS_PATTERN.sub("", raw).strip()

    try:
        options = json.loads(options_raw)
        if isinstance(options, list) and all(isinstance(o, str) for o in options):
            return message_text, options[:3], False  # cap at 3
    except (json.JSONDecodeError, TypeError):
        pass

    return raw.strip(), [], False


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------

def render_welcome(scenario: str = "empty", summary: dict | None = None) -> None:
    """Print the styled welcome banner."""
    banner_text = Text(BANNER, style="bold cyan")
    console.print(banner_text)

    if scenario == "existing" and summary:
        lang = summary.get("language", "unknown")
        fw = summary.get("framework", "unknown")
        fc = summary.get("file_count", 0)
        cache_hit = summary.get("_cache_hit", False)
        cache_tag = " [green](cached)[/green]" if cache_hit else " [yellow](fresh scan)[/yellow]"

        info_lines = [
            f"  **Project:** {lang} / {fw}",
            f"  **Files:** {fc}{cache_tag}",
        ]
        classes = summary.get("classes", [])
        routes = summary.get("routes", [])
        if classes or routes:
            info_lines.append(
                f"  **Classes:** {len(classes)}  |  **Routes:** {len(routes)}"
            )

        info_md = Markdown("\n".join(info_lines))
        console.print(Panel(info_md, title="[dim]Project Detected[/dim]",
                            border_style="dim", padding=(0, 2)))
    elif scenario == "empty":
        console.print(
            Panel("[dim]No existing files found. Starting fresh.[/dim]",
                  border_style="dim", padding=(0, 2))
        )
    console.print()


def render_agent_message(text: str, options: list[str] | None = None) -> None:
    """Render the agent's response as a Markdown panel (no options rendering here)."""
    md = Markdown(text)
    console.print(Panel(
        md,
        title="[agent]Agent[/agent]",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
    ))


# ---------------------------------------------------------------------------
# Arrow-key selection (questionary)
# ---------------------------------------------------------------------------

# Style matching our cyan/purple theme
_Q_STYLE = QStyle([
    ("qmark", "fg:cyan bold"),
    ("question", "fg:white bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
    ("answer", "fg:green bold"),
])

CUSTOM_LABEL = "✏  Custom answer..."
BACK_LABEL = "↩  Go back to previous question"


def get_user_choice(options: list[str], multi: bool = False) -> str:
    """Arrow-key selection menu.

    Args:
        options: List of option strings to display.
        multi: If True, use checkbox (space to toggle, enter to confirm).
               User can select multiple items. If False, single-select.

    Returns:
        Chosen option(s) as a string. Multiple selections joined with ' + '.
        Returns BACK_COMMAND if user selects "Go back".
    """
    if multi:
        # Multi-select: checkbox style — retry once on empty selection
        choices = list(options) + [CUSTOM_LABEL, BACK_LABEL]
        for attempt in range(2):
            selected = questionary.checkbox(
                "Select all that apply:",
                choices=choices,
                style=_Q_STYLE,
                instruction="(↑↓ move, Space to toggle, Enter to confirm)",
            ).ask()

            if selected is None:
                # Ctrl-C
                return ""

            # /back takes priority over everything
            if BACK_LABEL in selected:
                return BACK_COMMAND

            # Handle custom answer in multi-select
            has_custom = CUSTOM_LABEL in selected
            picked = [s for s in selected if s != CUSTOM_LABEL]

            if has_custom:
                custom = questionary.text(
                    "Type additional answer:",
                    style=_Q_STYLE,
                ).ask()
                if custom and custom.strip():
                    picked.append(custom.strip())

            if picked:
                result = " + ".join(picked)
                console.print(f"  [dim]→ {result}[/dim]")
                return result

            # Empty selection — hint and retry (or fall through to text input)
            if attempt == 0:
                console.print(
                    "  [warning]No items selected.[/warning] "
                    "[dim]Use Space to toggle items, then Enter to confirm. "
                    "Trying again...[/dim]"
                )

        # Two empty attempts — fall back to free-text input
        console.print("  [dim]Switching to text input...[/dim]")
        custom = questionary.text(
            "Type your answer:",
            style=_Q_STYLE,
        ).ask()
        return custom.strip() if custom else ""

    # Single-select: original behavior
    choices = list(options) + [CUSTOM_LABEL, BACK_LABEL]

    selected = questionary.select(
        "Pick an option:",
        choices=choices,
        style=_Q_STYLE,
        instruction="(↑↓ to move, Enter to select)",
    ).ask()

    if selected is None:
        return ""

    if selected == BACK_LABEL:
        return BACK_COMMAND

    if selected == CUSTOM_LABEL:
        custom = questionary.text(
            "Type your answer:",
            style=_Q_STYLE,
        ).ask()
        return custom.strip() if custom else ""

    console.print(f"  [dim]→ {selected}[/dim]")
    return selected


def get_user_input() -> str:
    """Simple free-text input (no options). Supports /back command."""
    try:
        raw = Prompt.ask("[user]You[/user]  [dim](/back to revise previous answer)[/dim]").strip()
        return raw
    except (KeyboardInterrupt, EOFError):
        return ""


# ---------------------------------------------------------------------------
# Turn saved / go-back notifications
# ---------------------------------------------------------------------------

def render_turn_saved(turn_num: int) -> None:
    """Subtle notification that the turn was saved to context."""
    console.print(
        f"  [dim]\u2714 Turn {turn_num} saved to context[/dim]"
    )


def render_go_back(previous_question: str) -> None:
    """Show the previous agent question when user goes back."""
    console.print(Panel(
        f"[dim]Going back to previous question:[/dim]\n\n{previous_question}",
        title="[warning]\u21a9 Revising Previous Answer[/warning]",
        title_align="left",
        border_style="yellow",
        padding=(1, 2),
    ))


def render_go_back_unavailable() -> None:
    """Shown when user tries /back but there's nothing to go back to."""
    console.print("  [dim]Nothing to go back to — this is the first question.[/dim]")


def strip_code_blocks(text: str) -> str:
    """Remove all fenced code blocks from text (planning mode — no code)."""
    return CODE_BLOCK_PATTERN.sub("", text).strip()


# ---------------------------------------------------------------------------
# Streaming into a Live panel (no raw stdout)
# ---------------------------------------------------------------------------

def stream_to_panel(chat_fn, messages: list, label: str) -> str:
    """Call chat() with streaming and render tokens live inside a Rich panel.

    Uses transient=True so the streaming panel disappears when done.
    The caller is responsible for rendering the final clean panel via
    render_agent_message (which strips ```options blocks, etc.).

    Args:
        chat_fn: The llm.chat function.
        messages: The message list to pass to chat().
        label: Label for token tracking.

    Returns:
        The full response text.
    """
    chunks: list[str] = []

    def _on_token(token: str):
        chunks.append(token)
        text_so_far = "".join(chunks)
        try:
            md = Markdown(text_so_far + " ▌")
        except Exception:
            md = Text(text_so_far + " ▌")
        panel = Panel(
            md,
            title="[agent]Agent[/agent]",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
        )
        live.update(panel)

    with Live(Panel("[dim]Thinking...[/dim]",
                    title="[agent]Agent[/agent]",
                    title_align="left",
                    border_style="cyan",
                    padding=(1, 2)),
              console=console, refresh_per_second=12,
              transient=True) as live:
        response = chat_fn(messages, stream=True, label=label,
                           stream_callback=_on_token)

    return response


# ---------------------------------------------------------------------------
# Token panel
# ---------------------------------------------------------------------------

def render_token_panel(tracker, turn_count: int) -> None:
    """Compact token usage bar after each turn."""
    calls = tracker.calls
    last = calls[-1] if calls else {}

    # Categorize
    cats: dict[str, dict] = {}
    for c in calls:
        label = c.get("label", "")
        if "conversation_turn" in label:
            cat = "chat"
        elif "state_extraction" in label:
            cat = "extract"
        elif "conversation_summary" in label:
            cat = "summary"
        elif "_cached" in label:
            cat = "cached"
        else:
            cat = "other"
        if cat not in cats:
            cats[cat] = {"in": 0, "out": 0}
        cats[cat]["in"] += c.get("input_tokens", 0)
        cats[cat]["out"] += c.get("output_tokens", 0)

    # Build parts
    this_turn = (
        f"[token.input]{last.get('input_tokens', 0)}[/token.input] in / "
        f"[token.output]{last.get('output_tokens', 0)}[/token.output] out"
    )

    breakdown_parts = []
    for cat in ["chat", "extract", "summary", "cached"]:
        if cat in cats:
            total = cats[cat]["in"] + cats[cat]["out"]
            breakdown_parts.append(f"{cat}=[token.total]{total}[/token.total]")

    total_line = (
        f"[bold]Total:[/bold] [token.total]{tracker.total}[/token.total] tokens "
        f"([token.input]{tracker.total_input}[/token.input] in + "
        f"[token.output]{tracker.total_output}[/token.output] out) · "
        f"{len(calls)} calls"
    )

    body = f"This turn: {this_turn}"
    if breakdown_parts:
        body += f"\n{' │ '.join(breakdown_parts)}"
    body += f"\n{total_line}"

    console.print(Panel(
        body,
        title=f"[dim]Turn {turn_count}[/dim]",
        title_align="right",
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Session summary panel
# ---------------------------------------------------------------------------

def render_session_summary(tracker) -> None:
    """Final session token usage panel."""
    calls = tracker.calls
    cats: dict[str, dict] = {}
    for c in calls:
        label = c.get("label", "")
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
        if cat not in cats:
            cats[cat] = {"in": 0, "out": 0, "count": 0}
        cats[cat]["in"] += c.get("input_tokens", 0)
        cats[cat]["out"] += c.get("output_tokens", 0)
        cats[cat]["count"] += 1

    lines = []
    for cat in ["chat", "extract", "summary", "plan", "cached", "other"]:
        if cat in cats:
            d = cats[cat]
            total = d["in"] + d["out"]
            lines.append(
                f"  **{cat.ljust(8)}**  {d['count']} calls │ "
                f"[token.input]{d['in']}[/token.input] in + "
                f"[token.output]{d['out']}[/token.output] out = "
                f"[token.total]{total}[/token.total]"
            )
    lines.append(
        f"\n  **{'TOTAL'.ljust(8)}**  {len(calls)} calls │ "
        f"[token.input]{tracker.total_input}[/token.input] in + "
        f"[token.output]{tracker.total_output}[/token.output] out = "
        f"[token.total]{tracker.total}[/token.total]"
    )

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Session Token Usage[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


# ---------------------------------------------------------------------------
# Proposal display (upgraded)
# ---------------------------------------------------------------------------

def render_version_history(versions: list[dict]) -> None:
    """Render plan version history as a styled table."""
    if not versions:
        console.print(Panel("[dim]No version history found.[/dim]",
                            border_style="dim", padding=(0, 2)))
        return

    lines = []
    for v in versions:
        ver = v.get("version", "?")
        ts = v.get("timestamp", "")
        # Compact timestamp
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                ts = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                pass
        msg = v.get("message", "")
        files = v.get("files", [])
        lines.append(
            f"  **v{ver}**  {ts}  │  {msg}  │  {len(files)} files"
        )

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Plan Version History[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


def render_diff(diff_text: str, title: str = "Diff") -> None:
    """Render a unified diff with color coding."""
    if not diff_text or "No changes" in diff_text:
        console.print(Panel(f"[dim]{diff_text}[/dim]",
                            border_style="dim", padding=(0, 2)))
        return

    styled_lines = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            styled_lines.append(f"[bold]{line}[/bold]")
        elif line.startswith("@@"):
            styled_lines.append(f"[cyan]{line}[/cyan]")
        elif line.startswith("+"):
            styled_lines.append(f"[green]{line}[/green]")
        elif line.startswith("-"):
            styled_lines.append(f"[red]{line}[/red]")
        else:
            styled_lines.append(line)

    console.print(Panel(
        "\n".join(styled_lines),
        title=f"[bold purple]{title}[/bold purple]",
        border_style="purple",
        padding=(1, 2),
    ))
    console.print()


def render_edit_result(section: str, success: bool, token_count: int = 0) -> None:
    """Render result of a targeted plan edit."""
    if success:
        msg = f"[green]✅ Updated [bold]{section}[/bold] successfully[/green]"
        if token_count:
            msg += f"  [dim]({token_count} tokens)[/dim]"
    else:
        msg = f"[red]❌ Failed to update [bold]{section}[/bold][/red]"
    console.print(Panel(msg, border_style="dim", padding=(0, 2)))
    console.print()


def render_revision_welcome(plan: dict) -> None:
    """Welcome banner for revision mode showing current plan summary."""
    name = plan.get("project_name", "Project")
    desc = plan.get("description", "")
    v1_count = len(plan.get("modules_v1", []))
    v2_count = len(plan.get("modules_v2", []))
    ep_count = len(plan.get("api_endpoints", []))

    lines = [
        f"  **Project:** {name}",
        f"  **Description:** {desc[:120]}",
        f"  **Modules:** {v1_count} V1, {v2_count} V2  |  **Endpoints:** {ep_count}",
    ]

    ts = plan.get("tech_stack", {})
    if ts:
        stack_parts = [f"{v}" for k, v in ts.items() if v and k != "other_tools"]
        if stack_parts:
            lines.append(f"  **Stack:** {', '.join(stack_parts[:5])}")

    console.print(Panel(
        Markdown("\n".join(lines)),
        title="[bold cyan]Revision Mode — Existing Plan Loaded[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()


EDIT_SECTIONS = {
    "modules": ("modules_v1", "V1 Modules"),
    "modules_v2": ("modules_v2", "V2 Modules"),
    "api": ("api_endpoints", "API Endpoints"),
    "stack": ("tech_stack", "Tech Stack"),
    "folders": ("folder_structure", "Folder Structure"),
    "patterns": ("design_patterns", "Design Patterns"),
    "description": ("description", "Description"),
}


def render_proposal(plan: dict) -> None:
    """Render the architecture proposal in a styled panel."""
    lines = []
    lines.append(f"# {plan.get('project_name', 'N/A')}\n")
    lines.append(f"{plan.get('description', '')}\n")

    # Tech stack
    stack = plan.get("tech_stack", {})
    if stack:
        lines.append("## Tech Stack\n")
        for k, v in stack.items():
            if v:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    # V1 modules
    v1 = plan.get("modules_v1", [])
    if v1:
        lines.append(f"## V1 Modules ({len(v1)})\n")
        for m in v1:
            name = m["name"] if isinstance(m, dict) else m.name
            desc = m.get("description", "") if isinstance(m, dict) else m.description
            lines.append(f"- **{name}**: {desc}")
        lines.append("")

    # V2 modules
    v2 = plan.get("modules_v2", [])
    if v2:
        lines.append(f"## V2 Modules ({len(v2)})\n")
        for m in v2:
            name = m["name"] if isinstance(m, dict) else m.name
            desc = m.get("description", "") if isinstance(m, dict) else m.description
            lines.append(f"- **{name}**: {desc}")
        lines.append("")

    # API endpoints
    eps = plan.get("api_endpoints", [])
    if eps:
        lines.append(f"## API Endpoints ({len(eps)})\n")
        lines.append("| Method | Path | Description | Auth |")
        lines.append("|--------|------|-------------|------|")
        for ep in eps:
            auth = "✓" if ep.get("auth_required") else "—"
            lines.append(
                f"| {ep.get('method', '')} | `{ep.get('path', '')}` | "
                f"{ep.get('description', '')} | {auth} |"
            )
        lines.append("")

    # Design patterns
    patterns = plan.get("design_patterns", [])
    if patterns:
        lines.append(f"## Design Patterns\n")
        for p in patterns:
            lines.append(f"- {p}")
        lines.append("")

    # Folder structure
    folders = plan.get("folder_structure", [])
    if folders:
        lines.append("## Folder Structure\n")
        lines.append("```")
        lines.extend(folders)
        lines.append("```")

    md = Markdown("\n".join(lines))
    console.print(Panel(
        md,
        title="[bold purple]Architecture Proposal[/bold purple]",
        border_style="purple",
        padding=(1, 2),
    ))
    console.print()
