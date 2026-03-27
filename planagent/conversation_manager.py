import json
import sys
from pathlib import Path
from rich.console import Console
from planagent.llm import chat, build_messages, tracker

console = Console()
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"
DONE_SIGNAL = "CONVERSATION_COMPLETE"

# ---------------------------------------------------------------------------
# Sliding window config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 6  # keep last N turn-pairs in full; older turns get summarized


# ---------------------------------------------------------------------------
# Context builder — now includes tier1 project summary + dynamic trimming
# ---------------------------------------------------------------------------

def _build_context(state: dict) -> str:
    """Build the CURRENT PROJECT CONTEXT block injected into system prompt.
    Includes tier1 project summary + scan-based signals + collected conversation state.
    Dynamically omits sections already collected to save tokens."""
    lines = [f"Scenario: {state.get('scenario', 'unknown')}"]

    # Tier1 project summary from context_reader (rich, compact)
    tier1 = state.get("context_tier1")
    if tier1:
        lines.append(f"\n--- Project Scan Summary ---\n{tier1}")
    else:
        # Fallback to basic existing_summary
        ex = state.get("existing_summary")
        if ex:
            lines += [
                "Existing project detected:",
                f"  Language: {ex.get('language', 'unknown')}",
                f"  Framework: {ex.get('framework', 'unknown')}",
                f"  Files: {ex.get('file_count', 0)}",
                f"  Folders: {', '.join(ex.get('top_folders', []))}",
            ]

    # Scan-based signals for gap detection
    ex = state.get("existing_summary", {})
    if ex:
        signals = []
        if not ex.get("has_tests"):
            signals.append("NO tests detected")
        ft = ex.get("file_types", {})
        if not ft.get("migration"):
            signals.append("NO migration files detected")
        if not ft.get("config"):
            signals.append("NO config files (.env) detected")
        if signals:
            lines.append(f"\n--- Scan Flags ---\n" + "\n".join(f"⚠ {s}" for s in signals))

    # Only show fields that are already collected (avoid repeating questions)
    if state.get("project_goal"):
        lines.append(f"Project Goal: {state['project_goal']}")
    if state.get("user_types"):
        lines.append(f"User types: {', '.join(state['user_types'])}")
    if state.get("tech_stack"):
        ts = state["tech_stack"]
        if isinstance(ts, dict):
            lines.append(f"Stack confirmed: {json.dumps(ts)}")
        else:
            lines.append(f"Stack confirmed: {ts}")
    if state.get("features_v1"):
        lines.append(f"V1 features: {', '.join(state['features_v1'])}")
    if state.get("features_v2"):
        lines.append(f"V2 features: {', '.join(state['features_v2'])}")
    if state.get("constraints"):
        lines.append(f"Constraints: {', '.join(state['constraints'])}")
    if state.get("gaps_flagged"):
        lines.append(f"Gaps flagged: {', '.join(state['gaps_flagged'])}")
    if state.get("gaps_confirmed"):
        lines.append(f"Gaps confirmed: {', '.join(state['gaps_confirmed'])}")
    if state.get("gaps_deferred"):
        lines.append(f"Gaps deferred: {', '.join(state['gaps_deferred'])}")

    # Show collection progress so the LLM knows what's left
    collected = []
    missing = []
    for field, label in [
        ("project_goal", "project goal"),
        ("user_types", "user types"),
        ("features_v1", "v1 features"),
        ("tech_stack", "tech stack"),
        ("constraints", "constraints"),
    ]:
        if state.get(field):
            collected.append(label)
        else:
            missing.append(label)
    lines.append(f"\nAlready collected: {', '.join(collected) if collected else 'nothing yet'}")
    lines.append(f"Still needed: {', '.join(missing) if missing else 'all collected — ready to finalize'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic system prompt — trim question sections already answered
# ---------------------------------------------------------------------------

def _build_system_prompt(state: dict) -> str:
    """Load system prompt and trim sections for already-collected fields."""
    raw = SYSTEM_PROMPT_PATH.read_text()
    context = _build_context(state)
    prompt = raw.replace("{context}", context)

    # Trim question priority items that are already collected
    # This saves ~20-40 tokens per already-answered section
    if state.get("project_goal"):
        prompt = prompt.replace(
            "1. what are they building (project type, one-line description)\n", "")
    if state.get("user_types"):
        prompt = prompt.replace(
            "2. who are the users (user types, roles, permissions needed)\n", "")
    if state.get("features_v1"):
        prompt = prompt.replace(
            "3. what are the core features (module by module, v1 scope)\n", "")
    if state.get("tech_stack"):
        prompt = prompt.replace(
            "4. what is the tech stack (language, framework, database, auth)\n", "")
    if state.get("constraints"):
        prompt = prompt.replace(
            "5. any constraints (deadline, team size, existing infrastructure)\n", "")
    if state.get("features_v2"):
        prompt = prompt.replace(
            "6. what goes in v2 (features mentioned but not critical for launch)\n", "")

    return prompt


# ---------------------------------------------------------------------------
# Incremental state extraction — only send NEW turns, not full history
# ---------------------------------------------------------------------------

STATE_EXTRACT_PROMPT = """Extract confirmed information from the NEW conversation turns below.
Merge with the existing state provided. Return ONLY a JSON object:
{
  "project_goal": "one-line description or null",
  "user_types": ["list of user roles"] or [],
  "features_v1": ["confirmed v1 features"] or [],
  "features_v2": ["deferred features"] or [],
  "tech_stack": {"language": "...", "framework": "...", "database": "..."} or {},
  "constraints": ["any constraints mentioned"] or [],
  "gaps_flagged": ["missing things agent noticed"] or [],
  "gaps_confirmed": ["gaps dev said yes to"] or [],
  "gaps_deferred": ["gaps dev said no/later to"] or []
}
Return ONLY valid JSON, no extra text."""


def _extract_state_incremental(state: dict) -> dict:
    """Extract state from ONLY the new turns since last extraction.
    Sends existing state + new turns, not the entire conversation."""
    history = state["conversation_history"]
    last_idx = state.get("last_extracted_turn", 0)

    # Only extract if there are new turns
    new_turns = history[last_idx:]
    if len(new_turns) < 2:
        return state

    # Build a compact representation of existing state for context
    existing = {}
    for key in ["project_goal", "user_types", "features_v1", "features_v2",
                "tech_stack", "constraints", "gaps_flagged",
                "gaps_confirmed", "gaps_deferred"]:
        val = state.get(key)
        if val:
            existing[key] = val

    new_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in new_turns
    )

    user_content = f"EXISTING STATE:\n{json.dumps(existing, default=str)}\n\nNEW TURNS:\n{new_text}"

    messages = [
        {"role": "system", "content": STATE_EXTRACT_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = chat(messages, stream=False, label="state_extraction",
                   json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        data = json.loads(raw)
        for key in ["project_goal", "user_types", "features_v1", "features_v2",
                     "tech_stack", "constraints", "gaps_flagged",
                     "gaps_confirmed", "gaps_deferred"]:
            val = data.get(key)
            if val:  # only update if non-empty
                state[key] = val
    except (json.JSONDecodeError, Exception):
        pass  # extraction failed, continue without updating

    state["last_extracted_turn"] = len(history)
    return state


# ---------------------------------------------------------------------------
# Sliding window — summarize old turns, keep recent ones in full
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """Summarize the following older conversation turns into a brief paragraph (max 100 words).
Keep all confirmed decisions, tech choices, and feature lists. Drop greetings and filler.
Return ONLY the summary text, no extra formatting."""


def _apply_sliding_window(state: dict) -> tuple[str, list]:
    """Return (summary_of_old_turns, recent_turns) for message building.
    Keeps the last WINDOW_SIZE messages in full, summarizes older ones."""
    history = state["conversation_history"]

    if len(history) <= WINDOW_SIZE:
        return state.get("conversation_summary", ""), history

    old_turns = history[:-WINDOW_SIZE]
    recent_turns = history[-WINDOW_SIZE:]

    # Check if we already summarized up to this point
    existing_summary = state.get("conversation_summary", "")
    if existing_summary and len(old_turns) <= state.get("_last_summarized_count", 0):
        return existing_summary, recent_turns

    # Summarize old turns (including any previous summary)
    old_text = ""
    if existing_summary:
        old_text = f"Previous summary: {existing_summary}\n\n"
    old_text += "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_turns
    )

    messages = [
        {"role": "system", "content": SUMMARIZE_PROMPT},
        {"role": "user", "content": old_text},
    ]
    try:
        summary = chat(messages, stream=False, label="conversation_summary",
                       use_cache=True)
        state["conversation_summary"] = summary.strip()
        state["_last_summarized_count"] = len(old_turns)
    except Exception:
        pass  # keep existing summary if summarization fails

    return state.get("conversation_summary", ""), recent_turns


# ---------------------------------------------------------------------------
# Live token usage panel — printed after every turn
# ---------------------------------------------------------------------------

def _print_live_token_panel(turn_count: int) -> None:
    """Print a compact live token breakdown after each conversation turn."""
    calls = tracker.calls
    last = calls[-1] if calls else {}

    # Categorize all calls so far
    cats = {}
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

    # Build compact display
    console.print(f"\n  [dim]┌─ Turn {turn_count} ─────────────────────────────────┐[/dim]")
    console.print(
        f"  [dim]│[/dim] This turn: "
        f"[cyan]{last.get('input_tokens', 0)}[/cyan] in / "
        f"[green]{last.get('output_tokens', 0)}[/green] out"
    )

    # Category breakdown
    parts = []
    for cat in ["chat", "extract", "summary", "cached"]:
        if cat in cats:
            total = cats[cat]["in"] + cats[cat]["out"]
            parts.append(f"{cat}=[yellow]{total}[/yellow]")
    if parts:
        console.print(f"  [dim]│[/dim] Breakdown: {' │ '.join(parts)}")

    console.print(
        f"  [dim]│[/dim] [bold]Total: [yellow]{tracker.total}[/yellow] tokens[/bold] "
        f"([cyan]{tracker.total_input}[/cyan] in + [green]{tracker.total_output}[/green] out) "
        f"across {len(calls)} calls"
    )
    console.print(f"  [dim]└──────────────────────────────────────────┘[/dim]\n")


# ---------------------------------------------------------------------------
# Opening message
# ---------------------------------------------------------------------------

def _opening_message(state: dict) -> str:
    """Returns first message based on detected scenario."""
    if state["scenario"] == "empty":
        return "what are you building? Even one line is fine to start."
    s = state["existing_summary"]
    lang = s.get("language", "project")
    fw = s.get("framework", "unknown framework")
    flds = ", ".join(s.get("top_folders", [])[:4])

    # If we have rich context, mention key classes/routes
    extras = []
    classes = s.get("classes", [])
    if classes:
        extras.append(f"Key classes: {', '.join(c.split('::')[-1] for c in classes[:5])}")
    routes = s.get("routes", [])
    if routes:
        extras.append(f"Routes found: {len(routes)}")

    msg = (
        f"I can see you have an existing {lang} project using {fw}.\n"
        f"Top-level folders: {flds}."
    )
    if extras:
        msg += "\n" + "\n".join(extras)
    msg += "\n\nwhat do you want to add or change?"
    return msg


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def run_conversation(state: dict) -> dict:
    """Main phase 1 loop. Runs until conversation_complete.
    Uses sliding window + incremental extraction for token efficiency."""

    # Opening message
    opening = _opening_message(state)
    sys.stdout.write(f"\nAgent: {opening}\n\n")
    sys.stdout.flush()
    state["conversation_history"].append(
        {"role": "assistant", "content": opening}
    )

    turn_count = 0
    while not state["conversation_complete"]:
        # Get developer input
        try:
            user_input = console.input("[bold purple]You:[/bold purple] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        state["conversation_history"].append(
            {"role": "user", "content": user_input}
        )
        turn_count += 1

        # Incremental state extraction every 2 turns (only new turns)
        if turn_count % 2 == 0:
            console.print("[dim]Updating project context...[/dim]")
            state = _extract_state_incremental(state)

        # Build system prompt with dynamic trimming
        system = _build_system_prompt(state)

        # Apply sliding window — summarize old turns, keep recent in full
        summary, recent_history = _apply_sliding_window(state)

        # If there's a summary of old turns, prepend it as context
        if summary:
            summary_msg = {"role": "assistant",
                           "content": f"[Earlier conversation summary: {summary}]"}
            windowed_history = [summary_msg] + recent_history[:-1]
        else:
            windowed_history = recent_history[:-1]

        messages = build_messages(system, windowed_history, user_input)

        sys.stdout.write("\nAgent: ")
        sys.stdout.flush()
        response = chat(messages, stream=True,
                        label=f"conversation_turn_{turn_count}")
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Live token usage — side-by-side with conversation
        _print_live_token_panel(turn_count)

        state["conversation_history"].append(
            {"role": "assistant", "content": response}
        )

        # Check completion signal
        if DONE_SIGNAL in response:
            state["conversation_complete"] = True
            # Final extraction to capture everything from last turns
            state = _extract_state_incremental(state)
            console.print("\n[dim]Got everything I need. Generating plan...[/dim]\n")

    # Save token usage to state
    state["token_usage"] = tracker.summary()

    # Final summary panel
    calls = tracker.calls
    cats = {}
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

    console.print("\n[bold cyan]═══ Session Token Usage ═══[/bold cyan]")
    for cat in ["chat", "extract", "summary", "plan", "cached", "other"]:
        if cat in cats:
            d = cats[cat]
            total = d["in"] + d["out"]
            console.print(
                f"  [bold]{cat.ljust(8)}[/bold] "
                f"{d['count']} calls │ "
                f"[cyan]{d['in']}[/cyan] in + [green]{d['out']}[/green] out = "
                f"[yellow]{total}[/yellow]"
            )
    console.print(
        f"  [bold]{'TOTAL'.ljust(8)}[/bold] "
        f"{len(calls)} calls │ "
        f"[cyan]{tracker.total_input}[/cyan] in + [green]{tracker.total_output}[/green] out = "
        f"[bold yellow]{tracker.total}[/bold yellow]"
    )
    console.print("[bold cyan]═══════════════════════════[/bold cyan]\n")

    return state