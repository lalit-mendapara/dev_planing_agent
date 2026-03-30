import json
from pathlib import Path
from planagent.llm import chat, build_messages, tracker
from planagent.ui import (
    console,
    parse_agent_response,
    render_agent_message,
    render_token_panel,
    render_session_summary,
    stream_to_panel,
    get_user_choice,
    get_user_input,
)

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"
DONE_SIGNAL = "CONVERSATION_COMPLETE"


def prefill_state_from_scan(state: dict) -> dict:
    """Auto-populate state fields from the cached project scan.
    This prevents the LLM from asking about things the scan already reveals."""
    ex = state.get("existing_summary")
    if not ex or state.get("scenario") != "existing":
        return state

    # Tech stack — confirmed from scan, never re-ask
    lang = ex.get("language", "unknown")
    fw = ex.get("framework", "unknown")
    if lang != "unknown" or fw != "unknown":
        state["tech_stack"] = {
            "language": lang,
            "framework": fw,
            "database": "unknown",  # scan can't detect DB, still ask
        }

    # User types — infer from class names like User, Admin, etc.
    classes = ex.get("classes", [])
    inferred_users = []
    for cls in classes:
        name = cls.split("::")[-1].lower()
        if name in ("user", "admin", "customer", "moderator", "coach",
                     "nutritionist", "seller", "buyer", "driver", "vendor"):
            inferred_users.append(cls.split("::")[-1])
    if inferred_users:
        state["user_types"] = inferred_users

    # Gaps — from scan flags
    ft = ex.get("file_types", {})
    gaps = []
    if not ex.get("has_tests"):
        gaps.append("no tests detected")
    if not ft.get("migration"):
        gaps.append("no migration files")
    if not ft.get("config"):
        gaps.append("no config files (.env)")
    if gaps:
        state["gaps_flagged"] = gaps

    return state

# ---------------------------------------------------------------------------
# Sliding window config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 6   # keep last N turn-pairs in full; older turns get summarized
MAX_TURNS = 8     # hard cap — force CONVERSATION_COMPLETE after this many turns


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
    Uses sliding window + incremental extraction for token efficiency.
    Renders Claude Code-style panels and multiple-choice options."""

    # Opening message — rendered as a panel
    opening = _opening_message(state)
    render_agent_message(opening)
    state["conversation_history"].append(
        {"role": "assistant", "content": opening}
    )

    turn_count = 0
    # pending_choice holds user input from an arrow-key selection so we
    # skip the free-text prompt and go straight to the next LLM turn.
    pending_choice: str | None = None

    while not state["conversation_complete"]:
        # --- Get user input -----------------------------------------------
        if pending_choice is not None:
            user_input = pending_choice
            pending_choice = None
        else:
            user_input = get_user_input()

        if not user_input:
            console.print("[dim]Session ended.[/dim]")
            break

        state["conversation_history"].append(
            {"role": "user", "content": user_input}
        )
        turn_count += 1

        # Incremental state extraction every turn (keeps "Still needed" current)
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

        # --- Stream response into a Live panel (no raw stdout) -----------
        response = stream_to_panel(chat, messages,
                                   label=f"conversation_turn_{turn_count}")

        # Parse out ```options blocks → clean text + option list
        message_text, options = parse_agent_response(response)

        # The streaming panel was transient (disappears when done).
        # Now render the final clean panel with options block stripped.
        render_agent_message(message_text)

        # Live token usage panel
        render_token_panel(tracker, turn_count)

        # Store the FULL response (including options block) in history
        # so the LLM has context, but the user sees the clean version
        state["conversation_history"].append(
            {"role": "assistant", "content": response}
        )

        # Check completion signal
        if DONE_SIGNAL in response:
            state["conversation_complete"] = True
            state = _extract_state_incremental(state)
            console.print(
                "[dim]Got everything I need. Generating plan...[/dim]\n"
            )
            break

        # Hard turn limit — force completion if LLM keeps going
        if turn_count >= MAX_TURNS:
            state["conversation_complete"] = True
            state = _extract_state_incremental(state)
            console.print(
                "[dim]Reached turn limit. Generating plan with what we have...[/dim]\n"
            )
            break

        # --- If options were presented, arrow-key select ----------------
        if options:
            choice = get_user_choice(options)
            if not choice:
                console.print("[dim]Session ended.[/dim]")
                break
            # Feed the choice directly into the next iteration
            # (no extra get_user_input prompt)
            pending_choice = choice

    # Save token usage to state
    state["token_usage"] = tracker.summary()

    # Final session summary panel
    render_session_summary(tracker)

    return state