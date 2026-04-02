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
from planagent.guardrails.guard import check_input, check_output

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"
DONE_SIGNAL = "CONVERSATION_COMPLETE"


def prefill_state_from_scan(state: dict) -> dict:
    """Auto-populate state fields from the cached project scan.
    This prevents the LLM from asking about things the scan already reveals.
    Now leverages deep context: models, enums, infra, env, dependencies, tests."""
    ex = state.get("existing_summary")
    if not ex or state.get("scenario") != "existing":
        return state

    # Tech stack — use comprehensive detection from context_reader
    lang = ex.get("language", "unknown")
    fw = ex.get("framework", "unknown")
    full_ts = ex.get("tech_stack", {})  # from _detect_full_tech_stack

    # Build the state tech_stack from the comprehensive detection
    tech = {
        "language": lang,
        "framework": fw,
    }
    # Pull specific categories from the full tech stack
    if full_ts.get("database"):
        tech["database"] = ", ".join(full_ts["database"])
    elif full_ts.get("database_cache"):
        tech["database"] = ", ".join(full_ts["database_cache"])
    else:
        tech["database"] = "unknown"
    if full_ts.get("orm"):
        tech["orm"] = ", ".join(full_ts["orm"])
    if full_ts.get("auth"):
        tech["auth"] = ", ".join(full_ts["auth"])
    if full_ts.get("queue"):
        tech["message_queue"] = ", ".join(full_ts["queue"])
    if full_ts.get("api_style"):
        tech["api_style"] = ", ".join(full_ts["api_style"])
    if full_ts.get("testing"):
        tech["testing"] = ", ".join(full_ts["testing"])
    if full_ts.get("monitoring"):
        tech["monitoring"] = ", ".join(full_ts["monitoring"])
    if full_ts.get("cloud"):
        tech["cloud"] = ", ".join(full_ts["cloud"])
    if full_ts.get("container"):
        tech["container"] = ", ".join(full_ts["container"])
    if full_ts.get("ci_cd"):
        tech["ci_cd"] = ", ".join(full_ts["ci_cd"])
    if full_ts.get("payment"):
        tech["payment"] = ", ".join(full_ts["payment"])
    if full_ts.get("realtime"):
        tech["realtime"] = ", ".join(full_ts["realtime"])
    if full_ts.get("ai"):
        tech["ai_ml"] = ", ".join(full_ts["ai"])
    if full_ts.get("validation"):
        tech["validation"] = ", ".join(full_ts["validation"])
    if full_ts.get("package_manager"):
        tech["package_manager"] = ", ".join(full_ts["package_manager"])
    if full_ts.get("linter"):
        tech["linter"] = ", ".join(full_ts["linter"])
    if full_ts.get("build"):
        tech["build_tools"] = ", ".join(full_ts["build"])

    if lang != "unknown" or fw != "unknown":
        state["tech_stack"] = tech

    # User types — infer from class names like User, Admin, etc.
    _USER_NAMES = {"user", "admin", "customer", "moderator", "coach",
                   "nutritionist", "seller", "buyer", "driver", "vendor",
                   "merchant", "manager", "staff", "member", "subscriber",
                   "author", "editor", "reviewer", "patient", "doctor"}
    classes = ex.get("classes", [])
    inferred_users = []
    for cls in classes:
        # Handle both "file::ClassName" and "file::ClassName (docstring)"
        raw_name = cls.split("::")[-1].split("(")[0].strip().lower()
        if raw_name in _USER_NAMES:
            inferred_users.append(cls.split("::")[-1].split("(")[0].strip())
    if inferred_users:
        state["user_types"] = inferred_users

    # Gaps — from scan flags (now much richer)
    ft = ex.get("file_types", {})
    gaps = []
    if not ex.get("has_tests"):
        gaps.append("no tests detected")
    if not ft.get("migration"):
        gaps.append("no migration files")
    env_keys = ex.get("env_keys", [])
    if not ft.get("config") and not env_keys:
        gaps.append("no config files (.env)")
    infra = ex.get("infra", {})
    if not infra.get("dockerfiles"):
        gaps.append("no Dockerfile")
    if not infra.get("github_actions"):
        gaps.append("no CI/CD pipeline")
    # Check test coverage gaps
    test_map = ex.get("test_map", {})
    if test_map:
        tested_modules = {v.get("tests_module") for v in test_map.values()
                          if v.get("tests_module")}
        # Find code files without corresponding tests
        code_files = [rel for rel, info in
                      state.get("context_index", {}).get("files", {}).items()
                      if info.get("type") == "code"]
        untested = []
        for cf in code_files:
            stem = Path(cf).stem
            if stem not in tested_modules and not stem.startswith("__"):
                untested.append(stem)
        if untested:
            gaps.append(f"untested modules: {', '.join(untested[:5])}")

    if gaps:
        state["gaps_flagged"] = gaps

    # Discovered features — pre-populate features_v1 from scan
    discovered = ex.get("discovered_features", [])
    if discovered:
        # Use high/medium confidence features as confirmed v1 features
        confirmed = [f["name"] for f in discovered
                     if f.get("confidence") in ("high", "medium")]
        if confirmed:
            state["features_v1"] = confirmed

    return state

# ---------------------------------------------------------------------------
# Sliding window config
# ---------------------------------------------------------------------------
WINDOW_SIZE = 6   # keep last N turn-pairs in full; older turns get summarized


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

    # Scan-based signals for gap detection (enriched)
    ex = state.get("existing_summary", {})
    if ex:
        signals = []
        if not ex.get("has_tests"):
            signals.append("NO tests detected")
        ft = ex.get("file_types", {})
        if not ft.get("migration"):
            signals.append("NO migration files detected")
        env_keys = ex.get("env_keys", [])
        if not ft.get("config") and not env_keys:
            signals.append("NO config files (.env) detected")
        infra = ex.get("infra", {})
        if not infra.get("dockerfiles"):
            signals.append("NO Dockerfile detected")
        if not infra.get("github_actions"):
            signals.append("NO CI/CD pipeline detected")
        # Untested modules
        test_map = ex.get("test_map", {})
        if test_map:
            tested = {v.get("tests_module") for v in test_map.values() if v.get("tests_module")}
            idx_files = state.get("context_index", {}).get("files", {})
            untested = [Path(r).stem for r, i in idx_files.items()
                        if i.get("type") == "code" and Path(r).stem not in tested
                        and not Path(r).stem.startswith("__")]
            if untested:
                signals.append(f"Untested modules: {', '.join(untested[:5])}")
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

    # Show collection progress (informational, not prescriptive)
    collected = []
    for field, label in [
        ("project_goal", "project goal"),
        ("user_types", "user types"),
        ("features_v1", "v1 features"),
        ("tech_stack", "tech stack"),
        ("constraints", "constraints"),
    ]:
        if state.get(field):
            collected.append(label)
    lines.append(f"\nAlready collected: {', '.join(collected) if collected else 'nothing yet'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic system prompt — trim question sections already answered
# ---------------------------------------------------------------------------

def _build_system_prompt(state: dict) -> str:
    """Load system prompt and trim sections for already-collected fields."""
    raw = SYSTEM_PROMPT_PATH.read_text()
    context = _build_context(state)
    prompt = raw.replace("{context}", context)

    # Trim topic items that are already collected to save tokens
    if state.get("project_goal"):
        prompt = prompt.replace(
            "- Project goal — what are they building / adding / changing\n", "")
    if state.get("features_v1"):
        prompt = prompt.replace(
            "- V1 features — the core modules for first launch\n", "")
    if state.get("tech_stack"):
        prompt = prompt.replace(
            "- Database choice — if not already detected by scan\n", "")
    if state.get("constraints"):
        prompt = prompt.replace(
            "- Constraints — deadline, team size (ask ONCE, accept whatever they say)\n", "")
    if state.get("features_v2"):
        prompt = prompt.replace(
            "- V2 features — anything mentioned but not critical for launch\n", "")

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
    """Returns first message based on detected scenario.
    Now shows richer context: models, entry points, test coverage, infra."""
    # Revision mode — acknowledge existing plan
    if state.get("is_revision"):
        plan = state.get("revision_base", {})
        name = plan.get("project_name", "your project")
        v1_count = len(plan.get("modules_v1", []))
        ep_count = len(plan.get("api_endpoints", []))
        return (
            f"I've loaded your existing plan for **{name}** "
            f"({v1_count} modules, {ep_count} endpoints).\n\n"
            "What do you want to change? You can add features, swap tech, "
            "refine modules, or restructure the architecture."
        )

    if state["scenario"] == "empty":
        return "what are you building? Even one line is fine to start."
    s = state["existing_summary"]
    lang = s.get("language", "project")
    fw = s.get("framework", "unknown framework")
    flds = s.get("top_folders", [])[:4]

    # Extract data for display
    manifest = s.get("manifest", {})
    classes = s.get("classes", [])
    models = s.get("models", [])
    routes = s.get("routes", [])
    entry_points = s.get("entry_points", [])
    test_map = s.get("test_map", {})
    infra = s.get("infra", {})
    discovered = s.get("discovered_features", [])

    # Header
    lines = [f"Scanned your **{lang} / {fw}** project:\n"]

    # Project identity
    if manifest.get("description"):
        proj_name = manifest.get('name', '')
        lines.append(f"- **Project** — {proj_name}: {manifest['description']}")
    elif s.get("readme_summary"):
        lines.append(f"- **About** — {s['readme_summary'][:150]}")

    # Structure
    if flds:
        lines.append(f"- **Folders** — {', '.join(flds)}")

    # Code structure
    if classes:
        cls_names = [c.split("::")[-1].split("(")[0].strip() for c in classes[:6]]
        lines.append(f"- **Classes** — {', '.join(cls_names)}")

    # Models
    if models:
        model_parts = []
        for m in models[:5]:
            fields = m.get("fields", [])
            if fields:
                model_parts.append(f"{m['name']}({', '.join(fields[:4])})")
            else:
                model_parts.append(m['name'])
        lines.append(f"- **DB Models** — {', '.join(model_parts)}")

    # Routes
    if routes:
        lines.append(f"- **Routes** — {len(routes)} endpoints")

    # Entry points
    if entry_points:
        eps = [e["file"] for e in entry_points[:3]]
        lines.append(f"- **Entry pts** — {', '.join(eps)}")

    # Testing
    if test_map:
        test_count = sum(len(v.get("tests", [])) for v in test_map.values())
        lines.append(f"- **Tests** — {test_count} tests across {len(test_map)} files")

    # Infrastructure
    if infra:
        infra_bits = []
        if infra.get("dockerfiles"):
            infra_bits.append("Docker")
        if infra.get("compose_services"):
            infra_bits.append(f"Compose({', '.join(infra['compose_services'][:3])})")
        if infra.get("github_actions"):
            infra_bits.append("GitHub Actions")
        if infra_bits:
            lines.append(f"- **Infra** — {', '.join(infra_bits)}")

    # Discovered features
    if discovered:
        high = [f["name"] for f in discovered if f.get("confidence") == "high"]
        med = [f["name"] for f in discovered if f.get("confidence") == "medium"]
        if high:
            lines.append(f"- **Confirmed** — {', '.join(high[:8])}")
        if med:
            lines.append(f"- **Likely** — {', '.join(med[:6])}")

    lines.append("\nWhat do you want to add or change?")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Revision mode — system prompt addendum for continuing from existing plan
# ---------------------------------------------------------------------------

REVISION_SYSTEM_ADDENDUM = """
IMPORTANT — REVISION MODE:
You are continuing a conversation about an EXISTING architecture plan.
The current plan is shown below. The developer wants to modify it.
Do NOT start from scratch. Focus ONLY on what they want to change.
Keep questions minimal — the plan already has most answers.

CURRENT PLAN:
{plan_summary}
"""


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
        from_choice = False
        if pending_choice is not None:
            user_input = pending_choice
            pending_choice = None
            from_choice = True
        else:
            user_input = get_user_input()

        if not user_input:
            console.print("[dim]Session ended.[/dim]")
            break

        # --- Guardrails: check user input is on-topic ----------------
        # Skip guardrails for selections from agent-suggested options —
        # the agent itself proposed them, so they are inherently on-topic.
        is_allowed, refusal = (True, "") if from_choice else check_input(user_input)
        if not is_allowed:
            render_agent_message(refusal)
            state["conversation_history"].append(
                {"role": "user", "content": user_input}
            )
            state["conversation_history"].append(
                {"role": "assistant", "content": refusal}
            )
            continue

        state["conversation_history"].append(
            {"role": "user", "content": user_input}
        )
        turn_count += 1

        # Incremental state extraction every turn
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

        # --- Guardrails: check agent output is on-topic ---------------
        is_on_topic, cleaned = check_output(response)
        if not is_on_topic:
            response = cleaned

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