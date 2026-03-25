import json
import sys
from pathlib import Path
from rich.console import Console
from planagent.llm import chat,build_messages,tracker

console = Console()
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"
DONE_SIGNAL = "CONVERSATION_COMPLETE"

def _build_context(state:dict) -> str:
    """Summarize current state into text for the system prompt."""
    lines = [f"Scenario: {state.get('scenario', 'unknown')}"]
    ex = state.get("existing_summary")
    if ex:
        lines += [
            f"Existing project detected:",
            f" Language: {ex.get('language', 'unknown')}",
            f"Framework: {ex.get('framework', 'unknown')}",
            f"Files: {ex.get('file_count', 0)}",
            f"Folders: {','.join(ex.get('top_folders',[]))}",
        ]
    if state.get("project_goal"):
        lines.append(f"Project Goal: {state['project_goal']}")
    if state.get("user_types"):
        lines.append(f"User types: {', '.join(state['user_types'])}")
    if state.get("tech_stack"):
        lines.append(f"Stack confirmed: {state['tech_stack']}")
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


STATE_EXTRACT_PROMPT = """Based on the conversation so far, extract any information that has been confirmed by the developer.
Return ONLY a JSON object with these keys (use null or empty list if not yet discussed):
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


def _extract_state(state: dict) -> dict:
    """Use a short LLM call to extract structured info from conversation history."""
    if len(state["conversation_history"]) < 3:
        return state
    messages = [
        {"role": "system", "content": STATE_EXTRACT_PROMPT},
        {"role": "user", "content": "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in state["conversation_history"]
        )},
    ]
    try:
        raw = chat(messages, stream=False, label="state_extraction")
        # Try to parse JSON from the response
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
    return state

def _opening_message(state:dict) -> str:
    """ returns first message based on detected scenario. """
    if state["scenario"] == "empty":
        return "what are you building? Even one line is fine to start."
    s = state["existing_summary"]
    lang = s.get("language","project")
    fw = s.get("framework","unknown framework")
    flds = ", ".join(s.get("top_folders",[])[:4])
    return (
        f"I can see you have an existing {lang} project using {fw}.\n"
        f"Top-level folders: {flds}.\n\n"
        f"what do you want to add or change?"
    )

def run_conversation(state:dict) -> dict:
    """ Main phase 1 loop.Runs until conversation_complete"""
    system_prompt = SYSTEM_PROMPT_PATH.read_text()

    #opening message
    opening = _opening_message(state)
    sys.stdout.write(f"\nAgent: {opening}\n\n")
    sys.stdout.flush()
    state["conversation_history"].append(
        {"role":"assistant","content":opening}
    )

    turn_count = 0
    while not state["conversation_complete"]:
        # Get developer input
        try:
            user_input = console.input("[bold purple]You:[/bold purple] ").strip()
        except (KeyboardInterrupt,EOFError):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue

        state["conversation_history"].append(
            {"role":"user","content":user_input}
        )
        turn_count += 1

        # Extract structured state every 2 turns
        if turn_count % 2 == 0:
            console.print("[dim]Updating project context...[/dim]")
            state = _extract_state(state)

        #Build context and call LLM
        context = _build_context(state)
        system  = system_prompt.replace("{context}",context)
        messages = build_messages(system,state["conversation_history"][:-1],user_input)

        sys.stdout.write("\nAgent: ")
        sys.stdout.flush()
        response = chat(messages, stream=True, label=f"conversation_turn_{turn_count}")
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Show token usage for this turn
        last_call = tracker.calls[-1]
        sys.stdout.write(
            f"  [{last_call['input_tokens']} in / "
            f"{last_call['output_tokens']} out / "
            f"total so far: {tracker.total}]\n\n"
        )
        sys.stdout.flush()

        state["conversation_history"].append(
            {"role":"assistant","content":response}
        )

        # check completion signal
        if DONE_SIGNAL in response:
            state["conversation_complete"] = True
            # Do a final extraction to capture everything
            state = _extract_state(state)
            console.print("\n[dim]Got everything I need. Generating plan...[/dim]\n")

    # Save token usage to state
    state["token_usage"] = tracker.summary()
    sys.stdout.write(
        f"\n--- Token Usage Summary ---\n"
        f"  Total calls:   {tracker.summary()['total_calls']}\n"
        f"  Input tokens:  {tracker.total_input}\n"
        f"  Output tokens: {tracker.total_output}\n"
        f"  Total tokens:  {tracker.total}\n"
        f"----------------------------\n"
    )
    sys.stdout.flush()

    return state




        
        
    
    
    