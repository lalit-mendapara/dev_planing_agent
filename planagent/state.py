def create_initial_state() -> dict:
    return {
        # --- Set by context_reader.py ------------------------------
        "scenario": None,           # "empty" or "existing"
        "existing_summary": None,   # dict if existing, else None
        "project_root": None,       # absolute path string
        "context_index": None,      # full file index (signatures, hashes, etc.)
        "context_tier1": None,      # compressed text summary for system prompt
        "cache_hit": False,         # True if loaded from cache (no re-scan)

        # --- Set by conversation_manager.py -------------------------
        "conversation_history": [],  # [{role, content}, ....]
        "conversation_summary": "",  # rolling summary of older turns
        "last_extracted_turn": 0,    # track which turns have been state-extracted
        "project_goal": None,        # one-line description
        "user_types": [],            # ["customers","admins","moderators"]
        "features_v1": [],           # confirmed v1 features
        "features_v2": [],           # deferred v2 features
        "tech_stack": {},            # {language:version, framework:version}
        "gaps_flagged": [],          # things that are missing or unclear
        "gaps_confirmed": [],        # gaps dev said yes to
        "gaps_deferred": [],         # gaps dev said no/ later
        "constraints": [],           # deadlines, team size, budget, etc.
        "conversation_complete": False,

        # --- Token tracking -------------------------
        "token_usage": {
            "total_input": 0,
            "total_output": 0,
            "total": 0,
            "calls": [],  # per-call breakdown
        },

        # --- Set by plan_generator.py -------------------------
        "proposal": None,            # full generated plan dict
        "proposal_approved": False,

        # --- set by output_writer.py -------------------------
        "files_written": [],  # paths of files written
    }
