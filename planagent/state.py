def create_initial_state() -> dict:
    return {
        # --- Set by context_reader.py ------------------------------
        "scenario": None, # "empty" or "existing"
        "existing_summary": None, # dict if existing,else None
        "project_root": None,  # absolute path string

        # --- Set by conversation_manager.py -------------------------
        "conversation_history":[], #[{role,content}, ....]
        "project_goal":None, # one-line description
        "user_types":[],# ["customers","admins","moderators"]
        "features_v1":[], # confirmed v1 features
        "features_v2":[], # deferred v2 features
        "tech_stack":{}, # {language:version, framework:version}
        "gaps_flagged":[],# things that are missing or unclear
        "gaps_confirmed":[], # gaps dev said yes to 
        "gaps_deferred":[], # gaps dev said no/ later
        "constraints":[], # deadlines,team size, budget, etc.
        "conversation_complete":False,

        # --- Token tracking -------------------------
        "token_usage":{
            "total_input":0,
            "total_output":0,
            "total":0,
            "calls":[], # per-call breakdown
        },

        # --- Set by plan_generator.py -------------------------
        "proposal":None, # full generated plan dict
        "proposal_approved":False,

        # --- set by output_writer.py -------------------------
        "files_written":[], # paths of files written
    }
