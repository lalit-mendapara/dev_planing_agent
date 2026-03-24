from pathlib import Path

STACK_SIGNALS = {
    "requirements.txt" : "python",
    "pyproject.toml":"python",
    "package.json":"javascript/Node",
    "go.mod":"Go",
    "Cargo.toml":"Rust",
}

FRAMEWORK_SIGNALS = {
    "django":"python",
    "flask":"python",
    "fastapi":"python",
    "express":"javascript/Node",
    "koa":"javascript/Node",
    "hapi":"javascript/Node",
    "nestjs":"javascript/Node",
    "nextjs":"javascript/Node",
    "react":"javascript/Node",
    "vue":"javascript/Node",
    "angular":"javascript/Node",
    "svelte":"javascript/Node",
    "rust":"Rust",
    "go":"Go",
}

IGNORE = {".git", "venv", "node_modules","__pycache__",".planagent"}

def read_context(project_root:str,state:dict) -> dict:
    root = Path(project_root)
    state["project_root"] = root
    
    files = [
        f for f in root.rglob("*")
        if f.is_file()
        and not any (p in IGNORE for p in f.parts)
    ]

    if not files:
        state["scenario"] = "empty"
        return state

    state["scenario"] = "existing"
    state["existing_summary"] = _build_summary(root,files)
    return state
    
def _build_summary(root,files):
    return {
        "language":_detect_language(files),
        "framework" :_detect_framework(files),
        "file_count":len(files),
        "has_tests":any("test" in str(f).lower() for f in files),
        "top_folders":_top_folders(root,files),
    }

def _detect_language(files):
    for f in files:
        if f.name in STACK_SIGNALS:
            return STACK_SIGNALS[f.name]
    return "unknown"

def _detect_framework(files):
    for f in files:
        if f.name in ("requirements.txt","package.json"):
            try:
                content = f.read_text(error="ignore").lower()
                for kw,name in FRAMEWORK_SIGNALS.items():
                    if kw in content:
                        return name
            except Exception:
                pass
    return "unknown"

def _top_folders(root,files):
    seen = set ()
    for f in files:
        try:
            parts = f.relative_to(root).parts
            if len(parts) > 1:
                seen.add(parts[0])
        except ValueError:
            pass
    return list(seen)[:8]
        