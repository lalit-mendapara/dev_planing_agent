"""
Plan management: load, version, diff, and targeted editing of existing plans.
Designed for low token usage — only sends relevant sections to LLM for edits.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from difflib import unified_diff

from planagent.llm import chat, tracker
from planagent.plan_generator import ArchitecturePlan


# ---------------------------------------------------------------------------
# Load existing plan from .planagent/
# ---------------------------------------------------------------------------

def load_existing_plan(project_root: str) -> dict | None:
    """Load context.json from .planagent/ and return as dict.
    Returns None if no plan exists."""
    ctx_path = Path(project_root) / ".planagent" / "context.json"
    if not ctx_path.exists():
        return None
    try:
        return json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def reconstruct_state_from_plan(project_root: str, plan: dict) -> dict:
    """Rebuild a minimal state dict from an existing plan for conversation resumption.
    Token-efficient: only populates fields that matter for continued planning."""
    from planagent.state import create_initial_state

    state = create_initial_state()
    state["project_root"] = project_root
    state["scenario"] = plan.get("scenario", "existing")
    state["proposal"] = plan
    state["proposal_approved"] = True

    # Reconstruct structured fields from plan
    state["project_goal"] = plan.get("description", "")
    state["features_v1"] = [
        m["name"] for m in plan.get("modules_v1", [])
    ]
    state["features_v2"] = [
        m["name"] for m in plan.get("modules_v2", [])
    ]

    ts = plan.get("tech_stack", {})
    if ts:
        state["tech_stack"] = ts if isinstance(ts, dict) else {"raw": ts}

    # Mark conversation as resumption
    state["is_revision"] = True
    state["revision_base"] = plan

    return state


# ---------------------------------------------------------------------------
# Version control
# ---------------------------------------------------------------------------

VERSIONS_DIR = "versions"
MANIFEST_FILE = "manifest.json"


def _versions_root(project_root: str) -> Path:
    return Path(project_root) / ".planagent" / VERSIONS_DIR


def _load_manifest(project_root: str) -> dict:
    """Load or create the version manifest."""
    manifest_path = _versions_root(project_root) / MANIFEST_FILE
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"versions": [], "current": 0}


def _save_manifest(project_root: str, manifest: dict):
    vroot = _versions_root(project_root)
    vroot.mkdir(parents=True, exist_ok=True)
    (vroot / MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )


def snapshot_current(project_root: str, message: str = "") -> int | None:
    """Snapshot current .planagent/ files into versions/v{N}/.
    Returns the version number, or None if nothing to snapshot."""
    planagent_dir = Path(project_root) / ".planagent"
    if not planagent_dir.exists():
        return None

    # Check if there's anything to snapshot
    plan_files = [f for f in planagent_dir.iterdir()
                  if f.is_file() and f.name != ".gitkeep"]
    if not plan_files:
        return None

    manifest = _load_manifest(project_root)
    version_num = len(manifest["versions"]) + 1

    # Copy files to version directory
    version_dir = _versions_root(project_root) / f"v{version_num}"
    version_dir.mkdir(parents=True, exist_ok=True)

    for f in plan_files:
        shutil.copy2(f, version_dir / f.name)

    # Update manifest
    manifest["versions"].append({
        "version": version_num,
        "timestamp": datetime.now().isoformat(),
        "message": message or f"Auto-snapshot before update",
        "files": [f.name for f in plan_files],
    })
    manifest["current"] = version_num
    _save_manifest(project_root, manifest)

    return version_num


def list_versions(project_root: str) -> list[dict]:
    """Return list of version entries from manifest."""
    return _load_manifest(project_root).get("versions", [])


def get_version_plan(project_root: str, version: int) -> dict | None:
    """Load context.json from a specific version."""
    ctx_path = _versions_root(project_root) / f"v{version}" / "context.json"
    if not ctx_path.exists():
        return None
    try:
        return json.loads(ctx_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def get_version_file(project_root: str, version: int, filename: str) -> str | None:
    """Read a specific file from a version snapshot."""
    fpath = _versions_root(project_root) / f"v{version}" / filename
    if not fpath.exists():
        return None
    return fpath.read_text(encoding="utf-8")


def diff_versions(project_root: str, v1: int, v2: int, filename: str = "plan.md") -> str:
    """Generate unified diff between two versions of a file.
    Returns diff string or a message if files don't exist."""
    content1 = get_version_file(project_root, v1, filename)
    content2 = get_version_file(project_root, v2, filename)

    if content1 is None and content2 is None:
        return f"File '{filename}' not found in v{v1} or v{v2}."
    if content1 is None:
        content1 = ""
    if content2 is None:
        content2 = ""

    lines1 = content1.splitlines(keepends=True)
    lines2 = content2.splitlines(keepends=True)

    diff = unified_diff(
        lines1, lines2,
        fromfile=f"v{v1}/{filename}",
        tofile=f"v{v2}/{filename}",
        lineterm=""
    )
    result = "\n".join(diff)
    return result if result else f"No changes in '{filename}' between v{v1} and v{v2}."


def diff_current_vs_version(project_root: str, version: int, filename: str = "plan.md") -> str:
    """Diff current plan file against a versioned snapshot."""
    current_path = Path(project_root) / ".planagent" / filename
    versioned = get_version_file(project_root, version, filename)

    current = current_path.read_text(encoding="utf-8") if current_path.exists() else ""
    versioned = versioned or ""

    lines1 = versioned.splitlines(keepends=True)
    lines2 = current.splitlines(keepends=True)

    diff = unified_diff(
        lines1, lines2,
        fromfile=f"v{version}/{filename}",
        tofile=f"current/{filename}",
        lineterm=""
    )
    result = "\n".join(diff)
    return result if result else f"No changes in '{filename}' since v{version}."


def rollback_to_version(project_root: str, version: int) -> bool:
    """Restore .planagent/ files from a version snapshot.
    Auto-snapshots current state first."""
    version_dir = _versions_root(project_root) / f"v{version}"
    if not version_dir.exists():
        return False

    # Snapshot current before rollback
    snapshot_current(project_root, message=f"Auto-snapshot before rollback to v{version}")

    # Copy version files to .planagent/
    planagent_dir = Path(project_root) / ".planagent"
    for f in version_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, planagent_dir / f.name)

    return True


# ---------------------------------------------------------------------------
# Targeted plan editing — low token, section-specific LLM calls
# ---------------------------------------------------------------------------

SECTION_EDIT_PROMPT = """You are editing a SPECIFIC SECTION of an existing backend architecture plan.
Current section data (JSON):
{current_section}

User's edit request: {edit_request}

Return ONLY the updated section as valid JSON matching the original structure. No explanation, no markdown fences.
Keep everything the user didn't mention unchanged. Only modify what they asked for."""


def edit_plan_section(
    project_root: str,
    section: str,
    edit_request: str,
) -> dict | None:
    """Edit a specific section of the plan using a targeted LLM call.

    section: one of 'modules_v1', 'modules_v2', 'api_endpoints',
             'tech_stack', 'folder_structure', 'design_patterns', 'description'
    edit_request: natural language description of the change

    Returns updated full plan dict, or None on failure.
    """
    plan = load_existing_plan(project_root)
    if not plan:
        return None

    current_section = plan.get(section)
    if current_section is None:
        return None

    prompt = SECTION_EDIT_PROMPT.replace(
        "{current_section}", json.dumps(current_section, indent=2, default=str)
    ).replace("{edit_request}", edit_request)

    messages = [
        {"role": "system", "content": prompt},
    ]

    try:
        raw = chat(messages, stream=False, label=f"edit_{section}",
                   json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        updated = json.loads(raw)
        plan[section] = updated
        return plan
    except (json.JSONDecodeError, Exception):
        return None


def apply_plan_update(project_root: str, updated_plan: dict) -> list[str]:
    """Write an updated plan back to .planagent/ files.
    Auto-snapshots before writing. Returns list of written file paths."""
    from planagent.output_writer import write_all_outputs, write_token_report

    # Build a minimal state for the output writer
    state = {
        "project_root": project_root,
        "proposal": updated_plan,
        "scenario": updated_plan.get("scenario", "existing"),
        "token_usage": tracker.summary(),
    }

    state = write_all_outputs(state)
    write_token_report(state)
    return state.get("files_written", [])


# ---------------------------------------------------------------------------
# Revise conversation — resume planning with existing plan as base
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


def build_revision_context(plan: dict) -> str:
    """Build a compact summary of the existing plan for revision conversations.
    Keeps token count low by summarizing rather than dumping full JSON."""
    parts = [f"Project: {plan.get('project_name', 'N/A')}"]
    parts.append(f"Description: {plan.get('description', 'N/A')}")

    ts = plan.get("tech_stack", {})
    if ts:
        stack_str = ", ".join(f"{k}: {v}" for k, v in ts.items() if v)
        parts.append(f"Stack: {stack_str}")

    v1 = plan.get("modules_v1", [])
    if v1:
        names = [m["name"] for m in v1]
        parts.append(f"V1 modules ({len(v1)}): {', '.join(names)}")

    v2 = plan.get("modules_v2", [])
    if v2:
        names = [m["name"] for m in v2]
        parts.append(f"V2 modules ({len(v2)}): {', '.join(names)}")

    eps = plan.get("api_endpoints", [])
    if eps:
        parts.append(f"API endpoints: {len(eps)}")

    patterns = plan.get("design_patterns", [])
    if patterns:
        parts.append(f"Patterns: {', '.join(patterns)}")

    return "\n".join(parts)
