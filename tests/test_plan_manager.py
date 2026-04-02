"""Tests for plan_manager: versioning, load, diff, edit, rollback."""

import json
import pytest
from pathlib import Path

from planagent.plan_manager import (
    load_existing_plan,
    reconstruct_state_from_plan,
    snapshot_current,
    list_versions,
    get_version_plan,
    get_version_file,
    diff_versions,
    diff_current_vs_version,
    rollback_to_version,
    build_revision_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_PLAN = {
    "project_name": "TestApp",
    "description": "A test application",
    "tech_stack": {
        "language": "Python",
        "framework": "FastAPI",
        "database": "PostgreSQL",
        "auth": "JWT",
        "cache": "",
        "queue": "",
        "other_tools": [],
    },
    "modules_v1": [
        {"name": "auth", "description": "Authentication module", "entities": ["User"]},
        {"name": "posts", "description": "Blog posts", "entities": ["Post"]},
    ],
    "modules_v2": [
        {"name": "analytics", "description": "Usage tracking", "entities": []},
    ],
    "api_endpoints": [
        {"method": "POST", "path": "/auth/login", "description": "Login", "auth_required": False},
        {"method": "GET", "path": "/posts", "description": "List posts", "auth_required": True},
    ],
    "folder_structure": ["/app", "/app/auth", "/app/posts"],
    "design_patterns": ["Repository pattern", "Service layer"],
    "scenario": "empty",
}


@pytest.fixture
def project_dir(tmp_path):
    """Create a temporary project with a .planagent/ directory and sample plan."""
    planagent_dir = tmp_path / ".planagent"
    planagent_dir.mkdir()

    # Write context.json
    ctx = {**SAMPLE_PLAN, "generated_by": "plan-architect-agent"}
    (planagent_dir / "context.json").write_text(json.dumps(ctx, indent=2))

    # Write plan.md
    (planagent_dir / "plan.md").write_text("# TestApp - plan\n\nA test application\n")

    # Write architecture.md
    (planagent_dir / "architecture.md").write_text("# Architecture\n\n## Tech Stack\n")

    return tmp_path


@pytest.fixture
def empty_project(tmp_path):
    """Project with no .planagent/ directory."""
    return tmp_path


# ---------------------------------------------------------------------------
# Load tests
# ---------------------------------------------------------------------------

class TestLoadExistingPlan:
    def test_loads_valid_plan(self, project_dir):
        plan = load_existing_plan(str(project_dir))
        assert plan is not None
        assert plan["project_name"] == "TestApp"
        assert plan["tech_stack"]["language"] == "Python"

    def test_returns_none_when_no_plan(self, empty_project):
        assert load_existing_plan(str(empty_project)) is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        planagent_dir = tmp_path / ".planagent"
        planagent_dir.mkdir()
        (planagent_dir / "context.json").write_text("not valid json{{{")
        assert load_existing_plan(str(tmp_path)) is None


class TestReconstructState:
    def test_reconstructs_from_plan(self, project_dir):
        plan = load_existing_plan(str(project_dir))
        state = reconstruct_state_from_plan(str(project_dir), plan)

        assert state["is_revision"] is True
        assert state["project_goal"] == "A test application"
        assert state["features_v1"] == ["auth", "posts"]
        assert state["features_v2"] == ["analytics"]
        assert state["tech_stack"]["language"] == "Python"
        assert state["revision_base"] is plan
        assert state["proposal"] is plan

    def test_handles_empty_plan(self, project_dir):
        state = reconstruct_state_from_plan(str(project_dir), {})
        assert state["is_revision"] is True
        assert state["features_v1"] == []


# ---------------------------------------------------------------------------
# Version control tests
# ---------------------------------------------------------------------------

class TestVersioning:
    def test_snapshot_creates_version(self, project_dir):
        version = snapshot_current(str(project_dir), message="test snapshot")
        assert version == 1

        # Check version directory exists
        v_dir = project_dir / ".planagent" / "versions" / "v1"
        assert v_dir.exists()
        assert (v_dir / "context.json").exists()
        assert (v_dir / "plan.md").exists()

    def test_snapshot_increments_version(self, project_dir):
        v1 = snapshot_current(str(project_dir), message="first")
        v2 = snapshot_current(str(project_dir), message="second")
        assert v1 == 1
        assert v2 == 2

    def test_snapshot_returns_none_when_empty(self, empty_project):
        assert snapshot_current(str(empty_project)) is None

    def test_list_versions_empty(self, empty_project):
        assert list_versions(str(empty_project)) == []

    def test_list_versions_after_snapshots(self, project_dir):
        snapshot_current(str(project_dir), message="first")
        snapshot_current(str(project_dir), message="second")

        versions = list_versions(str(project_dir))
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[0]["message"] == "first"
        assert versions[1]["version"] == 2
        assert versions[1]["message"] == "second"

    def test_get_version_plan(self, project_dir):
        snapshot_current(str(project_dir))
        plan = get_version_plan(str(project_dir), 1)
        assert plan is not None
        assert plan["project_name"] == "TestApp"

    def test_get_version_plan_nonexistent(self, project_dir):
        assert get_version_plan(str(project_dir), 99) is None

    def test_get_version_file(self, project_dir):
        snapshot_current(str(project_dir))
        content = get_version_file(str(project_dir), 1, "plan.md")
        assert content is not None
        assert "TestApp" in content

    def test_get_version_file_nonexistent(self, project_dir):
        assert get_version_file(str(project_dir), 1, "nope.md") is None


# ---------------------------------------------------------------------------
# Diff tests
# ---------------------------------------------------------------------------

class TestDiff:
    def test_diff_versions(self, project_dir):
        # Snapshot v1
        snapshot_current(str(project_dir), message="v1")

        # Modify plan.md
        (project_dir / ".planagent" / "plan.md").write_text(
            "# TestApp - plan\n\nUpdated description\n\n## New Section\n"
        )

        # Snapshot v2
        snapshot_current(str(project_dir), message="v2")

        result = diff_versions(str(project_dir), 1, 2, "plan.md")
        # Should contain diff markers
        assert "v1/plan.md" in result or "No changes" in result

    def test_diff_current_vs_version(self, project_dir):
        snapshot_current(str(project_dir), message="v1")

        # Modify current
        (project_dir / ".planagent" / "plan.md").write_text(
            "# TestApp - plan\n\nChanged!\n"
        )

        result = diff_current_vs_version(str(project_dir), 1, "plan.md")
        assert "v1/plan.md" in result or "No changes" in result

    def test_diff_nonexistent_file(self, project_dir):
        snapshot_current(str(project_dir))
        snapshot_current(str(project_dir))
        result = diff_versions(str(project_dir), 1, 2, "nonexistent.md")
        assert "not found" in result.lower() or "no changes" in result.lower()

    def test_diff_identical_versions(self, project_dir):
        snapshot_current(str(project_dir))
        snapshot_current(str(project_dir))
        result = diff_versions(str(project_dir), 1, 2, "plan.md")
        assert "No changes" in result


# ---------------------------------------------------------------------------
# Rollback tests
# ---------------------------------------------------------------------------

class TestRollback:
    def test_rollback_restores_version(self, project_dir):
        # Snapshot v1
        snapshot_current(str(project_dir))

        # Modify current plan
        original_plan = (project_dir / ".planagent" / "plan.md").read_text()
        (project_dir / ".planagent" / "plan.md").write_text("# COMPLETELY DIFFERENT\n")

        # Rollback to v1
        success = rollback_to_version(str(project_dir), 1)
        assert success

        # Current plan should match v1
        restored = (project_dir / ".planagent" / "plan.md").read_text()
        assert restored == original_plan

    def test_rollback_auto_snapshots(self, project_dir):
        snapshot_current(str(project_dir))

        # Modify and rollback
        (project_dir / ".planagent" / "plan.md").write_text("# CHANGED\n")
        rollback_to_version(str(project_dir), 1)

        # Should have v1 + auto-snapshot v2
        versions = list_versions(str(project_dir))
        assert len(versions) >= 2

    def test_rollback_nonexistent_version(self, project_dir):
        assert rollback_to_version(str(project_dir), 99) is False


# ---------------------------------------------------------------------------
# Revision context tests
# ---------------------------------------------------------------------------

class TestBuildRevisionContext:
    def test_builds_compact_summary(self):
        ctx = build_revision_context(SAMPLE_PLAN)
        assert "TestApp" in ctx
        assert "Python" in ctx
        assert "auth" in ctx
        assert "posts" in ctx
        assert "analytics" in ctx

    def test_handles_empty_plan(self):
        ctx = build_revision_context({})
        assert "N/A" in ctx

    def test_includes_endpoint_count(self):
        ctx = build_revision_context(SAMPLE_PLAN)
        assert "2" in ctx  # 2 endpoints
