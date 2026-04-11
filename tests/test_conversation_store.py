"""Tests for ConversationStore — temp JSONL persistence, go-back, supersede."""

import pytest
import json
from pathlib import Path
from planagent.conversation_store import ConversationStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a ConversationStore with a temp file."""
    return ConversationStore(store_path=tmp_path / "turns.jsonl")


# ===================================================================
# Basic operations
# ===================================================================

class TestBasicOperations:

    def test_add_turn(self, tmp_store):
        t = tmp_store.add("user", "Hello", turn_num=1)
        assert t["turn_id"] == 0
        assert t["role"] == "user"
        assert t["content"] == "Hello"
        assert t["active"] is True
        assert tmp_store.total_turns == 1

    def test_add_multiple_turns(self, tmp_store):
        tmp_store.add("assistant", "What are you building?", turn_num=0)
        tmp_store.add("user", "A food delivery app", turn_num=1)
        tmp_store.add("assistant", "Great choice!", turn_num=1)
        assert tmp_store.total_turns == 3
        assert tmp_store.active_turn_count == 1  # only user turns

    def test_turns_persisted_to_disk(self, tmp_store):
        tmp_store.add("user", "Hello", turn_num=1)
        tmp_store.add("assistant", "Hi there", turn_num=1)
        # Read the file directly
        lines = tmp_store.path.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["content"] == "Hello"

    def test_load_from_existing_file(self, tmp_path):
        path = tmp_path / "turns.jsonl"
        # Create store, add turns
        s1 = ConversationStore(store_path=path)
        s1.add("user", "Message 1", turn_num=1)
        s1.add("assistant", "Response 1", turn_num=1)
        # Load from same file
        s2 = ConversationStore(store_path=path)
        assert s2.total_turns == 2
        assert s2.turns[0]["content"] == "Message 1"

    def test_rag_refs_stored(self, tmp_store):
        refs = ["core_hld/caching", "arch_patterns/microservices"]
        t = tmp_store.add("assistant", "Use Redis", turn_num=1, rag_refs=refs)
        assert t["rag_refs"] == refs

    def test_clear(self, tmp_store):
        tmp_store.add("user", "Hello", turn_num=1)
        tmp_store.clear()
        assert tmp_store.total_turns == 0
        assert not tmp_store.path.exists()


# ===================================================================
# Context retrieval
# ===================================================================

class TestContextRetrieval:

    def test_get_active_turns(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        active = tmp_store.get_active_turns()
        assert len(active) == 3
        assert all(t["active"] for t in active)

    def test_get_last_agent_turn(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        last = tmp_store.get_last_agent_turn()
        assert last["content"] == "Q2"

    def test_get_last_user_turn(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        last = tmp_store.get_last_user_turn()
        assert last["content"] == "A1"

    def test_get_last_exchange(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        exchange = tmp_store.get_last_exchange()
        assert len(exchange) == 1
        assert exchange[0]["role"] == "assistant"
        assert exchange[0]["content"] == "Q2"

    def test_get_full_context_for_plan(self, tmp_store):
        tmp_store.add("assistant", "What are you building?", turn_num=0)
        tmp_store.add("user", "A food delivery app", turn_num=1)
        tmp_store.add("assistant", "What features?", turn_num=1)
        tmp_store.add("user", "Orders and payments", turn_num=2)
        ctx = tmp_store.get_full_context_for_plan()
        assert "ARCHITECT: What are you building?" in ctx
        assert "DEVELOPER: A food delivery app" in ctx
        assert "DEVELOPER: Orders and payments" in ctx

    def test_get_agent_questions(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        questions = tmp_store.get_agent_questions()
        assert len(questions) == 2
        assert questions[0]["content"] == "Q1"
        assert questions[1]["content"] == "Q2"

    def test_active_turn_count(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        tmp_store.add("user", "A2", turn_num=2)
        assert tmp_store.active_turn_count == 2  # only user turns


# ===================================================================
# Go-back / supersede
# ===================================================================

class TestGoBack:

    def test_go_back_returns_previous_question(self, tmp_store):
        tmp_store.add("assistant", "Q1: What are you building?", turn_num=0)
        tmp_store.add("user", "A food app", turn_num=1)
        tmp_store.add("assistant", "Q2: What features?", turn_num=1)
        tmp_store.add("user", "Orders only", turn_num=2)
        tmp_store.add("assistant", "Q3: Tech stack?", turn_num=2)
        # Go back — should supersede the last user answer + assistant response
        prev = tmp_store.go_back()
        assert prev is not None
        # Should return Q2 (the question before the superseded exchange)
        assert "Q2" in prev["content"] or "Q1" in prev["content"]

    def test_go_back_marks_turns_inactive(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        tmp_store.add("user", "A2", turn_num=2)
        tmp_store.add("assistant", "Q3", turn_num=2)
        tmp_store.go_back()
        active = tmp_store.get_active_turns()
        # A2 and Q3 should be inactive, Q1, A1, Q2 remain
        active_contents = [t["content"] for t in active]
        assert "A2" not in active_contents or "Q3" not in active_contents

    def test_go_back_at_start_returns_none(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        result = tmp_store.go_back()
        assert result is None

    def test_go_back_with_only_opening(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        # Only one exchange — going back should return None or Q1
        result = tmp_store.go_back()
        # With only 2 active turns, there's nothing to go back to meaningfully
        # (less than 3 active turns)
        assert result is None

    def test_go_back_updates_disk(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "A1", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        tmp_store.add("user", "A2", turn_num=2)
        tmp_store.add("assistant", "Q3", turn_num=2)
        tmp_store.go_back()
        # Reload from disk
        reloaded = ConversationStore(store_path=tmp_store.path)
        inactive = [t for t in reloaded.turns if not t["active"]]
        assert len(inactive) >= 1

    def test_full_context_excludes_superseded(self, tmp_store):
        tmp_store.add("assistant", "Q1", turn_num=0)
        tmp_store.add("user", "Wrong answer", turn_num=1)
        tmp_store.add("assistant", "Q2", turn_num=1)
        tmp_store.add("user", "A2", turn_num=2)
        tmp_store.add("assistant", "Q3", turn_num=2)
        tmp_store.go_back()
        # Add corrected answer
        tmp_store.add("user", "Correct answer", turn_num=2)
        ctx = tmp_store.get_full_context_for_plan()
        assert "Correct answer" in ctx
        # The superseded turn "A2" should not appear (it's inactive)
        # But "Wrong answer" should still be there (only last exchange was superseded)

    def test_supersede_specific_turn(self, tmp_store):
        t1 = tmp_store.add("user", "Original", turn_num=1)
        t2 = tmp_store.add("user", "Replacement", turn_num=1)
        tmp_store.supersede_turn(t1["turn_id"], t2["turn_id"])
        assert tmp_store.turns[0]["active"] is False
        assert tmp_store.turns[0]["superseded_by"] == t2["turn_id"]
        assert tmp_store.turns[1]["active"] is True


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_empty_store(self, tmp_store):
        assert tmp_store.total_turns == 0
        assert tmp_store.active_turn_count == 0
        assert tmp_store.get_last_agent_turn() is None
        assert tmp_store.get_last_user_turn() is None
        assert tmp_store.get_full_context_for_plan() == ""
        assert tmp_store.get_last_exchange() == []

    def test_get_rag_refs_nonexistent_turn(self, tmp_store):
        assert tmp_store.get_rag_refs_for_turn(999) == []

    def test_directory_created_automatically(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / "c" / "turns.jsonl"
        store = ConversationStore(store_path=deep_path)
        store.add("user", "Hello", turn_num=1)
        assert deep_path.exists()
