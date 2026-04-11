"""Conversation store — persists every turn to a temp JSONL file.

During conversation, only the LAST agent message + current user input are sent
to the LLM (plus system prompt with state/RAG). This slashes token usage by
~70% compared to sending windowed history.

At plan generation time, the full conversation is loaded from the temp file
so the LLM has complete context for producing the architecture plan.

Supports "go back" — user can revisit a previous question and change their
answer. Superseded turns are marked inactive so the final context is clean.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


class ConversationStore:
    """Append-only JSONL store for conversation turns.

    Each turn is written to disk immediately so nothing is lost on crash.
    The in-memory list mirrors the file for fast access.

    Turn schema::
        {
            "turn_id": int,          # monotonic ID
            "turn_num": int,         # logical turn number (user+assistant pair)
            "role": "user"|"assistant",
            "content": str,
            "timestamp": float,
            "active": bool,          # False when superseded by go-back
            "superseded_by": int|null,  # turn_id that replaced this one
            "rag_refs": [str],       # RAG source/section refs used for this turn
        }
    """

    def __init__(self, store_path: Optional[Path] = None):
        if store_path is None:
            store_path = Path(".planagent") / "conversation_turns.jsonl"
        self.path = Path(store_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.turns: list[dict] = []
        self._next_id = 0
        # Load existing turns if resuming
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(
        self,
        role: str,
        content: str,
        turn_num: int,
        rag_refs: Optional[list[str]] = None,
    ) -> dict:
        """Append a turn and flush to disk. Returns the turn dict."""
        turn = {
            "turn_id": self._next_id,
            "turn_num": turn_num,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "active": True,
            "superseded_by": None,
            "rag_refs": rag_refs or [],
        }
        self.turns.append(turn)
        self._next_id += 1
        self._append_to_disk(turn)
        return turn

    def supersede_turn(self, turn_id: int, replaced_by_id: int) -> None:
        """Mark a turn as superseded (user went back and changed answer)."""
        for t in self.turns:
            if t["turn_id"] == turn_id:
                t["active"] = False
                t["superseded_by"] = replaced_by_id
                break
        self._flush_all()

    # ------------------------------------------------------------------
    # Go-back support
    # ------------------------------------------------------------------

    def get_agent_questions(self) -> list[dict]:
        """Return all active assistant turns (the questions asked)."""
        return [t for t in self.turns if t["role"] == "assistant" and t["active"]]

    def get_last_agent_turn(self) -> Optional[dict]:
        """Return the most recent active assistant turn."""
        for t in reversed(self.turns):
            if t["role"] == "assistant" and t["active"]:
                return t
        return None

    def get_last_user_turn(self) -> Optional[dict]:
        """Return the most recent active user turn."""
        for t in reversed(self.turns):
            if t["role"] == "user" and t["active"]:
                return t
        return None

    def go_back(self) -> Optional[dict]:
        """Supersede the last user+assistant exchange and return the
        agent question BEFORE that exchange so the user can re-answer.

        Returns the previous agent question turn, or None if at start.
        """
        active = [t for t in self.turns if t["active"]]
        if len(active) < 3:
            return None  # nothing to go back to

        # Find last user turn and its corresponding assistant response
        last_user = None
        last_assistant = None
        for t in reversed(active):
            if t["role"] == "assistant" and last_assistant is None and last_user is not None:
                last_assistant_before = t
                break
            if t["role"] == "user" and last_user is None:
                last_user = t
            if t["role"] == "assistant" and last_user is None:
                last_assistant = t

        if last_user is None:
            return None

        # Supersede the last assistant response and last user answer
        if last_assistant is not None:
            last_assistant["active"] = False
        last_user["active"] = False

        self._flush_all()

        # Return the agent question that preceded the superseded user answer
        prev_agent = None
        for t in reversed(self.turns):
            if t["role"] == "assistant" and t["active"]:
                prev_agent = t
                break
        return prev_agent

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def get_active_turns(self) -> list[dict]:
        """Return all active (non-superseded) turns in order."""
        return [t for t in self.turns if t["active"]]

    def get_full_context_for_plan(self) -> str:
        """Build full conversation text from active turns for plan generation."""
        active = self.get_active_turns()
        lines = []
        for t in active:
            prefix = "DEVELOPER" if t["role"] == "user" else "ARCHITECT"
            lines.append(f"{prefix}: {t['content']}")
        return "\n\n".join(lines)

    def get_last_exchange(self) -> list[dict]:
        """Return the last agent message + user response as message dicts.
        This is the ONLY history sent during conversation turns."""
        active = self.get_active_turns()
        result = []
        # Find the last assistant message
        for t in reversed(active):
            if t["role"] == "assistant":
                result.insert(0, {"role": "assistant", "content": t["content"]})
                break
        return result

    def get_rag_refs_for_turn(self, turn_id: int) -> list[str]:
        """Get RAG references used in a specific turn."""
        for t in self.turns:
            if t["turn_id"] == turn_id:
                return t.get("rag_refs", [])
        return []

    @property
    def active_turn_count(self) -> int:
        """Number of active user turns (logical conversation length)."""
        return sum(1 for t in self.turns if t["active"] and t["role"] == "user")

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append_to_disk(self, turn: dict) -> None:
        """Append a single turn to the JSONL file."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(turn, default=str) + "\n")

    def _flush_all(self) -> None:
        """Rewrite the entire file (needed after superseding turns)."""
        with open(self.path, "w", encoding="utf-8") as f:
            for t in self.turns:
                f.write(json.dumps(t, default=str) + "\n")

    def _load(self) -> None:
        """Load turns from existing JSONL file."""
        self.turns = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    turn = json.loads(line)
                    self.turns.append(turn)
        if self.turns:
            self._next_id = max(t["turn_id"] for t in self.turns) + 1

    def clear(self) -> None:
        """Remove all turns and delete the file."""
        self.turns = []
        self._next_id = 0
        if self.path.exists():
            self.path.unlink()
