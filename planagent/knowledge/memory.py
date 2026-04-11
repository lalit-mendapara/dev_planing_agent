"""Conversation memory — in-memory vector store for conversation turns.

Replaces the sliding window + LLM summarization approach. Each turn is
embedded and stored. Older turns are retrieved by relevance (cosine
similarity) instead of being sent in full or summarized by an LLM call.

This saves tokens by:
1. Not sending full conversation history — only relevant past context
2. Eliminating the LLM summarization call entirely
3. Compacting retrieved turns to a token budget
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ConversationMemory:
    """In-memory embedding store for conversation turns.

    Usage:
        mem = ConversationMemory()
        mem.add("user said X", turn_num=1)
        mem.add("assistant said Y", turn_num=1)
        relevant = mem.retrieve("query about X", top_k=3, skip_last_n=2)
    """

    def __init__(self):
        self.entries: list[dict] = []  # {text, compact, embedding, turn_num, role}
        self._embeddings: list[np.ndarray] = []

    @property
    def size(self) -> int:
        return len(self.entries)

    def add(self, text: str, turn_num: int, role: str = "user") -> None:
        """Embed and store a conversation turn.

        Embedding uses only the first 500 chars (enough for semantic
        matching without wasting compute on long assistant responses).
        """
        try:
            from planagent.knowledge.embedder import embed_query
            compact = text[:500]
            vec = embed_query(compact)
            self.entries.append({
                "text": text,
                "compact": compact,
                "turn_num": turn_num,
                "role": role,
            })
            self._embeddings.append(vec)
        except Exception:
            pass  # never block conversation on embedding failure

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        skip_last_n: int = 2,
        max_chars_per_turn: int = 300,
    ) -> list[dict]:
        """Retrieve most relevant past turns, skipping the last N entries.

        Returns list of {text, turn_num, role, score} sorted by relevance.
        Text is truncated to max_chars_per_turn for token efficiency.
        """
        searchable_count = len(self.entries) - skip_last_n
        if searchable_count <= 0:
            return []

        try:
            from planagent.knowledge.embedder import embed_query
            query_vec = embed_query(query[:500])
        except Exception:
            return []

        # Build embedding matrix for searchable entries
        emb_matrix = np.array(self._embeddings[:searchable_count], dtype=np.float32)
        if emb_matrix.shape[0] == 0:
            return []

        # Normalize
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        q_vec = query_vec / q_norm

        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_normed = emb_matrix / norms

        scores = emb_normed @ q_vec

        # Get top-k
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        top_indices = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s < 0.25:  # relevance threshold
                continue
            entry = self.entries[idx]
            text = entry["text"]
            if len(text) > max_chars_per_turn:
                text = text[:max_chars_per_turn] + "..."
            results.append({
                "text": text,
                "turn_num": entry["turn_num"],
                "role": entry["role"],
                "score": s,
            })
        return results

    def format_for_prompt(self, retrieved: list[dict], max_chars: int = 800) -> str:
        """Format retrieved turns into a compact context block."""
        if not retrieved:
            return ""
        lines = []
        total = 0
        for r in retrieved:
            prefix = "DEV" if r["role"] == "user" else "AGENT"
            line = f"[Turn {r['turn_num']}] {prefix}: {r['text']}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)
