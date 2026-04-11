"""RAG retriever — full pipeline: query → embed → search → rerank → dedup.

Reranking uses a lightweight cross-encoder score approximation based on
keyword overlap + embedding score (avoids heavy cross-encoder dependency).
For production, swap _rerank() with a real cross-encoder (FlashRank/BGE).

Deduplication removes chunks sharing >30 % text overlap.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional

# Pattern to strip fenced code blocks from knowledge chunks
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")

from planagent.knowledge.embedder import embed_query
from planagent.knowledge.vectorstore import search as vector_search, collection_exists


# ---------------------------------------------------------------------------
# Reranker — lightweight keyword-overlap boost (no extra model needed)
# ---------------------------------------------------------------------------

def _keyword_overlap(query: str, text: str) -> float:
    """Fraction of query words found in text (case-insensitive)."""
    q_words = set(query.lower().split())
    t_lower = text.lower()
    if not q_words:
        return 0.0
    return sum(1 for w in q_words if w in t_lower) / len(q_words)


def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates: 0.7 * cosine_score + 0.3 * keyword_overlap.
    Drop anything below combined threshold 0.25, return top_k."""
    for c in candidates:
        kw = _keyword_overlap(query, c["text"])
        c["rerank_score"] = 0.7 * c["score"] + 0.3 * kw
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return [c for c in candidates[:top_k] if c["rerank_score"] > 0.25]


# ---------------------------------------------------------------------------
# Deduplication — remove chunks sharing >30 % content
# ---------------------------------------------------------------------------

_DEDUP_THRESHOLD = 0.30


def _deduplicate(chunks: list[dict]) -> list[dict]:
    """Remove near-duplicate chunks (>30 % text overlap)."""
    kept: list[dict] = []
    for c in chunks:
        is_dup = False
        for k in kept:
            ratio = SequenceMatcher(None, c["text"][:500], k["text"][:500]).ratio()
            if ratio > _DEDUP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Query builder — constructs targeted query from state
# ---------------------------------------------------------------------------

def _build_query_from_state(state: dict, user_message: str = "") -> str:
    """Build a targeted retrieval query from conversation state.

    Empty state  → broad query from user message / project goal
    Existing state → gap-targeted query using tech stack + gaps
    Revision → narrow query from the change request
    """
    parts: list[str] = []

    # Always include user's latest message if given
    if user_message:
        parts.append(user_message)

    # Project goal
    goal = state.get("project_goal")
    if goal:
        parts.append(goal)

    # Tech stack keywords
    ts = state.get("tech_stack", {})
    if isinstance(ts, dict):
        for v in ts.values():
            if v and v != "unknown":
                parts.append(str(v))

    # Gaps — most valuable for retrieval
    for gap in state.get("gaps_flagged", []):
        parts.append(gap)

    # Scenario hint
    scenario = state.get("scenario", "")
    if scenario == "empty" and not parts:
        parts.append("system design architecture planning new project")

    return " ".join(parts)[:500]  # cap length


# ---------------------------------------------------------------------------
# RAG budget — fewer chunks as state fills up
# ---------------------------------------------------------------------------

def _get_rag_budget(state: dict) -> int:
    """Return how many RAG chunks to inject based on state completeness."""
    collected = sum(
        1 for k in ["project_goal", "features_v1", "tech_stack", "constraints", "user_types"]
        if state.get(k)
    )
    if state.get("is_revision"):
        return 3  # revision — very targeted
    if collected >= 4:
        return 3  # state is rich
    elif collected >= 2:
        return 5
    else:
        return 8  # early conversation — maximum guidance


# ---------------------------------------------------------------------------
# Topic filter builder
# ---------------------------------------------------------------------------

def _infer_topic_filter(state: dict) -> Optional[list[str]]:
    """Infer topic filters from state to narrow vector search.

    Returns None (no filter) for empty/early state — cast wide net.
    Returns topic list for existing state to focus retrieval.
    """
    if state.get("scenario") == "empty" and not state.get("project_goal"):
        return None  # wide net

    topics: list[str] = []
    goal = (state.get("project_goal") or "").lower()
    ts = state.get("tech_stack", {})

    # Infer from goal keywords
    goal_topic_map = {
        "delivery": ["microservices", "event_driven", "message_queue", "scaling"],
        "ecommerce": ["microservices", "caching", "scaling", "database"],
        "e-commerce": ["microservices", "caching", "scaling", "database"],
        "chat": ["event_driven", "rag", "agentic", "message_queue"],
        "ai": ["rag", "agentic", "llm_inference", "ai_native", "observability"],
        "agent": ["agentic", "multi_agent", "llm_inference", "ai_native"],
        "rag": ["rag", "ai_native", "llm_inference"],
        "saas": ["microservices", "auth", "scaling", "deployment"],
        "marketplace": ["microservices", "event_driven", "scaling"],
        "social": ["caching", "event_driven", "scaling", "cdn"],
        "fintech": ["auth", "database", "scaling", "observability"],
        "hospital": ["microservices", "auth", "database"],
    }
    for keyword, t_list in goal_topic_map.items():
        if keyword in goal:
            topics.extend(t_list)

    # Infer from gaps
    for gap in state.get("gaps_flagged", []):
        gap_lower = gap.lower()
        if "test" in gap_lower:
            topics.append("testing")
        if "docker" in gap_lower or "ci" in gap_lower:
            topics.append("deployment")
        if "auth" in gap_lower:
            topics.append("auth")

    return list(set(topics)) if topics else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    state: dict,
    user_message: str = "",
    override_top_k: int | None = None,
) -> list[dict]:
    """Full RAG retrieval pipeline.

    1. Build query from state + user message
    2. Embed query
    3. Vector search with metadata filter (top_k * 3 candidates)
    4. Rerank (keyword + cosine hybrid)
    5. Deduplicate (>30 % overlap removal)
    6. Return top_k clean chunks

    Returns list of {text, score, rerank_score, metadata}.
    Returns empty list if knowledge base not built yet.
    """
    if not collection_exists():
        return []

    budget = override_top_k or _get_rag_budget(state)
    query = _build_query_from_state(state, user_message)
    if not query.strip():
        return []

    # Embed
    query_vec = embed_query(query)

    # Search with 3x over-fetch for reranking headroom
    topic_filter = _infer_topic_filter(state)
    candidates = vector_search(
        query_vec,
        top_k=budget * 3,
        topic_filter=topic_filter,
    )

    if not candidates:
        # Fallback: try without filter
        candidates = vector_search(query_vec, top_k=budget * 3)

    # Rerank
    reranked = _rerank(query, candidates, top_k=budget + 2)

    # Deduplicate
    final = _deduplicate(reranked)

    return final[:budget]


def _strip_code(text: str) -> str:
    """Remove fenced and inline code blocks from chunk text.
    Planning mode doesn't need code examples — only concepts."""
    text = _CODE_BLOCK_RE.sub("", text)
    text = _INLINE_CODE_RE.sub("", text)
    # Collapse multiple blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_rag_refs(chunks: list[dict]) -> list[str]:
    """Extract compact source/section references from chunks.
    Used to tag conversation turns with their RAG sources."""
    refs = []
    for c in chunks:
        meta = c.get("metadata", {})
        ref = f"{meta.get('source', '?')}/{meta.get('section', '?')}"
        if ref not in refs:
            refs.append(ref)
    return refs


def format_chunks_for_prompt(chunks: list[dict], max_chars: int = 4000) -> str:
    """Format retrieved chunks into a compact text block for LLM injection.

    Code blocks are stripped — planning mode needs concepts, not code.
    Keeps total size under *max_chars* to control token usage.
    """
    if not chunks:
        return ""

    lines: list[str] = []
    total = 0
    for i, c in enumerate(chunks):
        header = f"[{c['metadata']['source']}/{c['metadata']['section']}]"
        text = _strip_code(c["text"])
        if not text:
            continue
        # Trim if adding this chunk would exceed budget
        remaining = max_chars - total - len(header) - 5
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining] + "..."
        entry = f"{header}\n{text}"
        lines.append(entry)
        total += len(entry) + 1  # +1 for separator newline

    return "\n\n".join(lines)
