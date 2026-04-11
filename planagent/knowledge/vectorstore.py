"""In-memory vector store — loads pre-built index shipped with the package.

No external DB (Qdrant removed). Uses numpy cosine similarity for search.
Pre-built data lives in planagent/knowledge/prebuilt/:
  - chunks.json    — chunk text + metadata
  - embeddings.npy — (N, 384) float32 vectors

To regenerate:  python -m planagent.knowledge.prebuild
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

PREBUILT_DIR = Path(__file__).parent / "prebuilt"

# Lazy-loaded singleton index
_chunks: list[dict] | None = None
_embeddings: np.ndarray | None = None


def _load_index():
    """Load pre-built chunks + embeddings into memory (once)."""
    global _chunks, _embeddings
    if _chunks is not None:
        return

    chunks_path = PREBUILT_DIR / "chunks.json"
    emb_path = PREBUILT_DIR / "embeddings.npy"

    if not chunks_path.exists() or not emb_path.exists():
        _chunks = []
        _embeddings = np.empty((0, 384), dtype=np.float32)
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    _embeddings = np.load(emb_path).astype(np.float32)

    # Pre-normalize embeddings for fast cosine similarity
    norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    _embeddings = _embeddings / norms


def collection_exists() -> bool:
    """Check if pre-built index is available."""
    _load_index()
    return _chunks is not None and len(_chunks) > 0


def get_chunk_count() -> int:
    """Return number of chunks in the index."""
    _load_index()
    return len(_chunks) if _chunks else 0


def search(
    query_vector: np.ndarray,
    top_k: int = 10,
    topic_filter: Optional[list[str]] = None,
    source_filter: Optional[list[str]] = None,
) -> list[dict]:
    """Cosine similarity search with optional metadata filters.

    Returns list of dicts: {text, score, metadata}.
    """
    _load_index()
    if not _chunks or _embeddings is None or _embeddings.shape[0] == 0:
        return []

    # Normalize query vector
    q_norm = np.linalg.norm(query_vector)
    if q_norm == 0:
        return []
    q_vec = query_vector / q_norm

    # Compute cosine similarities (embeddings already normalized)
    scores = _embeddings @ q_vec  # (N,)

    # Apply metadata filters (mask out non-matching chunks)
    if topic_filter or source_filter:
        mask = np.ones(len(_chunks), dtype=bool)
        for i, chunk in enumerate(_chunks):
            meta = chunk["metadata"]
            if topic_filter:
                # Match if ANY requested topic is in chunk's topics
                if not any(t in meta.get("topics", []) for t in topic_filter):
                    mask[i] = False
            if source_filter:
                if meta.get("source") not in source_filter:
                    mask[i] = False
        scores = np.where(mask, scores, -1.0)

    # Get top-k indices
    if top_k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        s = float(scores[idx])
        if s <= 0:
            break
        chunk = _chunks[idx]
        results.append({
            "text": chunk["text"],
            "score": s,
            "metadata": {
                "source": chunk["metadata"]["source"],
                "section": chunk["metadata"]["section"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "topics": chunk["metadata"]["topics"],
            },
        })
    return results
