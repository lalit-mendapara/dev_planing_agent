"""Knowledge base index — pre-built and shipped with the package.

The knowledge base is pre-computed at development time and stored as
planagent/knowledge/prebuilt/chunks.json + embeddings.npy.

To regenerate:  python -m planagent.knowledge.prebuild

No runtime building needed. No separate CLI command.
"""

from __future__ import annotations

from pathlib import Path

from planagent.knowledge.vectorstore import collection_exists


def build_knowledge_base(sources_dir: Path | None = None, quiet: bool = False) -> dict:
    """Delegate to prebuild script.  Kept for backwards compatibility."""
    from planagent.knowledge.prebuild import prebuild
    return prebuild(sources_dir)


def ensure_knowledge_base(quiet: bool = False) -> bool:
    """Check if pre-built index is available.  No runtime building."""
    return collection_exists()


def _count_sources(chunks: list[dict]) -> int:
    return len({c["metadata"]["source"] for c in chunks})
