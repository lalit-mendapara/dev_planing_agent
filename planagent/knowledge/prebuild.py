"""Pre-build the knowledge index from source files.

Run ONCE during development to generate shipped index files:
    python -m planagent.knowledge.prebuild

Outputs:
    planagent/knowledge/prebuilt/chunks.json   — chunk text + metadata
    planagent/knowledge/prebuilt/embeddings.npy — (N, 384) float32 vectors

These files ship with the package — no runtime building needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from planagent.knowledge.chunker import chunk_all_sources
from planagent.knowledge.embedder import embed_texts

PREBUILT_DIR = Path(__file__).parent / "prebuilt"


def prebuild(sources_dir: Path | None = None) -> dict:
    """Chunk all sources, embed, and save to prebuilt/ directory."""
    PREBUILT_DIR.mkdir(parents=True, exist_ok=True)

    print("Chunking source files...")
    chunks = chunk_all_sources(sources_dir)
    if not chunks:
        print("ERROR: No chunks produced from sources.")
        return {"status": "no_sources", "chunks": 0}

    n_sources = len({c["metadata"]["source"] for c in chunks})
    print(f"  {len(chunks)} chunks from {n_sources} source files.")

    print("Embedding chunks (this may take 30-60s on first run)...")
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    # Save chunks (text + metadata, no embeddings)
    chunks_path = PREBUILT_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"  Saved {chunks_path} ({chunks_path.stat().st_size // 1024} KB)")

    # Save embeddings as numpy array
    emb_path = PREBUILT_DIR / "embeddings.npy"
    np.save(emb_path, vectors)
    print(f"  Saved {emb_path} ({emb_path.stat().st_size // 1024} KB)")

    print(f"\n✅ Pre-built index: {len(chunks)} chunks, {n_sources} sources, {vectors.shape[1]}d vectors.")
    return {"status": "ok", "chunks": len(chunks), "sources": n_sources}


if __name__ == "__main__":
    prebuild()
