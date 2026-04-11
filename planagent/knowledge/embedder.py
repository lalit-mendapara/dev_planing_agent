"""Embedding wrapper — uses fastembed for local, zero-API-key embeddings.

fastembed runs ONNX models locally (no PyTorch/TensorFlow needed).
Default model: BAAI/bge-small-en-v1.5 (384 dims, fast, good quality).
"""

from __future__ import annotations

import numpy as np
from typing import List

# Lazy-loaded singleton to avoid import cost on every CLI invocation
_model = None
_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 384 dims, ~50 MB, fast
_EMBEDDING_DIM = 384


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts. Returns (N, 384) float32 array."""
    model = _get_model()
    # fastembed returns a generator; materialise into ndarray
    vectors = list(model.embed(texts))
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns (384,) float32 array."""
    model = _get_model()
    vectors = list(model.query_embed(text))
    return np.array(vectors[0], dtype=np.float32)


def get_embedding_dim() -> int:
    return _EMBEDDING_DIM
