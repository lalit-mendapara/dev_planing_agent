"""Section-aware chunking engine for system design source files.

Splits source .txt files by SECTION headers, then sub-chunks each section
into 300-500 token pieces with ~20% overlap.  Each chunk carries metadata
(source file, section name, topic tags) used for filtered retrieval.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Section detection — all 6 source files use the same format:
#   ----------------------------------------------------------------
#   SECTION N: TITLE
#   ----------------------------------------------------------------
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^-{10,}\n"           # dashed line
    r"SECTION\s+\d+:\s*(.+?)\n"  # SECTION N: TITLE
    r"-{10,}",             # dashed line
    re.MULTILINE,
)

# Rough chars-per-token ratio (English text averages ~4 chars/token)
_CHARS_PER_TOKEN = 4
_TARGET_TOKENS = 400       # middle of 300-500 range
_OVERLAP_RATIO = 0.20      # 20 % overlap
_TARGET_CHARS = _TARGET_TOKENS * _CHARS_PER_TOKEN          # 1600
_OVERLAP_CHARS = int(_TARGET_CHARS * _OVERLAP_RATIO)       # 320

# Topic tags inferred from chunk text (lightweight keyword match)
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "load_balancing":  ["load balanc", "round robin", "least connection", "nginx", "haproxy", "alb", "nlb"],
    "caching":         ["cach", "redis", "memcached", "ttl", "cache-aside", "write-through"],
    "message_queue":   ["message queue", "kafka", "rabbitmq", "celery", "pub/sub", "consumer", "producer", "dlq"],
    "db_sharding":     ["shard", "partition", "replication", "read replica"],
    "api_gateway":     ["api gateway", "kong", "traefik"],
    "cdn":             ["cdn", "cloudfront", "cloudflare", "edge"],
    "rate_limiting":   ["rate limit", "token bucket", "leaky bucket"],
    "microservices":   ["microservice", "service discovery", "circuit breaker", "saga pattern"],
    "monolithic":      ["monolith"],
    "event_driven":    ["event-driven", "event driven", "event sourcing", "cqrs"],
    "serverless":      ["serverless", "lambda", "faas", "cloud function"],
    "layered":         ["layered", "n-tier", "3-tier", "presentation layer", "business logic layer"],
    "rag":             ["rag", "retrieval augmented", "vector db", "embedding", "chunk", "rerank"],
    "agentic":         ["agent", "react pattern", "reflection", "tool use", "langgraph"],
    "multi_agent":     ["multi-agent", "orchestrator", "specialist agent"],
    "llm_inference":   ["llm inference", "litellm", "model routing", "semantic cach"],
    "observability":   ["observability", "langfuse", "tracing", "monitoring", "drift"],
    "mlops":           ["mlops", "training pipeline", "model registry", "fine-tun", "lora", "qlora"],
    "database":        ["postgresql", "mysql", "mongodb", "database", "sql"],
    "auth":            ["auth", "jwt", "oauth", "session"],
    "testing":         ["test", "tdd", "unit test", "integration test"],
    "deployment":      ["deploy", "canary", "blue-green", "docker", "kubernetes"],
    "design_pattern":  ["repository pattern", "factory pattern", "singleton", "observer pattern", "strategy pattern"],
    "hld":             ["high-level design", "hld", "architecture diagram", "data flow"],
    "lld":             ["low-level design", "lld", "class diagram", "erd", "api contract"],
    "scaling":         ["scal", "horizontal", "vertical", "throughput"],
    "ai_native":       ["ai-native", "ai native", "prompt management", "human-in-the-loop"],
}


def _detect_topics(text: str) -> list[str]:
    """Return topic tags present in *text* (case-insensitive keyword scan)."""
    lower = text.lower()
    return [tag for tag, kws in _TOPIC_KEYWORDS.items() if any(k in lower for k in kws)]


def _split_into_sections(text: str, source_name: str) -> list[tuple[str, str]]:
    """Split *text* into (section_name, section_body) pairs."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return [(source_name, text)]

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


def _sub_chunk(text: str, max_chars: int = _TARGET_CHARS, overlap: int = _OVERLAP_CHARS) -> list[str]:
    """Split *text* into overlapping pieces respecting paragraph/sentence boundaries."""
    # Prefer splitting on double-newline (paragraph), then single newline, then ". "
    separators = ["\n\n", "\n", ". ", " "]
    return _recursive_split(text, separators, max_chars, overlap)


def _recursive_split(text: str, separators: list[str], max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    sep = separators[0] if separators else " "
    parts = text.split(sep)
    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            # Overlap: keep tail of current chunk
            tail = current[-overlap:] if overlap else ""
            current = tail + sep + part if tail else part
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    # If any chunk is still too long and we have more separators, recurse
    if len(separators) > 1:
        refined: list[str] = []
        for c in chunks:
            if len(c) > max_chars:
                refined.extend(_recursive_split(c, separators[1:], max_chars, overlap))
            else:
                refined.append(c)
        return refined

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_file(filepath: Path) -> list[dict]:
    """Chunk a single source file into metadata-rich pieces.

    Returns list of dicts:
        {text, metadata: {source, section, chunk_index, topics, char_count}}
    """
    text = filepath.read_text(encoding="utf-8")
    source = filepath.stem  # e.g. "core_hld_concepts"
    sections = _split_into_sections(text, source)

    all_chunks: list[dict] = []
    for section_name, section_body in sections:
        pieces = _sub_chunk(section_body)
        for idx, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "metadata": {
                    "source": source,
                    "section": section_name.lower().replace(" ", "_"),
                    "chunk_index": idx,
                    "topics": _detect_topics(piece),
                    "char_count": len(piece),
                },
            })
    return all_chunks


def chunk_all_sources(sources_dir: Path | None = None) -> list[dict]:
    """Chunk every .txt file in the sources directory."""
    if sources_dir is None:
        sources_dir = Path(__file__).parent / "sources"
    chunks: list[dict] = []
    for f in sorted(sources_dir.glob("*.txt")):
        chunks.extend(chunk_file(f))
    return chunks
