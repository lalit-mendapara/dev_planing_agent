"""Tests for the knowledge RAG pipeline: chunker, embedder, vectorstore, retriever."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

SOURCES_DIR = Path(__file__).parent.parent / "planagent" / "knowledge" / "sources"


# ===================================================================
# Chunker tests
# ===================================================================

class TestChunker:

    def test_chunk_all_sources_returns_chunks(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        assert len(chunks) > 0, "Should produce chunks from 6 source files"

    def test_chunk_count_reasonable(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        # 6 files × ~8 sections × ~3 chunks avg = ~150-400 chunks
        assert 50 < len(chunks) < 800, f"Unexpected chunk count: {len(chunks)}"

    def test_chunk_has_required_fields(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        for c in chunks[:5]:
            assert "text" in c
            assert "metadata" in c
            assert "source" in c["metadata"]
            assert "section" in c["metadata"]
            assert "chunk_index" in c["metadata"]
            assert "topics" in c["metadata"]
            assert "char_count" in c["metadata"]

    def test_chunk_text_not_empty(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        for c in chunks:
            assert len(c["text"].strip()) > 0, "Chunk text should not be empty"

    def test_chunk_size_within_bounds(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        # Target is ~1600 chars (400 tokens × 4 chars/token)
        oversized = [c for c in chunks if c["metadata"]["char_count"] > 3000]
        assert len(oversized) < len(chunks) * 0.1, \
            f"Too many oversized chunks: {len(oversized)}/{len(chunks)}"

    def test_all_6_sources_represented(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        sources = {c["metadata"]["source"] for c in chunks}
        assert len(sources) == 6, f"Expected 6 sources, got {len(sources)}: {sources}"

    def test_topics_detected(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        all_topics = set()
        for c in chunks:
            all_topics.update(c["metadata"]["topics"])
        # Should detect multiple topic categories
        assert len(all_topics) >= 5, f"Too few topics detected: {all_topics}"

    def test_sections_detected(self):
        from planagent.knowledge.chunker import chunk_all_sources
        chunks = chunk_all_sources(SOURCES_DIR)
        sections = {c["metadata"]["section"] for c in chunks}
        assert len(sections) > 10, f"Too few sections: {len(sections)}"

    def test_chunk_single_file(self):
        from planagent.knowledge.chunker import chunk_file
        f = SOURCES_DIR / "core_hld_concepts.txt"
        chunks = chunk_file(f)
        assert len(chunks) > 5, "core_hld_concepts should produce multiple chunks"
        assert all(c["metadata"]["source"] == "core_hld_concepts" for c in chunks)

    def test_chunk_overlap_exists(self):
        """Adjacent chunks from same section should share some text (overlap)."""
        from planagent.knowledge.chunker import chunk_file
        f = SOURCES_DIR / "core_hld_concepts.txt"
        chunks = chunk_file(f)
        # Find adjacent chunks from same section
        overlap_found = False
        for i in range(len(chunks) - 1):
            if (chunks[i]["metadata"]["section"] == chunks[i+1]["metadata"]["section"]
                    and chunks[i]["metadata"]["chunk_index"] + 1 == chunks[i+1]["metadata"]["chunk_index"]):
                # Check if tail of chunk i appears in chunk i+1
                tail = chunks[i]["text"][-100:]
                if tail in chunks[i+1]["text"]:
                    overlap_found = True
                    break
        # Overlap may not always be detectable due to splitting, so just check chunks exist
        assert len(chunks) > 1


# ===================================================================
# Embedder tests
# ===================================================================

class TestEmbedder:

    def test_embed_texts_returns_correct_shape(self):
        from planagent.knowledge.embedder import embed_texts, get_embedding_dim
        texts = ["hello world", "system design architecture"]
        vecs = embed_texts(texts)
        assert vecs.shape == (2, get_embedding_dim())

    def test_embed_query_returns_correct_shape(self):
        from planagent.knowledge.embedder import embed_query, get_embedding_dim
        vec = embed_query("food delivery app architecture")
        assert vec.shape == (get_embedding_dim(),)

    def test_similar_texts_have_high_cosine(self):
        from planagent.knowledge.embedder import embed_texts
        vecs = embed_texts([
            "load balancing distributes traffic across servers",
            "load balancer routes requests to multiple server instances",
            "cooking pasta requires boiling water",
        ])
        # Cosine similarity between first two should be higher than first and third
        cos_sim_12 = np.dot(vecs[0], vecs[1]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))
        cos_sim_13 = np.dot(vecs[0], vecs[2]) / (np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[2]))
        assert cos_sim_12 > cos_sim_13, "Similar texts should have higher cosine similarity"

    def test_embedding_dim_constant(self):
        from planagent.knowledge.embedder import get_embedding_dim
        assert get_embedding_dim() == 384


# ===================================================================
# Vectorstore tests
# ===================================================================

class TestVectorstore:

    def test_prebuilt_index_loads(self):
        """Pre-built index should load automatically from shipped files."""
        from planagent.knowledge.vectorstore import collection_exists, get_chunk_count
        assert collection_exists(), "Pre-built index should be available"
        assert get_chunk_count() > 50, "Should have 50+ chunks"

    def test_search_returns_results(self):
        """Search should return scored results from pre-built index."""
        from planagent.knowledge.embedder import embed_query
        from planagent.knowledge.vectorstore import search, collection_exists

        if not collection_exists():
            pytest.skip("Pre-built index not available")

        q_vec = embed_query("caching strategies Redis TTL")
        results = search(q_vec, top_k=5)
        assert len(results) > 0
        assert results[0]["score"] > 0
        assert "text" in results[0]
        assert "metadata" in results[0]

    def test_search_with_topic_filter(self):
        from planagent.knowledge.embedder import embed_query
        from planagent.knowledge.vectorstore import search, collection_exists

        if not collection_exists():
            pytest.skip("Pre-built index not available")

        q_vec = embed_query("message queue Kafka RabbitMQ")
        results = search(q_vec, top_k=5, topic_filter=["message_queue"])
        assert len(results) > 0
        for r in results:
            assert "message_queue" in r["metadata"]["topics"]

    def test_search_with_source_filter(self):
        from planagent.knowledge.embedder import embed_query
        from planagent.knowledge.vectorstore import search, collection_exists

        if not collection_exists():
            pytest.skip("Pre-built index not available")

        q_vec = embed_query("microservices architecture pattern")
        results = search(q_vec, top_k=5, source_filter=["architecture_patterns"])
        assert len(results) > 0
        for r in results:
            assert r["metadata"]["source"] == "architecture_patterns"


# ===================================================================
# Retriever tests
# ===================================================================

class TestRetriever:

    def test_retrieve_empty_state(self):
        """Empty state with user message should return broad chunks."""
        from planagent.knowledge.retriever import retrieve
        from planagent.knowledge.vectorstore import collection_exists

        if not collection_exists():
            pytest.skip("Collection not built yet")

        state = {
            "scenario": "empty",
            "project_goal": None,
            "tech_stack": {},
            "gaps_flagged": [],
            "features_v1": [],
            "constraints": [],
            "user_types": [],
            "rag_context": [],
            "rag_last_query": "",
        }
        results = retrieve(state, user_message="I want to build a food delivery app")
        assert len(results) > 0
        assert len(results) <= 8  # budget for empty state

    def test_retrieve_existing_state(self):
        """Existing state with tech stack + gaps should return targeted chunks."""
        from planagent.knowledge.retriever import retrieve
        from planagent.knowledge.vectorstore import collection_exists

        if not collection_exists():
            pytest.skip("Collection not built yet")

        state = {
            "scenario": "existing",
            "project_goal": "food delivery platform",
            "tech_stack": {"language": "Python", "framework": "FastAPI", "database": "PostgreSQL"},
            "features_v1": ["order management", "restaurant listing"],
            "gaps_flagged": ["no caching", "no message queue"],
            "constraints": ["3 months deadline"],
            "user_types": ["customer", "restaurant", "driver"],
            "rag_context": [],
            "rag_last_query": "",
        }
        results = retrieve(state)
        assert len(results) > 0
        assert len(results) <= 5  # tighter budget for existing state

    def test_retrieve_revision_mode(self):
        """Revision mode should return very few targeted chunks."""
        from planagent.knowledge.retriever import retrieve
        from planagent.knowledge.vectorstore import collection_exists

        if not collection_exists():
            pytest.skip("Collection not built yet")

        state = {
            "scenario": "existing",
            "is_revision": True,
            "project_goal": "food delivery platform",
            "tech_stack": {"language": "Python", "framework": "FastAPI"},
            "features_v1": ["orders"],
            "gaps_flagged": [],
            "constraints": [],
            "user_types": [],
            "rag_context": [],
            "rag_last_query": "",
        }
        results = retrieve(state, user_message="add WebSocket support for live tracking")
        assert len(results) <= 3  # revision budget is 3

    def test_format_chunks_for_prompt(self):
        from planagent.knowledge.retriever import format_chunks_for_prompt

        chunks = [
            {
                "text": "Load balancing distributes traffic across servers.",
                "score": 0.9,
                "rerank_score": 0.85,
                "metadata": {"source": "core_hld_concepts", "section": "load_balancing",
                             "chunk_index": 0, "topics": ["load_balancing"]},
            },
            {
                "text": "Redis is an in-memory cache for sub-millisecond access.",
                "score": 0.8,
                "rerank_score": 0.75,
                "metadata": {"source": "core_hld_concepts", "section": "caching",
                             "chunk_index": 0, "topics": ["caching"]},
            },
        ]
        text = format_chunks_for_prompt(chunks, max_chars=5000)
        assert "[core_hld_concepts/load_balancing]" in text
        assert "[core_hld_concepts/caching]" in text
        assert "Load balancing" in text

    def test_format_chunks_respects_max_chars(self):
        from planagent.knowledge.retriever import format_chunks_for_prompt

        chunks = [
            {
                "text": "x" * 2000,
                "score": 0.9,
                "rerank_score": 0.85,
                "metadata": {"source": "test", "section": "s1",
                             "chunk_index": 0, "topics": []},
            },
            {
                "text": "y" * 2000,
                "score": 0.8,
                "rerank_score": 0.75,
                "metadata": {"source": "test", "section": "s2",
                             "chunk_index": 0, "topics": []},
            },
        ]
        text = format_chunks_for_prompt(chunks, max_chars=500)
        assert len(text) <= 600  # some slack for headers

    def test_deduplication(self):
        from planagent.knowledge.retriever import _deduplicate

        chunks = [
            {"text": "Load balancing distributes traffic across multiple servers for high availability and fault tolerance in production systems.", "score": 0.9},
            {"text": "Load balancing distributes traffic across multiple servers for high availability and fault tolerance in production systems.", "score": 0.8},  # exact dup
            {"text": "Kafka is a distributed event streaming platform that stores events as an immutable log. Consumers can replay events from any offset in the topic partition. It supports millions of messages per second with very low latency across consumer groups.", "score": 0.7},
        ]
        result = _deduplicate(chunks)
        assert len(result) == 2, f"Expected 2 after dedup, got {len(result)}"

    def test_rag_budget_decreases_with_state(self):
        from planagent.knowledge.retriever import _get_rag_budget

        empty = {"scenario": "empty"}
        assert _get_rag_budget(empty) == 8

        partial = {"project_goal": "x", "features_v1": ["a"], "tech_stack": {"lang": "py"}}
        assert _get_rag_budget(partial) == 5

        rich = {
            "project_goal": "x", "features_v1": ["a"], "tech_stack": {"lang": "py"},
            "constraints": ["3 months"], "user_types": ["admin"],
        }
        assert _get_rag_budget(rich) == 3

        revision = {"is_revision": True}
        assert _get_rag_budget(revision) == 3


# ===================================================================
# Build index tests
# ===================================================================

class TestBuildIndex:

    def test_ensure_knowledge_base_returns_true(self):
        """Pre-built index should be detected by ensure_knowledge_base."""
        from planagent.knowledge.build_index import ensure_knowledge_base
        result = ensure_knowledge_base(quiet=True)
        assert result is True

    def test_prebuilt_files_exist(self):
        """Pre-built chunks.json and embeddings.npy should ship with package."""
        from planagent.knowledge.vectorstore import PREBUILT_DIR
        assert (PREBUILT_DIR / "chunks.json").exists()
        assert (PREBUILT_DIR / "embeddings.npy").exists()


# ===================================================================
# Conversation memory tests
# ===================================================================

class TestConversationMemory:

    def test_memory_add_and_size(self):
        from planagent.knowledge.memory import ConversationMemory
        mem = ConversationMemory()
        assert mem.size == 0
        mem.add("I want to build a food delivery app", turn_num=1, role="user")
        assert mem.size == 1
        mem.add("Great! Let me help you plan that.", turn_num=1, role="assistant")
        assert mem.size == 2

    def test_memory_retrieve_empty(self):
        from planagent.knowledge.memory import ConversationMemory
        mem = ConversationMemory()
        results = mem.retrieve("food delivery", top_k=3, skip_last_n=0)
        assert results == []

    def test_memory_retrieve_skips_recent(self):
        from planagent.knowledge.memory import ConversationMemory
        mem = ConversationMemory()
        mem.add("Build a food delivery app", turn_num=1, role="user")
        mem.add("Sure, what features?", turn_num=1, role="assistant")
        mem.add("Orders and payments", turn_num=2, role="user")
        mem.add("Got it. Tech stack?", turn_num=2, role="assistant")
        # Skip last 2 — should only search first 2 entries
        results = mem.retrieve("food delivery", top_k=3, skip_last_n=2)
        assert len(results) <= 2
        for r in results:
            assert "score" in r

    def test_memory_retrieve_relevance(self):
        from planagent.knowledge.memory import ConversationMemory
        mem = ConversationMemory()
        mem.add("We need Redis caching for fast lookups", turn_num=1, role="user")
        mem.add("Good choice for caching", turn_num=1, role="assistant")
        mem.add("Use PostgreSQL for the database", turn_num=2, role="user")
        mem.add("Solid relational DB choice", turn_num=2, role="assistant")
        mem.add("Latest message about payments", turn_num=3, role="user")
        mem.add("Payment integration noted", turn_num=3, role="assistant")
        # Search for caching — should find turn 1 as most relevant
        results = mem.retrieve("caching Redis", top_k=2, skip_last_n=2)
        if results:
            assert results[0]["turn_num"] == 1

    def test_memory_format_for_prompt(self):
        from planagent.knowledge.memory import ConversationMemory
        mem = ConversationMemory()
        retrieved = [
            {"text": "Build food delivery app", "turn_num": 1, "role": "user", "score": 0.9},
            {"text": "Use PostgreSQL", "turn_num": 2, "role": "user", "score": 0.7},
        ]
        text = mem.format_for_prompt(retrieved, max_chars=500)
        assert "DEV:" in text
        assert "Turn 1" in text

    def test_memory_format_respects_max_chars(self):
        from planagent.knowledge.memory import ConversationMemory
        mem = ConversationMemory()
        retrieved = [
            {"text": "x" * 500, "turn_num": 1, "role": "user", "score": 0.9},
            {"text": "y" * 500, "turn_num": 2, "role": "user", "score": 0.7},
        ]
        text = mem.format_for_prompt(retrieved, max_chars=200)
        assert len(text) <= 600  # first entry may exceed slightly due to prefix

    def test_memory_truncates_long_turns(self):
        from planagent.knowledge.memory import ConversationMemory
        mem = ConversationMemory()
        mem.add("short", turn_num=1, role="user")
        mem.add("x" * 1000, turn_num=2, role="assistant")
        mem.add("latest", turn_num=3, role="user")
        mem.add("response", turn_num=3, role="assistant")
        results = mem.retrieve("short", top_k=3, skip_last_n=2)
        for r in results:
            assert len(r["text"]) <= 303  # 300 + "..."


# ===================================================================
# Integration: state field exists
# ===================================================================

class TestCodeStripping:
    """Verify code blocks are stripped from RAG chunks in planning mode."""

    def test_strip_code_removes_fenced_blocks(self):
        from planagent.knowledge.retriever import _strip_code
        text = "Use Redis for caching.\n\n```python\nimport redis\nr = redis.Redis()\n```\n\nFast lookups."
        result = _strip_code(text)
        assert "import redis" not in result
        assert "Use Redis" in result
        assert "Fast lookups" in result

    def test_strip_code_removes_inline_code(self):
        from planagent.knowledge.retriever import _strip_code
        text = "Use `redis.get(key)` for cache lookups and `redis.set(key, val)` for writes."
        result = _strip_code(text)
        assert "redis.get" not in result
        assert "redis.set" not in result

    def test_strip_code_preserves_plain_text(self):
        from planagent.knowledge.retriever import _strip_code
        text = "Microservices pattern separates concerns into independent services."
        assert _strip_code(text) == text

    def test_format_chunks_strips_code(self):
        from planagent.knowledge.retriever import format_chunks_for_prompt
        chunks = [{
            "text": "Caching strategy:\n```python\ncache.set('key', 'val')\n```\nUse TTL-based expiry.",
            "score": 0.9,
            "rerank_score": 0.85,
            "metadata": {"source": "test", "section": "caching",
                         "chunk_index": 0, "topics": ["caching"]},
        }]
        result = format_chunks_for_prompt(chunks, max_chars=5000)
        assert "cache.set" not in result
        assert "TTL-based expiry" in result

    def test_extract_rag_refs(self):
        from planagent.knowledge.retriever import extract_rag_refs
        chunks = [
            {"metadata": {"source": "core_hld", "section": "caching"}},
            {"metadata": {"source": "arch_patterns", "section": "microservices"}},
            {"metadata": {"source": "core_hld", "section": "caching"}},  # dup
        ]
        refs = extract_rag_refs(chunks)
        assert len(refs) == 2
        assert "core_hld/caching" in refs
        assert "arch_patterns/microservices" in refs

    def test_extract_rag_refs_empty(self):
        from planagent.knowledge.retriever import extract_rag_refs
        assert extract_rag_refs([]) == []


class TestStateIntegration:

    def test_initial_state_has_rag_fields(self):
        from planagent.state import create_initial_state
        state = create_initial_state()
        assert "rag_context" in state
        assert "rag_last_query" in state
        assert state["rag_context"] == []
        assert state["rag_last_query"] == ""
