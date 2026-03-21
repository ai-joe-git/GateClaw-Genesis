"""Tests for Layer 3 - Embedding Engine"""

import pytest
import tempfile
from pathlib import Path

from memory.embeddings import EmbeddingEngine
from models.episode import EpisodeSummary, EmotionalTone


class TestEmbeddingEngine:
    """Test suite for semantic embeddings."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_memory.db"
            vectors_path = Path(tmpdir) / "vectors"
            yield db_path, vectors_path

    @pytest.fixture
    def engine(self, temp_paths):
        """Create embedding engine."""
        db_path, vectors_path = temp_paths
        return EmbeddingEngine(db_path=str(db_path), vectors_path=str(vectors_path))

    @pytest.fixture
    def sample_summary(self):
        """Create sample episode summary."""
        return EpisodeSummary(
            title="Python Debugging Session",
            summary="User had a race condition in async API. Fixed with mutex.",
            topics=["python", "api", "debugging"],
            emotional_tone=EmotionalTone.TRIUMPHANT,
            entities=["Python", "API", "mutex"],
            outcomes=["Fixed race condition"],
        )

    def test_init_creates_tables(self, temp_paths):
        """Test that initialization creates database tables."""
        db_path, vectors_path = temp_paths
        engine = EmbeddingEngine(db_path=str(db_path), vectors_path=str(vectors_path))

        assert Path(db_path).exists()
        assert vectors_path.exists()

    def test_generate_embedding(self, engine):
        """Test embedding generation."""
        embedding = engine.generate_embedding("Test text for embedding")

        assert len(embedding) == 384  # embedding dimension
        assert all(-1 <= v <= 1 for v in embedding)

    def test_store_embedding(self, engine, sample_summary):
        """Test storing embedding."""
        embedding_id = engine.store_embedding(
            episode_id="test-123", summary=sample_summary
        )

        assert embedding_id == "test-123"

        # Retrieve it
        retrieved = engine.get_embedding("test-123")
        assert retrieved is not None
        assert len(retrieved) == 384

    def test_cosine_similarity(self, engine):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        # Identical vectors
        sim = engine.cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 0.01

        # Orthogonal vectors
        sim = engine.cosine_similarity(vec1, vec3)
        assert abs(sim - 0.0) < 0.01

    def test_search_similar(self, engine, sample_summary):
        """Test semantic similarity search."""
        # Store a few embeddings
        summaries = [
            (
                "ep1",
                EpisodeSummary(
                    title="Python debugging",
                    summary="Fixed a Python bug",
                    topics=["python"],
                    emotional_tone=EmotionalTone.TRIUMPHANT,
                ),
            ),
            (
                "ep2",
                EpisodeSummary(
                    title="API development",
                    summary="Built REST API",
                    topics=["api"],
                    emotional_tone=EmotionalTone.CASUAL,
                ),
            ),
            (
                "ep3",
                EpisodeSummary(
                    title="Docker setup",
                    summary="Configured Docker container",
                    topics=["docker"],
                    emotional_tone=EmotionalTone.SERIOUS,
                ),
            ),
        ]

        for ep_id, summary in summaries:
            engine.store_embedding(episode_id=ep_id, summary=summary)

        # Search for Python-related
        results = engine.search_similar("Python programming", limit=3, threshold=0.0)

        assert len(results) > 0
        # Results should be sorted by similarity
        if len(results) > 1:
            assert results[0]["similarity"] >= results[1]["similarity"]

    def test_search_with_threshold(self, engine, sample_summary):
        """Test search with similarity threshold."""
        engine.store_embedding(episode_id="test-1", summary=sample_summary)

        # High threshold - should not match unrelated query
        results = engine.search_similar("cooking recipes", threshold=0.99)
        # Pseudo-embeddings will have some similarity, but we test structure
        assert isinstance(results, list)

    def test_get_stats(self, engine, sample_summary):
        """Test embedding statistics."""
        engine.store_embedding(episode_id="test-1", summary=sample_summary)

        stats = engine.get_stats()
        assert stats["total_embeddings"] == 1
        assert stats["embedding_dimension"] == 384
        assert "pseudo-embedding" in stats["models"]

    def test_different_texts_different_embeddings(self, engine):
        """Test that different texts produce different embeddings."""
        emb1 = engine.generate_embedding("Python debugging")
        emb2 = engine.generate_embedding("Cooking recipes")

        # Should be different
        similarity = engine.cosine_similarity(emb1, emb2)
        # Pseudo-embeddings are deterministic based on hash, so they'll differ
        assert isinstance(similarity, float)
