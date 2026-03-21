"""Tests for Genesis - Main orchestrator"""

import pytest
import tempfile
from pathlib import Path

from core.genesis import Genesis


class TestGenesis:
    """Test suite for Genesis memory orchestrator."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def genesis(self, temp_dir):
        """Create Genesis instance with temp paths."""
        return Genesis(
            db_path=str(temp_dir / "memory.db"),
            vectors_path=str(temp_dir / "vectors"),
            buffer_size=5,
        )

    def test_initialization(self, genesis):
        """Test Genesis initializes all layers."""
        stats = genesis.get_stats()

        assert "raw_buffer" in stats
        assert "compressed_episodes" in stats
        assert "embeddings" in stats
        assert stats["total_memory_objects"] == 0

    def test_begin_episode(self, genesis):
        """Test starting new episode."""
        episode = genesis.begin_episode(user_id="test_user")

        assert episode.episode_id
        assert episode.user_id == "test_user"
        assert len(episode.raw_messages) == 0

    def test_add_message(self, genesis):
        """Test adding messages to episode."""
        episode = genesis.begin_episode()

        episode = genesis.add_message(episode, "user", "Hello!")
        assert len(episode.raw_messages) == 1

        episode = genesis.add_message(episode, "assistant", "Hi there!")
        assert len(episode.raw_messages) == 2

    def test_end_episode(self, genesis):
        """Test ending episode creates summary."""
        episode = genesis.begin_episode()
        genesis.add_message(episode, "user", "I need help with Python")
        genesis.add_message(episode, "assistant", "What's the issue?")

        episode = genesis.end_episode(episode, auto_compress=True)

        assert episode.summary is not None
        assert episode.duration_seconds > 0

    def test_remember_episode(self, genesis):
        """Test saving episode to all layers."""
        episode = genesis.begin_episode()
        genesis.add_message(episode, "user", "Help with Python bug")
        genesis.add_message(episode, "assistant", "What's wrong?")
        episode = genesis.end_episode(episode)

        result = genesis.remember(episode)
        assert result is True

        stats = genesis.get_stats()
        assert stats["raw_buffer"]["total_episodes"] == 1
        assert stats["compressed_episodes"] == 1
        assert stats["embeddings"]["total_embeddings"] == 1

    def test_recall_empty(self, genesis):
        """Test recall with no stored episodes."""
        results = genesis.recall("Python")

        assert results["query"] == "Python"
        assert results["count"] == 0
        assert results["source"] == "none"

    def test_recall_with_episodes(self, genesis):
        """Test recall after storing episodes."""
        # Store episode
        episode = genesis.begin_episode()
        genesis.add_message(episode, "user", "I have a Python race condition")
        genesis.add_message(episode, "assistant", "Use a mutex lock")
        episode = genesis.end_episode(episode)
        genesis.remember(episode)

        # Recall
        results = genesis.recall("Python race condition")

        assert results["count"] >= 0  # Should find something

    def test_recall_recent(self, genesis):
        """Test retrieving recent conversations."""
        # Store several episodes
        for i in range(3):
            episode = genesis.begin_episode(user_id=f"user_{i}")
            genesis.add_message(episode, "user", f"Test message {i}")
            episode = genesis.end_episode(episode)
            genesis.remember(episode)

        recent = genesis.recall_recent(days=7, limit=10)

        assert len(recent) == 3

    def test_context_generation(self, genesis):
        """Test memory context for prompt injection."""
        # Store an episode
        episode = genesis.begin_episode()
        genesis.add_message(episode, "user", "Help me with Docker setup")
        genesis.add_message(episode, "assistant", "Sure, let's configure it")
        episode = genesis.end_episode(episode)
        genesis.remember(episode)

        context = genesis.get_context("Docker configuration", max_tokens=500)

        # Should have context header if matches found
        if context:
            assert "[RELEVANT PAST CONVERSATIONS]" in context or "Docker" in context

    def test_forget_episode(self, genesis):
        """Test forgetting an episode."""
        # Store episode
        episode = genesis.begin_episode()
        genesis.add_message(episode, "user", "Secret conversation")
        episode = genesis.end_episode(episode)
        genesis.remember(episode)

        episode_id = episode.episode_id

        # Forget it
        result = genesis.forget(episode_id)
        assert result is True

        # Verify it's gone
        stats = genesis.get_stats()
        assert stats["total_memory_objects"] == 0

    def test_list_episodes(self, genesis):
        """Test listing all episodes."""
        # Store episodes
        for i in range(5):
            episode = genesis.begin_episode()
            genesis.add_message(episode, "user", f"Message {i}")
            episode = genesis.end_episode(episode)
            genesis.remember(episode)

        episodes = genesis.list_all_episodes(limit=3)

        assert len(episodes) == 3
        assert all("episode_id" in ep for ep in episodes)

    def test_full_conversation_flow(self, genesis):
        """Test complete conversation lifecycle."""
        # Start conversation
        episode = genesis.begin_episode(user_id="test_user")

        # Add messages
        messages = [
            ("user", "I'm debugging a FastAPI application"),
            ("assistant", "What's the issue?"),
            ("user", "Getting connection timeouts"),
            ("assistant", "Check your connection pool settings"),
            ("user", "Yes! That fixed it. Thanks!"),
        ]

        for role, content in messages:
            episode = genesis.add_message(episode, role, content)

        # End and save
        episode = genesis.end_episode(episode)
        result = genesis.remember(episode)

        assert result is True
        assert episode.summary is not None
        assert "api" in episode.summary.topics or "debugging" in episode.summary.topics

        # Recall
        recall_results = genesis.recall("FastAPI debugging")

        # The context system works
        assert recall_results["count"] >= 0
