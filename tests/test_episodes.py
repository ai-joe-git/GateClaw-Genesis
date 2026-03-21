"""Tests for Layer 2 - Episode Manager"""

import pytest
import tempfile
from pathlib import Path

from memory.episodes import EpisodeManager, CompressedEpisode
from memory.buffer import RawBuffer
from models.episode import Episode, EmotionalTone


class TestEpisodeManager:
    """Test suite for episode compression."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_memory.db"

    @pytest.fixture
    def manager(self, temp_db):
        """Create episode manager."""
        return EpisodeManager(db_path=str(temp_db))

    @pytest.fixture
    def sample_episode(self):
        """Create sample conversation episode."""
        episode = Episode(user_id="test_user")
        episode.raw_messages = [
            {"role": "user", "content": "I have a Python bug with async code"},
            {"role": "assistant", "content": "What's the issue?"},
            {"role": "user", "content": "Race condition in my API calls"},
            {"role": "assistant", "content": "Need a mutex lock?"},
            {"role": "user", "content": "Yes! Fixed it. Now it works perfectly!"},
        ]
        episode.message_count = 5
        episode.emotional_tone = EmotionalTone.TRIUMPHANT
        return episode

    def test_compress_episode(self, manager, sample_episode):
        """Test episode compression."""
        summary = manager.compress_episode(sample_episode)

        assert summary.title
        assert summary.summary
        assert "python" in summary.topics or "api" in summary.topics
        assert summary.emotional_tone == EmotionalTone.TRIUMPHANT

    def test_extract_topics(self, manager):
        """Test topic extraction."""
        content = "I'm working with Python and Docker to deploy my FastAPI service."
        topics = manager._extract_topics(content)

        assert "python" in topics
        assert "docker" in topics
        assert "api" in topics

    def test_extract_outcomes(self, manager):
        """Test outcome extraction."""
        messages = [
            {"role": "user", "content": "I fixed the bug"},
            {"role": "user", "content": "Deployed to production"},
            {"role": "assistant", "content": "Great! It works now."},
        ]

        outcomes = manager._extract_outcomes(messages)
        assert len(outcomes) > 0
        assert any("fixed" in o.lower() for o in outcomes)

    def test_extract_key_moments(self, manager):
        """Test key moment extraction."""
        messages = [
            {"role": "user", "content": "How do I solve this?"},
            {"role": "assistant", "content": "Try this code:\n```python\npass\n```"},
            {"role": "user", "content": "Finally! It works!"},
        ]

        moments = manager._extract_key_moments(messages)
        assert any("Code shared" in m for m in moments)

    def test_save_compressed_episode(self, manager, sample_episode):
        """Test saving compressed episode."""
        summary = manager.compress_episode(sample_episode)
        sample_episode.summary = summary

        result = manager.save_compressed_episode(sample_episode)
        assert result is True

        # Retrieve it
        retrieved = manager.get_compressed_episode(sample_episode.episode_id)
        assert retrieved is not None
        assert retrieved.title == summary.title

    def test_get_recent_summaries(self, manager):
        """Test retrieving recent summaries."""
        # Create and save multiple episodes
        for i in range(3):
            episode = Episode(user_id="test")
            episode.raw_messages = [{"role": "user", "content": f"Test {i}"}]
            episode.message_count = 1
            summary = manager.compress_episode(episode)
            episode.summary = summary
            manager.save_compressed_episode(episode)

        summaries = manager.get_recent_summaries(limit=10)
        assert len(summaries) == 3

    def test_title_extraction(self, manager):
        """Test title extraction from first message."""
        title = manager._extract_title("Hello there! I need help with Python.")
        assert len(title) <= 53
        assert "hello" in title.lower() or "help" in title.lower()

    def test_summary_generation(self, manager):
        """Test summary generation."""
        messages = [
            {"role": "user", "content": "I have a database issue"},
            {"role": "assistant", "content": "What kind of issue?"},
            {"role": "user", "content": "Connection timeout"},
            {"role": "assistant", "content": "Check your connection pool"},
        ]

        summary = manager._generate_summary(messages)
        assert "database" in summary.lower() or "issue" in summary.lower()
