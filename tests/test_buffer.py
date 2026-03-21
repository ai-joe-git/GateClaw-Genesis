"""Tests for Layer 1 - Raw Buffer"""

import pytest
import tempfile
from pathlib import Path

from memory.buffer import RawBuffer
from models.episode import Episode, EmotionalTone


class TestRawBuffer:
    """Test suite for raw conversation buffer."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_memory.db"

    @pytest.fixture
    def buffer(self, temp_db):
        """Create buffer with temp database."""
        return RawBuffer(db_path=str(temp_db), buffer_size=5)

    @pytest.fixture
    def sample_episode(self):
        """Create sample episode for testing."""
        episode = Episode(user_id="test_user")
        episode = Episode(user_id="test_user")
        episode.raw_messages = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T12:00:00"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "timestamp": "2024-01-01T12:00:01",
            },
        ]
        episode.message_count = 2
        return episode

    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database."""
        buffer = RawBuffer(db_path=str(temp_db))
        assert Path(temp_db).exists()

    def test_add_message(self, buffer):
        """Test adding messages to episode."""
        episode = Episode(user_id="test")

        episode = buffer.add_message(episode, "user", "Hello")
        assert len(episode.raw_messages) == 1
        assert episode.message_count == 1

        episode = buffer.add_message(episode, "assistant", "Hi!")
        assert len(episode.raw_messages) == 2
        assert episode.message_count == 2

    def test_save_episode(self, buffer, sample_episode):
        """Test saving episode to buffer."""
        result = buffer.save_episode(sample_episode)
        assert result is True

        # Verify it's saved
        episodes = buffer.get_recent_episodes()
        assert len(episodes) == 1
        assert episodes[0].episode_id == sample_episode.episode_id

    def test_get_recent_episodes_ordered(self, buffer):
        """Test that recent episodes are ordered by time."""
        # Create multiple episodes
        for i in range(3):
            episode = Episode(user_id="test")
            episode.raw_messages = [{"role": "user", "content": f"Message {i}"}]
            episode.message_count = 1
            buffer.save_episode(episode)

        episodes = buffer.get_recent_episodes(limit=10)
        assert len(episodes) == 3
        # Most recent first
        assert episodes[0].created_at >= episodes[1].created_at

    def test_buffer_size_limit(self, temp_db):
        """Test that buffer pruning works."""
        buffer = RawBuffer(db_path=str(temp_db), buffer_size=3)

        # Add 5 episodes
        for i in range(5):
            episode = Episode(user_id="test")
            episode.raw_messages = [{"role": "user", "content": f"Message {i}"}]
            episode.message_count = 1
            buffer.save_episode(episode)

        # Should only have 3 (buffer_size)
        episodes = buffer.get_recent_episodes(limit=10)
        assert len(episodes) == 3

    def test_search_content(self, buffer):
        """Test content search in buffer."""
        episode = Episode(user_id="test")
        episode.raw_messages = [
            {"role": "user", "content": "I have a Python bug"},
            {"role": "assistant", "content": "Let's debug it"},
        ]
        episode.message_count = 2
        buffer.save_episode(episode)

        results = buffer.search_content("Python")
        assert len(results) == 1

        results = buffer.search_content("JavaScript")
        assert len(results) == 0

    def test_emotional_tone_inference(self, buffer):
        """Test that emotional tone is inferred from content."""
        episode = Episode(user_id="test")

        # Frustrated content
        episode = buffer.add_message(
            episode, "user", "This is so frustrating! I'm stuck!"
        )
        assert episode.emotional_tone == EmotionalTone.FRUSTRATED

    def test_weight_computation(self, buffer):
        """Test episode weight calculation."""
        episode = Episode(user_id="test")
        episode.emotional_tone = EmotionalTone.TRIUMPHANT
        episode.message_count = 10

        weight = episode.compute_weight()
        assert weight > 0.5
        assert weight <= 1.0

    def test_get_stats(self, buffer):
        """Test buffer statistics."""
        episode = Episode(user_id="test")
        episode.raw_messages = [{"role": "user", "content": "Test"}]
        episode.message_count = 1
        buffer.save_episode(episode)

        stats = buffer.get_stats()
        assert stats["total_episodes"] == 1
        assert stats["buffer_size"] == 5
        assert "average_weight" in stats
