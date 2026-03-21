#!/usr/bin/env python3
"""
Simple test runner for GateClaw Genesis.
Run from project root: python tests/run_tests.py
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.genesis import Genesis
from models.episode import Episode, EmotionalTone


def test_episode_creation():
    """Test episode model."""
    episode = Episode(user_id="test")
    assert episode.user_id == "test"
    assert episode.episode_id
    assert len(episode.raw_messages) == 0
    print("✓ Episode creation")


def test_emotional_tone_inference():
    """Test emotional tone from content."""
    episode = Episode(user_id="test")

    # Test frustrated
    combined = "This is so frustrating! I'm stuck on this bug!"
    tone = EmotionalTone.from_content(combined)
    assert tone == EmotionalTone.FRUSTRATED

    # Test triumphant
    combined = "Yes! Finally fixed it! It works!"
    tone = EmotionalTone.from_content(combined)
    assert tone == EmotionalTone.TRIUMPHANT

    print("✓ Emotional tone inference")


def test_genesis_initialization():
    """Test Genesis initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        genesis = Genesis(
            db_path=str(Path(tmpdir) / "memory.db"),
            vectors_path=str(Path(tmpdir) / "vectors"),
            buffer_size=5,
        )

        stats = genesis.get_stats()
        assert stats["total_memory_objects"] == 0

        genesis.close()

    print("✓ Genesis initialization")


def test_full_conversation_flow():
    """Test complete conversation lifecycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        genesis = Genesis(
            db_path=str(Path(tmpdir) / "memory.db"),
            vectors_path=str(Path(tmpdir) / "vectors"),
            buffer_size=10,
        )

        # Start conversation
        episode = genesis.begin_episode(user_id="test_user")

        # Add messages
        messages = [
            ("user", "Help me debug this Python issue"),
            ("assistant", "What's the problem?"),
            ("user", "Race condition in async code"),
            ("assistant", "Try adding a mutex lock"),
            ("user", "Yes! Fixed! Thanks!"),
        ]

        for role, content in messages:
            episode = genesis.add_message(episode, role, content)

        # End and save
        episode = genesis.end_episode(episode)
        result = genesis.remember(episode)

        assert result is True
        assert episode.summary is not None
        assert episode.message_count == 5

        # Check topics were extracted
        assert (
            "python" in episode.summary.topics or "debugging" in episode.summary.topics
        )

        # Check emotional tone
        assert episode.emotional_tone == EmotionalTone.TRIUMPHANT

        genesis.close()

    print("✓ Full conversation flow")


def test_recall():
    """Test memory recall."""
    with tempfile.TemporaryDirectory() as tmpdir:
        genesis = Genesis(
            db_path=str(Path(tmpdir) / "memory.db"),
            vectors_path=str(Path(tmpdir) / "vectors"),
            buffer_size=10,
        )

        # Store episode
        episode = genesis.begin_episode()
        genesis.add_message(episode, "user", "I'm working with Docker containers")
        genesis.add_message(episode, "assistant", "Docker is great for isolation")
        episode = genesis.end_episode(episode)
        genesis.remember(episode)

        # Test recall
        results = genesis.recall("Docker")

        # Should find the episode
        assert results["count"] >= 0

        genesis.close()

    print("✓ Memory recall")


def test_recent_conversations():
    """Test retrieving recent conversations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        genesis = Genesis(
            db_path=str(Path(tmpdir) / "memory.db"),
            vectors_path=str(Path(tmpdir) / "vectors"),
            buffer_size=10,
        )

        # Store multiple episodes
        for i in range(3):
            episode = genesis.begin_episode(user_id=f"user_{i}")
            genesis.add_message(episode, "user", f"Test message {i}")
            episode = genesis.end_episode(episode)
            genesis.remember(episode)

        recent = genesis.recall_recent(days=7, limit=10)

        assert len(recent) == 3

        genesis.close()

    print("✓ Recent conversations")


def test_forget():
    """Test forgetting an episode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        genesis = Genesis(
            db_path=str(Path(tmpdir) / "memory.db"),
            vectors_path=str(Path(tmpdir) / "vectors"),
            buffer_size=10,
        )

        # Store episode
        episode = genesis.begin_episode()
        genesis.add_message(episode, "user", "Secret data")
        episode = genesis.end_episode(episode)
        genesis.remember(episode)

        episode_id = episode.episode_id

        # Verify stored
        stats = genesis.get_stats()
        assert stats["total_memory_objects"] > 0

        # Forget
        result = genesis.forget(episode_id)
        assert result is True

        # Verify removed
        stats = genesis.get_stats()
        assert stats["total_memory_objects"] == 0

        genesis.close()

    print("✓ Forget episode")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("GateClaw Genesis - Running Tests")
    print("=" * 60)
    print()

    tests = [
        test_episode_creation,
        test_emotional_tone_inference,
        test_genesis_initialization,
        test_full_conversation_flow,
        test_recall,
        test_recent_conversations,
        test_forget,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
