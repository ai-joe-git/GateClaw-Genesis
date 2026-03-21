#!/usr/bin/env python3
"""
Demo script for GateClaw Genesis memory system.

Run with: python examples/demo.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.genesis import Genesis
from models.episode import Episode


def main():
    print("=" * 60)
    print("GateClaw Genesis - Episodic Memory Demo")
    print("=" * 60)
    print()

    # Initialize memory system
    print("Initializing Genesis...")
    genesis = Genesis(
        db_path="examples/memory.db", vectors_path="examples/vectors", buffer_size=10
    )

    # Show stats
    print("\nInitial stats:")
    stats = genesis.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Simulate a conversation
    print("\n" + "-" * 40)
    print("Simulating conversation...")
    print("-" * 40)

    episode = genesis.begin_episode(user_id="romain")

    conversations = [
        (
            "user",
            "Hey GateClaw, I'm working on a Python project and hitting a weird bug.",
        ),
        ("assistant", "Oh interesting, what kind of bug? I'm here to help debug."),
        ("user", "The API returns data but sometimes it's empty. It's intermittent."),
        ("assistant", "Intermittent issues are the worst. Race condition? Caching?"),
        (
            "user",
            "OH! You're right, it's probably a race condition with the async calls!",
        ),
        (
            "assistant",
            "Nice catch! Async race conditions are sneaky. Want to check the timing?",
        ),
        (
            "user",
            "Yes! Finally solved it. Added a mutex lock and now it's stable. Thanks!",
        ),
        ("assistant", "YES! That's satisfying when it clicks. Race condition solved."),
    ]

    for role, content in conversations:
        episode = genesis.add_message(episode, role, content)
        print(f"  [{role}]: {content[:50]}...")

    # End and save
    print("\nEnding episode...")
    episode = genesis.end_episode(episode)

    print("\nEpisode summary:")
    if episode.summary:
        print(f"  Title: {episode.summary.title}")
        print(f"  Topics: {', '.join(episode.summary.topics)}")
        print(f"  Emotional tone: {episode.summary.emotional_tone.value}")
        print(f"  Outcomes: {episode.summary.outcomes}")
        print(f"  Weight: {episode.weight:.2f}")

    # Save to memory
    print("\nSaving to memory...")
    genesis.remember(episode)

    # Show updated stats
    print("\nUpdated stats:")
    stats = genesis.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test recall
    print("\n" + "-" * 40)
    print("Testing recall...")
    print("-" * 40)

    # Recall by topic
    results = genesis.recall("race condition debugging", limit=3)

    print(f"\nQuery: 'race condition debugging'")
    print(f"Source: {results['source']}")
    print(f"Results: {results['count']}")

    for i, result in enumerate(results["results"], 1):
        print(f"\n  Result {i}:")
        print(f"    Episode ID: {result['episode_id']}")
        print(f"    Relevance: {result.get('relevance', 0):.2f}")
        print(f"    Preview: {result['preview'][:100]}...")

    # Test recent recall
    print("\n" + "-" * 40)
    print("Recent conversations:")
    print("-" * 40)

    recent = genesis.recall_recent(days=7, limit=5)

    for conv in recent:
        print(f"\n  [{conv['created_at'][:10]}] {conv['title']}")
        print(f"    Topics: {', '.join(conv['topics']) if conv['topics'] else 'none'}")
        print(f"    Tone: {conv['emotional_tone']}, Weight: {conv['weight']:.2f}")
        print(f"    Messages: {conv['message_count']}")

    # Test context generation
    print("\n" + "-" * 40)
    print("Memory context for prompt:")
    print("-" * 40)

    context = genesis.get_context("debugging Python issues", max_tokens=500)
    print(context)

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)

    genesis.close()


if __name__ == "__main__":
    main()
