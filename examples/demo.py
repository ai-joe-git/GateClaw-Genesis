#!/usr/bin/env python3
"""
Demo script for GateClaw Genesis memory system.

Run with: python examples/demo.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.genesis import Genesis
from models.episode import Episode, EmotionalFlag, FlagConfidence


def main():
    print("=" * 60)
    print("GateClaw Genesis - Full Memory System Demo")
    print("=" * 60)
    print()

    # Initialize memory system
    print("Initializing Genesis...")
    genesis = Genesis(
        db_path="examples/memory.db",
        vectors_path="examples/vectors",
        buffer_size=10,
    )

    # Show stats
    print("\nInitial stats:")
    stats = genesis.get_stats()
    for layer, data in stats.items():
        print(f"  {layer}: {data}")

    # ============================================================
    # Demo 1: Full conversation with emotional flags
    # ============================================================
    print("\n" + "=" * 60)
    print("Demo 1: Conversation with emotional flag detection")
    print("=" * 60)

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
            "Yes! I added a mutex lock and now it's stable. Finally! That did it!",
        ),
        ("assistant", "YES! That's satisfying when it clicks. Race condition solved."),
    ]

    for role, content in conversations:
        episode = genesis.add_message(episode, role, content)
        print(f"  [{role}]: {content[:50]}...")

    # End episode - this triggers auto-flag detection
    print("\nEnding episode (auto-detecting emotional flags)...")
    episode = genesis.end_episode(episode)

    print(
        f"  Detected flags: {[f'{f.value}({c.value})' for f, c in episode.emotional_state.flags]}"
    )
    print(f"  Resolved: {episode.emotional_state.resolved}")
    print(f"  Revisit count: {episode.emotional_state.revisit_count}")
    print(f"  Current weight: {episode.emotional_state.current_weight:.3f}")

    # Save to memory
    print("\nSaving to memory (Layer 1-4)...")
    genesis.remember(episode)

    # Wait for consolidation
    print("Waiting for consolidation to complete...")
    genesis.wait_consolidation(episode.episode_id, timeout=10.0)

    # ============================================================
    # Demo 2: Frustration episode (unresolved)
    # ============================================================
    print("\n" + "=" * 60)
    print("Demo 2: Unresolved frustration episode")
    print("=" * 60)

    episode2 = genesis.begin_episode(user_id="romain")

    conversations2 = [
        ("user", "I give up. Docker networking is completely broken for me."),
        ("assistant", "What's happening exactly?"),
        (
            "user",
            "Containers can't talk to each other. I've tried everything. This is so frustrating.",
        ),
        ("assistant", "Let's systematically debug this."),
        ("user", "I can't figure it out. My docker-compose.yml is a mess."),
    ]

    for role, content in conversations2:
        episode2 = genesis.add_message(episode2, role, content)

    episode2 = genesis.end_episode(episode2)
    print(
        f"  Detected flags: {[f'{f.value}({c.value})' for f, c in episode2.emotional_state.flags]}"
    )
    print(f"  Resolved: {episode2.emotional_state.resolved}")
    print(f"  Weight before decay: {episode2.emotional_state.current_weight:.3f}")

    genesis.remember(episode2)
    genesis.wait_consolidation(episode2.episode_id, timeout=10.0)

    # ============================================================
    # Demo 3: Decay test
    # ============================================================
    print("\n" + "=" * 60)
    print("Demo 3: Decay test (unresolved vs resolved)")
    print("=" * 60)

    # Simulate 30 days on unresolved frustration episode
    print("\nSimulating 30 days on unresolved frustration episode...")
    original_weight = episode2.emotional_state.current_weight

    # Manually set last_decay to 30 days ago
    episode2.emotional_state.last_decay = datetime.now() - timedelta(days=30)
    weight_after_30d = episode2.emotional_state.apply_decay()

    print(f"  Initial weight: {original_weight:.3f}")
    print(f"  Weight after 30 days (unresolved): {weight_after_30d:.3f}")
    print(f"  Unresolved floor: 0.7")
    assert weight_after_30d >= 0.7, (
        f"FAILED: Unresolved frustration should stay >= 0.7, got {weight_after_30d}"
    )
    print("  PASSED: Unresolved frustration stays above 0.7")

    # Now mark it resolved and simulate 30 more days
    print("\nMarking episode as resolved...")
    episode2.emotional_state.mark_resolved()
    episode2.emotional_state.last_decay = datetime.now() - timedelta(days=30)
    weight_after_resolved_30d = episode2.emotional_state.apply_decay()

    print(f"  Weight after resolution + 30 days: {weight_after_resolved_30d:.3f}")
    print(f"  Expected: < 0.5 (resolved decays faster)")
    assert weight_after_resolved_30d < 0.5, (
        f"FAILED: Resolved should decay below 0.5, got {weight_after_resolved_30d}"
    )
    print("  PASSED: Resolved episodes decay below 0.5")

    # ============================================================
    # Demo 4: Recall
    # ============================================================
    print("\n" + "=" * 60)
    print("Demo 4: Memory recall")
    print("=" * 60)

    # Semantic recall
    print("\nQuery: 'race condition debugging'")
    results = genesis.recall("race condition debugging", limit=3)
    print(f"  Source: {results['source']}")
    print(f"  Results: {results['count']}")
    for i, result in enumerate(results["results"], 1):
        print(
            f"  {i}. {result.get('title', 'Conversation')} (relevance: {result.get('relevance', 0):.2f})"
        )

    # Recall with unresolved
    print("\nQuery: 'docker' (with include_unresolved=True)")
    results = genesis.recall("docker", limit=3, include_unresolved=True)
    if results.get("unresolved"):
        print(f"  Unresolved episodes:")
        for u in results["unresolved"]:
            print(f"    - {u['value']} (weight: {u['weight']:.2f})")

    # Persistent facts
    print("\nPersistent facts:")
    facts = genesis.recall_facts(limit=5)
    for fact in facts:
        print(
            f"  - [{fact['topic']}] {fact['value']} (reinforced {fact['reinforcement_count']}x, weight: {fact['current_weight']:.2f})"
        )

    # ============================================================
    # Demo 5: Memory block generation
    # ============================================================
    print("\n" + "=" * 60)
    print("Demo 5: Memory block generation")
    print("=" * 60)

    print("\nStandard variant:")
    block_standard = genesis.get_memory_block(variant="standard")
    print(block_standard[:500] + "...")

    print("\nTelegram variant (short, natural, TTS-friendly):")
    block_telegram = genesis.get_memory_block(variant="telegram")
    print(block_telegram)

    # ============================================================
    # Final stats
    # ============================================================
    print("\n" + "=" * 60)
    print("Final memory system stats")
    print("=" * 60)

    stats = genesis.get_stats()
    for layer, data in stats.items():
        print(f"\n{layer}:")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {data}")

    genesis.close()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
