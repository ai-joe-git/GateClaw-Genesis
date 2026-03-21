"""Memory recall - Orchestrates retrieval across all layers."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import sqlite3

from memory.buffer import RawBuffer
from memory.episodes import EpisodeManager
from memory.embeddings import EmbeddingEngine
from models.episode import Episode


class MemoryRecall:
    """
    Orchestrates memory retrieval across all layers.

    Query flow:
    1. Check raw buffer first (exact recent matches)
    2. Search semantic store (meaning-based matches)
    3. Combine and rank results

    This is the main interface for "remembering" conversations.
    """

    def __init__(
        self, buffer: RawBuffer, episodes: EpisodeManager, embeddings: EmbeddingEngine
    ):
        self.buffer = buffer
        self.episodes = episodes
        self.embeddings = embeddings

    def recall(
        self,
        query: str,
        time_window_days: Optional[int] = None,
        include_raw: bool = True,
        include_semantic: bool = True,
        limit: int = 5,
    ) -> dict:
        """
        Main recall method - find relevant conversations.

        Args:
            query: Natural language query ("what did we say about APIs?")
            time_window_days: Only search within X days
            include_raw: Search raw buffer
            include_semantic: Search semantic store
            limit: Maximum results

        Returns:
            {
                "query": str,
                "results": [...],
                "source": str,  # "raw", "semantic", or "combined"
                "count": int
            }
        """
        results = []
        source = "none"

        # Layer 1: Raw buffer (exact content matches, recent only)
        if include_raw:
            raw_results = self.buffer.search_content(query, limit)
            raw_filtered = self._apply_time_filter(raw_results, time_window_days)

            for episode in raw_filtered:
                results.append(
                    {
                        "episode_id": episode.episode_id,
                        "source": "raw",
                        "created_at": episode.created_at.isoformat(),
                        "weight": episode.weight,
                        "preview": self._get_preview(episode),
                        "relevance": 1.0,  # Exact match in raw
                    }
                )

            if results:
                source = "raw"

        # Layer 3: Semantic search (meaning-based matches)
        if include_semantic and len(results) < limit:
            semantic_results = self.embeddings.search_similar(query, limit)

            for result in semantic_results:
                # Get full episode details
                compressed = self.episodes.get_compressed_episode(result["episode_id"])
                if compressed:
                    # Check time filter
                    if time_window_days:
                        cutoff = datetime.now() - timedelta(days=time_window_days)
                        if compressed.created_at < cutoff:
                            continue

                    results.append(
                        {
                            "episode_id": result["episode_id"],
                            "source": "semantic",
                            "created_at": compressed.created_at.isoformat(),
                            "weight": compressed.weight,
                            "preview": compressed.summary.summary,
                            "relevance": result["similarity"],
                            "title": compressed.summary.title,
                            "topics": compressed.summary.topics,
                            "emotional_tone": compressed.summary.emotional_tone.value,
                        }
                    )

            if results:
                source = "combined" if source == "raw" else "semantic"

        # Sort by relevance, then weight
        results.sort(
            key=lambda x: (x.get("relevance", 0), x.get("weight", 0)), reverse=True
        )

        return {
            "query": query,
            "results": results[:limit],
            "source": source,
            "count": len(results[:limit]),
        }

    def recall_recent(self, days: int = 7, limit: int = 10) -> list[dict]:
        """
        Get recent conversations with summaries.

        Lightweight recall - just shows what we've talked about lately.
        """
        episodes = self.buffer.get_recent_episodes(limit)

        results = []
        for episode in episodes:
            # Get compressed summary if available
            compressed = self.episodes.get_compressed_episode(episode.episode_id)

            preview = ""
            title = "Conversation"
            topics = []

            if compressed:
                preview = compressed.summary.summary
                title = compressed.summary.title
                topics = compressed.summary.topics
            else:
                preview = self._get_preview(episode)

            results.append(
                {
                    "episode_id": episode.episode_id,
                    "created_at": episode.created_at.isoformat(),
                    "title": title,
                    "preview": preview[:200],
                    "topics": topics,
                    "message_count": episode.message_count,
                    "emotional_tone": episode.emotional_tone.value,
                    "weight": episode.weight,
                }
            )

        return results

    def recall_topic(self, topic: str, limit: int = 5) -> list[dict]:
        """Find conversations about a specific topic."""
        # Use semantic search
        return self.recall(
            query=topic,
            include_raw=False,  # Rely on semantic for topic search
            limit=limit,
        )["results"]

    def remember_this(self, episode: Episode, compress: bool = True) -> bool:
        """
        Save an episode to all memory layers.

        Args:
            episode: Completed conversation
            compress: Whether to compress and embed

        Returns:
            True if saved successfully
        """
        # Layer 1: Raw buffer
        self.buffer.save_episode(episode)

        if compress:
            # Layer 2: Compress
            summary = self.episodes.compress_episode(episode)
            episode.summary = summary

            # Save compressed version
            self.episodes.save_compressed_episode(episode)

            # Layer 3: Embed
            self.embeddings.store_embedding(
                episode_id=episode.episode_id, summary=summary
            )

        return True

    def forget_episode(self, episode_id: str) -> bool:
        """
        Remove episode from all layers.

        Args:
            episode_id: Episode to forget

        Returns:
            True if removed, False if not found
        """
        # Remove from all tables
        conn = sqlite3.connect(self.buffer.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "DELETE FROM raw_episodes WHERE episode_id = ?", (episode_id,)
            )
            cursor.execute(
                "DELETE FROM compressed_episodes WHERE episode_id = ?", (episode_id,)
            )
            cursor.execute(
                "DELETE FROM episode_embeddings WHERE episode_id = ?", (episode_id,)
            )

            conn.commit()
            return True

        except Exception as e:
            print(f"Error forgetting episode: {e}")
            return False
        finally:
            conn.close()

    def get_memory_context(self, query: str, max_tokens: int = 1000) -> str:
        """
        Generate context string for injection into prompts.

        This is how memory "leaks" into conversations.
        Returns a formatted string of relevant past conversations
        that fits within token budget.
        """
        recall_result = self.recall(query, limit=5)

        if not recall_result["results"]:
            return ""

        context_parts = ["[RELEVANT PAST CONVERSATIONS]\n"]
        current_length = len(context_parts[0])

        for result in recall_result["results"]:
            entry = f"- {result.get('title', 'Conversation')} "
            entry += f"({result['created_at'][:10]}): "
            entry += result["preview"][:150] + "...\n"

            # Rough token estimate (4 chars per token)
            if current_length + len(entry) > max_tokens * 4:
                break

            context_parts.append(entry)
            current_length += len(entry)

        context_parts.append("[END MEMORY]\n")

        return "".join(context_parts)

    def _apply_time_filter(
        self, episodes: list[Episode], days: Optional[int]
    ) -> list[Episode]:
        """Filter episodes by time window."""
        if days is None:
            return episodes

        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in episodes if e.created_at >= cutoff]

    def _get_preview(self, episode: Episode, max_length: int = 200) -> str:
        """Generate preview text from episode."""
        if episode.summary:
            return episode.summary.summary[:max_length]

        # Use first user message as preview
        for msg in episode.raw_messages:
            if msg["role"] == "user":
                return msg["content"][:max_length] + "..."

        return "Conversation..."

    def get_stats(self) -> dict:
        """Get memory system statistics."""
        buffer_stats = self.buffer.get_stats()
        embedding_stats = self.embeddings.get_stats()

        conn = sqlite3.connect(self.buffer.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM compressed_episodes")
        compressed_count = cursor.fetchone()[0]

        conn.close()

        return {
            "raw_buffer": buffer_stats,
            "compressed_episodes": compressed_count,
            "embeddings": embedding_stats,
            "total_memory_objects": (
                buffer_stats["total_episodes"]
                + compressed_count
                + embedding_stats["total_embeddings"]
            ),
        }
