"""Genesis - Main memory system orchestrator."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from memory.buffer import RawBuffer
from memory.episodes import EpisodeManager
from memory.embeddings import EmbeddingEngine
from memory.recall import MemoryRecall
from models.episode import Episode, EpisodeSummary


class Genesis:
    """
    GateClaw Genesis - The birth of persistent memory.

    This is the main interface for the memory system.
    Coordinates all layers: raw buffer, compression, embeddings.

    Usage:
        genesis = Genesis()

        # Start a conversation
        episode = genesis.begin_episode(user_id="romain")

        # Add messages
        episode = genesis.add_message(episode, "user", "Hey, can you help with debugging?")
        episode = genesis.add_message(episode, "assistant", "Sure, what's the issue?")

        # End and save
        genesis.remember(episode)

        # Later, recall
        results = genesis.recall("debugging session last week")
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        vectors_path: str = "vectors",
        buffer_size: int = 10,
    ):
        """
        Initialize Genesis memory system.

        Args:
            db_path: Path to SQLite database
            vectors_path: Path to vector storage
            buffer_size: Number of raw episodes to keep
        """
        self.db_path = Path(db_path)
        self.vectors_path = Path(vectors_path)

        # Ensure paths exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectors_path.mkdir(parents=True, exist_ok=True)

        # Initialize layers
        self.buffer = RawBuffer(db_path=str(self.db_path), buffer_size=buffer_size)
        self.episodes = EpisodeManager(db_path=str(self.db_path))
        self.embeddings = EmbeddingEngine(
            db_path=str(self.db_path), vectors_path=str(self.vectors_path)
        )

        # Main recall interface
        self.recall_engine = MemoryRecall(
            buffer=self.buffer, episodes=self.episodes, embeddings=self.embeddings
        )

        # Active episode tracking
        self._active_episodes: dict[str, Episode] = {}

    def begin_episode(self, user_id: str = "default") -> Episode:
        """
        Start a new conversation episode.

        Returns:
            Empty episode ready for messages
        """
        episode = Episode(user_id=user_id)
        self._active_episodes[episode.episode_id] = episode
        return episode

    def add_message(self, episode: Episode, role: str, content: str) -> Episode:
        """
        Add a message to an active episode.

        Args:
            episode: Active episode
            role: "user" or "assistant"
            content: Message content

        Returns:
            Updated episode
        """
        return self.buffer.add_message(episode, role, content)

    def end_episode(self, episode: Episode, auto_compress: bool = True) -> Episode:
        """
        End an episode and prepare for storage.

        Args:
            episode: Completed episode
            auto_compress: Whether to auto-generate summary

        Returns:
            Finalized episode
        """
        # Set duration
        if episode.created_at:
            episode.duration_seconds = (
                datetime.now() - episode.created_at
            ).total_seconds()

        # Auto-compress if requested
        if auto_compress and not episode.summary:
            episode.summary = self.episodes.compress_episode(episode)

        # Remove from active
        self._active_episodes.pop(episode.episode_id, None)

        return episode

    def remember(self, episode: Episode) -> bool:
        """
        Save episode to all memory layers.

        This is the key method - it persists the conversation
        across all layers for future recall.

        Args:
            episode: Completed episode

        Returns:
            True if saved successfully
        """
        return self.recall_engine.remember_this(episode, compress=True)

    def forget(self, episode_id: str) -> bool:
        """
        Remove episode from all memory.

        Args:
            episode_id: Episode to forget

        Returns:
            True if removed
        """
        return self.recall_engine.forget_episode(episode_id)

    def recall(
        self, query: str, time_window_days: Optional[int] = None, limit: int = 5
    ) -> dict:
        """
        Search memory for relevant conversations.

        Args:
            query: Natural language query
            time_window_days: Only search within X days
            limit: Maximum results

        Returns:
            {
                "query": str,
                "results": [...],
                "source": str,
                "count": int
            }
        """
        return self.recall_engine.recall(
            query=query, time_window_days=time_window_days, limit=limit
        )

    def recall_recent(self, days: int = 7, limit: int = 10) -> list[dict]:
        """Get recent conversations."""
        return self.recall_engine.recall_recent(days=days, limit=limit)

    def recall_topic(self, topic: str, limit: int = 5) -> list[dict]:
        """Find conversations about a topic."""
        return self.recall_engine.recall_topic(topic=topic, limit=limit)

    def get_context(self, query: str, max_tokens: int = 1000) -> str:
        """
        Get memory context for prompt injection.

        Use this to inject relevant past conversations
        into new LLM prompts.

        Args:
            query: Current conversation topic
            max_tokens: Rough token budget

        Returns:
            Formatted context string
        """
        return self.recall_engine.get_memory_context(query, max_tokens)

    def get_stats(self) -> dict:
        """Get memory system statistics."""
        return self.recall_engine.get_stats()

    def get_active_episode(self, episode_id: str) -> Optional[Episode]:
        """Get active episode by ID."""
        return self._active_episodes.get(episode_id)

    def list_all_episodes(self, limit: int = 20) -> list[dict]:
        """List all stored episodes (both raw and compressed)."""
        raw = self.buffer.get_recent_episodes(limit)

        results = []
        for ep in raw:
            compressed = self.episodes.get_compressed_episode(ep.episode_id)

            result = {
                "episode_id": ep.episode_id,
                "created_at": ep.created_at.isoformat(),
                "message_count": ep.message_count,
                "emotional_tone": ep.emotional_tone.value,
                "weight": ep.weight,
            }

            if compressed:
                result["title"] = compressed.summary.title
                result["topics"] = compressed.summary.topics
                result["has_embedding"] = bool(
                    self.embeddings.get_embedding(ep.episode_id)
                )
            else:
                result["title"] = "Conversation"
                result["topics"] = []
                result["has_embedding"] = False

            results.append(result)

        return results

    def close(self):
        """Clean up resources."""
        # Nothing to close for SQLite
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
