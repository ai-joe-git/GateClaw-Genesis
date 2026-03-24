"""Genesis - Main memory system orchestrator for GateClaw."""

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from memory.buffer import RawBuffer
from memory.episodes import EpisodeManager
from memory.embeddings import EmbeddingEngine
from memory.recall import MemoryRecall
from memory.consolidation import ConsolidationEngine
from models.episode import Episode, EpisodeSummary


class Genesis:
    """
    GateClaw Genesis - The birth of persistent memory.

    This is the main interface for the memory system.
    Coordinates all 4 layers:
    - Layer 1: Raw buffer (lossless conversation storage)
    - Layer 2: Compression (summarization, topic extraction)
    - Layer 3: Semantic search (vector embeddings via Ollama)
    - Layer 4: Consolidation (fact extraction, relationship graph, decay)

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

        # Get context for prompt injection
        context = genesis.get_context("I need help with an API issue")
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        vectors_path: str = "vectors",
        buffer_size: int = 10,
        embedding_model: str = "qwen3-embedding:0.6b",
        llm_model: str = "qwen3.5:2b",
    ):
        """
        Initialize Genesis memory system.

        Args:
            db_path: Path to SQLite database
            vectors_path: Path to vector storage
            buffer_size: Number of raw episodes to keep
            embedding_model: Ollama model for embeddings
            llm_model: Ollama model for LLM-based fact extraction
        """
        self.db_path = Path(db_path)
        self.vectors_path = Path(vectors_path)
        self.llm_model = llm_model

        # Ensure paths exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectors_path.mkdir(parents=True, exist_ok=True)

        # Initialize layers
        self.buffer = RawBuffer(db_path=str(self.db_path), buffer_size=buffer_size)
        self.episodes = EpisodeManager(db_path=str(self.db_path))
        self.embeddings = EmbeddingEngine(
            db_path=str(self.db_path),
            vectors_path=str(self.vectors_path),
            embedding_model=embedding_model,
        )

        # Layer 4: Consolidation
        self.consolidation = ConsolidationEngine(
            db_path=str(self.db_path),
            llm_model=llm_model,
        )

        # Main recall interface
        self.recall_engine = MemoryRecall(
            buffer=self.buffer,
            episodes=self.episodes,
            embeddings=self.embeddings,
        )

        # Async task tracking
        self._consolidation_tasks: dict[str, asyncio.Task] = {}

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

    def end_episode(
        self,
        episode: Episode,
        auto_compress: bool = True,
        auto_detect_flags: bool = True,
    ) -> Episode:
        """
        End an episode and prepare for storage.

        Args:
            episode: Completed episode
            auto_compress: Whether to auto-generate summary
            auto_detect_flags: Whether to auto-detect emotional flags

        Returns:
            Finalized episode
        """
        # Set duration
        if episode.created_at:
            episode.duration_seconds = (
                datetime.now() - episode.created_at
            ).total_seconds()

        # Auto-detect emotional flags from content
        if auto_detect_flags:
            episode.detect_flags_from_content()

        # Compute weight from emotional state
        episode.compute_weight()

        # Auto-compress if requested
        if auto_compress and not episode.summary:
            episode.summary = self.episodes.compress_episode(episode)

        # Remove from active
        self._active_episodes.pop(episode.episode_id, None)

        return episode

    def remember(self, episode: Episode, consolidate: bool = True) -> bool:
        """
        Save episode to all memory layers.

        This is the key method - it persists the conversation
        across all layers for future recall.

        Args:
            episode: Completed episode
            consolidate: Whether to run Layer 4 consolidation (async, non-blocking)

        Returns:
            True if saved successfully
        """
        # Layer 1 + 2 + 3: Save to buffer, compress, embed
        result = self.recall_engine.remember_this(episode, compress=True)

        # Layer 4: Consolidation runs async to not block
        if consolidate:
            self._schedule_consolidation(episode)

        return result

    def _schedule_consolidation(self, episode: Episode):
        """
        Schedule Layer 4 consolidation as a non-blocking async task.

        Consolidation (fact extraction) runs after the session ends,
        routed to a small local model. This ensures zero latency
        impact on the user-facing conversation flow.
        """

        async def _consolidate():
            try:
                facts = self.consolidation.consolidate_episode(
                    episode,
                    llm_extract=True,
                )
                return facts
            except Exception as e:
                print(f"Consolidation error for episode {episode.episode_id}: {e}")
                return []

        try:
            # In async context: fire and forget via create_task
            loop = asyncio.get_running_loop()
            task = loop.create_task(_consolidate())
            self._consolidation_tasks[episode.episode_id] = task
        except RuntimeError:
            # No running loop: use asyncio.run() for proper async execution
            asyncio.run(_consolidate())

    def wait_consolidation(self, episode_id: str, timeout: float = 5.0) -> bool:
        """
        Wait for consolidation to complete for a specific episode.

        Args:
            episode_id: Episode to wait for
            timeout: Max seconds to wait

        Returns:
            True if consolidation completed, False if timeout
        """
        task = self._consolidation_tasks.get(episode_id)
        if not task:
            return True

        try:
            # Check if already done
            if task.done():
                return True
            # Use a simple polling approach
            import time

            start = time.time()
            while time.time() - start < timeout:
                if task.done():
                    return True
                time.sleep(0.1)
            return False
        except Exception:
            return True

        try:
            # Check if already done
            if task.done():
                return True
            # Task is running, wait for it
            import concurrent.futures

            with concurrent.futures.TimeoutException:
                task.result(timeout=timeout)
            return True
        except concurrent.futures.TimeoutException:
            return False
        except Exception:
            return True

    def forget(self, episode_id: str) -> bool:
        """
        Remove episode from all memory layers.

        Args:
            episode_id: Episode to forget

        Returns:
            True if removed
        """
        return self.recall_engine.forget_episode(episode_id)

    def recall(
        self,
        query: str,
        time_window_days: Optional[int] = None,
        limit: int = 5,
        include_unresolved: bool = False,
    ) -> dict:
        """
        Search memory for relevant conversations.

        Args:
            query: Natural language query
            time_window_days: Only search within X days
            limit: Maximum results
            include_unresolved: Also surface unresolved high-weight episodes

        Returns:
            {
                "query": str,
                "results": [...],
                "source": str,
                "count": int,
                "unresolved": [...]  # If include_unresolved=True
            }
        """
        result = self.recall_engine.recall(
            query=query,
            time_window_days=time_window_days,
            limit=limit,
        )

        # Layer 4: Surface unresolved high-weight episodes
        if include_unresolved:
            unresolved = self.consolidation.get_unresolved(min_weight=0.7, limit=3)
            result["unresolved"] = unresolved

        return result

    def recall_recent(self, days: int = 7, limit: int = 10) -> list[dict]:
        """Get recent conversations."""
        return self.recall_engine.recall_recent(days=days, limit=limit)

    def recall_topic(self, topic: str, limit: int = 5) -> list[dict]:
        """Find conversations about a topic."""
        return self.recall_engine.recall_topic(topic=topic, limit=limit)

    def recall_facts(self, topic: Optional[str] = None, limit: int = 10) -> list[dict]:
        """
        Recall persistent facts from Layer 4.

        Args:
            topic: Filter by topic (None = all topics)
            limit: Maximum facts to return

        Returns:
            List of PersistentFact dicts
        """
        facts = self.consolidation.get_facts(topic=topic, limit=limit)
        return [f.to_dict() for f in facts]

    def get_context(
        self,
        query: str,
        max_tokens: int = 1000,
        include_unresolved: bool = True,
        include_facts: bool = True,
    ) -> str:
        """
        Get memory context for prompt injection.

        This is how Genesis "leaks" into conversations. Generates a context
        string with:
        - Relevant past episodes (semantic search)
        - Unresolved issues (inject_unresolved behavior)
        - Relevant persistent facts

        Args:
            query: Current conversation topic
            max_tokens: Rough token budget
            include_unresolved: Include unresolved high-weight episodes
            include_facts: Include persistent facts

        Returns:
            Formatted context string for LLM injection
        """
        context_parts = ["[MEMORY CONTEXT]\n"]

        # Layer 4: Persistent facts — most reliable, surface first
        if include_facts:
            facts = self.consolidation.get_facts(topic=None, limit=8)
            if facts:
                context_parts.append("[PERSISTENT FACTS]\n")
                for fact in facts[:8]:
                    context_parts.append(
                        f"- {fact.value} (reinforced {fact.reinforcement_count}x)\n"
                    )
                context_parts.append("\n")

        # Layer 3: Semantic recall — episode titles
        recall_result = self.recall(query, limit=3, include_unresolved=False)

        if recall_result["results"]:
            context_parts.append("[RELEVANT CONVERSATIONS]\n")
            for result in recall_result["results"][:3]:
                title = result.get("title", "Conversation")
                if title == "Conversation" or not title:
                    continue
                entry = f"- {title} "
                entry += f"({result['created_at'][:10]}): "
                entry += result["preview"][:150] + "...\n"
                context_parts.append(entry)
            context_parts.append("\n")

        # Layer 4: Unresolved
        if include_unresolved:
            unresolved = self.consolidation.get_unresolved(min_weight=0.7, limit=2)
            if unresolved:
                context_parts.append("[UNRESOLVED - may be relevant]\n")
                for u in unresolved:
                    context_parts.append(
                        f"- {u['value']} (weight: {u['weight']:.2f})\n"
                    )
                context_parts.append("\n")

        context_parts.append("[END MEMORY]\n")
        return "".join(context_parts)

    def get_memory_block(self, variant: str = "standard") -> str:
        """
        Generate memory block for GateClaw agent injection.

        Two variants:
        - "standard": Full markdown, rich fact list
        - "telegram": 2-3 natural sentences, no markdown, TTS-friendly

        Args:
            variant: "standard" or "telegram"

        Returns:
            Formatted memory block string
        """
        if variant == "telegram":
            # Short, natural, spoken — max 3 sentences, no markdown, TTS-friendly
            # Get facts sorted by reinforcement (most referenced = most important)
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT fact_key, value, topic, reinforcement_count
                   FROM persistent_facts
                   ORDER BY reinforcement_count DESC, current_weight DESC
                   LIMIT 10"""
            ).fetchall()
            conn.close()

            if not rows:
                return "You don't have any specific memory of Romain yet."

            # Build natural sentences from fact VALUES only (fact_key is internal)
            # Group by topic using flexible matching
            project_facts = []
            problem_facts = []
            tech_facts = []
            other_facts = []

            for fact_key, value, topic, count in rows:
                if not value or len(value) < 10:
                    continue
                t = (topic or "").lower()
                # Flexible topic matching
                if any(x in t for x in ["project", "software", "gateclaw", "bot"]):
                    project_facts.append((value, count))
                elif any(
                    x in t
                    for x in ["problem", "bug", "issue", "favorite", "unresolved"]
                ):
                    problem_facts.append((value, count))
                elif any(
                    x in t
                    for x in [
                        "docker",
                        "config",
                        "technical",
                        "setup",
                        "tui",
                        "interface",
                        "python",
                    ]
                ):
                    tech_facts.append((value, count))
                else:
                    other_facts.append((value, count))

            parts = []

            # Project context — highest reinforced first
            project_facts.sort(key=lambda x: -x[1])
            if project_facts:
                parts.append(f"Romain is building {project_facts[0][0][:100]}.")

            # Known issues / bugs — specific and actionable
            problem_facts.sort(key=lambda x: -x[1])
            if problem_facts:
                prob = problem_facts[0][0]
                if len(prob) > 10:
                    parts.append(f"Known issue: {prob[:120]}.")

            # Technical setup — only if space allows
            if tech_facts and len(parts) < 3:
                tech_facts.sort(key=lambda x: -x[1])
                tech = tech_facts[0][0]
                if len(tech) > 10:
                    parts.append(f"Technical: {tech[:100]}.")

            if not parts:
                return "You don't have any specific memory of Romain yet."

            # Limit to 2 sentences max for telegram
            result = " ".join(parts[:2])
            return result

        # Standard: full markdown
        return self.get_context(query="", max_tokens=2000, include_unresolved=True)

    def get_stats(self) -> dict:
        """Get memory system statistics across all layers."""
        import sqlite3

        buffer_stats = self.recall_engine.buffer.get_stats()
        embedding_stats = self.embeddings.get_stats()
        consolidation_stats = self.consolidation.get_stats()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM compressed_episodes")
        compressed_count = cursor.fetchone()[0]
        conn.close()

        return {
            "layer1_raw_buffer": buffer_stats,
            "layer2_compressed_episodes": compressed_count,
            "layer3_embeddings": embedding_stats,
            "layer4_consolidation": consolidation_stats,
        }

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
                "consolidated": ep.consolidated,
            }

            if compressed:
                result["title"] = compressed.summary.title
                result["topics"] = compressed.summary.topics
                result["has_embedding"] = bool(
                    self.embeddings.get_embedding(ep.episode_id)
                )
                # Include emotional flags
                if ep.emotional_state.flags:
                    result["emotional_flags"] = [
                        f"{f.value}({c.value})" for f, c in ep.emotional_state.flags
                    ]
            else:
                result["title"] = "Conversation"
                result["topics"] = []
                result["has_embedding"] = False

            results.append(result)

        return results

    def close(self):
        """Clean up resources."""
        # Wait for consolidation tasks
        for episode_id, task in self._consolidation_tasks.items():
            if not task.done():
                try:
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(asyncio.wait([task], timeout=2.0))
                except:
                    pass
        self._consolidation_tasks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
