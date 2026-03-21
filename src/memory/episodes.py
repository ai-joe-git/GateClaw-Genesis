"""Layer 2: Episode manager for conversation compression and summarization."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import re

from models.episode import Episode, EpisodeSummary, EmotionalTone


# Keyword-based topic extraction (placeholder for more sophisticated NLP)
TOPIC_KEYWORDS = {
    "python": ["python", "py", "pip", "venv", "django", "flask", "fastapi"],
    "javascript": ["javascript", "js", "node", "npm", "react", "vue", "typescript"],
    "docker": ["docker", "container", "compose", "kubernetes", "k8s"],
    "database": ["database", "sql", "postgres", "mysql", "mongodb", "redis"],
    "api": ["api", "endpoint", "rest", "graphql", "request", "response"],
    "debugging": ["debug", "error", "bug", "fix", "issue", "problem", "stack trace"],
    "testing": ["test", "pytest", "jest", "unit", "integration", "coverage"],
    "git": ["git", "branch", "commit", "merge", "pull request", "conflict"],
    "deployment": ["deploy", "production", "staging", "ci/cd", "pipeline"],
    "architecture": ["architecture", "design", "pattern", "structure", "refactor"],
}


@dataclass
class CompressedEpisode:
    """A compressed episode ready for embedding storage."""

    episode_id: str
    summary: EpisodeSummary
    created_at: datetime
    weight: float
    user_id: str


class EpisodeManager:
    """
    Episode compression and summarization.

    Purpose: Transform raw conversations into meaningful, searchable summaries.
    This is the bridge between raw storage and semantic understanding.

    Compression ratio: ~10x (from raw text to structured summary)
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Initialize SQLite for compressed episodes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compressed_episodes (
                episode_id TEXT PRIMARY KEY,
                created_at TIMESTAMP,
                user_id TEXT,
                summary_json TEXT NOT NULL,
                weight REAL,
                embedding_id TEXT  -- Links to vector store
            )
        """)

        conn.commit()
        conn.close()

    def compress_episode(self, episode: Episode) -> EpisodeSummary:
        """
        Compress a raw episode into a structured summary.

        This is where the magic happens:
        1. Extract title from first meaningful exchange
        2. Summarize the conversation
        3. Extract topics using keyword matching
        4. Identify outcomes (solutions, decisions)
        5. Mark key moments

        TODO: Replace with LLM-based summarization for better quality.
        """
        messages = episode.raw_messages

        # Extract title from first user message (first 50 chars)
        first_user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), "Conversation"
        )
        title = self._extract_title(first_user_msg)

        # Generate summary (placeholder: key messages extraction)
        summary = self._generate_summary(messages)

        # Extract topics
        all_content = " ".join([m["content"] for m in messages])
        topics = self._extract_topics(all_content)

        # Extract outcomes
        outcomes = self._extract_outcomes(messages)

        # Extract entities (technical terms, names)
        entities = self._extract_entities(all_content)

        # Key moments (questions asked, decisions made)
        key_moments = self._extract_key_moments(messages)

        return EpisodeSummary(
            title=title,
            summary=summary,
            topics=topics,
            outcomes=outcomes,
            emotional_tone=episode.emotional_tone,
            key_moments=key_moments,
            entities=entities,
        )

    def _extract_title(self, first_message: str) -> str:
        """Extract a brief title from first message."""
        # Clean and truncate
        title = first_message.strip()
        title = re.sub(r"[^\w\s-]", "", title)

        # Take first 50 chars or until first sentence
        if "." in title:
            title = title.split(".")[0]

        if len(title) > 50:
            title = title[:47] + "..."

        return title.title()

    def _generate_summary(self, messages: list[dict]) -> str:
        """
        Generate a 2-3 sentence summary.

        Placeholder implementation - extracts key messages.
        TODO: Use LLM for proper summarization.
        """
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]

        # For now, just combine first user and first assistant message
        if user_msgs and assistant_msgs:
            summary = f"User asked about: {user_msgs[0][:100]}... "
            summary += f"Discussion covered: {', '.join(user_msgs[1:3])[:50] if len(user_msgs) > 1 else 'various topics'}."
            return summary

        return "Conversation about various topics."

    def _extract_topics(self, content: str) -> list[str]:
        """Extract topics using keyword matching."""
        content_lower = content.lower()
        topics = []

        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                topics.append(topic)

        return topics[:5]  # Max 5 topics

    def _extract_outcomes(self, messages: list[dict]) -> list[str]:
        """
        Extract outcomes (things achieved/solved).

        Looks for:
        - "Fixed", "Solved", "Works now"
        - "Deployed", "Completed"
        - Decisions made
        """
        outcomes = []
        outcome_patterns = [
            r"fixed[:\s]+([^.]+)",
            r"solved[:\s]+([^.]+)",
            r"works now",
            r"deployed[:\s]+([^.]+)",
            r"completed[:\s]+([^.]+)",
        ]

        for msg in messages:
            content = msg["content"].lower()
            for pattern in outcome_patterns:
                match = re.search(pattern, content)
                if match:
                    outcome = match.group(1) if match.groups() else match.group(0)
                    outcomes.append(outcome.strip())

        return outcomes[:3]  # Max 3 outcomes

    def _extract_entities(self, content: str) -> list[str]:
        """
        Extract named entities (technologies, names, etc).

        Placeholder - uses simple capitalization heuristics.
        TODO: Use spaCy or similar for proper NER.
        """
        # Simple: find capitalized words (not at sentence start)
        words = content.split()
        entities = []

        for i, word in enumerate(words):
            if word[0].isupper() and i > 0:
                # Not after punctuation (likely sentence start)
                prev = words[i - 1]
                if prev not in [".", "!", "?", ":"]:
                    clean = re.sub(r"[^\w]", "", word)
                    if len(clean) > 2:
                        entities.append(clean)

        # Deduplicate and limit
        entities = list(set(entities))[:10]
        return entities

    def _extract_key_moments(self, messages: list[dict]) -> list[str]:
        """
        Extract key conversation moments.

        Looks for:
        - Questions asked
        - Code shared
        - Decisions made
        - Breakthroughs (Yes!, Finally, etc.)
        """
        moments = []

        for msg in messages:
            content = msg["content"]

            # Questions
            if "?" in content and msg["role"] == "user":
                questions = re.findall(r"([^.?]+\?)", content)
                moments.extend([f"Asked: {q.strip()}" for q in questions[:2]])

            # Code blocks
            if "```" in content:
                moments.append("Code shared")

            # Breakthroughs
            breakthroughs = ["yes!", "finally", "works", "solved", "got it"]
            if any(b in content.lower() for b in breakthroughs):
                moments.append("Breakthrough moment")

        return moments[:5]

    def save_compressed_episode(
        self, episode: Episode, embedding_id: Optional[str] = None
    ) -> bool:
        """Save compressed episode to database."""
        if not episode.summary:
            episode.summary = self.compress_episode(episode)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO compressed_episodes
                (episode_id, created_at, user_id, summary_json, weight, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    episode.episode_id,
                    episode.created_at.isoformat(),
                    episode.user_id,
                    json.dumps(episode.summary.to_dict()),
                    episode.weight,
                    embedding_id,
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            print(f"Error saving compressed episode: {e}")
            return False
        finally:
            conn.close()

    def get_compressed_episode(self, episode_id: str) -> Optional[CompressedEpisode]:
        """Retrieve compressed episode by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT episode_id, created_at, user_id, summary_json, weight
            FROM compressed_episodes
            WHERE episode_id = ?
        """,
            (episode_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return CompressedEpisode(
                episode_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                user_id=row[2],
                summary=EpisodeSummary.from_dict(json.loads(row[3])),
                weight=row[4],
            )

        return None

    def get_recent_summaries(self, limit: int = 10) -> list[CompressedEpisode]:
        """Get most recent compressed episodes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT episode_id, created_at, user_id, summary_json, weight
            FROM compressed_episodes
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        episodes = []
        for row in cursor.fetchall():
            episodes.append(
                CompressedEpisode(
                    episode_id=row[0],
                    created_at=datetime.fromisoformat(row[1]),
                    user_id=row[2],
                    summary=EpisodeSummary.from_dict(json.loads(row[3])),
                    weight=row[4],
                )
            )

        conn.close()
        return episodes
