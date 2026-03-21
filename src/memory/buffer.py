"""Layer 1: Raw buffer for immediate, lossless conversation storage."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import sqlite3

from models.episode import Episode, EpisodeSummary, EmotionalTone


class RawBuffer:
    """
    Raw conversation buffer.

    Purpose: Store the last N conversations in full, uncompressed form.
    This enables queries like "what did we talk about yesterday?" with
    zero information loss.

    Storage: SQLite for persistence, in-memory cache for speed.
    """

    DEFAULT_BUFFER_SIZE: int = 10  # Keep last 10 conversations uncompressed

    def __init__(
        self, db_path: str = "memory.db", buffer_size: int = DEFAULT_BUFFER_SIZE
    ):
        self.db_path = Path(db_path)
        self.buffer_size = buffer_size
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for raw buffer."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_episodes (
                episode_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT DEFAULT 'default',
                messages_json TEXT NOT NULL,
                message_count INTEGER,
                emotional_tone TEXT,
                weight REAL
            )
        """)

        # Index for temporal queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at 
            ON raw_episodes(created_at DESC)
        """)

        conn.commit()
        conn.close()

    def add_message(self, episode: Episode, role: str, content: str) -> Episode:
        """
        Add a message to the current episode.

        Args:
            episode: Current episode being built
            role: 'user' or 'assistant'
            content: Message content

        Returns:
            Updated episode
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        episode.raw_messages.append(message)
        episode.message_count = len(episode.raw_messages)

        # Update emotional tone based on latest message
        combined_content = " ".join([m["content"] for m in episode.raw_messages])
        episode.emotional_tone = EmotionalTone.from_content(combined_content)

        return episode

    def save_episode(self, episode: Episode) -> bool:
        """
        Save completed episode to raw buffer.

        Args:
            episode: Completed episode to save

        Returns:
            True if saved successfully
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Calculate weight
        episode.weight = episode.compute_weight()

        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO raw_episodes 
                (episode_id, created_at, user_id, messages_json, message_count, emotional_tone, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    episode.episode_id,
                    episode.created_at.isoformat(),
                    episode.user_id,
                    json.dumps(episode.raw_messages),
                    episode.message_count,
                    episode.emotional_tone.value,
                    episode.weight,
                ),
            )

            conn.commit()

            # Prune old episodes if buffer is full
            self._prune_buffer(conn)

            return True

        except Exception as e:
            print(f"Error saving episode: {e}")
            return False
        finally:
            conn.close()

    def _prune_buffer(self, conn: sqlite3.Connection):
        """Remove oldest episodes if buffer exceeds size limit."""
        cursor = conn.cursor()

        # Get count
        cursor.execute("SELECT COUNT(*) FROM raw_episodes")
        count = cursor.fetchone()[0]

        if count > self.buffer_size:
            # Delete oldest episodes (lowest weight ones first)
            excess = count - self.buffer_size

            cursor.execute(
                """
                DELETE FROM raw_episodes 
                WHERE episode_id IN (
                    SELECT episode_id FROM raw_episodes 
                    ORDER BY weight ASC, created_at ASC 
                    LIMIT ?
                )
            """,
                (excess,),
            )

            conn.commit()

    def get_recent_episodes(self, limit: int = 5) -> list[Episode]:
        """
        Get most recent episodes from buffer.

        Args:
            limit: Maximum number to retrieve

        Returns:
            List of episodes, most recent first
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT episode_id, created_at, user_id, messages_json, 
                   message_count, emotional_tone, weight
            FROM raw_episodes
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        episodes = []
        for row in cursor.fetchall():
            episode = Episode(
                episode_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                user_id=row[2],
                raw_messages=json.loads(row[3]),
                message_count=row[4],
                emotional_tone=EmotionalTone(row[5]),
                weight=row[6],
            )
            episodes.append(episode)

        conn.close()
        return episodes

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        """Retrieve specific episode by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT episode_id, created_at, user_id, messages_json, 
                   message_count, emotional_tone, weight
            FROM raw_episodes
            WHERE episode_id = ?
        """,
            (episode_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return Episode(
                episode_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                user_id=row[2],
                raw_messages=json.loads(row[3]),
                message_count=row[4],
                emotional_tone=EmotionalTone(row[5]),
                weight=row[6],
            )

        return None

    def search_content(self, query: str, limit: int = 10) -> list[Episode]:
        """
        Simple text search through raw messages.

        Note: This is basic substring matching.
        For semantic search, use Layer 3 (EmbeddingEngine).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT episode_id, created_at, user_id, messages_json, 
                   message_count, emotional_tone, weight
            FROM raw_episodes
            WHERE messages_json LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """,
            (f"%{query}%", limit),
        )

        episodes = []
        for row in cursor.fetchall():
            episode = Episode(
                episode_id=row[0],
                created_at=datetime.fromisoformat(row[1]),
                user_id=row[2],
                raw_messages=json.loads(row[3]),
                message_count=row[4],
                emotional_tone=EmotionalTone(row[5]),
                weight=row[6],
            )
            episodes.append(episode)

        conn.close()
        return episodes

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM raw_episodes")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(weight) FROM raw_episodes")
        avg_weight = cursor.fetchone()[0] or 0

        cursor.execute("SELECT AVG(message_count) FROM raw_episodes")
        avg_length = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_episodes": total,
            "buffer_size": self.buffer_size,
            "average_weight": round(avg_weight, 2),
            "average_length": round(avg_length, 1),
        }
