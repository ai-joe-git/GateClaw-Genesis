"""Layer 3: Semantic search using vector embeddings."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

from models.episode import Episode, EpisodeSummary


class EmbeddingEngine:
    """
    Vector embedding engine for semantic memory search.

    Purpose: Enable natural language queries like "find conversations about API debugging"
    without needing exact keyword matches.

    How it works:
    1. Generate embeddings for each episode summary
    2. Store in vector database (ChromaDB or simple SQLite + cosine similarity)
    3. Query with natural language, find semantically similar episodes

    TODO: Integrate with actual embedding model (all-MiniLM-L6-v2, ChromaDB)
    For now, uses simple TF-IDF as placeholder.
    """

    def __init__(self, db_path: str = "memory.db", vectors_path: str = "vectors"):
        self.db_path = Path(db_path)
        self.vectors_path = Path(vectors_path)
        self.vectors_path.mkdir(exist_ok=True)

        self._init_db()

        # Placeholder: embedding model
        # In production: Load sentence-transformers model
        self.embedding_model = None
        self.embedding_dimension = 384  # all-MiniLM-L6-v2 dimension

    def _init_db(self):
        """Initialize embedding storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Store embeddings as JSON arrays
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episode_embeddings (
                embedding_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(episode_id)
            )
        """)

        # Search cache for frequently used queries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT,
                results_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text.

        TODO: Replace with actual embedding model.
        Placeholder: Uses hash-based pseudo-embedding for testing.
        """
        # Placeholder: Generate deterministic pseudo-embedding from text
        # In production: Use sentence-transformers
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # return model.encode(text).tolist()

        # Pseudo-embedding: Deterministic hash-based
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Convert hash to pseudo-vector
        embedding = []
        for i in range(self.embedding_dimension):
            chunk = text_hash[i % len(text_hash) : (i % len(text_hash)) + 8]
            val = int(chunk, 16) / (16**8)  # Normalize to 0-1
            embedding.append(val * 2 - 1)  # Scale to -1 to 1

        return embedding

    def store_embedding(
        self,
        episode_id: str,
        summary: EpisodeSummary,
        model_name: str = "pseudo-embedding",
    ) -> str:
        """
        Generate and store embedding for an episode summary.

        Args:
            episode_id: Episode to embed
            summary: Episode summary
            model_name: Name of embedding model used

        Returns:
            Embedding ID
        """
        # Combine summary text for embedding
        embed_text = f"{summary.title} {summary.summary} {' '.join(summary.topics)} {' '.join(summary.entities)}"

        embedding = self.generate_embedding(embed_text)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Use episode_id as embedding_id
        embedding_id = episode_id

        cursor.execute(
            """
            INSERT OR REPLACE INTO episode_embeddings
            (embedding_id, episode_id, embedding_json, embedding_model)
            VALUES (?, ?, ?, ?)
        """,
            (embedding_id, episode_id, json.dumps(embedding), model_name),
        )

        conn.commit()
        conn.close()

        return embedding_id

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a**2 for a in vec1) ** 0.5
        norm2 = sum(b**2 for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search_similar(
        self, query: str, limit: int = 5, threshold: float = 0.5
    ) -> list[dict]:
        """
        Search for episodes similar to query.

        Args:
            query: Natural language query
            limit: Maximum results
            threshold: Minimum similarity score

        Returns:
            List of {episode_id, similarity, summary}
        """
        query_embedding = self.generate_embedding(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding_id, episode_id, embedding_json
            FROM episode_embeddings
        """)

        results = []
        for row in cursor.fetchall():
            stored_embedding = json.loads(row[2])
            similarity = self.cosine_similarity(query_embedding, stored_embedding)

            if similarity >= threshold:
                results.append(
                    {
                        "embedding_id": row[0],
                        "episode_id": row[1],
                        "similarity": similarity,
                    }
                )

        conn.close()

        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:limit]

    def get_embedding(self, episode_id: str) -> Optional[list[float]]:
        """Get embedding for specific episode."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT embedding_json FROM episode_embeddings
            WHERE episode_id = ?
        """,
            (episode_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def get_stats(self) -> dict:
        """Get embedding engine statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM episode_embeddings")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT DISTINCT embedding_model FROM episode_embeddings")
        models = [row[0] for row in cursor.fetchall()]

        conn.close()

        return {
            "total_embeddings": total,
            "embedding_dimension": self.embedding_dimension,
            "models": models,
        }

    def semantic_recall(
        self,
        query: str,
        time_window_days: Optional[int] = None,
        min_weight: float = 0.0,
        limit: int = 5,
    ) -> list[dict]:
        """
        Advanced semantic recall with filters.

        Combines:
        - Semantic similarity (vector search)
        - Temporal filtering (recent conversations)
        - Weight threshold (important conversations)

        Args:
            query: Natural language query
            time_window_days: Only search within X days (None = all time)
            min_weight: Minimum conversation weight
            limit: Maximum results

        Returns:
            Ranked list of matching episodes
        """
        # This would normally join with episode tables
        # For now, just use search_similar
        results = self.search_similar(query, limit * 2)  # Get extra for filtering

        # TODO: Apply time and weight filters when joining with episodes

        return results[:limit]

    def clear_cache(self):
        """Clear search cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_cache")
        conn.commit()
        conn.close()
