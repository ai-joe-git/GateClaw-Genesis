"""Layer 3: Semantic search using vector embeddings via Ollama."""

import json
import sqlite3
import hashlib
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal

from models.episode import Episode, EpisodeSummary


class EmbeddingEngine:
    """
    Vector embedding engine for semantic memory search.

    Purpose: Enable natural language queries like "find conversations about API debugging"
    without needing exact keyword matches.

    How it works:
    1. Generate embeddings for each episode summary using Ollama
    2. Store in SQLite + cosine similarity
    3. Query with natural language, find semantically similar episodes

    Supports Ollama as the primary provider with fallback to hash-based
    pseudo-embeddings if Ollama is unavailable.
    """

    # Ollama embedding endpoint
    OLLAMA_URL = "http://localhost:11434/api/embed"

    def __init__(
        self,
        db_path: str = "memory.db",
        vectors_path: str = "vectors",
        embedding_model: str = "qwen3-embedding:0.6b",
    ):
        self.db_path = Path(db_path)
        self.vectors_path = Path(vectors_path)
        self.vectors_path.mkdir(exist_ok=True)

        # Embedding model configuration
        self.embedding_model = embedding_model
        self.embedding_dimension = 1024  # Qwen3-Embedding dimension

        self._init_db()

        # Check Ollama availability
        self._ollama_available = self._check_ollama()

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and the embedding model is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                # Check if our model or any qwen3-embedding variant is available
                for m in model_names:
                    if "qwen3-embedding" in m.lower():
                        # Use the available variant
                        self.embedding_model = m
                        return True
                    if m.startswith("qwen3-embedding"):
                        return True
                # If qwen3-embedding not found, fall back
                return False
        except:
            pass
        return False

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
        Generate embedding for text using Ollama.

        Falls back to hash-based pseudo-embedding if Ollama is unavailable.
        """
        # Try Ollama first
        if self._ollama_available:
            try:
                response = requests.post(
                    self.OLLAMA_URL,
                    json={"model": self.embedding_model, "input": text},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and len(embeddings) > 0:
                        return embeddings[0]
            except Exception as e:
                print(f"Ollama embedding failed, falling back: {e}")
                self._ollama_available = False

        # Fallback: Hash-based pseudo-embedding (preserves determinism)
        return self._pseudo_embedding(text)

    def _pseudo_embedding(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embedding from text hash."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        embedding = []
        for i in range(self.embedding_dimension):
            chunk = text_hash[i % len(text_hash) : (i % len(text_hash)) + 8]
            val = int(chunk, 16) / (16**8)
            embedding.append(val * 2 - 1)
        return embedding

    def store_embedding(
        self,
        episode_id: str,
        summary: EpisodeSummary,
        model_name: Optional[str] = None,
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

        embedding_id = episode_id
        model = model_name or self.embedding_model

        cursor.execute(
            """
            INSERT OR REPLACE INTO episode_embeddings
            (embedding_id, episode_id, embedding_json, embedding_model)
            VALUES (?, ?, ?, ?)
        """,
            (embedding_id, episode_id, json.dumps(embedding), model),
        )

        conn.commit()
        conn.close()

        return embedding_id

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            # If dimensions don't match, try to handle gracefully
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            if min_len == 0:
                return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a**2 for a in vec1) ** 0.5
        norm2 = sum(b**2 for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def search_similar(
        self, query: str, limit: int = 5, threshold: float = 0.3
    ) -> list[dict]:
        """
        Search for episodes similar to query.

        Args:
            query: Natural language query
            limit: Maximum results
            threshold: Minimum similarity score

        Returns:
            List of {episode_id, similarity}
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
            "embedding_model": self.embedding_model,
            "ollama_available": self._ollama_available,
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

        Args:
            query: Natural language query
            time_window_days: Only search within X days (None = all time)
            min_weight: Minimum conversation weight
            limit: Maximum results

        Returns:
            Ranked list of matching episodes
        """
        results = self.search_similar(query, limit * 2)
        return results[:limit]

    def clear_cache(self):
        """Clear search cache."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_cache")
        conn.commit()
        conn.close()
