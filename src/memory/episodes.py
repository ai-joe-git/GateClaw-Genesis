"""Layer 2: Episode manager for conversation compression and summarization."""

import json
import sqlite3
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import re

from dataclasses import dataclass, field

from models.episode import Episode, EpisodeSummary, EmotionalTone, EmotionalState


def try_parse_json(content: str):
    """Try to extract and parse valid JSON from LLM response.
    Returns (parsed_object, None) on success or (None, error_msg) on failure.
    """
    # Strip markdown code blocks
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        for part in parts[1:]:
            if part.startswith("json"):
                content = part[4:]
                break
            elif part.strip():
                content = part
                break

    # Strategy: find balanced {...} by progressively shrinking from the end
    json_start = content.find("{")
    if json_start >= 0:
        json_end = content.rfind("}")
        if json_end > json_start:
            for end_pos in range(json_end, json_start - 1, -1):
                candidate = content[json_start : end_pos + 1]
                try:
                    return json.loads(candidate), None
                except json.JSONDecodeError:
                    continue

    return None, "No valid JSON found in response"


# llama-swap endpoint for Layer 2 summarization
LLAMA_SWAP_URL = "http://localhost:8888/v1/chat/completions"
SUMMARIZE_MODEL = "Nemotron-3-Nano-4B"  # Summarization model (Layer 2)
FACTS_MODEL = "Claude-4.6-Opus-4B"  # Fact extraction model (Layer 4)

TOPIC_KEYWORDS = {
    "gateclaw": [
        "gateclaw",
        "genesis",
        "memory system",
        "persistent memory",
        "episodic",
    ],
    "python": ["python", "async", "asyncio", "uvloop", "pip", "virtualenv", "venv"],
    "typescript": ["typescript", "ts", "node", "npm", "yarn", "tsx", "deno"],
    "opencode": ["opencode", "open code", "ide", "editor", "cursor", "vscode"],
    "telegram": ["telegram", "bot", "chat_id", "tg", "message", "send"],
    "docker": [
        "docker",
        "container",
        "dockerfile",
        "image",
        "compose",
        "kubectl",
        "k8s",
    ],
    "database": [
        "sqlite",
        "postgres",
        "mysql",
        "mongodb",
        "redis",
        "db",
        "query",
        "sql",
    ],
    "api": ["api", "rest", "graphql", "endpoint", "http", "webhook", "json"],
    "bug": ["bug", "crash", "error", "issue", "fix", "broken", "not working", "failed"],
    "feature": ["feature", "new", "add", "implement", "enhancement", "improve"],
    "deployment": [
        "deploy",
        "production",
        "server",
        "hosting",
        "cloud",
        "aws",
        "azure",
    ],
    "testing": ["test", "testing", "unit test", "integration", "pytest", "jest"],
    "config": [
        "config",
        "configuration",
        "env",
        ".env",
        "settings",
        "ini",
        "yaml",
        "toml",
    ],
    "tui": ["tui", "terminal", "console", "cli", "command line", "interface"],
    "models": [
        "model",
        "llm",
        "gpt",
        "claude",
        "ollama",
        "nemotron",
        "embedding",
        "vector",
    ],
}


@dataclass
class CompressedEpisode:
    """Layer 2 compressed episode — summarized representation of a raw conversation."""

    episode_id: str
    session_id: str
    title: str
    summary: str
    topics: list[str]
    outcomes: list[str]
    key_moments: list[str]
    emotional_tone: EmotionalTone
    emotional_state: EmotionalState
    created_at: datetime
    weight: float = 1.0
    resolved: bool = False
    revisit_count: int = 0
    embedding: list[float] = field(default_factory=list)


class EpisodeManager:
    """
    Episode compression and summarization.

    Purpose: Transform raw conversations into meaningful, searchable summaries.
    This is the bridge between raw storage and semantic understanding.

    Compression ratio: ~10x (from raw text to structured summary)
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self.LLAMA_SWAP_URL = LLAMA_SWAP_URL
        self.SUMMARIZE_MODEL = SUMMARIZE_MODEL
        self.FACTS_MODEL = FACTS_MODEL
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

        First tries LLM-based summarization via llama-swap, then falls back to
        keyword-based extraction if LLM unavailable.
        """
        messages = episode.raw_messages

        # Try LLM summarization first
        llm_summary = self._llm_summarize(messages)
        if llm_summary:
            return llm_summary

        # Fallback to keyword-based extraction
        return self._keyword_based_compress(episode)

    def _llm_summarize(self, messages: list[dict]) -> Optional[EpisodeSummary]:
        """Call Claude-4.6-Opus-2B via llama-swap for LLM-based episode summarization."""
        try:
            # Format messages into readable conversation
            conversation = "\n".join(
                [
                    f"{m.get('role', 'unknown').upper()}: {m.get('content', '')[:500]}"
                    for m in messages
                    if m.get("content")
                ]
            )

            prompt = f"""You are a memory summarization system. Summarize this conversation into structured JSON only.
No explanation. No markdown. No code blocks. Return only valid JSON.

Output format:
{{
  "title": "one line description of what this conversation was about",
  "summary": "2-3 sentences capturing what happened, what was decided, what was built",
  "topics": ["topic1", "topic2", "topic3"],
  "outcome": "resolved|unresolved|ongoing",
  "key_moments": ["specific thing that happened", "specific decision made"],
  "emotional_tone": "frustrated|neutral|productive|breakthrough|collaborative"
}}

Conversation:
{conversation}"""

            response = requests.post(
                self.LLAMA_SWAP_URL,
                json={
                    "model": SUMMARIZE_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a memory summarization system. You respond with valid JSON only. No explanation. No markdown. No text outside the JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
                timeout=600,
            )

            if response.status_code == 200:
                msg = response.json()["choices"][0]["message"]
                content = (
                    msg.get("content", "").strip()
                    or msg.get("reasoning_content", "").strip()
                )

                data, err = try_parse_json(content)
                if data is not None and isinstance(data, dict):
                    return EpisodeSummary(
                        title=data.get("title", "Conversation")[:50],
                        summary=data.get("summary", ""),
                        topics=data.get("topics", [])[:5],
                        outcomes=data.get("key_moments", [])[:3],
                        emotional_tone=EmotionalTone(
                            data.get("emotional_tone", "neutral")
                        ),
                        key_moments=data.get("key_moments", [])[:5],
                        entities=data.get("topics", [])[:5],
                    )
                else:
                    print(f"[Genesis] LLM summarization parse failed: {err}")

        except Exception as e:
            print(f"[Genesis] LLM summarization failed: {e}")

        return None

    def _keyword_based_compress(self, episode: Episode) -> EpisodeSummary:
        """Fallback keyword-based compression when LLM unavailable."""
        messages = episode.raw_messages

        # Extract title from first user message
        first_user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), "Conversation"
        )
        title = self._extract_title(first_user_msg)

        # Generate summary
        summary = self._generate_summary(messages)

        # Extract topics
        all_content = " ".join([m["content"] for m in messages])
        topics = self._extract_topics(all_content)

        # Extract outcomes
        outcomes = self._extract_outcomes(messages)

        # Extract entities
        entities = self._extract_entities(all_content)

        # Key moments
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
