"""Layer 4: Consolidation and relationship graph for Genesis memory system."""

import json
import sqlite3
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

from models.episode import Episode, PersistentFact, EmotionalFlag, EmotionalState


class ConsolidationEngine:
    """
    Layer 4: Consolidates episodes into persistent facts and forms relationships.

    This is where memory becomes knowledge:
    - Episodes are processed to extract facts
    - Facts are connected to source episodes
    - Relationships emerge from shared flags and topics
    - Facts persist through reinforcement, decay slowly

    The relationship graph is emergent, not designed:
    - Two episodes both tagged FRUSTRATION + "Docker" get implicit connection
    - Surface bring-up based on topic overlap + flag patterns
    """

    # llama-swap endpoint for LLM-based fact extraction (CPU-friendly text gen)
    LLAMA_SWAP_URL = "http://localhost:8888/v1/chat/completions"
    SUMMARIZE_MODEL = "Nemotron-3-Nano-4B"  # Layer 2 — summarization
    FACTS_MODEL = (
        "Claude-4.6-Opus-4B"  # Layer 4 — fact extraction (pure transformer, no SSM)
    )

    def __init__(
        self,
        db_path: str = "memory.db",
        llm_model: str = "unused",  # llama-swap used instead
    ):
        self.db_path = Path(db_path)
        # Use llama-swap model, ignore Ollama param
        self.llm_model = self.FACTS_MODEL
        self._llm_available = self._check_llama_swap()
        self._init_db()

    def _check_llama_swap(self) -> bool:
        """Check if llama-swap is running at port 8888."""
        try:
            response = requests.get("http://localhost:8888/v1/models", timeout=2)
            return response.status_code == 200
        except:
            pass
        return False

    def _init_db(self):
        """Initialize Layer 4 storage: persistent facts and relationships."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Persistent facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persistent_facts (
                fact_key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                topic TEXT,
                source_episodes_json TEXT NOT NULL,
                reinforcement_count INTEGER DEFAULT 1,
                base_weight REAL DEFAULT 0.5,
                current_weight REAL DEFAULT 0.5,
                last_reinforced TEXT NOT NULL,
                last_decay TEXT NOT NULL,
                flag_types_json TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Episode-to-fact relationships
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episode_fact_links (
                episode_id TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                PRIMARY KEY (episode_id, fact_key)
            )
        """)

        # Relationship graph edges (emergent, not explicit)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episode_relationships (
                episode_id_1 TEXT NOT NULL,
                episode_id_2 TEXT NOT NULL,
                relationship_strength REAL DEFAULT 0.0,
                shared_topics_json TEXT,
                shared_flags_json TEXT,
                last_updated TEXT NOT NULL,
                PRIMARY KEY (episode_id_1, episode_id_2)
            )
        """)

        conn.commit()
        conn.close()

    def consolidate_episode(
        self,
        episode: Episode,
        llm_extract: bool = True,
    ) -> list[PersistentFact]:
        """
        Consolidate a single episode into persistent facts.

        Args:
            episode: The episode to consolidate
            llm_extract: Whether to use LLM for fact extraction (or fallback to rules)

        Returns:
            List of PersistentFact records created/updated
        """
        facts = []

        # Step 1: Extract facts (LLM or rule-based)
        if llm_extract and self._llm_available:
            facts = self._extract_facts_llm(episode)
        else:
            facts = self._extract_facts_rules(episode)

        # Step 2: Store facts and link to episode
        for fact in facts:
            self._store_fact(fact, episode.episode_id)

        # Step 3: Update episode relationships
        self._update_relationships(episode)

        # Step 4: Mark episode as consolidated
        episode.consolidated = True

        return facts

    def _extract_facts_llm(self, episode: Episode) -> list[PersistentFact]:
        """
        Use LLM via llama-swap to extract facts from episode.

        llama-swap uses OpenAI-compatible API at localhost:8888.
        Uses FACTS_MODEL (Claude-4.6-Opus-4B) for fact extraction.
        """
        if not episode.summary:
            return self._extract_facts_rules(episode)

        # Build the tight prompt for fact extraction
        prompt = f"""Extract persistent facts about the user from this conversation summary.
Return a JSON array only. No explanation. No markdown. No code blocks. No intro text.
If no clear facts, return empty array: []

Facts should be specific and useful for future context:
- Technical preferences (languages, tools, frameworks)
- Project names and what they do
- Problems encountered and whether resolved
- Decisions made
- Personal context (location, hardware, workflow)

Output format: [{{"key": "fact_name", "value": "fact_value", "confidence": 0.0-1.0}}]

Summary: {episode.summary.summary}
Topics: {", ".join(episode.summary.topics) if episode.summary.topics else "none"}
Outcome: {episode.summary.outcomes[0] if episode.summary.outcomes else "unknown"}"""

        try:
            response = requests.post(
                self.LLAMA_SWAP_URL,
                json={
                    "model": self.FACTS_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a structured data extraction system. Return valid JSON array only. No explanation. No markdown. No text outside the array.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 600,
                },
                timeout=600,
            )

            if response.status_code == 200:
                data = response.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                response_text = (
                    msg.get("content", "").strip()
                    or msg.get("reasoning_content", "").strip()
                )

                # Parse JSON from response
                try:
                    facts_data = json.loads(response_text)
                    if isinstance(facts_data, list):
                        facts = []
                        for fd in facts_data:
                            fact = PersistentFact(
                                fact_key=fd.get("fact_key", fd.get("topic", "unknown")),
                                value=fd.get("value", ""),
                                topic=fd.get("topic", ""),
                            )

                            # Set initial weight from episode emotional state
                            if episode.emotional_state.flags:
                                max_weight = max(
                                    episode.emotional_state.current_weight,
                                    episode.weight,
                                )
                                fact.current_weight = fact.base_weight = max_weight

                            facts.append(fact)
                        return facts
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"LLM extraction failed: {e}")

        # Fallback to rule-based
        return self._extract_facts_rules(episode)

    def _extract_facts_rules(self, episode: Episode) -> list[PersistentFact]:
        """
        Rule-based fact extraction from episode.

        Uses topics, entities, and emotional flags to generate facts.
        """
        facts = []

        if not episode.summary:
            return facts

        # Fact 1: Primary topic from summary
        if episode.summary.topics:
            primary_topic = episode.summary.topics[0]
            topic_fact_key = self._slugify(primary_topic)

            # Add emotional context to fact key
            dominant_flag = self._get_dominant_flag(episode.emotional_state)
            if dominant_flag:
                fact_key = f"{topic_fact_key}_{dominant_flag.value}"
            else:
                fact_key = topic_fact_key

            value = f"Conversation about {primary_topic}"
            if episode.emotional_state.resolved:
                value = f"Solved {primary_topic}: {episode.summary.summary[:100]}"
            elif episode.emotional_state.flags:
                flag_str = ", ".join(
                    f.value for f, c in episode.emotional_state.flags[:2]
                )
                value = f"User had {flag_str} around {primary_topic}"

            fact = PersistentFact(
                fact_key=fact_key,
                value=value,
                topic=primary_topic,
            )

            # Set weight from emotional state
            fact.base_weight = fact.current_weight = (
                episode.emotional_state.current_weight
            )
            facts.append(fact)

        # Fact 2: Outcome if present
        if episode.summary.outcomes:
            outcome = episode.summary.outcomes[0]
            outcome_key = self._slugify(f"outcome_{outcome}")
            fact = PersistentFact(
                fact_key=outcome_key,
                value=f"Outcome: {outcome}",
                topic="outcomes",
            )
            fact.base_weight = fact.current_weight = min(0.9, episode.weight + 0.1)
            facts.append(fact)

        # Fact 3: Entity-based facts
        for entity in episode.summary.entities[:2]:
            entity_key = self._slugify(f"entity_{entity}")
            fact = PersistentFact(
                fact_key=entity_key,
                value=f"Discussion involved {entity}",
                topic=entity,
            )
            fact.base_weight = fact.current_weight = episode.weight * 0.8
            facts.append(fact)

        return facts

    def _get_dominant_flag(self, state: EmotionalState) -> Optional[EmotionalFlag]:
        """Get the most significant emotional flag from state."""
        if not state.flags:
            return None

        # Priority order for dominance
        priority = [
            EmotionalFlag.FRUSTRATION,
            EmotionalFlag.BREAKTHROUGH,
            EmotionalFlag.SOLVED,
            EmotionalFlag.CONFUSION,
        ]

        for flag in priority:
            for f, c in state.flags:
                if f == flag:
                    return f

        return state.flags[0][0] if state.flags else None

    def _slugify(self, text: str) -> str:
        """Convert text to a fact key slug."""
        # Simple slugification
        slug = "".join(c if c.isalnum() else "_" for c in text.lower())
        slug = "_".join(w for w in slug.split("_") if w)
        return slug[:50]

    def _store_fact(self, fact: PersistentFact, episode_id: str):
        """Store or update a persistent fact."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if fact exists
        cursor.execute(
            "SELECT fact_key, reinforcement_count, current_weight FROM persistent_facts WHERE fact_key = ?",
            (fact.fact_key,),
        )
        existing = cursor.fetchone()

        if existing:
            # Reinforce existing fact
            reinforcement_count = existing[1] + 1
            current_weight = existing[2]
            # Weight increases with reinforcement
            current_weight = min(0.95, current_weight + (0.05 / reinforcement_count))

            cursor.execute(
                """
                UPDATE persistent_facts
                SET reinforcement_count = ?,
                    current_weight = ?,
                    last_reinforced = ?,
                    source_episodes_json = (
                        SELECT source_episodes_json || ? || ','
                        FROM persistent_facts WHERE fact_key = ?
                    )
                WHERE fact_key = ?
                """,
                (
                    reinforcement_count,
                    current_weight,
                    datetime.now().isoformat(),
                    episode_id,
                    fact.fact_key,
                    fact.fact_key,
                ),
            )
        else:
            # Insert new fact
            cursor.execute(
                """
                INSERT INTO persistent_facts
                (fact_key, value, topic, source_episodes_json, reinforcement_count,
                 base_weight, current_weight, last_reinforced, last_decay,
                 flag_types_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact.fact_key,
                    fact.value,
                    fact.topic,
                    episode_id + ",",
                    fact.reinforcement_count,
                    fact.base_weight,
                    fact.current_weight,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    json.dumps(fact.flag_types),
                    datetime.now().isoformat(),
                ),
            )

        # Link episode to fact
        cursor.execute(
            """
            INSERT OR IGNORE INTO episode_fact_links (episode_id, fact_key)
            VALUES (?, ?)
            """,
            (episode_id, fact.fact_key),
        )

        conn.commit()
        conn.close()

    def _update_relationships(self, episode: Episode):
        """
        Update relationship graph for an episode.

        Relationships emerge from:
        - Shared topics
        - Shared emotional flags
        - Temporal proximity
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get recent episodes for relationship computation
        cursor.execute(
            """
            SELECT episode_id, created_at FROM compressed_episodes
            WHERE episode_id != ?
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (episode.episode_id,),
        )

        recent_episodes = cursor.fetchall()

        for other_ep_id, other_created_at in recent_episodes:
            # Get other episode's summary and flags
            cursor.execute(
                """
                SELECT summary_json FROM compressed_episodes
                WHERE episode_id = ?
                """,
                (other_ep_id,),
            )
            row = cursor.fetchone()

            if not row:
                continue

            # Compute relationship strength
            strength = 0.0
            shared_topics = []
            shared_flags = []

            # Topic overlap
            if episode.summary and episode.summary.topics:
                # We'd need to store topics per episode for this
                # For now, skip topic-based similarity
                pass

            # Flag-based similarity
            if episode.emotional_state.flags:
                # We don't have other episode's flags stored per-episode yet
                # This would require enriching the compressed_episodes table
                pass

            # Simple: any shared entity
            if episode.summary and episode.summary.entities:
                # Again, would need entity storage per episode
                pass

            # For now, relationships are very basic
            # In a full implementation, we'd store topics/flags per compressed episode
            strength = 0.1  # Baseline relationship for recent conversations

            if strength > 0.05:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO episode_relationships
                    (episode_id_1, episode_id_2, relationship_strength,
                     shared_topics_json, shared_flags_json, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode.episode_id,
                        other_ep_id,
                        strength,
                        json.dumps(shared_topics),
                        json.dumps(shared_flags),
                        datetime.now().isoformat(),
                    ),
                )

        conn.commit()
        conn.close()

    def get_facts(
        self,
        topic: Optional[str] = None,
        min_weight: float = 0.0,
        limit: int = 10,
    ) -> list[PersistentFact]:
        """
        Retrieve persistent facts from memory.

        Args:
            topic: Filter by topic
            min_weight: Minimum fact weight
            limit: Maximum results

        Returns:
            List of PersistentFact objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if topic:
            cursor.execute(
                """
                SELECT * FROM persistent_facts
                WHERE topic = ? AND current_weight >= ?
                ORDER BY current_weight DESC
                LIMIT ?
                """,
                (topic, min_weight, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM persistent_facts
                WHERE current_weight >= ?
                ORDER BY current_weight DESC, reinforcement_count DESC
                LIMIT ?
                """,
                (min_weight, limit),
            )

        rows = cursor.fetchall()
        conn.close()

        facts = []
        for row in rows:
            # Apply decay before returning
            fact = PersistentFact(
                fact_key=row[0],
                value=row[1],
                topic=row[2],
                source_episodes=row[3].rstrip(",").split(",") if row[3] else [],
                reinforcement_count=row[4],
                base_weight=row[5],
                current_weight=row[6],
                last_reinforced=datetime.fromisoformat(row[7]),
                last_decay=datetime.fromisoformat(row[8]),
                flag_types=json.loads(row[9]) if row[9] else [],
            )
            fact.apply_decay()
            facts.append(fact)

        return facts

    def get_unresolved(self, min_weight: float = 0.7, limit: int = 5) -> list[dict]:
        """
        Get unresolved high-weight episodes/facts.

        This is what powers inject_unresolved=True in recall.

        Args:
            min_weight: Minimum weight threshold
            limit: Maximum results

        Returns:
            List of unresolved facts with source episode info
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Find unresolved facts (reinforced by frustration/confusion flags)
        cursor.execute(
            """
            SELECT pf.*, eal.episode_id
            FROM persistent_facts pf
            JOIN episode_fact_links eal ON pf.fact_key = eal.fact_key
            WHERE pf.current_weight >= ?
            ORDER BY pf.current_weight DESC
            LIMIT ?
            """,
            (min_weight, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append(
                {
                    "fact_key": row[0],
                    "value": row[1],
                    "topic": row[2],
                    "weight": row[6],
                    "episode_id": row[11] if len(row) > 11 else None,
                }
            )

        return results

    def get_stats(self) -> dict:
        """Get Layer 4 statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM persistent_facts")
        fact_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM episode_fact_links")
        link_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM episode_relationships")
        relationship_count = cursor.fetchone()[0]

        cursor.execute(
            "SELECT AVG(current_weight) FROM persistent_facts WHERE current_weight > 0"
        )
        avg_weight = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_facts": fact_count,
            "total_episode_fact_links": link_count,
            "total_relationships": relationship_count,
            "average_fact_weight": round(avg_weight, 3),
            "llm_extraction_available": self._llm_available,
        }

    def apply_global_decay(self):
        """
        Apply decay to all facts.

        Called periodically to decay fact weights over time.
        In production, this would run as a background task.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT fact_key, current_weight, last_decay FROM persistent_facts"
        )
        facts = cursor.fetchall()

        for fact_key, current_weight, last_decay_str in facts:
            last_decay = datetime.fromisoformat(last_decay_str)
            days_elapsed = (datetime.now() - last_decay).total_seconds() / 86400

            if days_elapsed < 1.0:
                continue

            # Slow decay
            new_weight = current_weight * (0.995**days_elapsed)
            new_weight = max(0.3, new_weight)

            cursor.execute(
                """
                UPDATE persistent_facts
                SET current_weight = ?, last_decay = ?
                WHERE fact_key = ?
                """,
                (new_weight, datetime.now().isoformat(), fact_key),
            )

        conn.commit()
        conn.close()
