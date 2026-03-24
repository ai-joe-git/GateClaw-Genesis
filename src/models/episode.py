"""Episode data structures for GateClaw Genesis memory system."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import uuid4
import json


class EmotionalFlag(Enum):
    """Categorical inflection flags detected during conversation."""

    FRUSTRATION = "frustration"  # User expressed frustration
    BREAKTHROUGH = "breakthrough"  # Problem solved, insight gained
    SOLVED = "solved"  # Problem marked as resolved
    REVISITED = "revisited"  # User returned to previous topic
    CONFUSION = "confusion"  # User re-asked same thing multiple times
    EXPLICIT_PRAISE = "explicit_praise"  # User said thanks or complimented
    COLLABORATION = "collaboration"  # Active joint problem-solving


class FlagConfidence(Enum):
    """Confidence tier for emotional flags."""

    EXPLICIT = "explicit"  # Direct signal: "YES!", "I give up", explicit patterns
    INFERRED = "inferred"  # Pattern-based: tone, revisit count, length


@dataclass
class EmotionalState:
    """Emotional flags with confidence tiers and decay tracking."""

    flags: list[tuple[EmotionalFlag, FlagConfidence]] = field(default_factory=list)
    revisit_count: int = 0
    resolved: bool = False

    # Decay state
    base_weight: float = 0.5
    current_weight: float = 0.5
    last_decay: datetime = field(default_factory=datetime.now)

    # Decay rates per flag type and confidence (exponential decay base)
    DECAY_RATES = {
        # Breakthrough with explicit confidence decays slowest (remember victories)
        ("breakthrough", "explicit"): 0.98,
        ("breakthrough", "inferred"): 0.95,
        # Solved problems also decay slowly
        ("solved", "explicit"): 0.97,
        ("solved", "inferred"): 0.94,
        # Frustration decays fast (you move on)
        ("frustration", "explicit"): 0.85,
        ("frustration", "inferred"): 0.75,
        # Confusion decays medium-fast
        ("confusion", "explicit"): 0.88,
        ("confusion", "inferred"): 0.80,
        # Revisited is a reinforcement signal
        ("revisited", "explicit"): 1.02,  # Grows slightly
        ("revisited", "inferred"): 1.01,
        # Praise decays normally
        ("explicit_praise", "explicit"): 0.95,
        ("explicit_praise", "inferred"): 0.92,
        # Collaboration decays medium
        ("collaboration", "explicit"): 0.93,
        ("collaboration", "inferred"): 0.90,
    }

    def add_flag(self, flag: EmotionalFlag, confidence: FlagConfidence):
        """Add an emotional flag if not already present."""
        if (flag, confidence) not in self.flags:
            self.flags.append((flag, confidence))
            self._recompute_weight()

    def increment_revisit(self):
        """Increment revisit count and trigger reinforcement."""
        self.revisit_count += 1
        if self.revisit_count >= 2:
            self.add_flag(EmotionalFlag.REVISITED, FlagConfidence.INFERRED)
        if self.revisit_count >= 4:
            self.add_flag(EmotionalFlag.REVISITED, FlagConfidence.EXPLICIT)
        self._recompute_weight()

    def mark_resolved(self):
        """Mark episode as resolved and add SOLVED flag."""
        self.resolved = True
        self.add_flag(EmotionalFlag.SOLVED, FlagConfidence.EXPLICIT)
        self._recompute_weight()

    def apply_decay(self) -> float:
        """
        Apply exponential decay to current weight.

        Returns:
            New current weight after decay
        """
        now = datetime.now()
        days_elapsed = (now - self.last_decay).total_seconds() / 86400

        if days_elapsed < 0.01:  # Less than ~15 minutes, skip
            return self.current_weight

        # Unresolved high-frustration episodes don't decay below 0.7
        if not self.resolved:
            frustration_flags = [
                f for f, c in self.flags if f == EmotionalFlag.FRUSTRATION
            ]
            if frustration_flags:
                min_weight = 0.7
            else:
                min_weight = 0.3
        else:
            min_weight = 0.1

        # Compute composite decay rate from all flags
        if self.flags:
            # Use the slowest-decaying (most important) flag's rate
            # Key format: (flag_value, confidence_value)
            decay_rate = min(
                self.DECAY_RATES.get((f.value, c.value), 0.90) for f, c in self.flags
            )
        else:
            decay_rate = 0.95  # Default decay for flagless episodes

        # Apply exponential decay
        self.current_weight = max(
            min_weight, self.current_weight * (decay_rate**days_elapsed)
        )
        self.last_decay = now

        return self.current_weight

    def _recompute_weight(self):
        """Recompute base weight from flags and revisit count."""
        weight = 0.5

        for flag, confidence in self.flags:
            if flag == EmotionalFlag.BREAKTHROUGH:
                weight += 0.15 if confidence == FlagConfidence.EXPLICIT else 0.08
            elif flag == EmotionalFlag.FRUSTRATION:
                weight += 0.12 if confidence == FlagConfidence.EXPLICIT else 0.05
            elif flag == EmotionalFlag.SOLVED:
                weight += 0.10
            elif flag == EmotionalFlag.CONFUSION:
                weight += 0.08 if confidence == FlagConfidence.EXPLICIT else 0.03
            elif flag == EmotionalFlag.EXPLICIT_PRAISE:
                weight += 0.05

        # Revisit count boosts weight
        weight += min(0.15, self.revisit_count * 0.03)

        # Resolved status
        if self.resolved:
            weight += 0.05

        self.base_weight = min(1.0, weight)
        self.current_weight = self.base_weight
        self.last_decay = datetime.now()

    def to_dict(self) -> dict:
        return {
            "flags": [(f.value, c.value) for f, c in self.flags],
            "revisit_count": self.revisit_count,
            "resolved": self.resolved,
            "base_weight": self.base_weight,
            "current_weight": self.current_weight,
            "last_decay": self.last_decay.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionalState":
        state = cls(
            revisit_count=data.get("revisit_count", 0),
            resolved=data.get("resolved", False),
            base_weight=data.get("base_weight", 0.5),
            current_weight=data.get("current_weight", 0.5),
        )
        if data.get("last_decay"):
            state.last_decay = datetime.fromisoformat(data["last_decay"])
        flags_data = data.get("flags", [])
        state.flags = [(EmotionalFlag(f), FlagConfidence(c)) for f, c in flags_data]
        return state


class EmotionalTone(Enum):
    """Legacy emotional tone classification (Layer 2 output)."""

    CASUAL = "casual"
    CURIOUS = "curious"
    FRUSTRATED = "frustrated"
    EXCITED = "excited"
    SERIOUS = "serious"
    REFLECTIVE = "reflective"
    COLLABORATIVE = "collaborative"
    STRESSED = "stressed"
    TRIUMPHANT = "triumphant"

    @classmethod
    def from_content(cls, content: str) -> "EmotionalTone":
        """Infer emotional tone from content using keyword heuristics."""
        lower = content.lower()

        celebration_words = ["yes!", "finally", "solved", "works", "amazing", "awesome"]
        frustration_words = ["stuck", "doesn't work", "error", "damn", "frustrating"]
        serious_words = ["important", "critical", "deadline", "production", "bug"]
        curious_words = ["how", "why", "what if", "wonder", "curious"]

        if any(w in lower for w in celebration_words):
            return cls.TRIUMPHANT
        elif any(w in lower for w in frustration_words):
            return cls.FRUSTRATED
        elif any(w in lower for w in serious_words):
            return cls.SERIOUS
        elif any(w in lower for w in curious_words):
            return cls.CURIOUS

        return cls.CASUAL


@dataclass
class EpisodeSummary:
    """Compressed summary of an episode (Layer 2 output)."""

    title: str
    summary: str
    topics: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    emotional_tone: EmotionalTone = EmotionalTone.CASUAL
    key_moments: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "topics": self.topics,
            "outcomes": self.outcomes,
            "emotional_tone": self.emotional_tone.value,
            "key_moments": self.key_moments,
            "entities": self.entities,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeSummary":
        return cls(
            title=data["title"],
            summary=data["summary"],
            topics=data.get("topics", []),
            outcomes=data.get("outcomes", []),
            emotional_tone=EmotionalTone(data.get("emotional_tone", "casual")),
            key_moments=data.get("key_moments", []),
            entities=data.get("entities", []),
        )


@dataclass
class Episode:
    """
    A complete conversation episode.

    Represents a single conversation session with:
    - Raw conversation data (Layer 1)
    - Compressed summary (Layer 2)
    - Embedding vector (Layer 3)
    - Emotional flags and decay state (Layer 3/4)
    """

    # Identity
    episode_id: str = field(default_factory=lambda: str(uuid4())[:8])
    session_id: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    # Content
    raw_messages: list[dict] = field(default_factory=list)
    message_count: int = 0

    # Compressed summary (Layer 2)
    summary: Optional[EpisodeSummary] = None

    # Embedding (Layer 3)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None

    # Metadata
    user_id: str = "default"
    emotional_tone: EmotionalTone = EmotionalTone.CASUAL

    # Weight and emotional state (Layer 3/4)
    weight: float = 1.0
    emotional_state: EmotionalState = field(default_factory=EmotionalState)

    # Consolidation status
    consolidated: bool = False  # True once Layer 4 has processed this

    def compute_weight(self) -> float:
        """Calculate conversation weight based on emotional state and flags."""
        # Use emotional state weight if available
        if self.emotional_state.flags or self.emotional_state.revisit_count > 0:
            return self.emotional_state.current_weight

        # Fallback to legacy computation
        base_weight = 0.5
        emotional_weights = {
            EmotionalTone.CASUAL: 0.3,
            EmotionalTone.CURIOUS: 0.5,
            EmotionalTone.FRUSTRATED: 0.7,
            EmotionalTone.EXCITED: 0.8,
            EmotionalTone.SERIOUS: 0.8,
            EmotionalTone.REFLECTIVE: 0.6,
            EmotionalTone.COLLABORATIVE: 0.7,
            EmotionalTone.STRESSED: 0.7,
            EmotionalTone.TRIUMPHANT: 0.9,
        }
        base_weight = emotional_weights.get(self.emotional_tone, 0.5)

        if self.message_count > 20:
            base_weight += 0.1
        if self.message_count > 50:
            base_weight += 0.1

        if self.summary and self.summary.outcomes:
            base_weight += min(0.2, len(self.summary.outcomes) * 0.05)

        self.weight = min(1.0, base_weight)
        return self.weight

    def add_emotional_flag(
        self, flag: EmotionalFlag, confidence: FlagConfidence = FlagConfidence.INFERRED
    ):
        """Add an emotional flag to this episode."""
        self.emotional_state.add_flag(flag, confidence)
        self.compute_weight()

    def detect_flags_from_content(self):
        """
        Detect emotional flags from raw message content.

        This is a lightweight pattern-based detection that runs
        on session end. Real LLM-based detection would come later.
        """
        all_content = " ".join(
            msg.get("content", "").lower() for msg in self.raw_messages
        )

        # Explicit patterns
        explicit_frustration = [
            "i give up",
            "i can't figure it out",
            "this is frustrating",
            "damn it",
            "unbelievable",
            "why is this so hard",
        ]
        explicit_breakthrough = [
            "yes!",
            "it works!",
            "finally!",
            "got it!",
            "that did it!",
            "solved it",
        ]
        explicit_solved = [
            "all good",
            "fixed it",
            "resolved",
            "working now",
            "thanks, that worked",
        ]
        explicit_praise = [
            "thank you",
            "thanks!",
            "you the best",
            "appreciate it",
            "nice!",
        ]

        # Check for explicit patterns
        for pattern in explicit_frustration:
            if pattern in all_content:
                self.add_emotional_flag(
                    EmotionalFlag.FRUSTRATION, FlagConfidence.EXPLICIT
                )
                break

        for pattern in explicit_breakthrough:
            if pattern in all_content:
                self.add_emotional_flag(
                    EmotionalFlag.BREAKTHROUGH, FlagConfidence.EXPLICIT
                )
                break

        for pattern in explicit_solved:
            if pattern in all_content:
                self.emotional_state.mark_resolved()
                break

        for pattern in explicit_praise:
            if pattern in all_content:
                self.add_emotional_flag(
                    EmotionalFlag.EXPLICIT_PRAISE, FlagConfidence.EXPLICIT
                )
                break

        # Inferred patterns
        exclamation_count = all_content.count("!")
        question_repeat = self._detect_repeated_questions()

        if question_repeat > 1:
            self.add_emotional_flag(EmotionalFlag.CONFUSION, FlagConfidence.INFERRED)

        if exclamation_count >= 3:
            self.add_emotional_flag(EmotionalFlag.BREAKTHROUGH, FlagConfidence.INFERRED)

        # Long collaborative session
        if (
            self.message_count >= 15
            and self.emotional_tone == EmotionalTone.COLLABORATIVE
        ):
            self.add_emotional_flag(
                EmotionalFlag.COLLABORATION, FlagConfidence.INFERRED
            )

    def _detect_repeated_questions(self) -> int:
        """Detect if user asked the same question multiple times."""
        user_messages = [
            msg.get("content", "").lower()
            for msg in self.raw_messages
            if msg.get("role") == "user"
        ]

        # Simple approach: look for similar question patterns
        question_words = [
            "how",
            "why",
            "what",
            "when",
            "where",
            "can i",
            "do i",
            "is it",
        ]
        questions = []
        for msg in user_messages:
            for qw in question_words:
                if qw in msg and "?" in msg:
                    questions.append(msg)
                    break

        if len(questions) < 2:
            return 0

        # Count duplicates or near-duplicates
        repeats = 0
        for i, q1 in enumerate(questions):
            for q2 in questions[i + 1 :]:
                if q1 == q2 or self._levenshtein_ratio(q1, q2) > 0.8:
                    repeats += 1

        return repeats

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_ratio(s2, s1)

        if len(s2) == 0:
            return 0.0

        # Simple Levenshtein distance
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "raw_messages": self.raw_messages,
            "message_count": self.message_count,
            "summary": self.summary.to_dict() if self.summary else None,
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "user_id": self.user_id,
            "emotional_tone": self.emotional_tone.value,
            "weight": self.weight,
            "emotional_state": self.emotional_state.to_dict(),
            "consolidated": self.consolidated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        episode = cls(
            episode_id=data["episode_id"],
            session_id=data.get("session_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            duration_seconds=data.get("duration_seconds", 0.0),
            raw_messages=data.get("raw_messages", []),
            message_count=data.get("message_count", 0),
            user_id=data.get("user_id", "default"),
            emotional_tone=EmotionalTone(data.get("emotional_tone", "casual")),
            weight=data.get("weight", 1.0),
            consolidated=data.get("consolidated", False),
        )

        if data.get("summary"):
            episode.summary = EpisodeSummary.from_dict(data["summary"])

        episode.embedding = data.get("embedding")
        episode.embedding_model = data.get("embedding_model")

        if data.get("emotional_state"):
            episode.emotional_state = EmotionalState.from_dict(data["emotional_state"])

        return episode


@dataclass
class PersistentFact:
    """
    Layer 4 output: A fact extracted and consolidated from episodes.

    Facts are distinct from episodes:
    - Episodes are WHAT HAPPENED (temporal, decays)
    - Facts are WHAT WAS LEARNED (extracted, reinforced across episodes, slower decay)

    Facts get reinforced when related episodes are created, creating
    quasi-permanent knowledge through repetition.
    """

    fact_key: str  # Slug: "docker_net_f3_frustration"
    value: str  # Human-readable: "Romain had Docker networking issues on Arc"
    topic: str  # Primary topic: "docker_networking"

    # Source tracking
    source_episodes: list[str] = field(default_factory=list)
    reinforcement_count: int = 1

    # Decay and weight
    base_weight: float = 0.5
    current_weight: float = 0.5
    last_reinforced: datetime = field(default_factory=datetime.now)
    last_decay: datetime = field(default_factory=datetime.now)

    # What kinds of episodes reinforced this fact
    flag_types: list[str] = field(default_factory=list)

    def reinforce(self, episode_id: str, flag: Optional[EmotionalFlag] = None):
        """Reinforce this fact from a new episode."""
        if episode_id not in self.source_episodes:
            self.source_episodes.append(episode_id)

        self.reinforcement_count += 1
        self.last_reinforced = datetime.now()

        # Add flag type if provided
        if flag and flag.value not in self.flag_types:
            self.flag_types.append(flag.value)

        # Weight grows with reinforcement
        self.current_weight = min(
            0.95, self.current_weight + (0.05 / self.reinforcement_count)
        )

    def apply_decay(self) -> float:
        """Apply slow decay to fact weight."""
        now = datetime.now()
        days_elapsed = (now - self.last_decay).total_seconds() / 86400

        if days_elapsed < 1.0:  # Minimum 1 day between decays
            return self.current_weight

        # Facts decay very slowly (they're meant to persist)
        decay_rate = 0.995  # ~1.8% per week
        self.current_weight = max(
            0.3,  # Facts don't decay below 0.3
            self.current_weight * (decay_rate**days_elapsed),
        )
        self.last_decay = now

        return self.current_weight

    def to_dict(self) -> dict:
        return {
            "fact_key": self.fact_key,
            "value": self.value,
            "topic": self.topic,
            "source_episodes": self.source_episodes,
            "reinforcement_count": self.reinforcement_count,
            "base_weight": self.base_weight,
            "current_weight": self.current_weight,
            "last_reinforced": self.last_reinforced.isoformat(),
            "last_decay": self.last_decay.isoformat(),
            "flag_types": self.flag_types,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersistentFact":
        return cls(
            fact_key=data["fact_key"],
            value=data["value"],
            topic=data.get("topic", ""),
            source_episodes=data.get("source_episodes", []),
            reinforcement_count=data.get("reinforcement_count", 1),
            base_weight=data.get("base_weight", 0.5),
            current_weight=data.get("current_weight", 0.5),
            last_reinforced=datetime.fromisoformat(
                data.get("last_reinforced", datetime.now().isoformat())
            ),
            last_decay=datetime.fromisoformat(
                data.get("last_decay", datetime.now().isoformat())
            ),
            flag_types=data.get("flag_types", []),
        )
