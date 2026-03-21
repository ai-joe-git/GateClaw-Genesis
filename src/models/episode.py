"""Episode data structures for memory system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class EmotionalTone(Enum):
    """Emotional weight of a conversation."""

    CASUAL = "casual"  # Light, no stress
    CURIOUS = "curious"  # Exploring, learning
    FRUSTRATED = "frustrated"  # Stuck, annoyed
    EXCITED = "excited"  # Breakthrough, success
    SERIOUS = "serious"  # Important, focused
    REFLECTIVE = "reflective"  # Deep thinking
    COLLABORATIVE = "collaborative"  # Working together
    STRESSED = "stressed"  # Under pressure
    TRIUMPHANT = "triumphant"  # Problem solved!

    @classmethod
    def from_content(cls, content: str) -> "EmotionalTone":
        """
        Infer emotional tone from conversation content.
        TODO: Implement with lightweight classifier or keyword heuristics.
        """
        # Placeholder - keyword-based heuristics
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
    """Compressed summary of an episode."""

    title: str  # Brief title: "API Debugging Session"
    summary: str  # 2-3 sentence summary
    topics: list[str] = field(
        default_factory=list
    )  # ["API", "debugging", "error-handling"]
    outcomes: list[str] = field(
        default_factory=list
    )  # ["Fixed race condition", "Deployed to staging"]
    emotional_tone: EmotionalTone = EmotionalTone.CASUAL
    key_moments: list[str] = field(
        default_factory=list
    )  # Important moments in conversation
    entities: list[str] = field(default_factory=list)  # ["Python", "Docker", "FastAPI"]

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

    This represents a single conversation session with:
    - Raw conversation data (Layer 1)
    - Compressed summary (Layer 2)
    - Embedding vector (Layer 3)
    """

    # Identity
    episode_id: str = field(default_factory=lambda: str(uuid4())[:8])
    session_id: Optional[str] = None  # Links to parent session

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0

    # Content
    raw_messages: list[dict] = field(default_factory=list)  # Full conversation
    message_count: int = 0

    # Compressed summary (Layer 2)
    summary: Optional[EpisodeSummary] = None

    # Embedding (Layer 3)
    embedding: Optional[list[float]] = None
    embedding_model: Optional[str] = None  # Which model generated embedding

    # Metadata
    user_id: str = "default"
    emotional_tone: EmotionalTone = EmotionalTone.CASUAL

    # Weight - conversations have different importance
    weight: float = 1.0  # 0.0 to 1.0, determines persistence priority

    def compute_weight(self) -> float:
        """
        Calculate conversation weight based on factors:
        - Emotional intensity (frustrated/triumphant > casual)
        - Length (longer might be more important)
        - Topics (technical discussions weight higher)
        - Outcomes (solved problems weight higher)
        """
        base_weight = 0.5

        # Emotional weighting
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

        # Length bonus
        if self.message_count > 20:
            base_weight += 0.1
        if self.message_count > 50:
            base_weight += 0.1

        # Outcomes bonus
        if self.summary and self.summary.outcomes:
            base_weight += min(0.2, len(self.summary.outcomes) * 0.05)

        return min(1.0, base_weight)

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
        )

        if data.get("summary"):
            episode.summary = EpisodeSummary.from_dict(data["summary"])

        episode.embedding = data.get("embedding")
        episode.embedding_model = data.get("embedding_model")

        return episode
