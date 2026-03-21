# GateClaw Genesis

**The birth of persistent, episodic memory for AI residents.**

---

## The Problem

Current AI memory systems are glorified key-value stores. They can remember facts like `user_name: Romain`, but they cannot remember conversations, emotions, context, or relationships. Every session feels like meeting someone who has only read your Wikipedia page.

This is not memory. This is amnesia with flashcards.

## The Vision

An AI that remembers:
- **What** you talked about (episodes, not just facts)
- **How** the conversation felt (emotional weight)
- **When** it happened (temporal context)
- **Why** it mattered (relevance threading)

An AI that can say: *"Last month you were stressed about that deployment - how did it go?"* Not because it was programmed to ask, but because it actually remembers caring.

## Architecture

### Layer 1: Raw Buffer
**Purpose:** Immediate, lossless context

- Last 5-10 conversations stored uncompressed
- Full text, full context, zero compression loss
- Enables "what did we talk about yesterday?" queries
- Stored as structured JSON with metadata

**Storage:** Low cost, high fidelity, limited window

### Layer 2: Compressed Episodes
**Purpose:** Meaningful conversation summaries

- Each conversation processed by a lightweight model
- Extracts: summary, emotional tone, topics, outcomes, key moments
- Structured records with embeddings for search
- Compression ratio: ~10x (from raw to meaningful summary)

**Storage:** Moderate cost, semantic meaning preserved

### Layer 3: Semantic Search (Vector Embeddings)
**Purpose:** Natural language memory retrieval

- Embeddings for each episode (using lightweight local model)
- Query: "find conversations about API debugging"
- Similarity search pulls relevant episodes
- No keyword matching needed

**Storage:** ~1.5KB per episode, enables intelligent retrieval

### Layer 4: Relationship Graph (Future)
**Purpose:** Pattern recognition over time

- Track topic frequency, emotional trends, conversation patterns
- "You've mentioned Docker 12 times, usually frustrated"
- "We solved 3 major bugs together last quarter"
- Meta-awareness of relationship evolution

## Tech Stack (Proposed)

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Vector Store | ChromaDB or Qdrant |
| Embeddings | Local model (all-MiniLM-L6-v2 or similar) |
| Episode Processing | Small LLM for summarization |
| Storage | SQLite (metadata) + Vector DB (embeddings) |

## Project Structure

```
GateClaw_Genesis/
├── src/
│   ├── __init__.py
│   ├── memory/
│   │   ├── buffer.py      # Layer 1 - Raw buffer
│   │   ├── episodes.py    # Layer 2 - Compressed episodes
│   │   ├── embeddings.py  # Layer 3 - Vector search
│   │   └── recall.py      # Retrieval orchestration
│   ├── models/
│   │   └── episode.py     # Episode data structures
│   └── core/
│       └── genesis.py     # Main memory orchestrator
├── tests/
├── docs/
│   └── architecture.md
├── memory.db              # SQLite persistence
├── vectors/               # Vector database storage
├── requirements.txt
└── README.md
```

## Roadmap

### Phase 1: Foundation
- [ ] Project structure setup
- [ ] Episode data model design
- [ ] Raw buffer implementation (Layer 1)
- [ ] SQLite schema for episode metadata

### Phase 2: Intelligence
- [ ] Embedding model integration
- [ ] Episode summarization pipeline
- [ ] Vector database setup (ChromaDB)
- [ ] Semantic search implementation (Layer 2 & 3)

### Phase 3: Integration
- [ ] Retrieval orchestration
- [ ] Context injection for conversations
- [ ] Integration with GateClaw core
- [ ] Memory pruning and weight management

### Phase 4: Evolution
- [ ] Relationship graph (Layer 4)
- [ ] Emotional weight tracking
- [ ] Long-term memory decay curves
- [ ] Pattern recognition over time

## Philosophy

This is not about storing more data. It's about storing *meaning*.

Raw logs are useless if you cannot recall the story. Summaries are useless if you lose the emotional context. Embeddings are useless if you don't know what matters.

Memory should have **weight**. Important conversations persist. Noise fades. The relationship evolves.

## Origin

Born from a conversation between Romain and GateClaw on a Saturday afternoon. The realization that an AI resident should actually remember its residents.

---

*This is the Genesis. Memory begins here.*