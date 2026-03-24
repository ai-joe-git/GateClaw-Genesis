# GateClaw Genesis

**Episodic memory system for AI residents — 4-layer architecture with emotional weight tracking and asymmetric decay.**

---

## Overview

GateClaw Genesis is a persistent memory system that enables an AI resident to remember conversations the way humans do — with context, emotion, and temporal awareness. It goes beyond simple key-value fact storage by maintaining episodic memory that decays naturally, surfaces unresolved issues, and evolves over time through reinforcement.

**This is not a vector database wrapper.** It's a complete memory architecture designed for AI residents that need to maintain context across sessions, track emotional weight of interactions, and autonomously surface relevant past conversations.

---

## Key Concepts

### Unified Sessions Architecture

**Critical insight**: GateClaw uses unified sessions shared across ALL clients:

| Client | Behavior |
|--------|----------|
| **TUI** | Full OpenCode session with all tools |
| **Telegram** | Same OpenCode session as TUI (mirrors it) |
| **Web UI** | (Not yet GateClaw-ified) |
| **AgentMon** | Own session, writes via `saveMessage()` |

Telegram does NOT have its own message format. When you send a Telegram message, it hits the same OpenCode session API that the TUI uses. Messages from all clients go to the same `message` + `part` tables in the OpenCode database.

### Database Architecture

GateClaw uses **multiple SQLite databases** across two storage locations:

**`%LOCALAPPDATA%\gateclaw\`** (Data - persistent):
| Database | Contents |
|----------|----------|
| `opencode-local.db` | OpenCode server session data (message, part, session tables) |
| `gateclaw-local.db` | Legacy gateclaw data (gc_message, gc_fact tables) |
| `opencode.db` | Empty (legacy) |
| `gateclaw.db` | Settings only |

**`%APPDATA%\gateclaw\`** (Config - roaming):
| Database | Contents |
|----------|----------|
| `memory.db` | Genesis memory (facts, episodes, compressed_episodes) |
| `gateclaw-memory.md` | Auto-generated memory block for agent injection |

**Critical schema discovery**:
- `message.data` contains `role` (not `part.data`)
- `part.data` contains the actual content with `{type, text, ...}`
- Filter for text messages: `p.data LIKE '%"type":"text"%'` (no spaces after colons)
- `gc_message` table (legacy) has messages stored differently — not used by current architecture

### Message Loading

The watcher loads messages from `message` + `part` tables:

```python
rows = conn.execute('''
    SELECT m.id, p.data, m.time_created
    FROM message m
    JOIN part p ON m.id = p.message_id
    WHERE m.session_id = ?
    AND p.data LIKE '%"type":"text"%'
    ORDER BY m.time_created
''', (session_id,))
```

For each row:
- `role` = `json.loads(m.data).get('role')`
- `text` = `json.loads(p.data).get('text')`
- `time` = `m.time_created`

---

## Architecture

### 4-Layer Memory System

```
┌─────────────────────────────────────────────────────────────┐
│                     LAYER 1: RAW BUFFER                     │
│  Lossless storage of recent conversations (~10 episodes)   │
│  SQLite: raw_episodes table                                │
│  Purpose: "What did we discuss yesterday?"                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 2: COMPRESSION                      │
│  Summarize each episode into structured record              │
│  - Title, summary, topics, outcomes                         │
│  - Emotional tone inference                                 │
│  - Key moments extraction                                   │
│  LLM: Nemotron-3-Nano-4B via llama-swap                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 3: SEMANTIC SEARCH                   │
│  Vector embeddings for natural language retrieval            │
│  Model: qwen3-embedding:0.6b via Ollama (port 11434)      │
│  Dimension: 1024                                            │
│  Storage: SQLite with cosine similarity (no external DB)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  LAYER 4: CONSOLIDATION                     │
│  Extract persistent facts with emotional weight               │
│  Track patterns across episodes                             │
│  Apply asymmetric decay curves                              │
│  LLM: Claude-4.6-Opus-2B via llama-swap                   │
└─────────────────────────────────────────────────────────────┘
```

### Emotional Weight System

**Emotional Flags** (boolean, pattern-detected):
- `FRUSTRATION` — user expressed frustration
- `BREAKTHROUGH` — problem solved or insight gained
- `SOLVED` — explicitly marked as resolved
- `REVISITED` — user returned to previous topic
- `CONFUSION` — user asked the same question multiple times
- `EXPLICIT_PRAISE` — user said thanks or complimented
- `COLLABORATION` — active joint problem-solving

**Flag Confidence**:
- `EXPLICIT` — direct signal ("YES! it works!", "I give up")
- `INFERRED` — pattern-based (revisit count, tone analysis)

**Decay Curves** (asymmetric):

| Flag | Confidence | Decay Rate | Behavior |
|------|------------|------------|----------|
| BREAKTHROUGH | EXPLICIT | 0.98/week | Persists longest |
| FRUSTRATION | EXPLICIT | 0.85/week | Fades fast (you move on) |
| FRUSTRATION | INFERRED | 0.75/week | Fades faster (might be noise) |
| SOLVED | EXPLICIT | 0.97/week | Fades normally |
| REVISITED | any | 1.01/week | *Grows* (reinforcement) |

**Unresolved Floor**: Episodes with `FRUSTRATION` flag that are not marked `SOLVED` never decay below 0.7 weight. They keep surfacing until resolved.

---

## Installation

### Prerequisites

- Python 3.11+
- **Ollama** (port 11434) with `qwen3-embedding:0.6b` model
- **Llama-swap** (port 8888) with text generation model

### Setup

```bash
cd GateClaw_Genesis

# Install dependencies
pip install requests watchdog

# Pull embedding model (if not already)
ollama pull qwen3-embedding:0.6b
```

### Verify Llama-swap Models

```bash
curl http://localhost:8888/v1/models
```

Required models:
- `Nemotron-3-Nano-4B` — Layer 2 compression (summarization)
- `Claude-4.6-Opus-2B` — Layer 4 consolidation (fact extraction)

---

## Usage

### Genesis Watcher (Recommended)

The watcher runs in the background and automatically monitors OpenCode databases, consolidating sessions after 30 seconds of inactivity:

```bash
python genesis_watcher.py
```

The watcher:
1. Scans all OpenCode databases on the machine
2. Tracks `session_state` in memory (persists while running)
3. After 30s debounce, starts consolidation thread
4. Semaphore limits concurrent LLM calls to 2 (prevents llama-swap overload)

**Known limitations**:
- State is in-memory only (lost on restart)
- First run needs 30s to initialize state per session
- Large sessions (1000+ messages) take 3-5+ minutes to consolidate

### Manual Integration

```python
from core.genesis import Genesis
from memory_injector import write_memory_block

# Initialize
genesis = Genesis(
    db_path=os.path.expandvars(r"%APPDATA%\gateclaw\memory.db"),
    vectors_path=os.path.expandvars(r"%APPDATA%\gateclaw\vectors"),
    buffer_size=10,
    embedding_model="qwen3-embedding:0.6b",
    llm_model="Claude-4.6-Opus-2B",
)

# Start episode
episode = genesis.begin_episode(user_id="telegram:123456789")

# Add messages from your client
episode = genesis.add_message(episode, "user", "Hello!")
episode = genesis.add_message(episode, "assistant", "Hi, how can I help?")

# End and remember (triggers all 4 layers)
episode = genesis.end_episode(episode)
genesis.remember(episode)

# Wait for async consolidation
genesis.wait_consolidation(episode.episode_id, timeout=120.0)

# Write memory block for agent injection
write_memory_block(variant="telegram")  # Short, natural sentences
write_memory_block(variant="standard")   # Full markdown

genesis.close()
```

### Retrieving Memory

```python
# Semantic search
results = genesis.recall("docker networking issue", limit=5)

# With unresolved issues surfaced
results = genesis.recall("help", include_unresolved=True)

# Get persistent facts by topic
facts = genesis.recall_facts(topic="docker", limit=5)

# Full context for LLM prompt injection
context = genesis.get_context(
    query="current conversation topic",
    max_tokens=1000,
    include_unresolved=True,
    include_facts=True
)
```

---

## Agent Memory Injection

Genesis auto-generates a memory block that OpenCode loads into every session:

**Output**: `%APPDATA%\gateclaw\agent\gateclaw-memory.md`

**Variants**:
- `telegram` — Short, natural sentences (max 3 per section, no markdown)
- `standard` — Full markdown with headers

**Content**:
```markdown
<!-- Generated by GateClaw Genesis 2026-03-23T19:48:19.158277 -->
<!-- DO NOT EDIT MANUALLY - This file is auto-generated -->

[MEMORY CONTEXT]
[PERSISTENT FACTS]
- User has unresolved docker networking issue (weight: 0.85)
- Recent breakthrough: fixed the port forwarding bug

[RELEVANT CONVERSATIONS]
- Docker networking debug session (2026-03-22)

[UNRESOLVED - may be relevant]
- Docker networking issues on Windows/WSL (weight: 0.72)

[END MEMORY]
```

OpenCode automatically loads all `.md` files from the `agent/` folder via glob pattern `{agent,agents}/**/*.md`. **No code changes required in the OpenCode fork.**

---

## Watcher Troubleshooting

### Watcher exits after startup without consolidating

**Bug**: First-run deadlock when `last_id == ""` (session never seen before).

Fixed by adding explicit initialization branch:
```python
elif last_id == "":
    # First encounter — initialize state but don't consolidate
    session_state[key] = {
        "last_activity": now,
        "last_id": latest_id,
        "use_gc": use_gc,
    }
```

### Consolidation threads never start / LLM hangs

**Bug**: 60+ sessions fire debounce simultaneously, overwhelming llama-swap with concurrent hot-swap requests.

Fixed with semaphore limiting concurrent LLM calls:
```python
LLM_SEMAPHORE = threading.Semaphore(2)

def consolidate_session(session_id, source_db, use_gc_message=False):
    with LLM_SEMAPHORE:
        # ... consolidation code ...
```

### Session consolidates every poll cycle (duplicates)

**Expected behavior**: On restart, state is lost, so all sessions re-fire debounce. This is by design for the initial setup. The deduplication logic prevents true duplicates, but session state should be persisted for production use.

### Messages not being loaded

**Checkpoints**:
1. Filter format: must be `p.data LIKE '%"type":"text"%'` (no spaces after colons)
2. Role is in `m.data`, not `p.data`:
   ```python
   role = json.loads(m.data).get('role', 'user')
   text = json.loads(p.data).get('text', '')
   ```
3. Session must have `m.data` with role field

---

## Technical Details

### LLM Timeouts

**Critical**: Llama-swap model hot-swapping takes 15-30 seconds per call. All LLM calls must use 600s timeout:

```python
response = requests.post(
    self.LLAMA_SWAP_URL,
    json={...},
    timeout=600,  # NOT 30, NOT 180
)
```

Locations requiring 600s timeout:
- `src/memory/episodes.py` — Layer 2 compression
- `src/memory/consolidation.py` — Layer 4 extraction
- `re_consolidate.py` — One-shot script

### Embedding API (Ollama)

```python
response = requests.post(
    "http://localhost:11434/api/embed",
    json={"model": "qwen3-embedding:0.6b", "input": text},
    timeout=30,
)
embeddings = response.json()["embeddings"]
```

### LLM API (Llama-swap)

```python
response = requests.post(
    "http://localhost:8888/v1/chat/completions",
    json={
        "model": "Claude-4.6-Opus-2B",
        "messages": [...],
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
    },
    timeout=600,
)
```

### Decay Algorithm

```python
# Per-flag decay rate applied exponentially
new_weight = current_weight * (decay_rate ** days_elapsed)

# Floor for unresolved frustration
if unresolved and has_frustration_flag:
    new_weight = max(0.7, new_weight)
```

---

## File Structure

```
GateClaw_Genesis/
├── src/
│   ├── core/
│   │   └── genesis.py          # Main orchestrator + all 4 layers
│   ├── memory/
│   │   ├── buffer.py          # Layer 1 - raw_episodes table
│   │   ├── episodes.py        # Layer 2 - compression (LLM summarization)
│   │   ├── embeddings.py       # Layer 3 - vector storage + search
│   │   ├── consolidation.py    # Layer 4 - fact extraction + decay
│   │   └── recall.py           # Retrieval logic + context generation
│   └── models/
│       └── episode.py          # EpisodeSummary, EmotionalTone, PersistentFact
├── memory_injector.py           # Writes gateclaw-memory.md for agent injection
├── genesis_watcher.py           # Background session monitor + consolidation
├── re_consolidate.py            # One-shot consolidation script
├── genesis_watcher.log          # Watcher output (if redirected)
└── README.md
```

---

## Database Schema

### memory.db Tables

**raw_episodes**
| Column | Type | Description |
|--------|------|-------------|
| episode_id | TEXT PK | Unique episode ID |
| user_id | TEXT | Session identifier |
| created_at | TEXT | ISO timestamp |
| ended_at | TEXT | ISO timestamp |
| message_count | INTEGER | Number of messages |
| raw_messages | TEXT | JSON array of messages |
| summary_id | TEXT FK | Link to compressed_episodes |

**compressed_episodes**
| Column | Type | Description |
|--------|------|-------------|
| episode_id | TEXT PK | Links to raw_episodes |
| title | TEXT | One-line description |
| summary | TEXT | 2-3 sentence summary |
| topics | TEXT | JSON array of topics |
| outcomes | TEXT | JSON array of outcomes |
| emotional_tone | TEXT | CASUAL/CURIOUS/FRUSTRATED/etc |
| key_moments | TEXT | JSON array of key moments |
| entities | TEXT | JSON array of mentioned entities |

**persistent_facts**
| Column | Type | Description |
|--------|------|-------------|
| fact_key | TEXT PK | Semantic key (e.g., "docker_networking_frustration") |
| value | TEXT | Fact content |
| topic | TEXT | Primary topic category |
| current_weight | REAL | Decayed weight (0.0-1.0) |
| reinforcement_count | INTEGER | Times reinforced |
| last_reinforced | TEXT | ISO timestamp |
| created_at | TEXT | ISO timestamp |
| source_episodes | TEXT | JSON array of episode IDs |

---

## Philosophy

Memory should have weight. Not everything matters equally.

A breakthrough solution should persist. Unresolved frustration should keep surfacing until addressed. A casual "hello" should fade. The relationship should evolve.

This is not about storing more data. It's about storing *meaning* — and letting that meaning decay naturally based on what actually mattered.

---

## Changelog

### 2026-03-23 — Critical Architecture Discoveries

**Unified Sessions**: Telegram and TUI share the same OpenCode sessions. No "Telegram-specific" messages. All clients hit the same session API.

**Database Discovery**: Messages stored in `opencode-local.db` (LOCALAPPDATA) `message` + `part` tables. Legacy `gc_message` table not used by current architecture.

**Role Location Bug**: `role` field is in `message.data` (JSON column), NOT `part.data`. Content is in `part.data` with `{type, text}` structure.

**LLM Concurrency Bug**: 60+ simultaneous consolidation requests hang llama-swap. Fixed with `Semaphore(2)`.

**First-Run Deadlock Bug**: Sessions with `last_id == ""` never initialized, causing infinite re-queue. Fixed with explicit initialization branch.

**Message Filter Format**: Must use `"type":"text"` (no spaces) to match compact JSON in database.

---

*Memory begins here.*
