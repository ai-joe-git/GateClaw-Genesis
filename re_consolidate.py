#!/usr/bin/env python3
"""
Re-consolidate: load raw messages, LLM-summarize, then extract facts.
This rebuilds both Layer 2 (summaries) and Layer 4 (facts) properly.
"""

import os
import sqlite3
import json
import sys
import requests
from pathlib import Path
from datetime import datetime

DB_PATH = os.path.expandvars(r"%APPDATA%\gateclaw\memory.db")
GENESIS_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(GENESIS_SRC))

LLAMA_SWAP_URL = "http://localhost:8888/v1/chat/completions"
SUMMARIZE_MODEL = "Nemotron-3-Nano-4B"  # Layer 2 — summarization
FACTS_MODEL = "Claude-4.6-Opus-4B"  # Layer 4 — fact extraction


def try_parse_json(content: str) -> tuple:
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

    # Strategy 1: find balanced {...} or [...]
    for start_char in ["{", "["]:
        # Find first occurrence
        json_start = content.find(start_char)
        if json_start < 0:
            continue

        # Find the last matching close brace
        if start_char == "{":
            end_char = "}"
        else:
            end_char = "]"

        # Find potential end position
        json_end = content.rfind(end_char)
        if json_end <= json_start:
            continue

        # Try progressively smaller slices from the end
        for end_pos in range(json_end, json_start - 1, -1):
            candidate = content[json_start : end_pos + 1]
            try:
                parsed = json.loads(candidate)
                return parsed, None
            except json.JSONDecodeError:
                continue

        # If no valid JSON found with this char, try the other
        continue

    return None, "No valid JSON found in response"


def llm_summarize(messages: list) -> dict:
    """Call Claude-4.6-Opus-2B to summarize raw messages."""
    if not messages:
        return None

    conversation = "\n".join(
        [
            f"{m.get('role', 'unknown').upper()}: {m.get('content', '')[:500]}"
            for m in messages
            if m.get("content")
        ]
    )

    prompt = f"""You are a memory summarization system. Return ONLY valid JSON. No explanation. No markdown. No code blocks.

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
{conversation[:3000]}"""

    try:
        response = requests.post(
            LLAMA_SWAP_URL,
            json={
                "model": SUMMARIZE_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a memory summarization system. Return valid JSON only. No explanation. No markdown. No text outside the JSON.",
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

            # Try to find JSON object or array in the response
            # Look for first { or [ and last } or ]
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            else:
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    content = content[json_start:json_end]

            content = content.strip()
            parsed, err = try_parse_json(content)
            if parsed is not None:
                if isinstance(parsed, dict):
                    return parsed
                elif isinstance(parsed, list):
                    # Wrap list in dict if needed
                    return {
                        "title": "Conversation",
                        "summary": " ".join(str(x) for x in parsed[:3]),
                        "topics": [],
                        "outcome": "unknown",
                        "emotional_tone": "neutral",
                    }
            else:
                print(f"  [LLM] JSON parse failed: {err}")
                return None
    except Exception as e:
        print(f"  [LLM] Failed: {e}")
    return None


def extract_facts_llm(summary: dict) -> list:
    """Extract facts using LLM from summary."""
    if not summary:
        return []

    prompt = f"""Extract persistent facts about the user from this conversation summary.
Return a JSON array only. No explanation. No markdown. No code blocks. No intro text.
If no clear facts, return empty array: []

Facts should be specific and useful for future context:
- Technical preferences (languages, tools, frameworks)
- Project names and what they do
- Problems encountered and whether resolved
- Decisions made
- Personal context (location, hardware, workflow)

Output format: [{{"key": "fact_name", "value": "fact_value", "topic": "topic"}}]

Summary: {summary.get("summary", "")}
Topics: {", ".join(summary.get("topics", []))}
Outcome: {summary.get("outcome", "unknown")}"""

    try:
        response = requests.post(
            LLAMA_SWAP_URL,
            json={
                "model": FACTS_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You extract facts from conversations. Return ONLY a JSON array. No explanation. No markdown. No text outside the array.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "max_tokens": 600,
            },
            timeout=600,
        )

        if response.status_code == 200:
            try:
                resp_data = response.json()
                msg = resp_data.get("choices", [{}])[0].get("message", {})
                content = (
                    msg.get("content", "").strip()
                    or msg.get("reasoning_content", "").strip()
                )
            except Exception as e:
                print(f"  [LLM facts] JSON parse error: {e}")
                content = ""

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
            content = content.strip()

            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass

            parsed, err = try_parse_json(content)
            if parsed is not None:
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    return [parsed]
            else:
                print(f"  [LLM facts] JSON parse failed: {err}")
            return []
            if parsed is not None:
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict):
                    # Single fact dict - wrap in list
                    return [parsed]
            else:
                print(f"  [LLM facts] JSON parse failed: {err}")
            return []
    except Exception as e:
        print(f"  [LLM facts] Failed: {e}")
    return []


def store_fact(conn, fact: dict, episode_id: str, base_weight: float = 0.5):
    """Store a single fact."""
    key = fact.get("key", "unknown")
    value = fact.get("value", "")

    if not key or key == "unknown" or len(value) < 3:
        return
    if any(c in key for c in ["#", "*", "_", "[", "]", "{", "}", "|", "\\"]):
        return
    if (
        value.startswith("#")
        or "## what i" in value.lower()
        or "*won't*" in value.lower()
    ):
        return

    topic = fact.get("topic", key)

    # Check if exists
    existing = conn.execute(
        "SELECT fact_key, reinforcement_count FROM persistent_facts WHERE fact_key = ?",
        (key,),
    ).fetchone()

    if existing:
        count = existing[1] + 1
        weight = min(0.95, base_weight + (0.05 / count))
        conn.execute(
            "UPDATE persistent_facts SET reinforcement_count = ?, current_weight = ?, last_reinforced = ? WHERE fact_key = ?",
            (count, weight, datetime.now().isoformat(), key),
        )
    else:
        conn.execute(
            """INSERT INTO persistent_facts 
               (fact_key, value, topic, source_episodes_json, reinforcement_count,
                base_weight, current_weight, last_reinforced, last_decay, flag_types_json, created_at)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, '[]', ?)""",
            (
                key,
                value,
                topic,
                episode_id + ",",
                base_weight,
                base_weight,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
            ),
        )

    # Link
    conn.execute(
        "INSERT OR IGNORE INTO episode_fact_links (episode_id, fact_key) VALUES (?, ?)",
        (episode_id, key),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-consolidate Genesis memory from source databases"
    )
    parser.add_argument(
        "--source-db",
        default=None,
        help="Specific source DB path to process (default: all)",
    )
    parser.add_argument("--memory-db", default=DB_PATH, help="Genesis memory.db path")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing facts before re-consolidating",
    )
    args = parser.parse_args()

    print("[Genesis] Starting full re-consolidation with LLM...")
    print(f"[Genesis] Memory DB: {args.memory_db}")

    memory_conn = sqlite3.connect(args.memory_db)

    if args.clear:
        memory_conn.execute("DELETE FROM persistent_facts")
        memory_conn.execute("DELETE FROM episode_fact_links")
        memory_conn.commit()
        print("[Genesis] Cleared existing facts")

    source_dbs = []
    if args.source_db:
        if os.path.exists(args.source_db):
            source_dbs = [(args.source_db, os.path.basename(args.source_db))]
        else:
            print(f"[Genesis] Source DB not found: {args.source_db}")
            return
    else:
        local_db = os.path.expandvars(r"%LOCALAPPDATA%\gateclaw\gateclaw-local.db")
        opencode_local = os.path.expandvars(
            r"%LOCALAPPDATA%\gateclaw\opencode-local.db"
        )
        opencode_main = os.path.expandvars(
            r"%USERPROFILE%\.local\share\opencode\opencode.db"
        )
        roaming_db = os.path.expandvars(r"%APPDATA%\gateclaw\opencode-local.db")
        for db, label in [
            (local_db, "gateclaw-local"),
            (opencode_local, "opencode-local"),
            (opencode_main, "opencode-main"),
            (roaming_db, "roaming"),
        ]:
            if os.path.exists(db):
                source_dbs.append((db, label))

    if not source_dbs:
        print("[Genesis] No source databases found")
        return

    total_facts = 0
    total_episodes = 0

    for source_db, source_label in source_dbs:
        print(f"\n[Genesis] Processing source: {source_label} ({source_db})")

        try:
            uri_path = source_db.replace("\\", "/")
            conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=30.0)
            conn.row_factory = sqlite3.Row
        except Exception as e:
            print(f"  [Genesis] Could not open {source_db}: {e}")
            continue

        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

        # Determine which table to use for messages
        # Prefer gc_message if it has data, otherwise use message table
        use_gc = False
        if "gc_message" in tables:
            gc_count = conn.execute("SELECT COUNT(*) FROM gc_message").fetchone()[0]
            use_gc = gc_count > 0

        if use_gc:
            rows = conn.execute(
                "SELECT DISTINCT session_key FROM gc_message ORDER BY time_created"
            ).fetchall()
            session_ids = [r[0] for r in rows]
            print(f"  [Genesis] {len(session_ids)} Telegram sessions")
        else:
            rows = conn.execute(
                "SELECT DISTINCT session_id FROM message ORDER BY time_created"
            ).fetchall()
            session_ids = [r[0] for r in rows]
            print(f"  [Genesis] {len(session_ids)} OpenCode sessions")

        for session_id in session_ids:
            if use_gc:
                msg_rows = conn.execute(
                    "SELECT role, content, time_created FROM gc_message WHERE session_key = ? ORDER BY time_created",
                    (session_id,),
                ).fetchall()
                messages = [
                    {"role": r[0], "content": r[1]}
                    for r in msg_rows
                    if r[1] and len(r[1]) > 1
                ]
            else:
                msg_rows = conn.execute(
                    """SELECT m.id, m.data, p.data FROM message m JOIN part p ON m.id = p.message_id
                       WHERE m.session_id = ? AND p.data LIKE '%"type":"text"%' ORDER BY m.time_created""",
                    (session_id,),
                ).fetchall()
                messages = []
                for rid, msg_data, part_data in msg_rows:
                    try:
                        md = json.loads(msg_data) if msg_data else {}
                        pd = json.loads(part_data) if part_data else {}
                        if pd.get("text"):
                            messages.append(
                                {
                                    "role": md.get("role", "assistant"),
                                    "content": pd["text"],
                                }
                            )
                    except:
                        continue

            if not messages:
                continue

            episode_id = f"{source_label}_{session_id[:16]}"

            already_consolidated = memory_conn.execute(
                "SELECT COUNT(*) FROM compressed_episodes WHERE episode_id = ?",
                (episode_id,),
            ).fetchone()[0]

            if already_consolidated:
                continue

            all_facts = []
            if len(messages) > 100:
                print(
                    f"  [Genesis] Chunking oversized session {episode_id} ({len(messages)} messages)"
                )
                for i in range(0, len(messages), 150):
                    chunk = messages[i : i + 150]
                    print(f"    Chunk {i // 150 + 1}: {len(chunk)} messages")
                    chunk_summary = llm_summarize(chunk)
                    if chunk_summary:
                        chunk_facts = extract_facts_llm(chunk_summary)
                        all_facts.extend(chunk_facts)
                    else:
                        chunk_facts = []
                    for fact in chunk_facts:
                        store_fact(memory_conn, fact, episode_id, base_weight=0.6)
                    memory_conn.commit()
                if all_facts:
                    merged_summary = {
                        "title": f"Chunked session ({len(messages)} messages)",
                        "summary": f"Large session split into {len(messages) // 80 + 1} chunks",
                        "topics": [],
                        "outcome": "unknown",
                        "emotional_tone": "neutral",
                    }
                    memory_conn.execute(
                        """INSERT OR REPLACE INTO compressed_episodes 
                           (episode_id, user_id, summary_json, created_at, weight)
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            episode_id,
                            session_id[:20],
                            json.dumps(merged_summary),
                            datetime.now().isoformat(),
                            1.0,
                        ),
                    )
                    memory_conn.commit()
                    print(f"    Facts from chunks: {len(all_facts)}")
                continue

            print(f"  [Genesis] Episode: {episode_id} ({len(messages)} messages)")

            summary = llm_summarize(messages)
            if not summary:
                summary = {
                    "title": f"{source_label} session",
                    "summary": "",
                    "topics": [],
                    "outcome": "unknown",
                    "emotional_tone": "neutral",
                }

            title = summary.get("title", "Conversation")[:60]
            print(f"    Summary: {title}")

            memory_conn.execute(
                """INSERT OR REPLACE INTO compressed_episodes 
                   (episode_id, user_id, summary_json, created_at, weight)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    episode_id,
                    session_id[:20],
                    json.dumps(summary),
                    datetime.now().isoformat(),
                    1.0,
                ),
            )
            memory_conn.commit()

            facts = extract_facts_llm(summary)
            print(f"    Facts: {len(facts)}")

            for fact in facts:
                store_fact(memory_conn, fact, episode_id, base_weight=0.6)
            memory_conn.commit()

            total_facts += len(facts)
            total_episodes += 1

        conn.close()

    memory_conn.close()

    print(f"\n[Genesis] Done — {total_facts} facts from {total_episodes} episodes")

    sys.path.insert(0, str(GENESIS_SRC))
    from memory_injector import write_memory_block

    write_memory_block(variant="telegram")
    write_memory_block(variant="standard")
    print("[Genesis] Memory blocks written")


if __name__ == "__main__":
    main()
