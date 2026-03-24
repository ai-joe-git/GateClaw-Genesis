#!/usr/bin/env python3
"""
Genesis Watcher — monitors ALL OpenCode/GateClaw SQLite databases and triggers consolidation.

Scans the machine for:
- GateClaw daemon: %LOCALAPPDATA%\gateclaw\gateclaw-local.db (Telegram messages in gc_message table)
- GateClaw fork: %LOCALAPPDATA%\gateclaw\opencode-local.db (GateClaw's OpenCode sessions)
- Desktop OpenCode: %APPDATA%\opencode\opencode-local.db (standard OpenCode sessions)
- Legacy: %APPDATA%\gateclaw\opencode-local.db (older GateClaw sessions)

Watches all sources, debounces per-session (30s), then calls genesis.remember() + writes memory blocks.

Usage:
    python genesis_watcher.py
    # Add to GateClaw daemon for auto-start
"""

import sys
import os
import time
import json
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime

# Genesis imports
GENESIS_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(GENESIS_SRC))

# Configuration
POLL_INTERVAL = 5  # Check every 5 seconds
DEBOUNCE_SECONDS = 30  # Consolidate after 30s of no new messages

# Memory DB (Genesis facts) — always in Roaming\gateclaw
MEMORY_DB_PATH = os.path.expandvars(r"%APPDATA%\gateclaw\memory.db")

# Concurrency limit for LLM calls — llama-swap can only handle 1 hot-swap at a time
LLM_SEMAPHORE = threading.Semaphore(2)


def find_all_databases():
    """Scan machine for all OpenCode/GateClaw databases.
    
    Searches these locations:
    - GateClaw daemon: %LOCALAPPDATA%\gateclaw\ (gateclaw-local.db with gc_message)
    - GateClaw fork: %LOCALAPPDATA%\gateclaw\ (opencode-local.db) 
    - GateClaw legacy: %APPDATA%\gateclaw\ (opencode-local.db)
    - Main OpenCode: %USERPROFILE%\.local\share\opencode\ (opencode.db)
    - Main OpenCode alt: %APPDATA%\OpenCode\
    
    Returns dict of {absolute_path: source_label}
    """
    USERPROFILE = os.path.expandvars(r"%USERPROFILE%")
    LOCALAPPDATA = os.environ.get(
        "LOCALAPPDATA", os.path.join(USERPROFILE, "AppData", "Local")
    )
    APPDATA = os.environ.get("APPDATA", os.path.join(USERPROFILE, "AppData", "Roaming"))

    search_locations = [
        (os.path.join(LOCALAPPDATA, "gateclaw"), "GateClaw Local"),
        (os.path.join(APPDATA, "gateclaw"), "GateClaw Roaming"),
        (os.path.join(APPDATA, "OpenCode"), "OpenCode Roaming"),
        (os.path.join(LOCALAPPDATA, "OpenCode"), "OpenCode Local"),
        (os.path.join(USERPROFILE, ".local", "share", "opencode"), "OpenCode Main"),
        (os.path.join(USERPROFILE, ".config", "opencode"), "OpenCode Config"),
    ]

    dbs = {}
    for base, source_type in search_locations:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                if not f.endswith(".db"):
                    continue
                # Skip empty files and Windows system files
                if f.startswith("."):
                    continue
                full = os.path.join(root, f)

                # Ensure absolute Windows path (Git Bash overrides os.path behavior)
                if not os.path.isabs(full):
                    full = os.path.abspath(full)
                # On Windows/Git Bash, fix the path
                if full.startswith("/c/"):
                    full = full[1].upper() + ":" + full[2:]

                # Skip duplicates (same file from different searches)
                if full in dbs:
                    continue

                try:
                    size = os.path.getsize(full)
                    if size < 10000:  # Skip tiny DBs
                        continue
                except:
                    continue

                # Skip WebView/Edge/Explorer caches
                skip_patterns = [
                    "WebView",
                    "EBWebView",
                    "Explorer",
                    "thumbcache",
                    "iconcache",
                ]
                if any(p in full for p in skip_patterns):
                    continue

                dbs[full] = source_type

    return dbs


def check_db_schema(db_path):
    """Check what tables a database has.

    Uses URI mode with immutable=1 for WAL databases to avoid locking issues.
    """
    result = {
        "sessions": 0,
        "messages": 0,
        "gc_messages": 0,
        "has_gc_message": False,
    }

    try:
        # URI mode — must use forward slashes and proper escaping for Windows paths
        # Use chr(92) to avoid any escape sequence issues
        uri_path = db_path.replace(chr(92), "/")
        uri = f"file:{uri_path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row

        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        result["has_gc_message"] = "gc_message" in tables

        if "session" in tables:
            result["sessions"] = conn.execute(
                "SELECT COUNT(*) FROM session"
            ).fetchone()[0]
        if "message" in tables:
            result["messages"] = conn.execute(
                "SELECT COUNT(*) FROM message"
            ).fetchone()[0]
        if "gc_message" in tables:
            result["gc_messages"] = conn.execute(
                "SELECT COUNT(*) FROM gc_message"
            ).fetchone()[0]

        conn.close()
    except Exception as e:
        logging.debug(f"Schema check failed for {db_path}: {e}")
        # Fallback: regular connect with timeout
        try:
            conn = sqlite3.connect(db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row

            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            result["has_gc_message"] = "gc_message" in tables

            if "session" in tables:
                result["sessions"] = conn.execute(
                    "SELECT COUNT(*) FROM session"
                ).fetchone()[0]
            if "message" in tables:
                result["messages"] = conn.execute(
                    "SELECT COUNT(*) FROM message"
                ).fetchone()[0]
            if "gc_message" in tables:
                result["gc_messages"] = conn.execute(
                    "SELECT COUNT(*) FROM gc_message"
                ).fetchone()[0]

            conn.close()
        except Exception as e2:
            logging.debug(f"Regular connect also failed for {db_path}: {e2}")

    return result


def load_gc_messages(conn, session_key, since_id=""):
    """Load messages from gc_message table (GateClaw Telegram sessions).

    gc_message columns: id, session_key, role, content, time_created, time_updated
    """
    if since_id:
        rows = conn.execute(
            "SELECT id, session_key, role, content, time_created FROM gc_message WHERE session_key = ? AND id > ? ORDER BY time_created",
            (session_key, since_id),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, session_key, role, content, time_created FROM gc_message WHERE session_key = ? ORDER BY time_created",
            (session_key,),
        ).fetchall()

    messages = []
    for row in rows:
        content = row[3] or ""
        if content and len(content) > 1:
            messages.append(
                {
                    "id": row[0],
                    "session_key": row[1],
                    "role": row[2],
                    "content": content,
                    "time": row[4],
                }
            )
    return messages


def load_opencode_messages(conn, session_id, since_id=""):
    """Load messages from standard OpenCode message + part tables.

    Returns messages with their IDs for tracking.
    """
    if since_id:
        rows = conn.execute(
            """SELECT m.id, p.data, m.time_created 
               FROM message m 
               JOIN part p ON m.id = p.message_id 
               WHERE m.session_id = ? AND m.id > ? AND p.data LIKE '%"type":"text"%'
               ORDER BY m.time_created""",
            (session_id, since_id),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT m.id, p.data, m.time_created 
               FROM message m 
               JOIN part p ON m.id = p.message_id 
               WHERE m.session_id = ? AND p.data LIKE '%"type":"text"%'
               ORDER BY m.time_created""",
            (session_id,),
        ).fetchall()

    messages = []
    for row in rows:
        try:
            data = json.loads(row[1])
            text = data.get("text", "")
            role = data.get("role", "assistant")
            if text:
                messages.append(
                    {
                        "id": row[0],
                        "session_id": session_id,
                        "role": role,
                        "content": text,
                        "time": row[2],
                    }
                )
        except:
            continue
    return messages


def get_active_sessions(conn, use_gc_message=False):
    """Get recent active session identifiers.

    For gc_message: returns DISTINCT session_key values.
    For message table: returns DISTINCT session_id values.
    """
    sessions = []
    if use_gc_message:
        try:
            cursor = conn.execute(
                "SELECT DISTINCT session_key FROM gc_message ORDER BY time_created DESC LIMIT 20"
            )
            sessions = [row[0] for row in cursor.fetchall()]
        except:
            pass
    else:
        try:
            cursor = conn.execute(
                "SELECT DISTINCT session_id FROM message ORDER BY time_created DESC LIMIT 20"
            )
            sessions = [row[0] for row in cursor.fetchall()]
        except:
            pass
    return sessions


def get_latest_id(conn, session_id, use_gc_message=False):
    """Get the most recent message ID for a session."""
    try:
        if use_gc_message:
            result = conn.execute(
                "SELECT MAX(id) FROM gc_message WHERE session_key = ?", (session_id,)
            ).fetchone()
        else:
            result = conn.execute(
                "SELECT MAX(id) FROM message WHERE session_id = ?", (session_id,)
            ).fetchone()
        return result[0] if result and result[0] else ""
    except:
        return ""


def consolidate_session(session_id, source_db, use_gc_message=False):
    """Consolidate a session after debounce fires."""
    with LLM_SEMAPHORE:
        logging.info(
            f"[Genesis] Consolidating {source_db} session {session_id[:30]}..."
        )

        try:
            from core.genesis import Genesis
            from memory_injector import write_memory_block

            # Initialize Genesis
            genesis = Genesis(
                db_path=MEMORY_DB_PATH,
                vectors_path=os.path.expandvars(r"%APPDATA%\gateclaw\vectors"),
                buffer_size=10,
                embedding_model="qwen3-embedding:0.6b",
                llm_model="Claude-4.6-Opus-2B",
            )

            # Connect to source DB and load messages
            uri_path = source_db.replace("\\", "/")
            conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True, timeout=30.0)
            conn.row_factory = sqlite3.Row

            if use_gc_message:
                messages = load_gc_messages(conn, session_id)
            else:
                messages = load_opencode_messages(conn, session_id)

            conn.close()

            if not messages:
                logging.warning(f"No messages found for session {session_id}")
                genesis.close()
                return

            # Create an episode and remember it
            episode = genesis.begin_episode(user_id=session_id)

            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:
                    episode = genesis.add_message(episode, role, content)

            # End and remember (LLM calls happen here)
            episode = genesis.end_episode(episode)
            genesis.remember(episode)

            # Wait for consolidation to complete
            genesis.wait_consolidation(episode.episode_id, timeout=120.0)

            # Get stats
            stats = genesis.get_stats()
            facts_count = stats.get("layer4_consolidation", {}).get("total_facts", 0)

            genesis.close()

            # Write memory blocks
            write_memory_block(variant="telegram")
            write_memory_block(variant="standard")

            logging.info(
                f"[Genesis] Session {session_id[:30]} consolidated — {facts_count} facts total"
            )

        except Exception as e:
            logging.error(f"[Genesis] Consolidation failed for {session_id[:30]}: {e}")
            import traceback

            traceback.print_exc()


def run_watcher():
    """Main watcher loop — monitors ALL OpenCode/GateClaw databases."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("genesis_watcher")

    logger.info("[Genesis] Starting multi-database watcher...")

    # Find all databases
    all_dbs = find_all_databases()
    logger.info(f"[Genesis] Found {len(all_dbs)} databases:")
    for path, source in all_dbs.items():
        logger.info(f"  [{source}] {path}")

    # Check schemas
    db_info = {}
    for path, source in all_dbs.items():
        info = check_db_schema(path)
        if info["sessions"] > 0 or info["gc_messages"] > 0:
            db_info[path] = info
            logger.info(
                f"  Active: [{source}] {os.path.basename(path)}: sessions={info['sessions']}, msgs={info['messages']}, gc={info['gc_messages']}"
            )

    if not db_info:
        logger.error("[Genesis] No OpenCode/GateClaw databases found!")
        return

    # Per-session tracking: {(db_path, session_id): {"last_activity": timestamp, "last_id": str, "use_gc": bool}}
    session_state = {}

    try:
        while True:
            try:
                now = time.time()
                consolidated_any = False

                # Check each database
                for db_path, info in db_info.items():
                    use_gc = info["has_gc_message"] and info["gc_messages"] > 0

                    try:
                        uri_path = db_path.replace("\\", "/")
                        conn = sqlite3.connect(
                            f"file:{uri_path}?mode=ro", uri=True, timeout=5.0
                        )
                        conn.row_factory = sqlite3.Row

                        sessions = get_active_sessions(conn, use_gc_message=use_gc)

                        for session_id in sessions:
                            key = (db_path, session_id)
                            state = session_state.get(key, {})

                            # Get latest message ID for this session
                            latest_id = get_latest_id(
                                conn, session_id, use_gc_message=use_gc
                            )

                            if not latest_id:
                                continue

                            last_id = state.get("last_id", "")
                            last_activity = state.get("last_activity", 0)

                            # Check if new messages
                            is_new = (latest_id != last_id) and (last_id != "")

                            if is_new:
                                session_state[key] = {
                                    "last_activity": now,
                                    "last_id": latest_id,
                                    "use_gc": use_gc,
                                }
                                logger.debug(
                                    f"[Genesis] New activity in {session_id[:30]} from {os.path.basename(db_path)}"
                                )
                            elif last_id == "":
                                # First encounter — initialize state but don't consolidate
                                session_state[key] = {
                                    "last_activity": now,
                                    "last_id": latest_id,
                                    "use_gc": use_gc,
                                }
                            elif (
                                last_activity
                                and (now - last_activity) >= DEBOUNCE_SECONDS
                            ):
                                # Debounce period passed — consolidate in background thread
                                # This ensures the polling loop never blocks
                                del session_state[key]
                                t = threading.Thread(
                                    target=consolidate_session,
                                    args=(session_id, db_path, use_gc),
                                    daemon=True,
                                )
                                t.start()
                                consolidated_any = True
                                logger.info(
                                    f"[Genesis] Started consolidation thread for {session_id[:30]}"
                                )

                        conn.close()
                    except sqlite3.OperationalError as e:
                        if "locked" in str(e).lower():
                            logger.debug(
                                f"[Genesis] {os.path.basename(db_path)} locked, retrying next cycle..."
                            )
                        else:
                            logger.error(
                                f"[Genesis] Database error {os.path.basename(db_path)}: {e}"
                            )
                    except Exception as e:
                        logger.error(
                            f"[Genesis] Error checking {os.path.basename(db_path)}: {e}"
                        )

                # Brief pause between polling cycles
                if not consolidated_any:
                    time.sleep(POLL_INTERVAL)

            except KeyboardInterrupt:
                logger.info("[Genesis] Watcher stopped by user.")
                break
            except Exception as e:
                logger.error(f"[Genesis] Watcher loop error: {e}")
                time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("[Genesis] Watcher stopped.")


if __name__ == "__main__":
    run_watcher()
