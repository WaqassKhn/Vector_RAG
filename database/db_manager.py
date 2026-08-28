"""
database/db_manager.py
───────────────────────
Persistent SQLite Database Engine with Write-Ahead Logging (WAL) mode.
Guarantees zero data loss across browser reloads, server restarts, and container redeploys.

Stores:
  - Chat Sessions & Messages (multi-session chat history, citations, grounding results)
  - Document Registry & Full Chunk Texts (persistent local store for PineconeDB)
  - Multi-Tier Cognitive Memory (Episodic episodes, User preferences, Semantic facts, Procedural recipes)
  - Token Consumption Logs
"""

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import DB_PATH

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """
    Thread-safe SQLite database manager for RAG_NTPC.
    Ensures persistent storage of all application state.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for SQLite database connection with row factory and WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes database tables and indexes if they do not already exist."""
        with self._get_connection() as conn:
            # 1. Sessions Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 2. Messages Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations TEXT DEFAULT '[]',
                    grounding_score REAL DEFAULT 0.0,
                    grounding_passed INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    latency_ms REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # 3. Documents Registry Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT UNIQUE NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    file_size INTEGER DEFAULT 0,
                    indexed_at TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}'
                )
            """)

            # 4. Chunk Texts Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunk_texts (
                    chunk_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    text TEXT NOT NULL,
                    page_number INTEGER DEFAULT 1,
                    has_table INTEGER DEFAULT 0,
                    numeric_count INTEGER DEFAULT 0,
                    header_context TEXT DEFAULT ''
                )
            """)

            # 5. Episodic Memory Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations TEXT DEFAULT '[]',
                    embedding_json TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                )
            """)

            # 6. User Preferences Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 7. Semantic Facts Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_facts (
                    id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source TEXT DEFAULT 'conversation',
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL
                )
            """)

            # 8. Procedural Recipes Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS procedural_recipes (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    trigger_patterns TEXT NOT NULL,
                    steps TEXT NOT NULL,
                    few_shot_examples TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL
                )
            """)

            # 9. Token Usage Logs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage_logs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    model TEXT,
                    task TEXT
                )
            """)

            # Indexes for high performance queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunk_texts(filename);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodic_memory(timestamp);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_token_logs_timestamp ON token_usage_logs(timestamp);")

        logger.info(f"[DatabaseManager] Database initialized at {self.db_path}")

    # ─── Sessions CRUD ────────────────────────────────────────────────────────

    def create_session(self, title: str = "New Conversation", session_id: Optional[str] = None) -> str:
        """Creates a new chat session and returns its ID."""
        sid = session_id or str(uuid.uuid4())
        now = _utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (sid, title, now, now),
            )
        return sid

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single session by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Lists all sessions sorted by last updated timestamp descending."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Updates a session's title."""
        now = _utc_now_iso()
        with self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
            return cur.rowcount > 0

    def touch_session(self, session_id: str) -> None:
        """Updates the session's updated_at timestamp."""
        now = _utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session and all its messages (via cascade)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    # ─── Messages CRUD ────────────────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[str]] = None,
        grounding_score: float = 0.0,
        grounding_passed: bool = False,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        message_id: Optional[str] = None,
    ) -> str:
        """Saves a message to the database and touches the session."""
        mid = message_id or str(uuid.uuid4())
        now = _utc_now_iso()
        citations_json = json.dumps(citations or [], ensure_ascii=False)

        with self._get_connection() as conn:
            # Ensure session exists
            sess = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not sess:
                # Auto-create session if it doesn't exist
                auto_title = content[:30] + ("..." if len(content) > 30 else "")
                conn.execute(
                    "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, auto_title or "New Conversation", now, now),
                )

            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, role, content, citations,
                    grounding_score, grounding_passed, tokens_used, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mid,
                    session_id,
                    role,
                    content,
                    citations_json,
                    grounding_score,
                    1 if grounding_passed else 0,
                    tokens_used,
                    latency_ms,
                    now,
                ),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

        return mid

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Returns all messages for a session in chronological order."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            messages = []
            for r in rows:
                item = dict(r)
                item["citations"] = json.loads(item.get("citations") or "[]")
                item["grounding_passed"] = bool(item.get("grounding_passed", 0))
                messages.append(item)
            return messages

    def clear_session_messages(self, session_id: str) -> None:
        """Clears all messages for a specific session."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    # ─── Documents & Chunks CRUD (PineconeDB backing) ─────────────────────────

    def upsert_document(
        self,
        filename: str,
        chunk_count: int,
        file_size: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registers or updates a document in the documents table."""
        now = _utc_now_iso()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        doc_id = str(uuid.uuid4())

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, filename, chunk_count, file_size, indexed_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(filename) DO UPDATE SET
                    chunk_count = chunk_count + excluded.chunk_count,
                    indexed_at = excluded.indexed_at,
                    metadata_json = excluded.metadata_json
                """,
                (doc_id, filename, chunk_count, file_size, now, meta_json),
            )

    def get_documents(self) -> Dict[str, Dict[str, Any]]:
        """Returns document registry matching the format expected by PineconeDB / UI."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY indexed_at DESC").fetchall()
            registry = {}
            for r in rows:
                registry[r["filename"]] = {
                    "chunk_count": r["chunk_count"],
                    "file_size": r["file_size"],
                    "indexed_at": r["indexed_at"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                }
            return registry

    def delete_document(self, filename: str) -> int:
        """Deletes document registry and associated chunk texts. Returns deleted chunk count."""
        with self._get_connection() as conn:
            cur_chunks = conn.execute(
                "DELETE FROM chunk_texts WHERE filename = ?", (filename,)
            )
            deleted_chunks = cur_chunks.rowcount
            conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            return deleted_chunks

    def save_chunk_texts(self, chunks: List[Dict[str, Any]]) -> None:
        """Batch inserts or updates chunk texts in the database."""
        with self._get_connection() as conn:
            for c in chunks:
                conn.execute(
                    """
                    INSERT INTO chunk_texts (
                        chunk_id, filename, text, page_number, has_table, numeric_count, header_context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        text = excluded.text,
                        page_number = excluded.page_number,
                        has_table = excluded.has_table,
                        numeric_count = excluded.numeric_count,
                        header_context = excluded.header_context
                    """,
                    (
                        c["chunk_id"],
                        c.get("filename", "unknown"),
                        c.get("text", ""),
                        c.get("page_number", 1),
                        1 if c.get("has_table") else 0,
                        int(c.get("numeric_count", 0)),
                        c.get("header_context", ""),
                    ),
                )

    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Fetches full text for a chunk by its chunk_id."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT text FROM chunk_texts WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
            if row:
                return row["text"]
        return None

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Returns all chunks metadata and text (used by HybridReranker BM25)."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM chunk_texts").fetchall()
            return [
                {
                    "chunk_id": r["chunk_id"],
                    "filename": r["filename"],
                    "text": r["text"],
                    "page_number": r["page_number"],
                    "has_table": bool(r["has_table"]),
                    "numeric_count": r["numeric_count"],
                    "header_context": r["header_context"],
                }
                for r in rows
            ]

    def clear_documents_and_chunks(self) -> None:
        """Deletes all documents and chunk texts."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM chunk_texts")

    # ─── Episodic Memory CRUD ────────────────────────────────────────────────

    def save_episode(
        self,
        query: str,
        answer: str,
        citations: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Saves an interaction episode to episodic_memory."""
        eid = str(uuid.uuid4())
        now = _utc_now_iso()
        citations_json = json.dumps(citations or [], ensure_ascii=False)
        emb_json = json.dumps(embedding or [])

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO episodic_memory (
                    id, session_id, query, answer, citations, embedding_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (eid, session_id, query, answer, citations_json, emb_json, now),
            )
        return eid

    def get_all_episodes(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Retrieves recent episodes."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM episodic_memory ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            episodes = []
            for r in rows:
                item = dict(r)
                item["citations"] = json.loads(item.get("citations") or "[]")
                item["embedding"] = json.loads(item.get("embedding_json") or "[]")
                episodes.append(item)
            return episodes

    def clear_episodes(self) -> None:
        """Clears all episodic memory records."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM episodic_memory")

    # ─── User Preferences CRUD ────────────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> None:
        """Sets a persistent user preference."""
        now = _utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, str(value), now),
            )

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Gets a user preference value."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT value FROM user_preferences WHERE key = ?", (key,)).fetchone()
            if row:
                return row["value"]
        return default

    def get_all_preferences(self) -> Dict[str, str]:
        """Returns all user preferences as a dict."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT key, value FROM user_preferences").fetchall()
            return {r["key"]: r["value"] for r in rows}

    def delete_preference(self, key: str) -> bool:
        """Deletes a user preference."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM user_preferences WHERE key = ?", (key,))
            return cur.rowcount > 0

    # ─── Semantic Facts CRUD ─────────────────────────────────────────────────

    def save_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        source: str = "conversation",
        confidence: float = 1.0,
    ) -> str:
        """Saves a domain fact or learned entity relationship."""
        fid = str(uuid.uuid4())
        now = _utc_now_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO semantic_facts (
                    id, subject, predicate, object, source, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (fid, subject, predicate, object_, source, confidence, now),
            )
        return fid

    def get_all_facts(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Returns all semantic facts."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_facts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_fact(self, fact_id: str) -> bool:
        """Deletes a fact by ID."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM semantic_facts WHERE id = ?", (fact_id,))
            return cur.rowcount > 0

    def clear_facts(self) -> None:
        """Clears all semantic facts."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM semantic_facts")

    # ─── Procedural Recipes CRUD ─────────────────────────────────────────────

    def save_recipe(
        self,
        name: str,
        trigger_patterns: List[str],
        steps: List[str],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Saves or updates a procedural task recipe."""
        rid = str(uuid.uuid4())
        now = _utc_now_iso()
        triggers_json = json.dumps(trigger_patterns, ensure_ascii=False)
        steps_json = json.dumps(steps, ensure_ascii=False)
        examples_json = json.dumps(few_shot_examples or [], ensure_ascii=False)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO procedural_recipes (
                    id, name, trigger_patterns, steps, few_shot_examples, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    trigger_patterns = excluded.trigger_patterns,
                    steps = excluded.steps,
                    few_shot_examples = excluded.few_shot_examples
                """,
                (rid, name, triggers_json, steps_json, examples_json, now),
            )
        return rid

    def get_recipe(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a recipe by its unique name."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM procedural_recipes WHERE name = ?", (name,)
            ).fetchone()
            if row:
                item = dict(row)
                item["trigger_patterns"] = json.loads(item.get("trigger_patterns") or "[]")
                item["steps"] = json.loads(item.get("steps") or "[]")
                item["few_shot_examples"] = json.loads(item.get("few_shot_examples") or "[]")
                return item
        return None

    def get_all_recipes(self) -> List[Dict[str, Any]]:
        """Returns all procedural recipes."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM procedural_recipes ORDER BY name ASC").fetchall()
            recipes = []
            for r in rows:
                item = dict(r)
                item["trigger_patterns"] = json.loads(item.get("trigger_patterns") or "[]")
                item["steps"] = json.loads(item.get("steps") or "[]")
                item["few_shot_examples"] = json.loads(item.get("few_shot_examples") or "[]")
                recipes.append(item)
            return recipes

    def delete_recipe(self, name: str) -> bool:
        """Deletes a procedural recipe by name."""
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM procedural_recipes WHERE name = ?", (name,))
            return cur.rowcount > 0

    # ─── Token Usage Logging ─────────────────────────────────────────────────

    def log_token_usage(
        self,
        session_id: Optional[str],
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "",
        task: str = "",
    ) -> None:
        """Logs token consumption for auditing and quota tracking."""
        lid = str(uuid.uuid4())
        now = _utc_now_iso()
        total_tokens = prompt_tokens + completion_tokens
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO token_usage_logs (
                    id, session_id, timestamp, prompt_tokens, completion_tokens, total_tokens, model, task
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (lid, session_id, now, prompt_tokens, completion_tokens, total_tokens, model, task),
            )

    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Returns aggregate token consumption metrics."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens
                FROM token_usage_logs
            """).fetchone()

            today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_row = conn.execute("""
                SELECT
                    COUNT(*) as today_requests,
                    SUM(total_tokens) as today_tokens
                FROM token_usage_logs
                WHERE timestamp LIKE ?
            """, (f"{today_prefix}%",)).fetchone()

            return {
                "total_requests": row["total_requests"] or 0,
                "total_prompt_tokens": row["total_prompt_tokens"] or 0,
                "total_completion_tokens": row["total_completion_tokens"] or 0,
                "total_tokens": row["total_tokens"] or 0,
                "today_requests": today_row["today_requests"] or 0,
                "today_tokens": today_row["today_tokens"] or 0,
            }


# ─── Global Singleton ─────────────────────────────────────────────────────────

_GLOBAL_DB: Optional[DatabaseManager] = None


def get_db(db_path: Optional[Union[str, Path]] = None) -> DatabaseManager:
    """Returns or creates the global DatabaseManager singleton."""
    global _GLOBAL_DB
    if _GLOBAL_DB is None or db_path is not None:
        _GLOBAL_DB = DatabaseManager(db_path=db_path)
    return _GLOBAL_DB
