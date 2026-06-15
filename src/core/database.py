"""
database.py — SQLite Connection & Schema Definitions (Layer 2 — Storage)

This module owns ALL database concerns:
  - Connection pool management (thread-safe via check_same_thread=False)
  - Table creation (idempotent — safe to call multiple times)
  - Low-level CRUD helpers consumed by every other layer

Schema overview
───────────────
  documents      : one row per ingested PDF  (Layer 0 writes, Layer 3 reads indirectly)
  chunks         : one row per text chunk    (Layer 1 writes, Layer 3 reads via fetch_chunk_by_id)
  sentences      : one row per sentence      (Layer 1 writes, FAISS row mapping lives here)
  entity_triples : one row per table triple  (Layer 0 writes, triple_lookup tool reads)

Usage:
    from database import init_db, fetch_all_chunks, fetch_chunk_by_id
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from src.core.config import DB_PATH

logger = logging.getLogger(__name__)

# ── Connection helper ─────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row-factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")   # Better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ── Schema creation ───────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they do not already exist.

    Idempotent — safe to call on every application start.
    Logs at INFO on first creation, DEBUG on subsequent calls.
    """
    logger.info("Initialising database at %s", DB_PATH)
    conn = _get_connection()
    try:
        cur = conn.cursor()

        # ── documents table (Layer 0 writes) ──────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id             INTEGER PRIMARY KEY AUTOINCREMENT,
                source             TEXT    NOT NULL,          -- original PDF filename
                date_issued        TEXT,                      -- ISO 8601 date string
                tags               TEXT,                      -- JSON array as string e.g. '["fire","cooling"]'
                version            TEXT,
                doc_type           TEXT,                      -- e.g. 'standard_operating_procedure'
                severity           TEXT,
                md_filepath        TEXT    NOT NULL,          -- relative path to data/parsed/*.md
                json_filepath      TEXT    NOT NULL,          -- relative path to data/parsed/*.json
                full_markdown_text TEXT    NOT NULL,          -- full parsed Markdown content
                ingested_at        TEXT    DEFAULT (datetime('now'))
            );
        """)

        # ── chunks table (Layer 1 writes) ─────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id      TEXT    PRIMARY KEY,            -- e.g. "42" (string for tool compatibility)
                doc_id        INTEGER REFERENCES documents(doc_id) ON DELETE CASCADE,
                markdown_text TEXT    NOT NULL,
                start_page    INTEGER,
                metadata_json TEXT                            -- full JSON blob with inherited doc metadata
            );
        """)

        # ── sentences table (Layer 1 writes, FAISS row-mapping) ───────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                sentence_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id      TEXT    REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                sentence_text TEXT    NOT NULL,
                faiss_index   INTEGER UNIQUE                 -- maps to FAISS vector row (0-based)
            );
        """)

        # ── entity_triples table (Layer 0 writes, triple_lookup reads) ────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entity_triples (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    TEXT    NOT NULL,
                chunk_id  INTEGER NOT NULL,
                entity    TEXT    NOT NULL,
                attribute TEXT    NOT NULL,
                value     TEXT    NOT NULL
            );
        """)

        # ── users table (Auth) ────────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                role          TEXT    DEFAULT 'user',
                created_at    TEXT    DEFAULT (datetime('now'))
            );
        """)

        # ── chat_sessions table ───────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                title         TEXT    NOT NULL,
                created_at    TEXT    DEFAULT (datetime('now')),
                updated_at    TEXT    DEFAULT (datetime('now'))
            );
        """)

        # ── chat_messages table ───────────────────────────────────────────────
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     INTEGER REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role           TEXT    NOT NULL,  -- 'user' or 'assistant'
                content        TEXT    NOT NULL,
                mentioned_docs TEXT,              -- JSON list of mentioned pdf names
                created_at     TEXT    DEFAULT (datetime('now'))
            );
        """)

        conn.commit()

        # ── Add section column to chunks if not present (idempotent) ──────────
        try:
            conn.execute("ALTER TABLE chunks ADD COLUMN section TEXT DEFAULT 'General';")
            conn.commit()
            logger.info("Added 'section' column to chunks table.")
        except Exception:  # Column already exists — safe to ignore
            pass
            
        try:
            conn.execute("ALTER TABLE sentences ADD COLUMN qdrant_id TEXT;")
            conn.commit()
            logger.info("Added 'qdrant_id' column to sentences table.")
        except Exception:  
            pass
            
        try:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN sources TEXT;")
            conn.commit()
            logger.info("Added 'sources' column to chat_messages table.")
        except Exception:  
            pass

        logger.info("Database schema ready (documents / chunks / sentences / entity_triples).")
    except sqlite3.Error as exc:
        logger.error("Failed to initialise database: %s", exc)
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Document helpers ──────────────────────────────────────────────────────────

def document_exists(source: str) -> bool:
    """Check if a document with the given source filename already exists."""
    conn = _get_connection()
    try:
        cur = conn.execute("SELECT 1 FROM documents WHERE source = ? LIMIT 1", (source,))
        return cur.fetchone() is not None
    except sqlite3.Error as exc:
        logger.error("Database error while checking document existence: %s", exc)
        return False
    finally:
        conn.close()



def insert_document(
    source: str,
    date_issued: str | None,
    tags: list[str] | None,
    version: str | None,
    doc_type: str | None,
    severity: str | None,
    md_filepath: str,
    json_filepath: str,
    full_markdown_text: str,
) -> int:
    """Insert one document row and return its auto-incremented doc_id.

    Args:
        source             : Original PDF filename (e.g. "UPS_Bypass_v2.pdf").
        date_issued        : ISO 8601 date string or None.
        tags               : List of tag strings; stored as JSON array string.
        version            : Document version string or None.
        doc_type           : Document type label or None.
        severity           : Severity label or None.
        md_filepath        : Relative path to the .md file in data/parsed/.
        json_filepath      : Relative path to the .json file in data/parsed/.
        full_markdown_text : Complete Markdown content of the document.

    Returns:
        doc_id (int): Auto-incremented primary key of the inserted row.

    Raises:
        sqlite3.Error: Rolls back and re-raises on any DB failure.
    """
    tags_str = json.dumps(tags) if tags is not None else json.dumps([])
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO documents
                (source, date_issued, tags, version, doc_type, severity,
                 md_filepath, json_filepath, full_markdown_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source, date_issued, tags_str, version, doc_type, severity,
             md_filepath, json_filepath, full_markdown_text),
        )
        conn.commit()
        doc_id: int = cur.lastrowid  # type: ignore[assignment]
        logger.debug("Inserted document: source=%s → doc_id=%d", source, doc_id)
        return doc_id
    except sqlite3.Error as exc:
        logger.error("insert_document failed for source=%s: %s", source, exc)
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Chunk helpers ─────────────────────────────────────────────────────────────

def insert_chunk(
    chunk_id: str,
    doc_id: int,
    markdown_text: str,
    start_page: int | None,
    metadata_json: str,
    section: str = "General",
) -> None:
    """Insert one chunk row into the chunks table.

    Args:
        chunk_id      : Unique string identifier (e.g. "0", "1", "42").
        doc_id        : Foreign key to the parent document.
        markdown_text : Full Markdown text of this chunk.
        start_page    : Starting page number within the source PDF (or None).
        metadata_json : JSON-serialised metadata dict (inherited from document + chunk fields).
        section       : Section label from context_aware_chunk (default 'General').

    Raises:
        sqlite3.Error: Rolls back and re-raises on any DB failure.
    """
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, markdown_text, start_page, metadata_json, section)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, doc_id, markdown_text, start_page, metadata_json, section),
        )
        conn.commit()
        logger.debug("Inserted chunk_id=%s for doc_id=%d (section=%s)", chunk_id, doc_id, section)
    except sqlite3.Error as exc:
        logger.error("insert_chunk failed for chunk_id=%s: %s", chunk_id, exc)
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_all_documents() -> list[dict[str, Any]]:
    """Return all document rows as a list of plain dicts.

    Returns:
        List of dicts with keys: doc_id, source, date_issued, doc_type, ingested_at.
        Use doc_id values to scope chunk queries to a specific document.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT doc_id, source, date_issued, doc_type, ingested_at FROM documents ORDER BY doc_id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_chunks_by_doc(doc_id: int) -> list[dict[str, Any]]:
    """Return all chunks belonging to a specific document.

    Args:
        doc_id: The integer primary key of the parent document.

    Returns:
        List of dicts with keys: chunk_id, doc_id, markdown_text, start_page, metadata_json.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, markdown_text, start_page, metadata_json FROM chunks WHERE doc_id = ? ORDER BY chunk_id",
            (doc_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_all_chunks() -> list[dict[str, Any]]:
    """Return all chunks as a list of plain dicts.

    Returns:
        List of dicts with keys: chunk_id, doc_id, markdown_text, start_page, metadata_json.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, markdown_text, start_page, metadata_json FROM chunks"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def fetch_chunk_by_id(chunk_id: str) -> dict[str, Any]:
    """Fetch a single chunk by its chunk_id.

    Args:
        chunk_id: String identifier of the chunk.

    Returns:
        Dict with keys: chunk_id, doc_id, markdown_text, start_page, metadata_json.

    Raises:
        KeyError: If no chunk with the given chunk_id exists.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT chunk_id, doc_id, markdown_text, start_page, metadata_json FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"No chunk found with chunk_id={chunk_id!r}")
        return dict(row)
    finally:
        conn.close()


# ── Sentence helpers ──────────────────────────────────────────────────────────

def insert_sentence(chunk_id: str, sentence_text: str, faiss_index: int = None, qdrant_id: str = None) -> int:
    """Insert one sentence row and return its sentence_id.

    Args:
        chunk_id      : Parent chunk identifier.
        sentence_text : Raw sentence string.
        faiss_index   : 0-based row in the FAISS index (deprecated).
        qdrant_id     : UUID string in Qdrant.

    Returns:
        sentence_id (int): Auto-incremented primary key.
    """
    conn = _get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO sentences (chunk_id, sentence_text, faiss_index, qdrant_id) VALUES (?, ?, ?, ?)",
            (chunk_id, sentence_text, faiss_index, qdrant_id),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    except sqlite3.Error as exc:
        logger.error("insert_sentence failed for chunk_id=%s: %s", chunk_id, exc)
        conn.rollback()
        raise
    finally:
        conn.close()


def get_parent_chunk_id(faiss_row: int) -> str:
    """Resolve a FAISS row index to its parent chunk_id.

    Args:
        faiss_row: 0-based integer index in the FAISS vector store.

    Returns:
        chunk_id (str) of the parent chunk.

    Raises:
        KeyError: If no sentence maps to the given faiss_row.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT chunk_id FROM sentences WHERE faiss_index = ?", (faiss_row,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No sentence found with faiss_index={faiss_row}")
        return row["chunk_id"]
    finally:
        conn.close()


def get_sentence_text(faiss_row: int) -> str:
    """Retrieve the sentence text for a given FAISS row index.

    Args:
        faiss_row: 0-based integer index in the FAISS vector store.

    Returns:
        sentence_text (str).

    Raises:
        KeyError: If no sentence maps to the given faiss_row.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT sentence_text FROM sentences WHERE faiss_index = ?", (faiss_row,)
        ).fetchone()
        if row is None:
            raise KeyError(f"No sentence found with faiss_index={faiss_row}")
        return row["sentence_text"]
    finally:
        conn.close()


def clear_sentences() -> None:
    """Delete all rows from the sentences table (used when rebuilding the FAISS index).
    
    Warning: This breaks the mapping between the DB and any old FAISS index on disk.
    """
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sentences;")
        conn.commit()
        logger.info("Cleared all sentences from DB.")
    except sqlite3.Error as exc:
        logger.error("clear_sentences failed: %s", exc)
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_sentences_and_chunks() -> None:
    """Delete all sentences and chunks rows (used before a full re-index).

    Warning: This is destructive. Call only when rebuilding the FAISS index from scratch.
    """
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sentences;")
        conn.execute("DELETE FROM chunks;")
        conn.commit()
        logger.info("Cleared all sentences and chunks from DB.")
    except sqlite3.Error as exc:
        logger.error("clear_sentences_and_chunks failed: %s", exc)
        conn.rollback()
        raise
    finally:
        conn.close()


def clear_all_data() -> None:
    """Delete all rows from all tables and reset auto-increment sequences.
    
    Warning: This completely wipes the database.
    """
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM sentences;")
        conn.execute("DELETE FROM chunks;")
        conn.execute("DELETE FROM documents;")
        conn.execute("DELETE FROM entity_triples;")
        conn.execute("DELETE FROM sqlite_sequence;")
        conn.commit()
        logger.info("Cleared ALL data from DB (including documents and sequences).")
    except sqlite3.Error as exc:
        logger.error("clear_all_data failed: %s", exc)
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Entity Triple helpers ─────────────────────────────────────────────────────

def save_triples(triples: list[dict]) -> None:
    """Insert a batch of entity-attribute-value triples into entity_triples.

    Args:
        triples: List of dicts produced by extract_triples(), each with keys:
                 doc_id, chunk_id, entity, attribute, value.

    Raises:
        sqlite3.Error: Rolls back and re-raises on any DB failure.
    """
    if not triples:
        return
    conn = _get_connection()
    try:
        conn.executemany(
            "INSERT INTO entity_triples (doc_id, chunk_id, entity, attribute, value) "
            "VALUES (:doc_id, :chunk_id, :entity, :attribute, :value)",
            triples,
        )
        conn.commit()
        logger.debug("Saved %d entity triples to DB.", len(triples))
    except sqlite3.Error as exc:
        logger.error("save_triples failed: %s", exc)
        conn.rollback()
        raise
    finally:
        conn.close()


def get_connection() -> sqlite3.Connection:
    """Public alias for _get_connection — used by triple_lookup tool."""
    return _get_connection()


# ── Auth & Chat helpers ───────────────────────────────────────────────────────

def create_user(username: str, password_hash: str, role: str = "user") -> int:
    conn = _get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def get_user_by_username(username: str) -> dict[str, Any] | None:
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def create_chat_session(user_id: int, title: str) -> int:
    conn = _get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)",
            (user_id, title)
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def get_chat_sessions(user_id: int) -> list[dict[str, Any]]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_chat_session(session_id: int) -> dict[str, Any] | None:
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def delete_chat_session(session_id: int) -> bool:
    conn = _get_connection()
    try:
        cur = conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()

def add_chat_message(session_id: int, role: str, content: str, mentioned_docs: list[str] = None, sources: list[str] = None) -> int:
    docs_json = json.dumps(mentioned_docs) if mentioned_docs else None
    sources_json = json.dumps(sources) if sources else None
    conn = _get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, mentioned_docs, sources) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, docs_json, sources_json)
        )
        conn.execute("UPDATE chat_sessions SET updated_at = datetime('now') WHERE session_id = ?", (session_id,))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()

def get_chat_messages(session_id: int) -> list[dict[str, Any]]:
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,)
        ).fetchall()
        
        result = []
        for r in rows:
            msg = dict(r)
            if msg["mentioned_docs"]:
                msg["mentioned_docs"] = json.loads(msg["mentioned_docs"])
            else:
                msg["mentioned_docs"] = []
                
            if "sources" in msg and msg["sources"]:
                msg["sources"] = json.loads(msg["sources"])
            else:
                msg["sources"] = []
            result.append(msg)
        return result
    finally:
        conn.close()

