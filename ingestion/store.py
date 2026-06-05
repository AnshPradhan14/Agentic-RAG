"""
store.py
────────────────────────────────────────────────────────────────────────────
Phase 13 — Vector Store & Disk Persistence.

Phase 14 Error Handling:
 - Vector store upsert failure per chunk → log error, collect into failed_chunks.
 - After all chunks processed, save failed chunks to {output_dir}/failed_chunks.json
   so the operator can inspect and retry without re-running the full pipeline.
 - FAISS write failure → logged, RuntimeError raised (total write failure is critical).

Architecture:
 - Uses existing embedding model wrapper (_get_embedding_model) — no duplicate load.
 - Normalises vectors (faiss.normalize_L2) before FAISS insert for cosine similarity.
 - Chunk "content" stored as document text; everything else goes into metadata_json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.core.config import FAISS_INDEX_PATH, INDEX_DIR, FINAL_TEXT_DIR, PARSED_DIR
from src.core.database import (
    insert_chunk,
    insert_document,
    insert_sentence,
)
from src.tools.rag_tools import _get_embedding_model

logger = logging.getLogger(__name__)


# ── FAISS helpers ─────────────────────────────────────────────────────────────

def _init_faiss_index(dim: int) -> faiss.IndexFlatIP:
    """Load the existing FAISS index from disk, or create a fresh one."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if FAISS_INDEX_PATH.exists():
        logger.info(f"[store] Loading existing FAISS index from {FAISS_INDEX_PATH}")
        return faiss.read_index(str(FAISS_INDEX_PATH))
    logger.info(f"[store] Creating new FAISS IndexFlatIP (dim={dim})")
    return faiss.IndexFlatIP(dim)


def _write_faiss_index(index: faiss.IndexFlatIP) -> None:
    """Persist FAISS index to disk. Raises RuntimeError on failure."""
    try:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        logger.info(f"[store] FAISS index saved to {FAISS_INDEX_PATH} (total={index.ntotal} vectors).")
    except Exception as exc:
        logger.error(f"[store] FAILED to write FAISS index to {FAISS_INDEX_PATH}: {exc}")
        raise RuntimeError(f"FAISS write failed: {exc}") from exc


# ── Failed chunk persistence ──────────────────────────────────────────────────

def _save_failed_chunks(
    failed: list[dict[str, Any]],
    output_dir: Path,
    pdf_stem: str,
) -> None:
    """
    Write failed chunks to {output_dir}/failed_chunks.json for manual retry.
    Appends to an existing file if it already exists.

    Phase 14: "Vector store upsert failure → save chunk to failed_chunks.json for retry."
    """
    if not failed:
        return

    failed_path = output_dir / "failed_chunks.json"

    # Load existing failures so we don't overwrite previous runs
    existing: list[dict[str, Any]] = []
    if failed_path.exists():
        try:
            existing = json.loads(failed_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []

    combined = existing + failed
    failed_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.error(
        f"[store] {len(failed)} chunk(s) failed to upsert for '{pdf_stem}'. "
        f"Saved to {failed_path} for retry."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def store_chunks(
    chunks: list[dict[str, Any]],
    output_dir: str,
    pdf_stem: str,
) -> None:
    """
    Embed, index, and persist all chunks for one document.

    Steps
    ─────
    1. Create a parent document row in SQLite (required by rag_tools FK).
    2. Encode all chunk content strings in one batched model call.
    3. Normalise embeddings for cosine similarity (Fix 1B).
    4. Add vectors to FAISS and persist index.
    5. For each chunk, insert SQLite rows (chunk + sentence).
       - Per-chunk failures are caught, logged, and deferred to failed_chunks.json.
    6. Persist chunks JSON to disk.
    7. If any chunks failed, write failed_chunks.json.

    Parameters
    ──────────
    chunks : list[dict]
        Normalised chunks produced by Phase 11 (normalizer).
    output_dir : str
        Directory where output artefacts are written.
    pdf_stem : str
        PDF filename stem used to name output files and DB records.
    """
    if not chunks:
        logger.warning(f"[store] No chunks to store for '{pdf_stem}'. Skipping.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Create document row in SQLite ─────────────────────────────────
    doc_id: int = insert_document(
        source=f"{pdf_stem}.pdf",
        date_issued=None,
        tags=[],
        version="1.0",
        doc_type="Ingested PDF",
        severity="Info",
        md_filepath=str(FINAL_TEXT_DIR / f"{pdf_stem}_structured.md"),
        json_filepath=str(FINAL_TEXT_DIR / f"{pdf_stem}_chunks.json"),
        full_markdown_text="",   # full MD already saved to disk by pipeline.py
    )
    logger.debug(f"[store] Inserted document row for '{pdf_stem}' → doc_id={doc_id}")

    # ── Step 2: Filter non-empty chunks, build text list ─────────────────────
    valid_chunks: list[dict[str, Any]] = []
    texts: list[str] = []
    for chunk in chunks:
        text = chunk.get("content", "").strip()
        if text:
            valid_chunks.append(chunk)
            texts.append(text)

    if not texts:
        logger.warning(f"[store] All chunks for '{pdf_stem}' have empty content. Nothing to embed.")
        return

    # ── Step 3: Generate embeddings (one batched call) ────────────────────────
    logger.info(f"[store] Encoding {len(texts)} chunk(s) for '{pdf_stem}'...")
    model = _get_embedding_model()
    embeddings: np.ndarray = model.encode(texts, convert_to_numpy=True).astype(np.float32)

    # ── Step 4: Normalise for cosine similarity + update FAISS ───────────────
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = _init_faiss_index(dim)
    start_faiss_row = index.ntotal      # record offset before we add

    index.add(embeddings)
    logger.info(
        f"[store] Added {len(embeddings)} vector(s) to FAISS index "
        f"(total now={index.ntotal})."
    )
    _write_faiss_index(index)           # raises RuntimeError on failure

    # ── Step 5: Persist each chunk in SQLite — per-chunk error isolation ──────
    failed_chunks: list[dict[str, Any]] = []

    for i, chunk in enumerate(valid_chunks):
        faiss_row = start_faiss_row + i
        chunk_id  = str(chunk.get("chunk_id", f"{pdf_stem}_{i}"))
        content   = chunk.get("content", "")

        # Metadata blob = everything except "content" (Phase 13 spec)
        meta: dict[str, Any] = {k: v for k, v in chunk.items() if k != "content"}
        meta_json = json.dumps(meta, ensure_ascii=False)

        try:
            insert_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                markdown_text=content,
                start_page=meta.get("page", 0),
                metadata_json=meta_json,
                section=meta.get("section", "General"),
            )
            insert_sentence(
                chunk_id=chunk_id,
                sentence_text=content,
                faiss_index=faiss_row,
            )
            logger.debug(f"[store] Upserted chunk_id={chunk_id} (faiss_row={faiss_row}).")

        except Exception as exc:
            # Phase 14: "Vector store upsert failure → log error, save to failed_chunks.json"
            logger.error(
                f"[store] Upsert FAILED for chunk_id={chunk_id} (faiss_row={faiss_row}): "
                f"{type(exc).__name__}: {exc}"
            )
            # Tag the chunk with failure metadata for operator retry
            failed_entry = {**chunk, "_failed_reason": str(exc), "_faiss_row": faiss_row}
            failed_chunks.append(failed_entry)

    # ── Step 6: Persist chunks JSON to disk ───────────────────────────────────
    chunks_file = out_path / f"{pdf_stem}_chunks.json"
    chunks_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"[store] Chunks JSON written → {chunks_file}")

    # ── Step 7: Write failed_chunks.json if any upserts failed ───────────────
    _save_failed_chunks(failed_chunks, out_path, pdf_stem)

    success_count = len(valid_chunks) - len(failed_chunks)
    logger.info(
        f"[store] Complete for '{pdf_stem}': "
        f"{success_count}/{len(valid_chunks)} chunk(s) stored successfully, "
        f"{len(failed_chunks)} failed."
    )


# ── FAISS Rebuild ─────────────────────────────────────────────────────────────

def rebuild_faiss_index() -> int:
    """
    Rebuild the FAISS index from ALL remaining sentences in the DB.

    Called after a document deletion to remove stale vectors (FAISS IndexFlatIP
    does not support in-place removal, so we must rebuild from scratch).

    Returns:
        int: Number of vectors in the rebuilt index (0 if DB is empty).
    """
    import sqlite3
    from src.core.config import DB_PATH
    from src.tools.rag_tools import _get_embedding_model, _get_faiss_index

    logger.info("[store:rebuild] Rebuilding FAISS index from remaining DB sentences...")

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT sentence_id, chunk_id, sentence_text FROM sentences ORDER BY sentence_id"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("[store:rebuild] No sentences remain. Clearing FAISS index file.")
        if FAISS_INDEX_PATH.exists():
            FAISS_INDEX_PATH.unlink()
        _get_faiss_index.cache_clear()
        return 0

    # Encode all remaining sentences
    model = _get_embedding_model()
    texts = [r["sentence_text"] for r in rows]
    embeddings = model.encode(texts, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embeddings)

    # Build fresh index
    dim = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(embeddings)

    # Update faiss_index values in DB to match new positions
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    try:
        for new_idx, row in enumerate(rows):
            conn.execute(
                "UPDATE sentences SET faiss_index=? WHERE sentence_id=?",
                (new_idx, row["sentence_id"])
            )
        conn.commit()
        logger.info("[store:rebuild] Updated faiss_index for %d sentences.", len(rows))
    except Exception as exc:
        conn.rollback()
        logger.error("[store:rebuild] Failed to update faiss_index in DB: %s", exc)
        raise
    finally:
        conn.close()

    # Persist the new index
    faiss.write_index(new_index, str(FAISS_INDEX_PATH))
    logger.info(
        "[store:rebuild] FAISS index rebuilt → %s (%d vectors).",
        FAISS_INDEX_PATH, new_index.ntotal
    )

    # Clear the lru_cache so the next query loads the fresh index
    _get_faiss_index.cache_clear()
    return new_index.ntotal
