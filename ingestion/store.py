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

import numpy as np
import uuid
from qdrant_client.http import models

from src.core.config import INDEX_DIR, FINAL_TEXT_DIR, PARSED_DIR, QDRANT_COLLECTION
from src.core.database import (
    insert_chunk,
    insert_document,
    insert_sentence,
)
from src.tools.rag_tools import _get_embedding_model, _get_qdrant_client

logger = logging.getLogger(__name__)

def _ensure_qdrant_collection(client, dim: int) -> None:
    """Ensure the Qdrant collection exists with the correct vector size."""
    collections = client.get_collections().collections
    if not any(c.name == QDRANT_COLLECTION for c in collections):
        logger.info(f"[store] Creating new Qdrant collection '{QDRANT_COLLECTION}' (dim={dim})")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


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

    # ── Step 4: Qdrant setup ───────────────
    dim = embeddings.shape[1]
    client = _get_qdrant_client()
    _ensure_qdrant_collection(client, dim)

    # ── Step 5: Persist each chunk in SQLite and Qdrant ──────
    failed_chunks: list[dict[str, Any]] = []

    qdrant_points = []
    
    for i, chunk in enumerate(valid_chunks):
        qdrant_id = uuid.uuid4().hex
        chunk_id  = str(chunk.get("chunk_id", f"{pdf_stem}_{i}"))
        content   = chunk.get("content", "")
        vector    = embeddings[i].tolist()

        # Metadata blob = everything except "content" (Phase 13 spec)
        meta: dict[str, Any] = {k: v for k, v in chunk.items() if k != "content"}
        meta_json = json.dumps(meta, ensure_ascii=False)
        
        qdrant_points.append(
            models.PointStruct(
                id=qdrant_id,
                vector=vector,
                payload={
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_name": f"{pdf_stem}.pdf",
                    "sentence_text": content,
                }
            )
        )

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
                faiss_index=None,
                qdrant_id=qdrant_id,
            )
            logger.debug(f"[store] Upserted chunk_id={chunk_id} to DB.")

        except Exception as exc:
            logger.error(
                f"[store] DB Insert FAILED for chunk_id={chunk_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            failed_entry = {**chunk, "_failed_reason": str(exc)}
            failed_chunks.append(failed_entry)

    # Upsert to Qdrant
    if qdrant_points:
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=qdrant_points
        )
        logger.info(f"[store] Upserted {len(qdrant_points)} vectors to Qdrant.")



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


# ── Qdrant Deletion ───────────────────────────────────────────────────────────

def delete_document_vectors(doc_id: int) -> None:
    """
    Delete all vectors associated with a specific document from Qdrant.
    This is an instant operation that doesn't require rebuilding the index.
    """
    client = _get_qdrant_client()
    try:
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            )
        )
        logger.info(f"[store:delete] Deleted vectors for doc_id={doc_id} from Qdrant.")
    except Exception as exc:
        logger.error(f"[store:delete] Failed to delete vectors for doc_id={doc_id}: {exc}")
        raise
