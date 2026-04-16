"""
layer1_indexer.py — Layer 1: Hierarchical Indexing (3 Stages)

Responsibility:
    Read parsed Markdown from SQLite → Semantic chunking with sentence-boundary
    alignment (Fix 3) → FAISS sentence embeddings with cosine similarity (Fix 1A)
    → Persist chunks to SQLite and FAISS index + map to disk.

Stages:
    Stage 1 — build_chunks(doc_id)    : Markdown → sentence-aligned chunks → SQLite `chunks`
    Stage 2 — build_embeddings()      : All chunks → sentence embeddings → FAISS IndexFlatIP
    Stage 3 — persist: save FAISS index + sentences_map.json to disk

Fix 1A (FAISS):  IndexFlatIP + normalize_L2 → exact cosine similarity
Fix 3   (Chunk): align_to_sentence_boundary() called after every token-boundary cut

Usage (standalone):
    python layer1_indexer.py --doc-id 1       # index a single document
    python layer1_indexer.py --rebuild-all    # re-index every document in the DB
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import tiktoken
from tqdm import tqdm

from src.core.config import (
    CHUNK_TOKEN_LIMIT,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    SENTENCES_MAP_PATH,
    RETRIEVAL_TOP_K,
)
from src.core.database import (
    fetch_all_chunks,
    init_db,
    insert_chunk,
    insert_sentence,
    clear_sentences,
    clear_sentences_and_chunks,
)

# ── NLTK data (safe download) ──────────────────────────────────────────────────
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK punkt not found. Attempting download...")
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        logger.warning(f"NLTK download failed: {e}. Falling back to simple splitting.")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 3 — Sentence Boundary Alignment
# ─────────────────────────────────────────────────────────────────────────────

def get_tokens(text: str, enc) -> int:
    """Safe token count helper."""
    try:
        return len(enc.encode(text))
    except Exception:
        # Fallback to simple word count + 30% overhead if tiktoken hangs/fails
        return int(len(text.split()) * 1.3)


def build_chunks(doc_id: int) -> list[dict[str, Any]]:
    """Stage 1: Read document Markdown from SQLite and create continuous token-based chunks."""
    import sqlite3
    from src.core.config import DB_PATH

    logger.info("Stage 1 — Building continuous token-based chunks for doc_id=%d", doc_id)

    # ── Fetch document row ────────────────────────────────────────────────────
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT source, date_issued, tags, version, doc_type, severity, full_markdown_text "
        "FROM documents WHERE doc_id = ?",
        (doc_id,),
    ).fetchone()
    conn.close()

    if row is None:
        raise ValueError(f"No document found with doc_id={doc_id}")

    source_md: str = row["full_markdown_text"]

    # ── Base metadata ─────────────────────────────────────────────────────────
    base_meta: dict[str, Any] = {
        "doc_id":      doc_id,
        "source":      row["source"],
        "date_issued": row["date_issued"],
        "tags":        json.loads(row["tags"] or "[]"),
        "version":     row["version"],
        "doc_type":    row["doc_type"],
        "severity":    row["severity"],
    }

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None # Fallback logic used in get_tokens

    # ── Helper: create and persist a single chunk ─────────────────────────────
    chunks: list[dict[str, Any]] = []
    chunk_counter = 0

    def _save_chunk(text: str) -> None:
        nonlocal chunk_counter
        if not text.strip():
            return

        chunk_id = str(_global_chunk_id_offset(doc_id, chunk_counter))
        chunk_counter += 1

        chunk_meta = {
            **base_meta,
            "chunk_index": chunk_counter - 1,
        }
        meta_json = json.dumps(chunk_meta)

        chunk_record: dict[str, Any] = {
            "chunk_id":      chunk_id,
            "doc_id":        doc_id,
            "markdown_text": text,
            "start_page":    None,
            "metadata_json": meta_json,
        }
        chunks.append(chunk_record)
        insert_chunk(
            chunk_id=chunk_id, doc_id=doc_id, markdown_text=text,
            start_page=None, metadata_json=meta_json,
        )

    # ── Process continuous sentences ──────────────────────────────────────────
    try:
        sentences = nltk.sent_tokenize(source_md.strip())
    except Exception:
        sentences = [s.strip() + "." for s in source_md.strip().split(".") if s.strip()]

    current_sentences = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = get_tokens(sent, enc)
        
        if current_tokens + sent_tokens > CHUNK_TOKEN_LIMIT and current_sentences:
            # Flush current chunk
            _save_chunk(" ".join(current_sentences))
            
            # Handle overlap: keep last N tokens worth of sentences
            overlap_sents = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                st = get_tokens(s, enc)
                if overlap_tokens + st > CHUNK_OVERLAP:
                    break
                overlap_sents.insert(0, s)
                overlap_tokens += st
            
            current_sentences = overlap_sents
            current_tokens = overlap_tokens

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Final flush
    if current_sentences:
        _save_chunk(" ".join(current_sentences))

    logger.info("Stage 1 complete — %d chunks created for doc_id=%d", len(chunks), doc_id)
    return chunks


def _global_chunk_id_offset(doc_id: int, local_index: int) -> int:
    """Generate a globally unique chunk ID integer from doc_id + local index.

    Formula: doc_id * 100_000 + local_index
    Supports up to 100,000 chunks per document — well beyond any realistic PDF.
    """
    return doc_id * 100_000 + local_index


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 + 3 — Sentence Embedding + FAISS Indexing (Fix 1A)
# ─────────────────────────────────────────────────────────────────────────────

def build_embeddings() -> None:
    """Stage 2 & 3: Embed all sentences in all chunks and save to FAISS.

    Implements Fix 1A from the A-RAG paper:
    - Uses IndexFlatIP (inner product) NOT IndexFlatL2.
    - Calls faiss.normalize_L2() on embeddings BEFORE adding to index.
    - Result: inner product on unit vectors == cosine similarity.

    Side effects:
        - Encodes every sentence from every chunk in SQLite.
        - Inserts sentence rows into the `sentences` table.
        - Saves the FAISS index to FAISS_INDEX_PATH.
        - Saves the row→(sentence_id, chunk_id) map to SENTENCES_MAP_PATH.
    """
    import faiss  # import here so the rest of the module works without faiss installed
    from sentence_transformers import SentenceTransformer

    logger.info("Stage 2 — Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embedding_dim = model.get_sentence_embedding_dimension()

    logger.info("Stage 2 — Fetching all chunks from DB ...")
    all_chunks = fetch_all_chunks()
    if not all_chunks:
        logger.warning("No chunks found in the database. Run build_chunks() first.")
        return

    # ── Collect all sentences + their parent chunk IDs ────────────────────────
    sentences_text: list[str]  = []
    sentences_meta: list[dict] = []   # {chunk_id, sentence_local_idx}

    for chunk in tqdm(all_chunks, desc="Collecting sentences"):
        sents = nltk.sent_tokenize(chunk["markdown_text"])
        for idx, sent in enumerate(sents):
            if sent.strip():
                sentences_text.append(sent.strip())
                sentences_meta.append({"chunk_id": chunk["chunk_id"], "local_idx": idx})

    if not sentences_text:
        logger.warning("No sentences extracted from chunks.")
        return

    logger.info("Stage 2 — Encoding %d sentences ...", len(sentences_text))
    # batch_size=64 is a safe default for CPU; increase if you have more RAM
    embeddings: np.ndarray = model.encode(
        sentences_text,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalise manually below (Fix 1A)
    )
    embeddings = embeddings.astype(np.float32)

    # FIX 1A: Normalise to unit length → inner product becomes cosine similarity
    faiss.normalize_L2(embeddings)

    logger.info("Stage 3 — Building FAISS IndexFlatIP (dim=%d) ...", embedding_dim)
    vector_index = faiss.IndexFlatIP(embedding_dim)
    vector_index.add(embeddings)   # add normalised embeddings

    # ── Insert sentence rows into SQLite + build on-disk map ─────────────────
    clear_sentences()  # Important: clear old sentences so new FAISS indices start from 0 without collision
    sentences_map: list[dict] = []   # [{faiss_row, sentence_id, chunk_id}]

    for faiss_row, (sent_text, sent_meta) in enumerate(
        tqdm(
            zip(sentences_text, sentences_meta),
            total=len(sentences_text),
            desc="Storing sentence rows",
        )
    ):
        sentence_id = insert_sentence(
            chunk_id      = sent_meta["chunk_id"],
            sentence_text = sent_text,
            faiss_index   = faiss_row,
        )
        sentences_map.append({
            "faiss_row":   faiss_row,
            "sentence_id": sentence_id,
            "chunk_id":    sent_meta["chunk_id"],
        })

    # ── Persist FAISS index to disk ───────────────────────────────────────────
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(vector_index, str(FAISS_INDEX_PATH))
    logger.info("FAISS index saved → %s (%d vectors)", FAISS_INDEX_PATH, vector_index.ntotal)

    # ── Persist sentence map to disk ──────────────────────────────────────────
    SENTENCES_MAP_PATH.write_text(
        json.dumps(sentences_map, indent=2), encoding="utf-8"
    )
    logger.info("Sentence map saved → %s (%d entries)", SENTENCES_MAP_PATH, len(sentences_map))

    logger.info("Stage 2 & 3 complete — FAISS index ready.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point (standalone usage)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer 1 — Build chunks and FAISS embeddings."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--doc-id",
        type=int,
        help="Build chunks for a specific doc_id (then rebuild embeddings).",
    )
    group.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Clear and rebuild chunks + FAISS index for ALL documents.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)
    init_db()

    if args.rebuild_all:
        logger.info("Rebuild-all: clearing existing chunks and sentences ...")
        clear_sentences_and_chunks()
        import sqlite3
        from src.core.config import DB_PATH as _DB_PATH
        conn = sqlite3.connect(str(_DB_PATH))
        doc_ids = [r[0] for r in conn.execute("SELECT doc_id FROM documents").fetchall()]
        conn.close()
        for did in doc_ids:
            build_chunks(did)
    elif args.doc_id:
        build_chunks(args.doc_id)

    build_embeddings()
    print("✅ Layer 1 indexing complete.")
    sys.exit(0)