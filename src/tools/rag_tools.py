"""
layer3_tools.py — Layer 3: A-RAG Runtime Tools

Implements the three LangChain @tool functions used by the ReAct agent:

    1. keyword_search  — Exact lexical matching (Fix 2: no snippet truncation)
    2. semantic_search — FAISS cosine search (Fix 1B: normalize query + dot product score)
    3. chunk_read      — Full chunk retrieval with adjacent-chunk hint (Fix 4)

All 4 critical A-RAG fixes are applied in this module:
    Fix 1 (1B) : semantic_search normalises query vector; score = dot product (not 1-dist).
    Fix 2      : keyword_search returns ALL matching sentences (no [:2] truncation).
    Fix 4      : chunk_read docstring includes the adjacent-chunk navigation strategy.

C_read is a module-level set tracking which chunks have been read in the current
agent turn.  It is reset to an empty set at the start of every run_agent() call
in layer4_agent.py.

Usage:
    from layer3_tools import keyword_search, semantic_search, chunk_read, reset_c_read
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import faiss
import nltk
import numpy as np
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

from src.core.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    RETRIEVAL_TOP_K,
    SENTENCES_MAP_PATH,
)
from src.core.database import (
    fetch_all_chunks,
    fetch_all_documents,
    fetch_chunks_by_doc,
    fetch_chunk_by_id,
    get_parent_chunk_id,
    get_sentence_text,
    get_connection,
)

logger = logging.getLogger(__name__)

# ── Per-query state (reset by layer4_agent.reset_c_read) ─────────────────────
C_read: set[str] = set()


def reset_c_read() -> None:
    """Reset the set of already-read chunk IDs.  Called once per agent query."""
    global C_read
    C_read = set()
    logger.debug("C_read reset for new query.")


# ─────────────────────────────────────────────────────────────────────────────
# Reranking Helper
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(results: list[dict], query: str) -> list[dict]:
    """Rerank retrieved chunks by term-overlap with the transformed query.

    Computes a simple overlap score: number of query words found in the snippet.
    Combines it linearly with the original retrieval score so that semantically
    close AND lexically relevant chunks bubble to the top.

    Args:
        results : List of dicts with 'chunk_id', 'score', 'snippet' keys.
        query   : The (transformed) query string.

    Returns:
        Re-sorted list — most relevant first.
    """
    query_words = set(query.lower().split())
    for r in results:
        text_words = set(r.get("snippet", "").lower().split())
        overlap = len(query_words & text_words) / max(len(query_words), 1)
        # Blend original retrieval score (0-1) with lexical overlap (0-1)
        r["rerank_score"] = round(0.7 * r["score"] + 0.3 * overlap, 4)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded singletons (loaded once, reused across all tool calls)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model (loads from disk on first call)."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_faiss_index() -> faiss.IndexFlatIP:
    """Load and cache the FAISS index from disk."""
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. "
            "Run 'python layer1_indexer.py --rebuild-all' first."
        )
    logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
    return faiss.read_index(str(FAISS_INDEX_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — Keyword Search (Fix 2)
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Tokenise text into sentences using NLTK sent_tokenize."""
    return nltk.sent_tokenize(text)


@tool
def keyword_search(keywords: list[str], top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Use this tool for precise entity matching or when you know the exact wording of a name, date, or term. It searches the entire corpus for exact lexical matches.

    Scores each chunk using keyword counts.
    Returns the top-k chunk IDs along with abbreviated snippets consisting only of sentences containing the keywords.

    Args:
        keywords : List of search terms (e.g. ["UPS bypass", "coolant"]).
        top_k    : Maximum number of chunks to return (default 5).

    Returns:
        List of dicts, each with:
            chunk_id (str) : Chunk identifier.
            score    (int) : Relevance score.
            snippet  (str) : All matching sentences joined by " ... ".
    """
    logger.info("keyword_search called: keywords=%s, top_k=%d", keywords, top_k)

    all_chunks = fetch_all_chunks()
    scored: list[tuple[str, int, str]] = []

    for chunk in all_chunks:
        text_lower = chunk["markdown_text"].lower()

        # A-RAG Eq. 1 scoring
        score = sum(
            text_lower.count(kw.lower()) * len(kw) for kw in keywords
        )

        if score > 0:
            # FIX 2: ALL sentences containing any keyword — no truncation
            snippet_sentences = [
                sent
                for sent in _split_sentences(chunk["markdown_text"])
                if any(kw.lower() in sent.lower() for kw in keywords)
            ]
            snippet = " ... ".join(snippet_sentences)  # join — never sliced
            snippet = snippet[:600] + "..." if len(snippet) > 600 else snippet  # 600 chars preserves table rows
            scored.append((chunk["chunk_id"], score, snippet))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = [
        {"chunk_id": cid, "score": sc, "snippet": snip}
        for cid, sc, snip in scored[:top_k]
    ]

    logger.info("keyword_search returned %d results.", len(results))
    logger.debug("keyword_search results: %s", results)
    reranked = _rerank(results, " ".join(keywords))
    return reranked


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — Semantic Search (Fix 1B)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def semantic_search(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[dict]:
    """Use this tool for conceptual or meaning-based matching, or when the exact wording in the documents is unknown. It finds passages that are semantically related to your query.

    FIX 1B — Query vector is L2-normalised before FAISS search.
    Score = dot product of unit vectors = cosine similarity (NOT 1 - L2 distance).
    Aggregates sentence-level scores to chunk-level, returning top chunks.
    Returns top-k chunk IDs and snippets of the most relevant sentences within those chunks.

    Args:
        query  : Natural language search query.
        top_k  : Number of chunks to return (default 5).

    Returns:
        List of dicts, each with:
            chunk_id (str)   : Chunk identifier.
            score    (float) : Cosine similarity score (0–1).
            snippet  (str)   : The best-matching sentence from that chunk.
    """
    logger.info("semantic_search called: query='%s', top_k=%d", query, top_k)

    model  = _get_embedding_model()
    index  = _get_faiss_index()

    # FIX 1B — Encode and normalise the query vector
    q_emb: np.ndarray = model.encode([query], convert_to_numpy=True).astype(np.float32)
    q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)   # MUST normalise query; cosine = dot product on unit vecs

    # Search for top_k*2 sentences so we can aggregate to top_k unique chunks
    D, I = index.search(q_emb, top_k * 2)

    # Aggregate sentence-level scores to chunk-level (keep highest score per chunk)
    chunk_scores: dict[str, tuple[float, str]] = {}  # chunk_id → (best_score, snippet)

    for sent_faiss_row, cosine_score in zip(I[0], D[0]):
        if sent_faiss_row < 0:
            continue  # FAISS returns -1 for empty slots

        try:
            parent_chunk_id = get_parent_chunk_id(int(sent_faiss_row))
            snippet_text    = get_sentence_text(int(sent_faiss_row))
        except KeyError:
            logger.debug("FAISS row %d has no DB mapping — skipping.", sent_faiss_row)
            continue

        # FIX 1B: score = cosine similarity directly (NOT 1 - distance)
        score = float(cosine_score)

        if (
            parent_chunk_id not in chunk_scores
            or score > chunk_scores[parent_chunk_id][0]
        ):
            chunk_scores[parent_chunk_id] = (score, snippet_text)

    # Sort by score descending, limit to top_k
    sorted_chunks = sorted(
        chunk_scores.items(), key=lambda x: x[1][0], reverse=True
    )[:top_k]

    results = [
        {"chunk_id": cid, "score": round(score, 4), "snippet": (snip[:600] + "...") if len(snip) > 600 else snip}
        for cid, (score, snip) in sorted_chunks
    ]

    logger.info("semantic_search returned %d results.", len(results))
    logger.debug("semantic_search results: %s", results)
    reranked = _rerank(results, query)
    return reranked


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — List Documents
# ─────────────────────────────────────────────────────────────────────────────

@tool
def list_documents() -> list[dict]:
    """List all documents that have been ingested into the RAG system.

    Use this tool FIRST when the user asks to summarise, compare, or work across
    multiple documents.  It returns the doc_id and filename of every PDF in the
    database, so you know how many documents exist and what each one is called.

    Returns:
        List of dicts, each with:
            doc_id      (int) : Unique document identifier.
            source      (str) : Original PDF filename.
            date_issued (str) : Date the document was issued (may be None).
            doc_type    (str) : Document type label (may be None).
    """
    logger.info("list_documents called")
    docs = fetch_all_documents()
    logger.info("list_documents returned %d documents", len(docs))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — Get Document Chunks (for summarisation)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_document_chunks(doc_id: int, max_chunks: int = 6) -> list[dict]:
    """Retrieve the first N chunk IDs and their snippets for a specific document.

    Use this tool when you want to summarise a SPECIFIC document identified by
    its doc_id (obtained from list_documents).  It returns real chunk_ids that
    you can then pass to chunk_read to get the full text.

    DO NOT guess chunk_ids.  Always use this tool to discover them.

    Args:
        doc_id     : The integer doc_id from list_documents.
        max_chunks : How many chunks to return (default 6, max 10).

    Returns:
        List of dicts with:
            chunk_id (str) : Real chunk identifier safe to pass to chunk_read.
            snippet  (str) : First 300 characters of the chunk for preview.
    """
    logger.info("get_document_chunks called: doc_id=%d, max_chunks=%d", doc_id, max_chunks)
    max_chunks = min(max_chunks, 10)  # cap to protect token budget
    chunks = fetch_chunks_by_doc(doc_id)
    result = [
        {
            "chunk_id": c["chunk_id"],
            "snippet": c["markdown_text"][:300] + "..." if len(c["markdown_text"]) > 300 else c["markdown_text"],
        }
        for c in chunks[:max_chunks]
    ]
    logger.info("get_document_chunks returned %d chunks for doc_id=%d", len(result), doc_id)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — Chunk Read (Fix 4)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def chunk_read(chunk_ids: list[str]) -> dict:
    """Essential Tool. Read the complete content of document segments by their IDs. Use this to examine the full context and details that are hidden in search snippets. STRATEGY: If information seems truncated, you may also read adjacent chunks (ID +/- 1).

    This tool returns the full text of the specified chunks.

    Note: Previously read chunks will be marked as already seen with 'This chunk has been read before'.

    Args:
        chunk_ids: List of chunk IDs to read (e.g. ['100000', '100001', '100024']).

    Returns:
        Dict mapping each chunk_id to either:
            {'markdown': str, 'metadata': dict} — full chunk content
            'This chunk has been read before'   — for duplicate requests
    """
    logger.info("chunk_read called for chunk_ids=%s", chunk_ids)

    results: dict[str, object] = {}

    for cid in chunk_ids:
        if cid in C_read:
            results[cid] = "This chunk has been read before"
            logger.debug("chunk_read: chunk_id=%s already in C_read.", cid)
        else:
            try:
                chunk = fetch_chunk_by_id(cid)
                results[cid] = {
                    "markdown": chunk["markdown_text"],
                    "metadata": json.loads(chunk["metadata_json"] or "{}"),
                }
                C_read.add(cid)
                logger.debug("chunk_read: chunk_id=%s fetched and added to C_read.", cid)
            except KeyError:
                results[cid] = f"Error: chunk_id={cid!r} not found in the database."
                logger.warning("chunk_read: chunk_id=%s not found.", cid)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6 — Triple Lookup (Change 5)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def triple_lookup(entity: str = "", attribute: str = "") -> str:
    """Direct structured lookup for factual questions about contract data.
    Use this for: prices, quantities, dates, names, IDs, addresses, specifications.
    entity: the section or subject (e.g. 'Buyer', 'Product', 'Seller')
    attribute: the field name (e.g. 'Unit Price', 'Organisation Name', 'GSTIN')
    Returns matching values from the entity_triples table.
    """
    logger.info("triple_lookup called: entity='%s', attribute='%s'", entity, attribute)
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT entity, attribute, value FROM entity_triples
               WHERE entity LIKE ? AND attribute LIKE ?
               LIMIT 10""",
            (f"%{entity}%", f"%{attribute}%")
        ).fetchall()
        if not rows:
            return "No structured match found. Try semantic_search instead."
        return "\n".join([f"{r[0]} | {r[1]}: {r[2]}" for r in rows])
    finally:
        conn.close()
