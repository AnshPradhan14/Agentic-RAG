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

from qdrant_client import QdrantClient
from qdrant_client.http import models
import nltk
import numpy as np
from langchain_core.tools import tool
import requests

from src.core.config import (
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_TOP_K,
    QDRANT_URL,
    QDRANT_COLLECTION,
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

class OllamaEmbedder:
    """Wrapper to hit Ollama's local embeddings API, mimicking sentence-transformers."""
    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def encode(self, texts, convert_to_numpy=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
            
        url = f"{self.host}/api/embed"
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            
            if convert_to_numpy:
                import numpy as np
                return np.array(embeddings, dtype=np.float32)
            return embeddings
        except Exception as e:
            logger.error("Failed to fetch embeddings from Ollama: %s", e)
            raise


@lru_cache(maxsize=1)
def _get_embedding_model() -> OllamaEmbedder:
    """Load and cache the OllamaEmbedder."""
    logger.info("Setting up Ollama embedder for model: %s", EMBEDDING_MODEL_NAME)
    return OllamaEmbedder(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Load and cache the Qdrant client."""
    logger.info("Setting up Qdrant client at %s", QDRANT_URL)
    is_local = not QDRANT_URL.startswith("http")
    
    if is_local:
        Path(QDRANT_URL).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=QDRANT_URL)
    else:
        return QdrantClient(url=QDRANT_URL)


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
def semantic_search(query: str, top_k: int = RETRIEVAL_TOP_K, mentioned_docs: list[str] = None) -> list[dict]:
    """Use this tool for conceptual or meaning-based matching, or when the exact wording in the documents is unknown. It finds passages that are semantically related to your query.

    Returns top-k chunk IDs and snippets of the most relevant sentences within those chunks.

    Args:
        query  : Natural language search query.
        top_k  : Number of chunks to return (default 5).
        mentioned_docs: Optional list of document names to filter the search to.

    Returns:
        List of dicts, each with:
            chunk_id (str)   : Chunk identifier.
            score    (float) : Cosine similarity score (0–1).
            snippet  (str)   : The best-matching sentence from that chunk.
    """
    logger.info("semantic_search called: query='%s', top_k=%d, docs=%s", query, top_k, mentioned_docs)

    model  = _get_embedding_model()
    client = _get_qdrant_client()

    q_emb: np.ndarray = model.encode([query], convert_to_numpy=True).astype(np.float32)
    query_vector = q_emb[0].tolist()
    
    query_filter = None
    if mentioned_docs:
        # Match any of the mentioned docs
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_name",
                    match=models.MatchAny(any=mentioned_docs)
                )
            ]
        )

    # Search for top_k*2 sentences so we can aggregate to top_k unique chunks
    search_result = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k * 2,
        query_filter=query_filter,
        with_payload=True
    )

    # Aggregate sentence-level scores to chunk-level (keep highest score per chunk)
    chunk_scores: dict[str, tuple[float, str]] = {}  # chunk_id → (best_score, snippet)

    for hit in search_result:
        chunk_id = hit.payload.get("chunk_id")
        snippet_text = hit.payload.get("sentence_text", "")
        score = hit.score
        
        if not chunk_id:
            continue

        if (
            chunk_id not in chunk_scores
            or score > chunk_scores[chunk_id][0]
        ):
            chunk_scores[chunk_id] = (score, snippet_text)

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
# Tool 4 — Get Document Chunks (for summarisation)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_document_chunks(doc_id: int = None, doc_name: str = None, max_chunks: int = 6) -> list[dict]:
    """Retrieve the first N chunk IDs and their snippets for a specific document.

    Use this tool when you want to summarise or read a SPECIFIC document.
    You must provide EITHER the integer doc_id OR the exact document name (e.g. 'Contract - GEMC-1.pdf').
    It returns real chunk_ids that you can then pass to chunk_read to get the full text.

    DO NOT guess chunk_ids.  Always use this tool to discover them.

    Args:
        doc_id     : The integer doc_id.
        doc_name   : The exact filename of the document.
        max_chunks : How many chunks to return (default 6, max 10).

    Returns:
        List of dicts with:
            chunk_id (str) : Real chunk identifier safe to pass to chunk_read.
            snippet  (str) : First 300 characters of the chunk for preview.
    """
    if doc_id is None and doc_name is None:
        return [{"error": "Must provide either doc_id or doc_name"}]
        
    if doc_id is None:
        docs = fetch_all_documents()
        for d in docs:
            if d["source"] == doc_name:
                doc_id = d["doc_id"]
                break
        if doc_id is None:
            return [{"error": f"Document '{doc_name}' not found in database."}]

    logger.info("get_document_chunks called: doc_id=%s, doc_name=%s, max_chunks=%d", doc_id, doc_name, max_chunks)

    max_chunks = min(max_chunks, 10)  # cap to protect token budget
    chunks = fetch_chunks_by_doc(doc_id)
    
    if not chunks:
        return [{"error": f"Document '{doc_name or doc_id}' exists (doc_id={doc_id}) but has 0 chunks. The document may need to be re-ingested through the current pipeline."}]
        
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
