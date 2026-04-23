"""
server.py — FastAPI Server for the A-RAG Pipeline

Runs as a persistent server so libraries and the embedding model are loaded
ONCE on startup and stay in RAM. This eliminates the ~20-second cold start
penalty of the CLI approach.

Endpoints:
    POST /upload               — Upload a PDF file directly (multipart/form-data)
    POST /ingest               — Ingest a PDF already on disk by path
    POST /ask                  — Ask a question, get an answer from the A-RAG agent
    GET  /health               — Check server status and loaded models
    GET  /documents            — List all ingested documents
    DELETE /documents/{doc_id} — Delete a document and its chunks
    GET  /stats                — Get overall stats (doc count, query count, etc.)
    GET  /docs                 — Auto-generated Swagger UI
"""

import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── Global state ───────────────────────────────────────────────────────────────
_state: dict = {
    "llm": None,
    "embedding_model_loaded": False,
    "faiss_index_loaded": False,
    "startup_time": None,
    "query_count": 0,
}


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    logger.info("=== A-RAG Server Starting Up ===")

    from src.core.database import init_db
    init_db()
    logger.info("✅ Database schema ready.")

    logger.info("⏳ Loading embedding model into RAM ...")
    from src.tools.rag_tools import _get_embedding_model, _get_faiss_index
    try:
        _get_embedding_model()
        _state["embedding_model_loaded"] = True
        logger.info("✅ Embedding model loaded.")
    except Exception as exc:
        import traceback
        logger.error("⚠️ Failed to pre-load embedding model: %s\n%s", exc, traceback.format_exc())
        logger.error("⚠️ Retrying embedding model load in 3 seconds...")
        import asyncio
        await asyncio.sleep(3)
        try:
            _get_embedding_model.cache_clear()
            _get_embedding_model()
            _state["embedding_model_loaded"] = True
            logger.info("✅ Embedding model loaded on retry.")
        except Exception as exc2:
            logger.error("⚠️ Embedding model retry also failed: %s", exc2)

    try:
        _get_faiss_index()
        _state["faiss_index_loaded"] = True
        logger.info("✅ FAISS index loaded.")
    except FileNotFoundError:
        logger.warning("⚠️  FAISS index not found. Ingest a PDF first.")

    from src.agents.rag_agent import build_llm
    try:
        _state["llm"] = build_llm()
        logger.info("✅ Ollama client ready.")
    except Exception as exc:
        logger.error("⚠️ Failed to build LLM client: %s", exc)
        _state["llm"] = None

    _state["startup_time"] = round(time.time() - t0, 2)
    logger.info(f"=== Server ready in {_state['startup_time']}s. ===")

    yield

    logger.info("=== A-RAG Server shutting down ===")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="A-RAG Pipeline API",
    description="Agentic RAG — Ask questions about your ingested documents.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the static frontend (port 8080) to call the API (port 8000) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Models ─────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    max_iterations: int = 5  # Reduced from 10 to prevent runaway loops and context blowout


class AskResponse(BaseModel):
    query: str
    answer: str
    iterations_used: int
    time_seconds: float


class IngestRequest(BaseModel):
    pdf_path: str


class IngestResponse(BaseModel):
    doc_id: int
    source: str
    chunks_created: int
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    from src.core.config import OLLAMA_MODEL, EMBEDDING_MODEL_NAME
    return {
        "status": "ok",
        "embedding_model_loaded": _state["embedding_model_loaded"],
        "faiss_index_loaded": _state["faiss_index_loaded"],
        "ollama_ready": _state["llm"] is not None,
        "startup_time_seconds": _state["startup_time"],
        "active_llm": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "query_count": _state["query_count"],
    }


@app.get("/stats")
def get_stats():
    """Return overall system stats for the admin dashboard."""
    from src.core.database import fetch_all_documents, fetch_all_chunks
    from src.core.config import OLLAMA_MODEL, EMBEDDING_MODEL_NAME

    docs = fetch_all_documents()
    chunks = fetch_all_chunks()

    return {
        "total_documents": len(docs),
        "total_chunks": len(chunks),
        "total_queries": _state["query_count"],
        "active_llm": OLLAMA_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "faiss_index_loaded": _state["faiss_index_loaded"],
        "embedding_model_loaded": _state["embedding_model_loaded"],
    }


@app.get("/documents")
def list_documents():
    """List all ingested documents with metadata."""
    from src.core.database import fetch_all_documents, fetch_all_chunks

    docs = fetch_all_documents()
    all_chunks = fetch_all_chunks()

    # Count chunks per doc
    chunk_counts: dict = {}
    for c in all_chunks:
        chunk_counts[c["doc_id"]] = chunk_counts.get(c["doc_id"], 0) + 1

    result = []
    for doc in docs:
        result.append({
            "doc_id": doc["doc_id"],
            "source": doc["source"],
            "date_issued": doc.get("date_issued"),
            "doc_type": doc.get("doc_type"),
            "ingested_at": doc.get("ingested_at"),
            "chunk_count": chunk_counts.get(doc["doc_id"], 0),
            "status": "indexed",
        })
    return result


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    """
    Delete a document and all its associated artifacts (DB rows, physical files, and FAISS vectors).
    Fixes:
    - Deletes source PDF from data/raw/
    - Deletes parsed JSON/MD from data/parsed/
    - Cleans up sentences, chunks, and entity triples
    - Resets sqlite_sequence if the DB is empty
    - Rebuilds or removes FAISS index to prevent stale searches
    """
    from src.core.config import DB_PATH, RAW_DIR, BASE_DIR
    from src.indexing.faiss_indexer import build_embeddings
    from src.tools.rag_tools import _get_faiss_index
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        # ── 1. Fetch paths before deletion ─────────────────────────────────────
        row = conn.execute(
            "SELECT source, md_filepath, json_filepath FROM documents WHERE doc_id=?",
            (doc_id,)
        ).fetchone()
        
        if row is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        source, md_filepath, json_filepath = row[0], row[1], row[2]
        source_pdf_path = RAW_DIR / source

        # ── 2. Delete DB rows ──────────────────────────────────────────────────
        # Chunks and Sentences will cascade if FKs are set, but we be explicit
        conn.execute("DELETE FROM entity_triples WHERE doc_id=?", (str(doc_id),))
        conn.execute("DELETE FROM sentences WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        
        # Check if DB is now empty to reset sequences (Fixes user request)
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if doc_count == 0:
            logger.info("Database is empty. Resetting AUTOINCREMENT sequences.")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('documents', 'sentences', 'entity_triples')")
        
        conn.commit()
        logger.info("Deleted DB records for doc_id=%d (%s)", doc_id, source)

        # ── 3. Delete physical files from disk ─────────────────────────────────
        deleted_files = []
        
        # Files to delete: [Source PDF, Parsed MD, Parsed JSON]
        file_paths_to_clean = [source_pdf_path]
        for fpath in [md_filepath, json_filepath]:
            if fpath:
                p = Path(fpath)
                if not p.is_absolute():
                    p = BASE_DIR / p
                file_paths_to_clean.append(p)

        for p in file_paths_to_clean:
            if p.exists():
                p.unlink()
                deleted_files.append(str(p))
                logger.info("Deleted physical file: %s", p)
            else:
                logger.warning("File not found for deletion: %s", p)

        # ── 4. Fix FAISS Inconsistency (Crucial) ──────────────────────────────
        # If we delete a doc but leave its vectors in FAISS, search results will be broken.
        _get_faiss_index.cache_clear() # Clear the singleton cache
        
        if doc_count > 0:
            logger.info("Rebuilding FAISS index for remaining %d documents...", doc_count)
            # This is a bit slow but ensures 100% consistency. 
            # In a production app, we'd use faiss.remove_ids.
            build_embeddings() 
            _state["faiss_index_loaded"] = True
        else:
            logger.info("No documents left. Clearing FAISS index files.")
            from src.core.config import FAISS_INDEX_PATH, SENTENCES_MAP_PATH
            if FAISS_INDEX_PATH.exists(): FAISS_INDEX_PATH.unlink()
            if SENTENCES_MAP_PATH.exists(): SENTENCES_MAP_PATH.unlink()
            _state["faiss_index_loaded"] = False

        return {
            "message": f"Document '{source}' and all related data deleted successfully",
            "doc_id": doc_id,
            "deleted_files": deleted_files,
            "system_reset": doc_count == 0
        }

    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        logger.exception("Failed to delete document %d: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file directly via multipart form.
    Saves it to data/raw/, then runs the full ingestion pipeline.
    Heavy I/O work is offloaded to a thread pool via run_in_threadpool to avoid
    blocking the FastAPI event loop.
    """
    from src.core.config import RAW_DIR
    from src.ingestion.pdf_ingestor import ingest_pdf
    from src.indexing.faiss_indexer import build_chunks, build_embeddings
    from src.tools.rag_tools import _get_faiss_index

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Check for duplicate
    dest = RAW_DIR / file.filename
    if dest.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Error: Document '{file.filename}' already exists in the database."
        )

    # Save uploaded file to data/raw/
    try:
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info("Uploaded PDF saved to: %s", dest)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    # Run ingestion pipeline in a thread pool to avoid blocking the event loop
    try:
        doc_id   = await run_in_threadpool(ingest_pdf, str(dest))
        chunks   = await run_in_threadpool(build_chunks, doc_id)
        await run_in_threadpool(build_embeddings)

        # Reload FAISS index
        _get_faiss_index.cache_clear()
        await run_in_threadpool(_get_faiss_index)
        _state["faiss_index_loaded"] = True

        logger.info("Upload & ingestion complete: doc_id=%d, chunks=%d", doc_id, len(chunks))
        return {
            "doc_id": doc_id,
            "source": file.filename,
            "chunks_created": len(chunks),
            "message": f"✅ Successfully ingested '{file.filename}'",
        }
    except Exception as exc:
        # Clean up the uploaded file if ingestion fails
        if dest.exists():
            dest.unlink()
        logger.exception("Ingestion failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Ask a question. The agent searches ingested documents and returns an answer."""
    if _state["llm"] is None:
        raise HTTPException(status_code=503, detail="LLM not loaded. Check server logs.")
    if not _state["embedding_model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded. Check server logs for the root cause (e.g. HuggingFace connectivity or missing model files)."
        )
    if not _state["faiss_index_loaded"]:
        raise HTTPException(
            status_code=400,
            detail="No FAISS index found. Ingest at least one PDF first via POST /upload"
        )

    from src.agents.rag_agent import build_llm, run_agent

    t0 = time.time()
    logger.info("Received query: '%s'", request.query)
    _state["query_count"] += 1

    try:
        llm = build_llm()
        answer = run_agent(
            query=request.query,
            max_iterations=request.max_iterations,
            llm_with_tools=llm,
        )
        elapsed = round(time.time() - t0, 2)
        logger.info("Query answered in %.2fs", elapsed)

        return AskResponse(
            query=request.query,
            answer=answer,
            iterations_used=0,
            time_seconds=elapsed,
        )

    except Exception as exc:
        logger.exception("Error during agent run: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ask_stream")
def ask_stream(request: AskRequest):
    """
    Ask a question with streaming response.
    Streams the final LLM answer token-by-token using Server-Sent Events (SSE).
    The ReAct tool-calling loop runs synchronously first, then the final answer
    is streamed back to the client as it is generated by the LLM.

    Event format:
        data: <token_text>\n\n          — a streamed token chunk
        data: [DONE]\n\n               — signals end of stream
        data: {"error": "..."}\n\n     — on failure
    """
    if _state["llm"] is None:
        raise HTTPException(status_code=503, detail="LLM not loaded. Check server logs.")
    if not _state["embedding_model_loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not loaded. Check server logs for the root cause (e.g. HuggingFace connectivity or missing model files)."
        )
    if not _state["faiss_index_loaded"]:
        raise HTTPException(
            status_code=400,
            detail="No FAISS index found. Ingest at least one PDF first via POST /upload"
        )

    from src.agents.rag_agent import build_llm, run_agent_stream

    logger.info("Received streaming query: '%s'", request.query)
    _state["query_count"] += 1

    def event_generator():
        try:
            llm = build_llm()
            for token in run_agent_stream(
                query=request.query,
                max_iterations=request.max_iterations,
                llm_with_tools=llm,
            ):
                # SSE format: each event is "data: <payload>\n\n"
                yield f"data: {json.dumps(token)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("Streaming agent error: %s", exc)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """Ingest a PDF already on disk by providing its path."""
    from src.ingestion.pdf_ingestor import ingest_pdf
    from src.indexing.faiss_indexer import build_chunks, build_embeddings
    from src.tools.rag_tools import _get_faiss_index

    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_path}")

    try:
        doc_id = ingest_pdf(str(pdf_path))
        chunks = build_chunks(doc_id)
        build_embeddings()

        _get_faiss_index.cache_clear()
        _get_faiss_index()
        _state["faiss_index_loaded"] = True

        return IngestResponse(
            doc_id=doc_id,
            source=str(pdf_path),
            chunks_created=len(chunks),
            message=f"✅ Successfully ingested '{pdf_path.name}'",
        )
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
