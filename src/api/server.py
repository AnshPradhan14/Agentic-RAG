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
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
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

    try:
        # LLM client is built per-request via _unified_chat now.
        _state["llm"] = "ready"
        logger.info("✅ LLM ready (unified chat).")
    except Exception as exc:
        logger.error("⚠️ Failed to setup LLM state: %s", exc)
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
    is_gemc: Optional[bool] = None


class IngestResponse(BaseModel):
    doc_id: int
    source: str
    chunks_created: int
    message: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    from src.core.config import LLM_PROVIDER, ACTIVE_MODEL, EMBEDDING_MODEL_NAME
    return {
        "status": "ok",
        "embedding_model_loaded": _state["embedding_model_loaded"],
        "faiss_index_loaded": _state["faiss_index_loaded"],
        "ollama_ready": _state["llm"] is not None,
        "startup_time_seconds": _state["startup_time"],
        "active_llm": f"{LLM_PROVIDER.upper()}: {ACTIVE_MODEL}",
        "llm_provider": LLM_PROVIDER,
        "active_model": ACTIVE_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "query_count": _state["query_count"],
    }


@app.get("/stats")
def get_stats():
    """Return overall system stats for the admin dashboard."""
    from src.core.database import fetch_all_documents, fetch_all_chunks
    from src.core.config import LLM_PROVIDER, ACTIVE_MODEL, EMBEDDING_MODEL_NAME

    docs = fetch_all_documents()
    chunks = fetch_all_chunks()

    return {
        "total_documents": len(docs),
        "total_chunks": len(chunks),
        "total_queries": _state["query_count"],
        "active_llm": f"{LLM_PROVIDER.upper()}: {ACTIVE_MODEL}",
        "llm_provider": LLM_PROVIDER,
        "active_model": ACTIVE_MODEL,
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
def delete_document(doc_id: int, background_tasks: BackgroundTasks):
    """
    Delete a document and all its associated artifacts (DB rows, physical files, and FAISS vectors).
    """
    from src.core.config import DB_PATH, RAW_DIR, PARSED_DIR
    from src.tools.rag_tools import _get_faiss_index
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        # 1. Fetch metadata before deletion
        row = conn.execute(
            "SELECT source FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
        
        if row is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        source_filename = row[0]

        # 2. Delete DB records
        conn.execute("DELETE FROM entity_triples WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM sentences WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        if doc_count == 0:
            logger.info("Database is empty. Resetting AUTOINCREMENT sequences.")
            conn.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('documents', 'sentences', 'entity_triples')"
            )
        conn.commit()
        logger.info("Deleted DB records for doc_id=%d (%s)", doc_id, source_filename)

    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        logger.exception("Failed to delete DB records for doc_id=%d: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        conn.close()

    # ── 3. Delete physical files (best-effort — never raise 500 here) ─────────
    from src.core.config import FINAL_TEXT_DIR
    pdf_stem = Path(source_filename).stem
    deleted_files: list[str] = []
    files_to_delete = [
        RAW_DIR        / source_filename,
        PARSED_DIR     / f"{pdf_stem}_raw.md",
        FINAL_TEXT_DIR / f"{pdf_stem}_structured.md",
        FINAL_TEXT_DIR / f"{pdf_stem}_chunks.json",
        FINAL_TEXT_DIR / f"{pdf_stem}_metadata.json",
    ]
    for p in files_to_delete:
        try:
            if p.exists():
                p.unlink()
                deleted_files.append(str(p))
                logger.info("Deleted physical file: %s", p)
            else:
                logger.warning("File not found for deletion (skipping): %s", p)
        except Exception as exc:
            logger.warning("Could not delete file %s: %s", p, exc)

    # ── 4. Schedule FAISS rebuild in background ───────────────────────────────
    _get_faiss_index.cache_clear()
    from ingestion.store import rebuild_faiss_index
    
    def _background_rebuild():
        try:
            remaining = rebuild_faiss_index()
            logger.info("FAISS index rebuilt in background with %d remaining vector(s).", remaining)
            _state["faiss_index_loaded"] = remaining > 0
        except Exception as exc:
            logger.error("FAISS background rebuild failed after deletion of doc_id=%d: %s", doc_id, exc)
            
    background_tasks.add_task(_background_rebuild)

    return {
        "message": f"Document '{source_filename}' deleted successfully",
        "doc_id": doc_id,
        "deleted_files": deleted_files,
        "system_reset": doc_count == 0,
    }



@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), is_gemc: Optional[bool] = Form(None)):
    """
    Upload a PDF file directly via multipart form.
    Saves it to data/raw/, then runs the full ingestion pipeline.
    Heavy I/O work is offloaded to a thread pool via run_in_threadpool to avoid
    blocking the FastAPI event loop.
    """
    from src.core.config import RAW_DIR, PARSED_DIR
    from ingestion.pipeline import run_ingestion
    from src.core.database import get_connection

    # Save to RAW_DIR
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RAW_DIR / file.filename
    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        logger.error(f"Failed to save uploaded file {file.filename}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {exc}")

    # Run ingestion
    try:
        summary = await run_in_threadpool(run_ingestion, str(dest_path), str(PARSED_DIR))
        
        if summary.get("status") == "skipped_duplicate":
            conn = get_connection()
            try:
                row = conn.execute("SELECT doc_id FROM documents WHERE source=?", (file.filename,)).fetchone()
                doc_id = row[0] if row else -1
            finally:
                conn.close()
            return IngestResponse(
                doc_id=doc_id,
                source=file.filename,
                chunks_created=0,
                message="Skipped: Document already ingested."
            )

        if summary.get("errors") and not summary.get("total_chunks"):
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {summary['errors']}")

        # Clear FAISS index cache in the server so next query loads updated index
        from src.tools.rag_tools import _get_faiss_index
        _get_faiss_index.cache_clear()
        _state["faiss_index_loaded"] = True

        conn = get_connection()
        try:
            row = conn.execute("SELECT doc_id FROM documents WHERE source=?", (file.filename,)).fetchone()
            doc_id = row[0] if row else -1
        finally:
            conn.close()

        return IngestResponse(
            doc_id=doc_id,
            source=file.filename,
            chunks_created=summary.get("total_chunks", 0),
            message="Document uploaded and ingested successfully."
        )
    except Exception as exc:
        logger.exception("Error during upload/ingestion: %s", exc)
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

    from src.agents.rag_agent import run_agent

    t0 = time.time()
    logger.info("Received query: '%s'", request.query)
    _state["query_count"] += 1

    try:
        answer = run_agent(
            query=request.query,
            max_iterations=request.max_iterations,
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

    from src.agents.rag_agent import run_agent_stream

    logger.info("Received streaming query: '%s'", request.query)
    _state["query_count"] += 1

    def event_generator():
        try:
            for token in run_agent_stream(
                query=request.query,
                max_iterations=request.max_iterations,
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
    from src.core.config import PARSED_DIR
    from ingestion.pipeline import run_ingestion
    from src.core.database import get_connection

    pdf_path = Path(request.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF file not found at path: {request.pdf_path}")

    filename = pdf_path.name
    try:
        summary = run_ingestion(str(pdf_path), str(PARSED_DIR))
        
        if summary.get("status") == "skipped_duplicate":
            conn = get_connection()
            try:
                row = conn.execute("SELECT doc_id FROM documents WHERE source=?", (filename,)).fetchone()
                doc_id = row[0] if row else -1
            finally:
                conn.close()
            return IngestResponse(
                doc_id=doc_id,
                source=filename,
                chunks_created=0,
                message="Skipped: Document already ingested."
            )

        if summary.get("errors") and not summary.get("total_chunks"):
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {summary['errors']}")

        # Clear FAISS index cache in the server so next query loads updated index
        from src.tools.rag_tools import _get_faiss_index
        _get_faiss_index.cache_clear()
        _state["faiss_index_loaded"] = True

        conn = get_connection()
        try:
            row = conn.execute("SELECT doc_id FROM documents WHERE source=?", (filename,)).fetchone()
            doc_id = row[0] if row else -1
        finally:
            conn.close()

        return IngestResponse(
            doc_id=doc_id,
            source=filename,
            chunks_created=summary.get("total_chunks", 0),
            message="Document ingested successfully."
        )
    except Exception as exc:
        logger.exception("Error during disk ingestion: %s", exc)
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
