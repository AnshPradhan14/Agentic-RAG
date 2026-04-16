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

import logging
import os
import shutil
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
        logger.error("⚠️ Failed to pre-load embedding model: %s", exc)

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
    max_iterations: int = 10


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
    """Delete a document and all its associated chunks/sentences."""
    from src.core.config import DB_PATH
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        # Check it exists
        row = conn.execute("SELECT source FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        source = row[0]
        conn.execute("DELETE FROM sentences WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id=?)", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
        conn.commit()
        logger.info("Deleted document doc_id=%d source=%s", doc_id, source)
        return {"message": f"Document '{source}' deleted successfully", "doc_id": doc_id}
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

    # Run ingestion pipeline
    try:
        doc_id = ingest_pdf(str(dest))
        chunks = build_chunks(doc_id)
        build_embeddings()

        # Reload FAISS index
        _get_faiss_index.cache_clear()
        _get_faiss_index()
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
