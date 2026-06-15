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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import bcrypt

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
    "qdrant_ready": False,
    "startup_time": None,
    "query_count": 0,
}


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    logger.info("=== A-RAG Server Starting Up ===")

    from src.core.config import LLM_PROVIDER
    
    # --- Auto-start Ollama if needed ---
    if LLM_PROVIDER.lower() == "ollama":
        import urllib.request
        import urllib.error
        import subprocess
        
        logger.info("⏳ Checking if Ollama is running...")
        ollama_ready = False
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            ollama_ready = True
            logger.info("✅ Ollama is already running.")
        except (urllib.error.URLError, ConnectionError):
            logger.warning("⚠️ Ollama is not running. Attempting to start it in the background...")
            
            # Start Ollama process silently
            # On Windows, 'ollama serve' or 'ollama app' could be used. We'll try 'ollama serve'
            try:
                # Use creationflags=subprocess.CREATE_NO_WINDOW to hide the console on Windows
                creationflags = 0
                if sys.platform == "win32":
                    creationflags = subprocess.CREATE_NO_WINDOW
                    
                subprocess.Popen(
                    ["ollama", "serve"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags
                )
                
                # Wait for it to wake up
                for i in range(10):
                    import asyncio
                    await asyncio.sleep(2)
                    try:
                        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
                        ollama_ready = True
                        logger.info(f"✅ Ollama successfully started after {i*2} seconds.")
                        break
                    except (urllib.error.URLError, ConnectionError):
                        pass
                        
                if not ollama_ready:
                    logger.error("❌ Failed to verify Ollama started. It may take longer, or the command failed.")
            except Exception as e:
                logger.error(f"❌ Failed to start Ollama process automatically: {e}")

    from src.core.database import init_db
    init_db()
    logger.info("✅ Database schema ready.")

    logger.info("⏳ Loading embedding model into RAM ...")
    from src.tools.rag_tools import _get_embedding_model, _get_qdrant_client
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

    logger.info("⏳ Pre-warming Docling DocumentConverter (loads layout/table ML models)...")
    try:
        from ingestion.pdf_parser import get_document_converter
        await run_in_threadpool(get_document_converter)
        logger.info("✅ Docling DocumentConverter ready.")
    except Exception as exc:
        logger.warning("⚠️  Docling pre-warm failed (will load on first upload): %s", exc)

    try:
        client = _get_qdrant_client()
        client.get_collections()
        _state["qdrant_ready"] = True
        logger.info("✅ Qdrant connection verified.")
    except Exception as exc:
        logger.warning("⚠️  Qdrant connection failed: %s", exc)

    try:
        if LLM_PROVIDER.lower() == "ollama" and not ollama_ready:
            _state["llm"] = "error"
            logger.error("⚠️ LLM not fully ready because Ollama failed to respond.")
        else:
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

# ── Static Files ───────────────────────────────────────────────────────────────
from src.core.config import RAW_DIR
RAW_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/raw", StaticFiles(directory=str(RAW_DIR)), name="raw")


# ── Models ─────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    session_id: Optional[int] = None
    mentioned_docs: Optional[list[str]] = None

class AuthRequest(BaseModel):
    username: str
    password: str

class ChatSessionRequest(BaseModel):
    user_id: int
    title: str


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

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # Fallback for old sha256 passwords for testing/backward compatibility
    if not hashed_password.startswith('$2b$'):
        import hashlib
        return hashed_password == hashlib.sha256(plain_password.encode()).hexdigest()
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

# ── Auth & Chat Endpoints ──────────────────────────────────────────────────────

@app.post("/auth/signup")
def signup(req: AuthRequest):
    from src.core.database import create_user, get_user_by_username
    import sqlite3
    
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    
    try:
        user_id = create_user(req.username, hash_password(req.password))
        return {"message": "User created successfully", "user_id": user_id, "username": req.username, "role": "user"}
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auth/login")
def login(req: AuthRequest):
    from src.core.database import get_user_by_username
    
    user = get_user_by_username(req.username)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    return {
        "message": "Login successful", 
        "user_id": user["user_id"], 
        "username": user["username"], 
        "role": user["role"]
    }

@app.post("/chats")
def create_chat(req: ChatSessionRequest):
    from src.core.database import create_chat_session
    try:
        session_id = create_chat_session(req.user_id, req.title)
        return {"session_id": session_id, "title": req.title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chats/user/{user_id}")
def get_user_chats(user_id: int):
    from src.core.database import get_chat_sessions
    return get_chat_sessions(user_id)

@app.get("/chats/{session_id}")
def get_chat_history(session_id: int):
    from src.core.database import get_chat_messages
    return get_chat_messages(session_id)

@app.delete("/chats/{session_id}")
def delete_chat(session_id: int):
    from src.core.database import delete_chat_session
    success = delete_chat_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"message": "Chat deleted successfully"}

# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Returns the operational status of the server and models."""
    from src.core.config import LLM_PROVIDER, ACTIVE_MODEL, EMBEDDING_MODEL_NAME, CHUNK_TOKEN_LIMIT, CHUNK_OVERLAP, RETRIEVAL_TOP_K, LLM_TEMPERATURE
    return {
        "status": "online",
        "llm_status": _state["llm"],
        "embedding_model_loaded": _state["embedding_model_loaded"],
        "qdrant_ready": _state["qdrant_ready"],
        "uptime_seconds": round(time.time() - _state.get("startup_time", time.time()), 2) if _state.get("startup_time") else 0,
        "active_llm": f"{LLM_PROVIDER.upper()}: {ACTIVE_MODEL}",
        "llm_provider": LLM_PROVIDER,
        "active_model": ACTIVE_MODEL,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "query_count": _state["query_count"],
        "chunk_token_limit": CHUNK_TOKEN_LIMIT,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "llm_temperature": LLM_TEMPERATURE,
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
        "qdrant_ready": _state.get("qdrant_ready", False),
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


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    from src.core.database import fetch_chunk_by_id
    chunk = fetch_chunk_by_id(chunk_id)
    if not chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return {"chunk_id": chunk_id, "markdown_text": chunk["markdown_text"]}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, background_tasks: BackgroundTasks):
    """
    Delete a document and all its associated artifacts (DB rows, physical files, and Qdrant vectors).
    """
    from src.core.config import DB_PATH, RAW_DIR, PARSED_DIR
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

    # ── 4. Delete vectors from Qdrant ───────────────────────────────
    from ingestion.store import delete_document_vectors
    
    try:
        delete_document_vectors(doc_id)
        logger.info("Vectors deleted for doc_id=%d.", doc_id)
    except Exception as exc:
        logger.error("Failed to delete vectors from Qdrant for doc_id=%d: %s", doc_id, exc)

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

        _state["qdrant_ready"] = True

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
        from src.core.database import add_chat_message
        
        # Save user message if session exists
        if request.session_id:
            add_chat_message(request.session_id, "user", request.query, request.mentioned_docs)

        # Modify query slightly if documents are mentioned
        query_text = request.query
        if request.mentioned_docs:
            docs_context = " ".join(request.mentioned_docs)
            query_text = f"[Focus on documents: {docs_context}] {query_text}"

        answer = run_agent(
            query=query_text,
        )
        elapsed = round(time.time() - t0, 2)
        logger.info("Query answered in %.2fs", elapsed)

        # Save assistant message if session exists
        if request.session_id:
            add_chat_message(request.session_id, "assistant", answer)

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
    if not _state.get("qdrant_ready", False):
        raise HTTPException(
            status_code=400,
            detail="Qdrant connection not established. Ingest at least one PDF first via POST /upload"
        )

    from src.agents.rag_agent import run_agent_stream

    logger.info("Received streaming query: '%s'", request.query)
    _state["query_count"] += 1

    from src.core.database import add_chat_message
    
    # Save user message if session exists
    if request.session_id:
        add_chat_message(request.session_id, "user", request.query, request.mentioned_docs)

    # Modify query slightly if documents are mentioned
    query_text = request.query
    if request.mentioned_docs:
        docs_context = " ".join(request.mentioned_docs)
        query_text = f"[Focus on documents: {docs_context}] {query_text}"

    def event_generator():
        final_answer = ""
        sources = []
        try:
            for token in run_agent_stream(
                query=query_text,
            ):
                if isinstance(token, dict):
                    t = token.get("type")
                    if t == "answer":
                        final_answer += token.get("content", "")
                    elif t == "convert_to_answer":
                        final_answer = token.get("content", "")
                    elif t == "sources":
                        sources = token.get("content", [])
                
                # SSE format: each event is "data: <payload>\n\n"
                yield f"data: {json.dumps(token)}\n\n"
            
            # Save assistant message if session exists
            if request.session_id and final_answer:
                add_chat_message(request.session_id, "assistant", final_answer, sources=sources)
            
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

        _state["qdrant_ready"] = True

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
    should_reload = "--reload" in sys.argv
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=should_reload,
        log_level="info",
    )
