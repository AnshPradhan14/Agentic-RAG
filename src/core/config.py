"""
config.py — Global Configuration for the A-RAG Pipeline
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Root ──────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

# ── Data Directories ──────────────────────────────────────────────────────────
RAW_DIR: Path          = BASE_DIR / "data" / "raw"
PARSED_DIR: Path       = BASE_DIR / "data" / "parsed"
RAW_PARSED_DIR: Path   = BASE_DIR / "data" / "raw_parsed"
FINAL_TEXT_DIR: Path   = BASE_DIR / "data" / "final_text"
DB_PATH: Path          = BASE_DIR / "data" / "rag.db"

# ── Index & Vector DB ────────────────────────────────────────────────────────
INDEX_DIR: Path          = BASE_DIR / "index"
QDRANT_URL: str          = os.environ.get("QDRANT_URL", "qdrant_storage").strip()
QDRANT_COLLECTION: str   = os.environ.get("QDRANT_COLLECTION", "rag_sentences").strip()

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME: str = "qwen3-embedding:0.6b"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_TOKEN_LIMIT: int = 400
CHUNK_OVERLAP: int     = 100

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 5

# ── LLM Configuration ─────────────────────────────────────────────────────────
LLM_PROVIDER: str      = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
GROQ_MODEL: str        = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
OLLAMA_MODEL: str      = os.environ.get("OLLAMA_MODEL", "llama3").strip()
ACTIVE_MODEL: str      = GROQ_MODEL if LLM_PROVIDER == "groq" else OLLAMA_MODEL
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.0"))


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

# ── Ensure directories exist ──────────────────────────────────────────────────
for _dir in [RAW_DIR, PARSED_DIR, RAW_PARSED_DIR, FINAL_TEXT_DIR, INDEX_DIR, BASE_DIR / "data"]:
    _dir.mkdir(parents=True, exist_ok=True)