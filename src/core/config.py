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
RAW_DIR: Path    = BASE_DIR / "data" / "raw"
PARSED_DIR: Path = BASE_DIR / "data" / "parsed"
DB_PATH: Path    = BASE_DIR / "data" / "rag.db"

# ── FAISS Index ───────────────────────────────────────────────────────────────
INDEX_DIR: Path          = BASE_DIR / "index"
FAISS_INDEX_PATH: Path   = INDEX_DIR / "sentences.index"
SENTENCES_MAP_PATH: Path = INDEX_DIR / "sentences_map.json"

# ── Embedding Model ───────────────────────────────────────────────────────────
_LOCAL_E5:     Path = BASE_DIR / "models" / "multilingual-e5-base"
_LOCAL_MINILM: Path = BASE_DIR / "models" / "all-MiniLM-L6-v2"
if _LOCAL_E5.exists():
    EMBEDDING_MODEL_NAME: str = str(_LOCAL_E5)
elif _LOCAL_MINILM.exists():
    EMBEDDING_MODEL_NAME: str = str(_LOCAL_MINILM)
else:
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"  # fallback: HF Hub download

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_TOKEN_LIMIT: int = 400
CHUNK_OVERLAP: int     = 100

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 5

# ── LLM — Ollama (local daemon, cloud-tagged model) ───────────────────────────
# Requires: local Ollama app running + `ollama signin`
# Model runs on Ollama's cloud, routed via local daemon (localhost:11434)
OLLAMA_MODEL: str      = "qwen3.5:cloud"
LLM_TEMPERATURE: float = 0.4

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

# ── Ensure directories exist ──────────────────────────────────────────────────
for _dir in [RAW_DIR, PARSED_DIR, INDEX_DIR, BASE_DIR / "data"]:
    _dir.mkdir(parents=True, exist_ok=True)