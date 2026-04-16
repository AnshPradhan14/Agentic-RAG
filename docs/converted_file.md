# Layer 0 --- Data Ingestion & Processing

### (Updated: Option A File Storage + Framework Selection + Full requirements.txt)

------------------------------------------------------------------------

## ⚙️ Framework Recommendation & Justification

### Orchestration: **LangChain** ✅ (Recommended over LlamaIndex)

  --------------------------------------------------------------------------
  Criterion              LangChain                 LlamaIndex
  ---------------------- ------------------------- -------------------------
  ReAct Agent Loop       Native `AgentExecutor` +  Requires more boilerplate
                         `@tool` decorator ---     for custom ReAct flows
                         fits the A-RAG paper's    
                         loop exactly              

  Local LLM (Ollama)     First-class `ChatOllama`  Supported but less
                         integration               ergonomic

  Tool binding           `llm.bind_tools([...])`   More verbose
                         --- clean, minimal        

  Custom tool control    Full control over         Abstracts away too much
                         `keyword_search`,         
                         `semantic_search`,        
                         `chunk_read`              

  SQLite + FAISS         Works natively alongside  Same
                         both, no lock-in          

  Community & docs       Larger ecosystem, more    Stronger for out-of-box
                         examples for custom RAG   RAG, weaker for custom

  **Verdict**            ✅ **Best fit for this    ❌ Over-abstracts the
                         custom agentic pipeline** agent loop we need to
                                                   control
  --------------------------------------------------------------------------

### Embedding: **sentence-transformers** (`all-MiniLM-L6-v2`) ✅

-   Fully offline, MIT license
-   384-dim embeddings --- fast on CPU
-   Best-in-class for sentence-level semantic similarity

### Vector Store: **FAISS** (`faiss-cpu`) ✅

-   `IndexFlatIP` with `normalize_L2` → exact cosine similarity (Fix 1)
-   Persisted to disk as `.index` file --- survives restarts
-   No server needed --- pure library

### PDF Parsing: **Docling** ✅

-   Offline, MIT license, handles OCR, tables, code blocks
-   Direct Markdown + JSON export --- matches Layer 0 design exactly

### Local LLM: **Ollama** (`llama3.2:3b` or `mistral:7b`) ✅

-   Runs on CPU, no API key, quantized models
-   Supports tool/function calling natively
-   `langchain-ollama` wrapper is stable and maintained

### Tokenizer: **tiktoken** ✅

-   Fast BPE tokenizer for counting tokens during chunking (Layer 1)
-   No model download needed for token counting

### Sentence Segmentation: **NLTK** (`punkt`) ✅

-   Lightweight, offline, reliable `sent_tokenize`
-   Required for Fix 3 (sentence boundary alignment) and Fix 2 (keyword
    snippet)

------------------------------------------------------------------------

## 📁 Directory Structure (Option A --- File System Storage)

```text
project/
│
├── data/
│   ├── raw/                   ← Place original PDFs here before ingestion
│   │   └── *.pdf
│   │
│   ├── parsed/                ← Docling output (auto-created by ingest.py)
│   │   ├── <doc_name>.md      ← Markdown output from Docling
│   │   └── <doc_name>.json    ← JSON metadata output from Docling
│   │
│   └── rag.db                 ← SQLite database (auto-created)
│
├── index/                     ← FAISS index files (auto-created by ingest.py)
│   ├── sentences.index        ← FAISS IndexFlatIP binary
│   └── sentences_map.json     ← Maps FAISS row → {sentence_id, chunk_id, sentence_text}
│
├── config.py
├── layer0_ingest.py
├── database.py
├── tools.py
├── agent.py
├── main.py
└── requirements.txt

Rule: data/parsed/ is written by
layer0_ingest.py during the Docling step.SQLite stores the content +
filepath reference. If the .md/.json files aredeleted, the DB still has
the full text. The files serve as a human-readable audit trail.

📦 Full
requirements.txt

# ── PDF Parsing
──────────────────────────────────────────────────────────────
docling\>=2.5.0 \# Offline PDF → Markdown + JSON (MIT license)

# ── NLP & Tokenization ───────────────────────────────────────────────────────

nltk\>=3.8.1 \# sent_tokenize for Fix 2 & Fix 3 tiktoken\>=0.7.0 \#
Token counting during chunking (Layer 1)

# ── Embeddings ───────────────────────────────────────────────────────────────

sentence-transformers\>=3.0.0 \# all-MiniLM-L6-v2 --- local sentence
embeddings torch\>=2.2.0 \# Backend for sentence-transformers (CPU
build)

# ── Vector Search ────────────────────────────────────────────────────────────

faiss-cpu\>=1.8.0 \# IndexFlatIP + normalize_L2 (Fix 1) --- CPU only

# ── Orchestration & Agent ────────────────────────────────────────────────────

langchain\>=0.3.0 \# Core: @tool, AgentExecutor, message history
langchain-ollama\>=0.2.0 \# ChatOllama wrapper for local LLM
langchain-core\>=0.3.0 \# Base abstractions (messages, tools, runnables)

# ── Storage ───────────────────────────────────────────────────────────────────

# sqlite3 is part of Python stdlib --- no pip install needed

# ── Utilities ────────────────────────────────────────────────────────────────

numpy\>=1.26.0 \# Array ops for FAISS embedding normalization
tqdm\>=4.66.0 \# Progress bars during ingestion 
python-dotenv\>=1.0.0 \#Optional: load config from .env file 

Python version: \>= 3.10 required (for list\[str\] type hints without from **future**)

Install command:Bashpip install -r requirements.txt NLTK data download (run once
after install):Pythonimport nltk nltk.download('punkt')
nltk.download('punkt_tab') Ollama model pull (run once):Bashollama pull
llama3.2:3b \# OR for better quality at cost of RAM: ollama pull
mistral:7b 


Layer 0 — Data Ingestion & Processing
Objective
Parse raw PDFs into structured Markdown and JSON using Docling, save both files to data/parsed/, then extract document-level metadata and store everything into SQLite for downstream use by the chunker (Layer 1).

PDF Parsing Tool: docling
100% offline, open-source (MIT license) — no API calls, no data leaving your infrastructure

High-fidelity parsing of complex layouts, tables, lists, code blocks, and mathematical formulas
Built-in OCR for scanned PDFs and legacy documents
Direct export to Markdown and JSON — no extra conversion steps
CPU-only efficient — runs on standard on-premise servers, no GPU required
Seamless integration with LangChain, FAISS, and SQLite

Output Format: Hybrid (Markdown + JSON) — Stored on File System
Markdown → saved to data/parsed/<doc_name>.md

For LLM comprehension and retrieval

Preserves original document structure: headers (#, ##), lists, tables, code blocks

LLMs are trained on Markdown — structural preservation improves retrieval accuracy

JSON → saved to data/parsed/<doc_name>.json

For metadata and structured filtering

Stores document-level fields:
Layer 0 --- Data Ingestion & Processing
Objective
Parse raw PDFs into structured Markdown and JSON using Docling, save both files to
data/parsed/, then extract document-level metadata and store everything
into SQLite for downstream use by the chunker (Layer 1).PDF Parsing
Tool: docling100% offline, open-source (MIT license) --- no API calls,
no data leaving your infrastructureHigh-fidelity parsing of complex
layouts, tables, lists, code blocks, and mathematical formulasBuilt-in
OCR for scanned PDFs and legacy documentsDirect export to Markdown and
JSON --- no extra conversion stepsCPU-only efficient --- runs on
standard on-premise servers, no GPU requiredSeamless integration with
LangChain, FAISS, and SQLiteOutput Format: Hybrid (Markdown + JSON) ---
Stored on File SystemMarkdown → saved to
data/parsed/`<doc_name>`{=html}.mdFor LLM comprehension and
retrievalPreserves original document structure: headers (#, ##), lists,
tables, code blocksLLMs are trained on Markdown --- structural
preservation improves retrieval accuracyJSON → saved to
data/parsed/`<doc_name>`{=html}.jsonFor metadata and structured
filteringStores document-level fields:JSON{ "source":
"UPS_Bypass_Procedure_v2.pdf", "date_issued": "2024-01-15", "tags":
\["electrical", "UPS", "bypass"\], "version": "2.0", "doc_type":
"standard_operating_procedure", "severity": "high" } (Note: page_number
and section_path are chunk-level metadata and are generated later in
Layer 1).Why hybrid + file system (Option A)?Clean separation of
concerns: Markdown for human/LLM reading, JSON for machine
filteringFiles on disk = human-readable audit trail and re-ingestion
without re-running OCR (expensive)SQLite stores the content from both
files + a filepath reference column for traceabilityIf files are
manually deleted, SQLite still retains all parsed content --- no data
lossProcessing StepsPlaintextPDF (data/raw/) │ ▼ \[1\] Run Docling on
each PDF │ ├─ Writes: data/parsed/`<doc_name>`{=html}.md │ └─ Writes:
data/parsed/`<doc_name>`{=html}.json │ ▼ \[2\] Read JSON → extract
DOCUMENT-LEVEL metadata fields: │ source, date_issued, tags, version,
doc_type, severity │ ▼ \[3\] Read Markdown → store full text │ ▼ \[4\]
Write to SQLite `documents` table: doc_id (auto), source, date_issued,
tags (JSON string), version, doc_type, severity, md_filepath,
json_filepath, full_markdown_text Code to Implement: layer0_ingest.py
--- Layer 0 SectionGenerate the following functions in
layer0_ingest.py:parse_pdf_with_docling(pdf_path: str) -\> tuple\[Path,
Path\]PlaintextPurpose : Run Docling on a single PDF file. Args :
pdf_path --- absolute or relative path to the source PDF Returns :
(md_path, json_path) --- paths to the saved .md and .json files in
data/parsed/ Side effects: - Creates data/parsed/ directory if it does
not exist - Saves `<doc_name>`{=html}.md and `<doc_name>`{=html}.json
into data/parsed/ - Logs the output paths at INFO level Raises :
FileNotFoundError if pdf_path does not exist RuntimeError if Docling
fails to produce both output files Python# Docling usage pattern
(implement this): from docling.document_converter import
DocumentConverter import json

converter = DocumentConverter() result = converter.convert(pdf_path)

markdown_text = result.document.export_to_markdown() json_data =
result.document.export_to_dict() \# or export_to_json()

# Save markdown

md_path = PARSED_DIR / f"{pdf_path.stem}.md"
md_path.write_text(markdown_text, encoding="utf-8")

# Save JSON

json_path = PARSED_DIR / f"{pdf_path.stem}.json"
json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
extract_metadata(json_path: Path) -\> dictPlaintextPurpose : Read the
Docling JSON output and extract structured DOCUMENT-LEVEL metadata
fields. Args : json_path --- path to the saved .json file in
data/parsed/ Returns : dict with keys: source, date_issued, tags,
version, doc_type, severity Notes : Use .get() with safe defaults for
all fields --- Docling JSON structure may vary by document type. Never
raise on missing fields.

          IMPORTANT: Do NOT extract `page_number` or `section_path` here. 
          Those are chunk-level attributes and will be handled by Layer 1 
          using the saved JSON file.

store_document(md_path: Path, json_path: Path, metadata: dict) -\>
intPlaintextPurpose : Insert a parsed document record into the SQLite
`documents` table. Args : md_path --- path to the .md file (stored as
md_filepath in DB) json_path --- path to the .json file (stored as
json_filepath in DB) metadata --- dict from extract_metadata() Returns :
doc_id (int) --- the auto-incremented primary key of the inserted row
Side effects: - Reads full markdown text from md_path - Inserts one row
into `documents` table with all metadata + full_markdown_text - Commits
the transaction SQLite documents table schema (implement in
database.py):SQLCREATE TABLE IF NOT EXISTS documents ( doc_id INTEGER
PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, -- original PDF
filename date_issued TEXT, -- ISO 8601 date string tags TEXT, -- JSON
array as string e.g. '\["fire","cooling"\]' version TEXT, doc_type TEXT,
-- e.g., 'standard_operating_procedure' severity TEXT,\
md_filepath TEXT NOT NULL, -- relative path to data/parsed/*.md
json_filepath TEXT NOT NULL, -- relative path to data/parsed/*.json
full_markdown_text TEXT NOT NULL, -- full parsed Markdown content (from
.md file) ingested_at TEXT DEFAULT (datetime('now')) );
ingest_pdf(pdf_path: str) -\> intPlaintextPurpose : Full Layer 0
pipeline for a single PDF --- orchestrates the three functions above in
sequence. Args : pdf_path --- path to the raw PDF in data/raw/ Returns :
doc_id (int) --- DB primary key, passed to Layer 1 (build_chunks) Flow
: 1. parse_pdf_with_docling(pdf_path) → (md_path, json_path) 2.
extract_metadata(json_path) → metadata dict 3. store_document(md_path,
json_path, metadata) → doc_id 4. Log success: f"Ingested '{source}' →
doc_id={doc_id}" 5. Return doc_id config.py values required by Layer
0Pythonimport os from pathlib import Path

# ── Project Root ──────────────────────────────────────────────────────────────

BASE_DIR = Path(**file**).resolve().parent

# ── Data Directories (Option A: File System Storage) ─────────────────────────

RAW_DIR = BASE_DIR / "data" / "raw" \# Drop PDFs here before ingesting
PARSED_DIR = BASE_DIR / "data" / "parsed" \# Docling .md and .json
outputs DB_PATH = BASE_DIR / "data" / "rag.db" \# SQLite database

# ── FAISS Index (Layer 1) ─────────────────────────────────────────────────────

INDEX_DIR = BASE_DIR / "index" FAISS_INDEX_PATH = INDEX_DIR /
"sentences.index" SENTENCES_MAP_PATH = INDEX_DIR / "sentences_map.json"

# ── Embedding Model (Layer 1) ─────────────────────────────────────────────────

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Chunking (Layer 1) ────────────────────────────────────────────────────────

CHUNK_TOKEN_LIMIT = 1000 \# Target chunk size in tokens (Fix 3 aligns to
sentence boundary)

# ── LLM (Layer 4) ────────────────────────────────────────────────────────────

OLLAMA_MODEL = "llama3.2:3b" \# Switch to "mistral:7b" for better
quality OLLAMA_TEMPERATURE = 0 Logging Standard for Layer 0Pythonimport
logging logger = logging.getLogger(**name**)

# Use these levels:

logger.info(f"Starting ingestion: {pdf_path}") logger.info(f"Docling
output saved → {md_path}, {json_path}") logger.info(f"Metadata
extracted: source={metadata\['source'\]}, tags={metadata\['tags'\]}")
logger.info(f"Stored to DB: doc_id={doc_id}") logger.debug(f"Full
metadata dict: {metadata}") logger.error(f"Docling failed for
{pdf_path}: {e}") Error Handling Rules for Layer 0ScenarioBehaviorPDF
not foundRaise FileNotFoundError with clear messageDocling fails /
crashesCatch exception, log at ERROR, re-raise as RuntimeErrorJSON
metadata field missingUse .get(key, default) --- never raise, log at
DEBUGSQLite insert failsRollback transaction, log at ERROR,
re-raisedata/parsed/ doesn't existAuto-create with mkdir(parents=True,
exist_ok=True)
