# Agentic-RAG Pipeline (A-RAG)

> **Agentic Retrieval-Augmented Generation** · 100% Offline · FastAPI + React + FAISS + Ollama + Docling

A production-grade, locally-hosted RAG pipeline implementing a comprehensive 5-Layer Architecture, advanced Table & OCR parsing, structured Entity extraction, and ReAct agent loops.

---

## 🌟 Key Features & Updates

This project has evolved into a full-stack, highly resilient Agentic-RAG application. Key updates include:

- **Full-Stack Application**: A modern React (Vite) frontend with real-time streaming, document management, and chat interface, backed by a robust FastAPI server.
- **Advanced OCR & Table Extraction**: Uses Docling with `force_full_page_ocr=True` fallback for scanned PDFs. Includes a custom `MarkdownCleaner` that structurally repairs malformed Markdown tables across page breaks.
- **Table-Aware Chunking**: Hierarchical and context-aware chunking that intelligently preserves Markdown tables and prevents chunking from breaking table rows.
- **Triple Lookup Tool (Knowledge Graph)**: Extracts Entity-Attribute-Value triples from tabular data during ingestion for deterministic fact retrieval and entity disambiguation.
- **Cross-Encoder Reranking**: Re-ranks vector search results using a cross-encoder model to dramatically improve context relevance.
- **100% Local Execution**: Completely offline operation using Ollama (Llama/Mistral), FAISS (IndexFlatIP with L2 normalization), and CPU-optimized Sentence Transformers.

---

## 🏗 Architecture: 5-Layer Pipeline

1. **Layer 0: Data Ingestion & Processing**
   - PDF ingestion via Docling (Advanced table structure extraction and OCR fallback).
   - Artifact cleanup (Hindi character removal, text normalization, and automatic Markdown table structural repair).
   - Structured `entity_triples` extraction from tables.
   - Storage in SQLite `documents` and `entity_triples` tables.
2. **Layer 1: Hierarchical Indexing**
   - Section-boundary and Table-aware chunking via NLTK `sent_tokenize`.
   - Embeddings generation via `all-MiniLM-L6-v2`.
3. **Layer 2: Storage Layer**
   - SQLite for relational metadata, chunks, and entity triples.
   - FAISS (`IndexFlatIP` + unit normalization) for exact cosine similarity vector storage.
4. **Layer 3: Runtime Agent Tools**
   - `semantic_search`: FAISS vector retrieval + Cross-encoder reranking.
   - `keyword_search`: SQL-based exact match retrieval.
   - `chunk_read`: Adjacent chunk reading for expanded context.
   - `triple_lookup`: Deterministic querying of extracted structured entities.
5. **Layer 4: Local LLM Generation**
   - LangChain ReAct loop powered by local ChatOllama.

---

## 📂 Project Structure

```text
RAG1/
├── data/                           ← Storage (SQLite, raw PDFs, parsed JSON/MD)
├── index/                          ← FAISS binary index and sentence map
├── frontend/                       ← React (Vite) User Interface
├── src/
│   ├── api/
│   │   └── server.py               ← FastAPI backend (Upload, Ingest, Ask, Stream)
│   ├── core/
│   │   ├── config.py               ← Global settings and paths
│   │   └── database.py             ← SQLite schema and CRUD operations
│   ├── ingestion/
│   │   ├── pdf_ingestor.py         ← Docling parsing & OCR Fallback
│   │   ├── markdown_cleaner.py     ← Table structure auto-correction
│   │   └── table_relation_extractor.py ← Entity Triple extractor
│   ├── indexing/
│   │   └── faiss_indexer.py        ← Table-aware chunking & FAISS embeddings
│   ├── tools/
│   │   └── rag_tools.py            ← Semantic, Keyword, Chunk, and Triple tools
│   └── agents/
│       └── rag_agent.py            ← LangChain ReAct Loop
├── run.py                          ← Unified CLI Entry Point
└── requirements.txt                ← Python dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python ≥ 3.10
- [Node.js](https://nodejs.org/) (for Frontend)
- [Ollama](https://ollama.com) installed and running

### 2. Install Backend Dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull the Local Models
```bash
ollama pull llama3.2:3b
# Or for better quality (requires ~5 GB RAM):
ollama pull mistral:7b
```

### 4. Run One-Time Setup
```bash
python run.py setup
```
*This downloads required NLTK models, initializes SQLite schemas, and prepares the directory structure.*

### 5. Start the Application

You need two terminal windows to run both the frontend and the backend.

**Terminal 1: Start Backend (FastAPI)**
```bash
python run.py server
```
*Server runs on `http://localhost:8000`*

**Terminal 2: Start Frontend (React)**
```bash
cd frontend
npm install
npm run dev  # Or python -m http.server 3000 if using static build
```
*UI runs on `http://localhost:3000`*

---

## 🛠 Usage & Features

### UI Dashboard
Navigate to `http://localhost:3000`. 
1. **Document Management:** Upload PDFs directly. The system automatically handles OCR fallback for scanned documents and fixes broken tables. You can also view and delete ingested documents.
2. **Chat Interface:** Ask questions. The agent streams responses token-by-token and intelligently calls tools (`semantic_search`, `triple_lookup`, etc.) to find the answer.

### CLI Usage (Headless Mode)
You can also run the pipeline purely from the terminal without the frontend:

**Ingest a Document:**
```bash
python run.py ingest --pdf data/raw/manual.pdf
```

**Ask a Question:**
```bash
python run.py ask --query "Summarize the cooling system procedures."
```

**Rebuild FAISS Index:**
```bash
python src/indexing/faiss_indexer.py --rebuild-all
```

---

## ⚙️ Configuration
All core parameters (model selection, chunk sizes, top-K retrieval) are centralized in `src/core/config.py`.

```python
CHUNK_TOKEN_LIMIT    = 1000
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL         = "llama3.2:3b"
RETRIEVAL_TOP_K      = 5
```
