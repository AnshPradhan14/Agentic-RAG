# A-RAG Pipeline — Layer 0 Implementation

> **Agentic Retrieval-Augmented Generation** · 100% offline · LangChain + FAISS + Ollama + Docling

A production-grade, locally-hosted RAG pipeline implementing all 4 critical fixes from the A-RAG paper ([arXiv:2602.03442](https://arxiv.org/abs/2602.03442)).

---

## Project Structure

```
RAG1/
│
├── data/                           ← Layer 0 & Layer 2 (Storage)
│   ├── raw/                        ← Drop source PDFs here
│   ├── parsed/                     ← Docling output (.md, .json per PDF)
│   └── rag.db                      ← SQLite: documents + chunks + sentences
│
├── index/                          ← Layer 1 & Layer 2 (Vector Storage)
│   ├── sentences.index             ← FAISS IndexFlatIP binary
│   └── sentences_map.json          ← Maps FAISS row → {sentence_id, chunk_id}
│
├── config.py                       ← ALL LAYERS: Global settings, paths, model names
├── database.py                     ← LAYER 2: SQLite schema + all CRUD helpers
│
├── layer0_ingest.py                ← LAYER 0: PDF → Docling → SQLite `documents`
├── layer1_indexer.py               ← LAYER 1: Chunks (Fix 3) + FAISS embed (Fix 1A)
│
├── layer3_tools.py                 ← LAYER 3: keyword_search, semantic_search, chunk_read
├── layer4_agent.py                 ← LAYER 4: LangChain ReAct loop + Ollama LLM
│
├── main.py                         ← ENTRY POINT: CLI (ingest / ask)
├── setup_once.py                   ← One-time setup (NLTK, DB init, dir creation)
└── requirements.txt                ← All dependencies
```

---

## Architecture: 5-Layer Pipeline

```
Layer 0 → Data Ingestion & Processing   (Docling, SQLite documents table)
Layer 1 → Hierarchical Indexing         (NLTK chunking, FAISS embeddings)
Layer 2 → Storage Layer                 (SQLite + FAISS on disk)
Layer 3 → Runtime Agent Tools           (keyword_search, semantic_search, chunk_read)
Layer 4 → Local LLM Generation          (ChatOllama + ReAct loop)
```

---

## 4 Critical A-RAG Fixes (all implemented)

| Fix | Location | What it fixes |
|-----|----------|---------------|
| **Fix 1** | `layer1_indexer.py` + `layer3_tools.py` | `IndexFlatL2` → `IndexFlatIP`; `normalize_L2` on embeddings **and** query; `score = dot_product` |
| **Fix 2** | `layer3_tools.py` `keyword_search` | Removes `[:2]` truncation — returns **all** matching sentences in snippet |
| **Fix 3** | `layer1_indexer.py` `align_to_sentence_boundary()` | Called after every token-boundary cut — chunks never break mid-sentence |
| **Fix 4** | `layer3_tools.py` `chunk_read` docstring | Includes adjacent-chunk navigation hint — LLM knows it can read `chunk_id ± 1` |

---

## Quick Start

### 1. Prerequisites

- Python ≥ 3.10
- [Ollama](https://ollama.com) installed and running

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the local LLM

```bash
ollama pull llama3.2:3b
# OR for better quality (needs ~5 GB RAM):
ollama pull mistral:7b
```

### 4. One-time setup

```bash
python setup_once.py
```

This downloads NLTK tokenizer data, creates directory structure, and initialises the SQLite schema.

### 5. Ingest a PDF

```bash
# Copy your PDF into data/raw/ first, then:
python main.py ingest --pdf data/raw/your_document.pdf
```

### 6. Ask a question

```bash
python main.py ask --query "What is the UPS bypass procedure?"
```

---

## Detailed Usage

### Ingest subcommand

```bash
python main.py ingest --pdf data/raw/manual.pdf
# Skip embedding step (batch multiple PDFs first, embed once at end):
python main.py ingest --pdf data/raw/manual.pdf --skip-embeddings
# Then rebuild all embeddings at once:
python layer1_indexer.py --rebuild-all
```

### Ask subcommand

```bash
python main.py ask --query "What are the safety risks for electrical bypass?"
python main.py ask --query "Summarise the cooling system procedures" --max-iterations 15
python main.py --log-level DEBUG ask --query "debug run"
```

### Standalone Layer 0 (ingest only, no embedding)

```bash
python layer0_ingest.py --pdf data/raw/my_doc.pdf
```

### Standalone Layer 1 (chunk + embed)

```bash
python layer1_indexer.py --doc-id 1        # process a specific document
python layer1_indexer.py --rebuild-all     # re-index all documents
```

---

## Technology Stack

| Component | Library | Why |
|-----------|---------|-----|
| PDF Parsing | `docling` | Offline, OCR, tables, code blocks, MIT license |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | 384-dim, CPU, MIT license |
| Vector Store | `faiss-cpu` (`IndexFlatIP`) | Exact cosine similarity, persisted to disk |
| Orchestration | `langchain` + `langchain-ollama` | Native ReAct, `@tool`, `bind_tools` |
| Local LLM | Ollama (`llama3.2:3b`) | CPU-quantized, tool-calling, no API key |
| Storage | SQLite (stdlib) | No server, portable, ACID transactions |
| Tokenizer | `tiktoken` | Fast BPE token counting for chunking |
| Sentence split | `nltk` (`punkt`) | Reliable sent_tokenize for Fix 2 & Fix 3 |

---

## Configuration

All settings are in `config.py` — no hardcoded values anywhere else.

```python
CHUNK_TOKEN_LIMIT    = 1000          # Target max tokens per chunk
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL         = "llama3.2:3b" # Switch to "mistral:7b" for quality
OLLAMA_TEMPERATURE   = 0             # Deterministic output
```

---

## Error Handling

| Scenario | Behaviour |
|----------|-----------|
| PDF not found | `FileNotFoundError` with clear message |
| Docling fails | Caught, logged at ERROR, re-raised as `RuntimeError` |
| JSON metadata field missing | `.get(key, default)` — never raises, logged at DEBUG |
| SQLite insert fails | Rollback, log at ERROR, re-raise |
| `data/parsed/` missing | Auto-created with `mkdir(parents=True, exist_ok=True)` |
| Ollama not running | `ConnectionRefusedError` caught in `main.py` with actionable message |
| FAISS index missing | `FileNotFoundError` with instructions to run ingestion first |
