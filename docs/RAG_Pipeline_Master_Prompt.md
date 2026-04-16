# 🧠 Master Prompt — Full Optimized A-RAG Pipeline Code Generation

> **Role:** You are a senior AI/ML engineer specializing in Retrieval-Augmented Generation (RAG) systems, LangChain, FAISS, and local LLM inference. You write production-grade, fully modular, well-commented Python code.

---

## TASK

Generate the **complete, end-to-end, production-ready implementation** of an Agentic RAG (A-RAG) pipeline based on the architecture described below. The code must be **fully runnable**, **modular**, and **incorporate all 4 critical fixes** derived from the A-RAG paper (arXiv:2602.03442).

---

## ARCHITECTURE OVERVIEW

The pipeline has 5 layers:

```
Layer 0 → Data Ingestion & Processing
Layer 1 → Hierarchical Indexing (3-level)
Layer 2 → Storage Layer (SQLite + FAISS)
Layer 3 → Runtime Agent Loop (ReAct-style)
Layer 4 → Local LLM Generation (Ollama)
```

---

## DETAILED SPECIFICATIONS PER LAYER

### LAYER 0 — Data Ingestion & Processing

**PDF Parsing Tool:** `docling`
- 100% offline, open-source (MIT license)
- Handles complex layouts, tables, code blocks, math formulas
- Built-in OCR for scanned PDFs
- Exports to both Markdown and JSON
- CPU-only; no GPU required

**Output Format: Hybrid (Markdown + JSON)**
- **Markdown** → for LLM comprehension and retrieval (preserves headers `#`, `##`, lists, tables)
- **JSON** → for metadata and filtering, storing:
  - `source` (document filename)
  - `date_issued`
  - `tags` (e.g., `["fire", "cooling"]`)
  - `version`
  - `page_number`
  - `section_path` (e.g., `["Electrical", "UPS", "Bypass"]`)

**Processing Steps:**
1. Run `docling` on each PDF → produces `*.md` and `*.json`
2. Extract metadata fields from JSON
3. Store raw Markdown + metadata in a local **SQLite** database with a unique `chunk_id`

---

### LAYER 1 — Hierarchical Indexing (3 Stages)

#### Stage 1 — Chunking (Coarse Granularity, Level 3)
- **Input:** Markdown content from each document
- **Method:** Semantic chunking respecting Markdown headers (`#`, `##`); never split tables or code blocks; target size ≈ 1000 tokens
- **Output:** Chunks `c_i` each containing:
  - `chunk_id` (unique, auto-incremented)
  - `markdown_text`
  - `metadata` (inherited from document JSON + chunk-specific `start_page`)
- **Storage:** SQLite table `chunks`

> ✅ **FIX 3 — Sentence Boundary Alignment (MANDATORY)**
>
> The A-RAG paper (Section 3.1) requires chunk boundaries to align with sentence boundaries.
> Implement a `align_to_sentence_boundary()` function using `nltk.sent_tokenize` that:
> - Takes `(text: str, token_limit: int, tokenizer)` as arguments
> - Walks forward sentence by sentence, stopping before exceeding `token_limit`
> - Returns `(chunk: str, remainder: str)` — clean sentence-aligned chunk + leftover text
> - Is called **after every token-boundary cut** in the chunking loop
>
> ```python
> import nltk
> nltk.download('punkt', quiet=True)
>
> def align_to_sentence_boundary(text: str, token_limit: int, tokenizer) -> tuple[str, str]:
>     """Walk back to nearest sentence end after hitting token limit."""
>     sentences = nltk.sent_tokenize(text)
>     current_tokens = 0
>     cut = 0
>     for i, sent in enumerate(sentences):
>         sent_tokens = len(tokenizer.encode(sent))
>         if current_tokens + sent_tokens > token_limit:
>             break
>         current_tokens += sent_tokens
>         cut = i + 1
>     chunk = ' '.join(sentences[:cut])
>     remainder = ' '.join(sentences[cut:])
>     return chunk, remainder
> ```

---

#### Stage 2 — Sentence Embedding (Fine Granularity, Level 2)
- **Input:** Each chunk's `markdown_text`
- **Method:**
  1. Split into sentences using `nltk.sent_tokenize`
  2. Generate dense vectors using `sentence-transformers` model `all-MiniLM-L6-v2` (local, no API)
  3. Store each sentence vector with its `parent_chunk_id`
- **Output:** FAISS vector index + SQLite mapping of `sentence_id → parent_chunk_id + sentence_text`

> ✅ **FIX 1 — FAISS Index Type + Cosine Similarity (MANDATORY)**
>
> The A-RAG paper (Eq. 3) defines semantic search using **cosine similarity**.
> The WRONG approach used `IndexFlatL2` (Euclidean distance) and `score = 1 - dist`.
>
> **Correct Implementation:**
>
> **Indexing Stage (Fix 1A):**
> ```python
> import faiss
>
> # CORRECT: Inner product on unit vectors = cosine similarity
> vector_index = faiss.IndexFlatIP(embedding_dim)
>
> embeddings = model.encode(sentences)       # shape: (N, dim)
> faiss.normalize_L2(embeddings)             # normalize IN-PLACE to unit length
> vector_index.add(embeddings)               # now dot product == cosine similarity
> ```
>
> **Search Stage (Fix 1B):**
> ```python
> def semantic_search(query: str, top_k: int = 5) -> list[dict]:
>     q_emb = model.encode([query])
>     q_emb = q_emb.reshape(1, -1)
>     faiss.normalize_L2(q_emb)              # MUST normalize query too
>     D, I = vector_index.search(q_emb, top_k * 2)
>     chunk_scores = {}
>     for sent_id, dist in zip(I[0], D[0]):
>         parent_id = get_parent_chunk_id(sent_id)
>         score = dist                        # CORRECT: dot product on unit vectors = cosine
>         if parent_id not in chunk_scores or score > chunk_scores[parent_id][0]:
>             chunk_scores[parent_id] = (score, get_sentence_text(sent_id))
>     sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
>     return [{'chunk_id': cid, 'snippet': snippet} for cid, (_, snippet) in sorted_chunks]
> ```

---

### LAYER 2 — Storage Layer

| Level | Name | Storage Tech | Content | Accessed By |
|-------|------|-------------|---------|-------------|
| Level 1 | Keyword (runtime) | None (on-the-fly scan) | Raw chunk text | `keyword_search` tool |
| Level 2 | Sentence vectors | FAISS `IndexFlatIP` + SQLite | Normalized sentence embeddings + parent chunk ID | `semantic_search` tool |
| Level 3 | Full chunks | SQLite + file system | Markdown text + JSON metadata | `chunk_read` tool |

**SQLite Schema to implement:**
```sql
-- Documents table
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    date_issued TEXT,
    tags TEXT,       -- JSON array as string
    version TEXT,
    filepath TEXT
);

-- Chunks table
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(doc_id),
    markdown_text TEXT NOT NULL,
    start_page INTEGER,
    metadata_json TEXT   -- full JSON blob
);

-- Sentences table
CREATE TABLE sentences (
    sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT REFERENCES chunks(chunk_id),
    sentence_text TEXT NOT NULL,
    faiss_index INTEGER UNIQUE  -- maps to FAISS vector row
);
```

---

### LAYER 3 — Runtime Agent Loop (ReAct-style)

#### In-Memory Data Structures
```python
C_read: set[str] = set()  # chunk IDs that have been fully read
messages: list[dict]       # full conversation history
```

#### Tool 1 — `keyword_search`

Performs exact lexical matching. Scores each chunk: `Σ count(kw, text) × len(kw)` (paper Eq. 1).

> ✅ **FIX 2 — Remove Snippet Truncation (MANDATORY)**
>
> The A-RAG paper (Eq. 2) defines snippet = **all sentences containing any keyword**.
> The WRONG code was: `snippet_sentences[:2]` — this truncates to just 2 sentences.
>
> **Correct Implementation:**
> ```python
> def keyword_search(keywords: list[str], top_k: int = 5) -> list[dict]:
>     scores = []
>     for chunk in fetch_all_chunks():
>         text = chunk['markdown_text'].lower()
>         score = sum(text.count(kw.lower()) * len(kw) for kw in keywords)
>         if score > 0:
>             snippet_sentences = [
>                 sent for sent in split_sentences(text)
>                 if any(kw.lower() in sent for kw in keywords)
>             ]
>             snippet = ' ... '.join(snippet_sentences)  # FIX: NO [:2] truncation
>             scores.append((chunk['chunk_id'], score, snippet))
>     scores.sort(key=lambda x: x[1], reverse=True)
>     return [{'chunk_id': cid, 'snippet': snip} for cid, _, snip in scores[:top_k]]
> ```

#### Tool 2 — `semantic_search`
See Fix 1B above. Encodes query → cosine search in FAISS → aggregate by parent chunk → return top-k.

#### Tool 3 — `chunk_read`

> ✅ **FIX 4 — chunk_read Tool Description with Adjacent Chunk Hint (MANDATORY)**
>
> The LangChain `@tool` docstring for `chunk_read` MUST include the adjacent chunk navigation hint from the A-RAG paper (Figure 8, Appendix E). Without it, the LLM will not know it can read neighbouring chunks.
>
> ```python
> @tool
> def chunk_read(chunk_ids: list[str]) -> dict:
>     """Read the complete content of document chunks by their IDs.
>
>     This tool returns the full text of the specified chunks.
>
>     IMPORTANT: Search results only show abbreviated snippets — they are NOT sufficient
>     for answering questions. You MUST use chunk_read to get the full content.
>
>     STRATEGY:
>     - Always read promising chunks identified by your searches.
>     - If information seems incomplete or truncated, read adjacent chunks (+/- 1).
>     - Reading full text is essential for accurate answers.
>
>     Note: Previously read chunks will be marked as already seen.
>
>     Parameters:
>         chunk_ids: List of chunk IDs (e.g., ['0', '24', '172'])
>     """
>     results = {}
>     for cid in chunk_ids:
>         if cid in C_read:
>             results[cid] = 'This chunk has been read before'
>         else:
>             chunk = fetch_chunk_by_id(cid)
>             results[cid] = {
>                 'markdown': chunk['markdown_text'],
>                 'metadata': json.loads(chunk['metadata_json'])
>             }
>             C_read.add(cid)
>     return results
> ```

#### Agent Loop

```python
def run_agent(query: str, max_iterations: int = 10) -> str:
    global C_read
    C_read = set()
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': query}
    ]
    for _ in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        if response.tool_calls:
            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                args = tool_call['args']
                if tool_name == 'keyword_search':
                    result = keyword_search.invoke(args)
                elif tool_name == 'semantic_search':
                    result = semantic_search.invoke(args)
                elif tool_name == 'chunk_read':
                    result = chunk_read.invoke(args)
                else:
                    result = f"Unknown tool: {tool_name}"
                messages.append({
                    'role': 'tool',
                    'content': str(result),
                    'tool_call_id': tool_call['id']
                })
        else:
            return response.content
    messages.append({'role': 'user', 'content': 'Answer the question based on the information gathered so far.'})
    return llm_with_tools.invoke(messages).content
```

#### System Prompt

```
You are a question-answering assistant with access to a document corpus through available tools.
Your goal is to answer questions accurately by finding and analyzing relevant information
from the provided documents.

[Available Tools]
keyword_search  : Find chunks by exact keyword matching (use short, specific terms).
semantic_search : Find chunks by semantic similarity (use natural language queries).
chunk_read      : Read the full content of a specific chunk (always read before answering).
                  If information seems incomplete or truncated, read adjacent chunks (+/- 1).

[Strategy]
Work iteratively: search → read → evaluate → search → read → ... → answer.
For multi-hop questions, decompose the problem and tackle each sub-question step by step.

[When Answering]
Ground your response in the retrieved documents.
Cite the specific chunks (source document, page number) that support your answer.
Provide clear, direct answers supported by evidence.
Avoid speculation beyond what the documents support.
```

---

### LAYER 4 — Local LLM Generation (Ollama)

| Requirement | Implementation |
|------------|---------------|
| LLM | Ollama with `llama3.2:3b` or `mistral:7b` (tool-calling capable) |
| Inference | CPU (or GPU if available) — quantized 4-bit models |
| Integration | LangChain `ChatOllama` + `bind_tools` |

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3.2:3b', temperature=0)
llm_with_tools = llm.bind_tools([keyword_search, semantic_search, chunk_read])
```

---

## CODE GENERATION REQUIREMENTS

Generate the following **complete Python files**, each as a separate, importable module:

### 1. `config.py`
- All constants: `DB_PATH`, `FAISS_INDEX_PATH`, `SENTENCES_MAP_PATH`, `EMBEDDING_MODEL_NAME`, `CHUNK_TOKEN_LIMIT`, `OLLAMA_MODEL`
- No hardcoded values elsewhere

### 2. `ingest.py`
- `ingest_pdf(filepath: str) -> None`
  - Uses `docling` to parse PDF → markdown + JSON
  - Calls Layer 0 logic
  - Persists to SQLite `documents` table
- `build_chunks(doc_id: int) -> list[dict]`
  - Implements semantic chunking with `align_to_sentence_boundary()` (Fix 3)
  - Persists to SQLite `chunks` table
- `build_embeddings() -> None`
  - Encodes all sentences from all chunks
  - Uses `faiss.IndexFlatIP` + `faiss.normalize_L2` (Fix 1A)
  - Saves FAISS index + sentence map to disk

### 3. `database.py`
- `init_db() -> None` — creates all SQLite tables
- `fetch_all_chunks() -> list[dict]`
- `fetch_chunk_by_id(chunk_id: str) -> dict`
- `get_parent_chunk_id(faiss_row: int) -> str`
- `get_sentence_text(faiss_row: int) -> str`

### 4. `tools.py`
- All 3 LangChain `@tool` decorated functions: `keyword_search`, `semantic_search`, `chunk_read`
- `C_read` as module-level set (reset per query)
- All 4 fixes applied
- Full type hints and docstrings

### 5. `agent.py`
- `SYSTEM_PROMPT` constant
- `build_llm() -> ChatOllama`
- `run_agent(query: str, max_iterations: int = 10) -> str`
- Proper ReAct loop with full message history management

### 6. `main.py`
- CLI entrypoint with `argparse`
- Subcommands:
  - `ingest --pdf <path>` — runs full ingestion pipeline
  - `ask --query "<question>"` — runs agent loop and prints answer
- Calls `init_db()` on startup

### 7. `requirements.txt`
- All dependencies with pinned or minimum versions

---

## CODE QUALITY STANDARDS

- **Type hints** on all functions
- **Docstrings** on all public functions explaining purpose, args, and return type
- **Logging** using Python `logging` module (not `print`) — log at INFO for milestones, DEBUG for detail
- **Error handling** with try/except and meaningful error messages — never silent failures
- **No global mutable state** except `C_read` (which is explicitly reset per query)
- **No hardcoded paths or magic numbers** — all in `config.py`
- All code must be **PEP 8 compliant**
- Each file must have a module-level docstring explaining its role in the pipeline

---

## EXPLICIT CONSTRAINTS

1. **Do NOT use OpenAI API** — everything must run 100% locally (offline-first)
2. **Do NOT use `IndexFlatL2`** anywhere — only `IndexFlatIP` with `normalize_L2`
3. **Do NOT truncate keyword_search snippets** with `[:2]` or any slice
4. **Always call `align_to_sentence_boundary()`** after every token-boundary cut
5. **The `chunk_read` docstring MUST include** the adjacent chunk navigation strategy
6. The FAISS index must be saved/loaded from disk (not rebuilt each run)
7. Use `SQLite` only — no external DB servers (Postgres, Redis, etc.)
8. Use `sentence-transformers` `all-MiniLM-L6-v2` — no other embedding model unless noted

---

## OUTPUT FORMAT

Return all 7 files in order, each beginning with a header comment:

```python
# === FILE: config.py ===
```

Separate files with a clear `---` divider. No explanation needed between files — only code. After all files, provide a **Setup & Usage** section in plain text (not code) covering:
- How to install dependencies
- How to pull the Ollama model
- How to ingest documents
- How to run a query

---

## SUMMARY OF ALL 4 MANDATORY FIXES

| Fix | Location | What to Change |
|-----|----------|---------------|
| Fix 1 | `ingest.py` + `tools.py` `semantic_search` | `IndexFlatL2` → `IndexFlatIP`; `normalize_L2` on both embeddings & query; `score = dist` (not `1 - dist`) |
| Fix 2 | `tools.py` `keyword_search` | Remove `[:2]` — return **all** matching sentences in snippet |
| Fix 3 | `ingest.py` chunking | Add `align_to_sentence_boundary()` post-processing after every token-boundary cut |
| Fix 4 | `tools.py` `chunk_read` docstring | Add "read adjacent chunks (+/- 1)" strategy line to LangChain tool description |

**All 4 fixes are non-negotiable. Generate code that implements each one explicitly.**
