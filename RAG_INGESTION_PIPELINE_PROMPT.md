# RAG Ingestion Pipeline — Full Rebuild Prompt
> Copy-paste this into Claude Code / Cursor / any AI coding agent.
> This replaces ALL existing parsing/ingestion code from scratch.

---

## YOUR MISSION

You are rebuilding the **entire PDF ingestion pipeline** for an enterprise RAG system from absolute zero.
Delete all existing parsing, chunking, and ingestion code first.
Then build the new pipeline exactly as specified below.
Do not preserve any old logic. Do not refactor old code. Full rewrite only.

---

## PHASE 0 — CLEANUP

Delete the following completely (ask user to confirm directory paths if unsure):
- All files related to PDF parsing, text extraction, markdown conversion
- All files related to chunking or text splitting
- All files related to document ingestion or indexing pipeline
- Any utility files that only served those deleted modules

**Keep:**
- Vector store connection/client code
- Embedding model wrapper code
- Environment config files (`.env`, `config.py`, etc.)
- Database schema files (we will extend them, not delete)
- Any existing API route files unrelated to ingestion

---

## PHASE 1 — NEW PROJECT STRUCTURE

Create this exact folder structure under the project root:

```
ingestion/
├── __init__.py
├── pipeline.py              # Master orchestrator — entry point
├── pdf_parser.py            # Docling-based raw markdown extraction
├── language_filter.py       # Strip Hindi/Devanagari text, keep English only
├── batch_processor.py       # Split PDFs >50 pages into 25-page batches
├── llm_restructurer.py      # LLM call to fix markdown hierarchy
├── metadata_builder.py      # LLM call to extract section JSON metadata
├── chunker.py               # Section-aware chunking using metadata JSON
├── normalizer.py            # Text normalization (whitespace, encoding, etc.)
├── store.py                 # Save final chunks to vector store + disk
└── prompts/
    ├── restructure_prompt.txt     # System prompt for LLM restructurer
    └── metadata_extract_prompt.txt # System prompt for metadata extractor
```

---

## PHASE 2 — DEPENDENCIES

Ensure these are installed and added to `requirements.txt`:

```
docling
langdetect
anthropic
regex
tiktoken
```

---

## PHASE 3 — `language_filter.py`

**Purpose:** Remove all Hindi/Devanagari text from raw Docling output.
Keep only English text. Operate at paragraph/line level.

**Logic:**
- Split text into paragraphs (split on double newline `\n\n`)
- For each paragraph, calculate ratio of Devanagari characters
  - Devanagari Unicode range: `\u0900–\u097F`
  - If Devanagari character ratio > 0.15 (15%), discard entire paragraph
- For lines within a retained paragraph: if a single line is >40% Devanagari, remove that line only
- Preserve blank lines and markdown heading markers (`#`) — only strip text content
- Preserve table structures — if a table row contains Hindi text, replace that cell content with `[HINDI_OMITTED]` rather than deleting the entire row (preserves table structure for LLM)
- After filtering, collapse 3+ consecutive blank lines into 2

```python
import regex

DEVANAGARI_PATTERN = regex.compile(r'[\u0900-\u097F]')

def devanagari_ratio(text: str) -> float:
    if not text.strip():
        return 0.0
    devanagari_chars = len(DEVANAGARI_PATTERN.findall(text))
    return devanagari_chars / len(text)

def filter_hindi(raw_text: str) -> str:
    # Implement paragraph-level and line-level filtering as described above
    # Return cleaned English-only text
    ...
```

---

## PHASE 4 — `pdf_parser.py`

**Purpose:** Use Docling to convert PDF pages to raw markdown.

```python
from docling.document_converter import DocumentConverter

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert entire PDF to raw markdown using Docling.
    Returns raw unstructured markdown string.
    """
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()
```

**Notes:**
- Do not post-process Docling output here — that is language_filter and llm_restructurer's job
- If Docling raises on a page, log the error with page number and continue (don't abort full pipeline)

---

## PHASE 5 — `batch_processor.py`

**Purpose:** Split large PDFs into 25-page batches before LLM restructuring.

**Logic:**
- Count total pages of PDF
- If total pages <= 50: process as single batch
- If total pages > 50: split into batches of 25 pages each
  - Batch 1: pages 1–25
  - Batch 2: pages 26–50
  - Batch N: pages ((N-1)*25)+1 to min(N*25, total_pages)
- Use PyMuPDF (`fitz`) or `pypdf` to extract page ranges as temp PDF files
- Each batch PDF gets parsed independently by `pdf_parser.py`
- After all batches are LLM-restructured, stitch them in order with a `\n\n---BATCH_BREAK---\n\n` separator
- Final stitching pass: remove `---BATCH_BREAK---` markers and ensure heading continuity across batches

```python
def get_batch_ranges(total_pages: int, batch_size: int = 25, threshold: int = 50) -> list[tuple[int,int]]:
    """Returns list of (start_page, end_page) tuples. Pages are 1-indexed."""
    if total_pages <= threshold:
        return [(1, total_pages)]
    ranges = []
    start = 1
    while start <= total_pages:
        end = min(start + batch_size - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges
```

---

## PHASE 6 — `prompts/restructure_prompt.txt`

**This is the exact system prompt for the LLM restructuring call.**
Save this verbatim as a `.txt` file and load it in `llm_restructurer.py`.

```
You are a precise document structure specialist. Your only job is to take raw, poorly-formatted markdown text extracted from a PDF and reformat it into clean, well-organized markdown that accurately reflects the original PDF's heading hierarchy and content structure.

STRICT RULES — violating any of these is a critical failure:

1. CONTENT PRESERVATION
   - You MUST NOT add, remove, summarize, paraphrase, or alter any English text content.
   - Every sentence, number, statistic, list item, table value, and technical term in the input MUST appear in the output unchanged.
   - Do not fix grammar. Do not rephrase. Do not improve wording. Copy text exactly.

2. HIERARCHY RECONSTRUCTION
   - Identify the document's heading hierarchy by analyzing font-size signals, numbering patterns (1., 1.1, 1.1.1), capitalization, and contextual clues in the raw text.
   - Map headings to correct markdown levels:
     - Document title or top-level chapter → # (H1)
     - Major section → ## (H2)
     - Subsection → ### (H3)
     - Sub-subsection → #### (H4)
     - Never go deeper than H4 unless the document explicitly has 5+ levels
   - Numbered headings (e.g., "3.2.1 Methodology") → preserve the number in the heading text

3. TABLE RECONSTRUCTION
   - If raw text contains data that was clearly a table in the PDF (aligned columns, repeated delimiters, grid-like structure), reconstruct it as a proper GitHub-flavored markdown table.
   - Preserve all cell values exactly.
   - If table structure is ambiguous or corrupted beyond recovery, wrap the raw data in a fenced code block with the label `table`.

4. LIST RECONSTRUCTION
   - Reconstruct bullet lists and numbered lists using proper markdown syntax.
   - Preserve nesting levels of lists.

5. HINDI/NON-ENGLISH TEXT
   - If any Devanagari or Hindi text appears in the input (it should have been filtered, but if remnants exist), silently omit it. Do not note its omission. Do not insert a placeholder.

6. WHAT YOU MUST NOT DO
   - Do not add an introduction, summary, or conclusion that wasn't in the original.
   - Do not add metadata headers (author, date, etc.) unless they were in the original text.
   - Do not wrap the output in triple backticks or any outer code block.
   - Do not output any explanation, commentary, or preamble — output ONLY the restructured markdown.

7. OUTPUT FORMAT
   - Output raw markdown only.
   - Start directly with the first heading or content line.
   - Use blank lines between sections.
   - Ensure every heading is preceded by a blank line and followed by a blank line.
```

---

## PHASE 7 — `prompts/metadata_extract_prompt.txt`

**This is the exact system prompt for the metadata extraction LLM call.**
Save verbatim as `.txt`.

```
You are a document structure analyzer. You will receive a well-structured markdown document and extract its complete section hierarchy as a JSON object.

Your output MUST be a single valid JSON object and nothing else. No preamble, no explanation, no markdown backticks. Only raw JSON.

Extract the following structure:

{
  "title": "<document title, from H1 or inferred>",
  "total_sections": <integer count of all sections including subsections>,
  "sections": [
    {
      "id": "<auto-generated: section_001, section_002, ... zero-padded to 3 digits>",
      "heading": "<exact heading text without # symbols>",
      "level": <integer: 1 for H1, 2 for H2, 3 for H3, 4 for H4>,
      "heading_path": "<full path using > separator, e.g.: Introduction > Background > Key Concepts>",
      "parent_id": "<id of parent section, or null if top-level>",
      "children": ["<id>", "<id>"],
      "approximate_page_range": [<start_page_int>, <end_page_int>]
    }
  ]
}

RULES:
- Every heading in the document (H1 through H4) gets exactly one entry in sections[].
- IDs are sequential integers formatted as section_001, section_002, etc. in document order (top to bottom).
- heading_path starts from H1 and builds down using " > " as separator.
- For approximate_page_range: the input markdown will contain page marker comments in format <!-- page: N --> — use these to determine page ranges. If no page markers, use [0, 0].
- parent_id of a top-level section (H1) is null.
- children[] contains IDs of direct children only (not grandchildren).
- If the document has no headings at all, return: {"title": "Untitled", "total_sections": 1, "sections": [{"id": "section_001", "heading": "Document Content", "level": 1, "heading_path": "Document Content", "parent_id": null, "children": [], "approximate_page_range": [1, 999]}]}
- Output only valid JSON. No trailing commas. No comments inside JSON.
```

---

## PHASE 8 — `llm_restructurer.py`

**Purpose:** Call LLM to restructure raw markdown. Temperature = 0. Always.

```python
import anthropic
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def load_prompt(filename: str) -> str:
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    with open(os.path.join(prompt_dir, filename), "r", encoding="utf-8") as f:
        return f.read()

RESTRUCTURE_SYSTEM_PROMPT = load_prompt("restructure_prompt.txt")

def restructure_markdown(raw_markdown: str) -> str:
    """
    Send raw markdown to LLM. Get back structured markdown.
    Temperature = 0. No exceptions.
    """
    response = client.messages.create(
        model="claude-opus-4-5",           # Use most capable model for restructuring
        max_tokens=8192,
        temperature=0,
        system=RESTRUCTURE_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Restructure the following raw markdown extracted from a PDF:\n\n{raw_markdown}"
            }
        ]
    )
    return response.content[0].text.strip()
```

**Batching integration:**
- If batch processing is active, call `restructure_markdown()` once per batch
- After all batches restructured, stitch results together
- Run one final `restructure_markdown()` pass on the stitched result ONLY if total tokens < 6000 (for continuity fix at batch seams); skip if too large

---

## PHASE 9 — `metadata_builder.py`

**Purpose:** Call LLM on the final structured markdown to extract section hierarchy JSON.

```python
import anthropic
import json
import os

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

METADATA_SYSTEM_PROMPT = load_prompt("metadata_extract_prompt.txt")  # reuse loader from llm_restructurer

def extract_metadata(structured_markdown: str, source_file: str) -> dict:
    """
    Extract section hierarchy from structured markdown.
    Returns parsed metadata dict.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # Haiku sufficient for JSON extraction — saves cost
        max_tokens=4096,
        temperature=0,
        system=METADATA_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": structured_markdown
            }
        ]
    )

    raw_json = response.content[0].text.strip()

    # Safety: strip accidental backtick fences if model adds them
    if raw_json.startswith("```"):
        raw_json = raw_json.split("```")[1]
        if raw_json.startswith("json"):
            raw_json = raw_json[4:]
    raw_json = raw_json.strip()

    metadata = json.loads(raw_json)
    metadata["source_file"] = source_file
    return metadata
```

**Validation after parse:**
- Assert `metadata["sections"]` is a list
- Assert each section has keys: `id`, `heading`, `level`, `heading_path`, `parent_id`, `children`, `approximate_page_range`
- If validation fails: log error, create fallback metadata with single root section covering full document

---

## PHASE 10 — `chunker.py`

**Purpose:** Use section metadata JSON + structured markdown to create one chunk per section/subsection.

**Core Logic:**
1. Parse structured markdown into a dict: `{section_id: section_content_string}`
   - Use heading markers in the markdown to find where each section starts and ends
   - Section content = everything from its heading line up to (but not including) the next same-or-higher-level heading
2. For each section in metadata:
   - Extract its content from the markdown
   - Build chunk object with full metadata

**Chunk Schema:**
```python
{
    "chunk_id": f"{source_file_stem}__{section_id}",   # e.g.: "annual_report__section_012"
    "source_file": str,                                  # original PDF filename
    "page_range": [int, int],                           # from metadata approximate_page_range
    "heading_path": str,                                 # "Introduction > Background > Key Concepts"
    "heading_path_list": list[str],                     # ["Introduction", "Background", "Key Concepts"]
    "section_level": int,                                # 1, 2, 3, or 4
    "section_heading": str,                              # "Key Concepts"
    "chunk_index": int,                                  # sequential 0-based index across all chunks in doc
    "content": str,                                      # section text (normalized)
    "content_length": int,                               # character count
    "token_estimate": int,                               # approx tokens (use tiktoken cl100k_base)
    "has_table": bool,                                   # True if content contains a markdown table
    "parent_heading": str | None,                        # immediate parent heading text, or null
    "child_headings": list[str],                         # list of immediate child heading texts
}
```

**Implementation notes:**
- Use `tiktoken` with `cl100k_base` encoding for token estimate
- `has_table`: check if content contains `| --- |` pattern (markdown table delimiter row)
- Minimum chunk content length: 50 characters — if section content is shorter, merge it into parent section chunk
- Do not split a section further — one section = one chunk, regardless of size
- If a section has no content (just a heading with all content in subsections), include only the heading line as content — do not create empty chunks

---

## PHASE 11 — `normalizer.py`

**Purpose:** Normalize text before storage. Apply to `content` field of each chunk.

Apply these normalizations in order:
1. Unicode normalization: `unicodedata.normalize("NFKC", text)`
2. Strip zero-width characters: `\u200b`, `\u200c`, `\u200d`, `\ufeff`
3. Normalize quotes: replace `"`, `"` with `"` and `'`, `'` with `'`
4. Normalize dashes: replace `–` (en-dash), `—` (em-dash) with ` - `
5. Collapse multiple spaces to single space (preserve newlines)
6. Collapse 3+ consecutive newlines to 2
7. Strip leading/trailing whitespace per line
8. Do NOT lowercase — preserve original casing for proper nouns, acronyms, headings

---

## PHASE 12 — `pipeline.py` — Master Orchestrator

```python
def run_ingestion(pdf_path: str, output_dir: str) -> dict:
    """
    Full pipeline. Returns summary dict with stats.
    
    Steps:
    1. Parse PDF with Docling → raw_markdown
    2. Filter Hindi → filtered_markdown
    3. Determine batch strategy (≤50 pages = single, >50 = 25-page batches)
    4. For each batch:
       a. LLM restructure → structured_markdown_batch
    5. Stitch batches → full_structured_markdown
    6. Extract section metadata (LLM) → metadata_json
    7. Validate metadata
    8. Chunk by sections → chunks[]
    9. Normalize each chunk content
    10. Save chunks to vector store
    11. Save metadata JSON to disk: {output_dir}/{pdf_stem}_metadata.json
    12. Save structured markdown to disk: {output_dir}/{pdf_stem}_structured.md
    13. Save chunks JSON to disk: {output_dir}/{pdf_stem}_chunks.json
    14. Return summary
    """
    
    summary = {
        "source_file": pdf_path,
        "total_pages": 0,
        "batches_processed": 0,
        "total_sections": 0,
        "total_chunks": 0,
        "chunks_with_tables": 0,
        "hindi_paragraphs_removed": 0,
        "errors": []
    }
    
    # ... implement each step with try/except per step
    # Log each step start/end with timestamp
    # On step failure: log to summary["errors"], raise if critical, continue if recoverable
    
    return summary
```

**Logging:**
- Use Python `logging` module
- Log level INFO for each step
- Log level WARNING for recoverable issues (e.g., empty batch after Hindi filter)
- Log level ERROR for failures
- Include document name and step name in every log message

---

## PHASE 13 — `store.py`

**Purpose:** Save final chunks to vector store + disk.

```python
def store_chunks(chunks: list[dict], output_dir: str, pdf_stem: str):
    """
    1. For each chunk: generate embedding → upsert to vector store
       - Use existing embedding model wrapper from project
       - Vector store document ID = chunk["chunk_id"]
       - Store full chunk dict as metadata (exclude "content" from metadata if store has size limits)
       - Store "content" as the vector store document text
    2. Save all chunks as JSON: {output_dir}/{pdf_stem}_chunks.json
    3. Save metadata JSON: {output_dir}/{pdf_stem}_metadata.json
    """
    ...
```

---

## PHASE 14 — ERROR HANDLING RULES

Apply these consistently across all modules:

| Failure Type | Behavior |
|---|---|
| Docling parse failure on a page | Log warning, skip page, continue |
| LLM API timeout | Retry 3x with exponential backoff (2s, 4s, 8s), then raise |
| LLM returns invalid JSON (metadata) | Log error, use fallback single-section metadata, continue |
| LLM returns empty content | Log error, use raw filtered markdown as-is, mark in summary |
| Batch stitch produces duplicate headings | Log warning, deduplicate headings on stitch |
| Chunk content < 50 chars | Merge into parent, log info |
| Vector store upsert failure | Log error, save chunk to failed_chunks.json for retry |

---

## PHASE 15 — TESTING

After build, create `ingestion/tests/test_pipeline.py`:

- `test_devanagari_filter`: feed mixed Hindi+English paragraph, assert Hindi removed, English preserved
- `test_batch_ranges`: test `get_batch_ranges(40)` → `[(1,40)]`, `get_batch_ranges(60)` → `[(1,25),(26,50),(51,60)]`
- `test_chunk_metadata_fields`: assert all required fields present in every chunk
- `test_no_empty_chunks`: assert no chunk has `content_length` < 50
- `test_heading_path_format`: assert all `heading_path` strings use ` > ` separator
- `test_temperature_zero`: mock LLM call, assert temperature param = 0

---

## FINAL CHECKLIST

Before marking done, verify:

- [ ] All old parsing/ingestion code deleted
- [ ] `ingestion/` folder created with all modules
- [ ] Both prompt `.txt` files saved in `ingestion/prompts/`
- [ ] All LLM calls use `temperature=0`
- [ ] Hindi filter uses Devanagari Unicode range `\u0900-\u097F`
- [ ] Batch threshold = 50 pages, batch size = 25 pages
- [ ] Chunk schema has all 14 fields listed in Phase 10
- [ ] `normalizer.py` preserves casing
- [ ] `pipeline.py` saves 3 output files per PDF (metadata JSON, structured MD, chunks JSON)
- [ ] `tests/test_pipeline.py` has all 6 tests
- [ ] `requirements.txt` updated with new dependencies
- [ ] Environment variable `ANTHROPIC_API_KEY` used (never hardcoded)
