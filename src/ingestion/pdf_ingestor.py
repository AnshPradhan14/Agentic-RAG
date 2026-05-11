"""
pdf_ingestor.py — Layer 0: Data Ingestion & Processing

Pipeline for a single PDF:
    1. parse_pdf()       — new modular parser (Docling + PaddleOCR fallback)
    2. extract_metadata()— pull document-level metadata from structured JSON
    3. store_document()  — INSERT into SQLite `documents` table
    4. ingest_pdf()      — orchestrator that wires steps 1-3 + triple extraction

The new parser (src.ingestion.parser) handles:
    - Bilingual Hindi/English documents (pipe-split + Devanagari removal)
    - Complex structured tables (not flattened — stored as JSON)
    - Noisy / OCR-based PDFs (PaddleOCR fallback when Docling yields <50 words)
    - Government-style key-value + mixed layout documents

Output artefacts written to data/parsed/:
    - <stem>.md   — human-readable markdown (for audit trail)
    - <stem>.json — structured sections + chunks (for indexing)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from src.core.config import PARSED_DIR, RAW_DIR
from src.core.database import init_db, insert_document, save_triples
from src.ingestion.table_relation_extractor import extract_triples
from src.ingestion.utils_cleaning import table_to_markdown

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — PDF Parsing (delegates to new modular parser)
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> tuple[Path, Path]:
    """
    Run the modular parsing pipeline on a single PDF.

    Uses src.ingestion.parser.parse_and_chunk which:
      - Tries Docling first (layout-aware, table-preserving)
      - Falls back to PaddleOCR if Docling yields insufficient text
      - Applies bilingual cleaning (Hindi|English → English only)
      - Extracts KV pairs, preserves table structure

    Writes:
      - data/parsed/<stem>.md   — markdown representation (audit trail)
      - data/parsed/<stem>.json — structured sections + chunks

    Returns:
        (md_path, json_path)
    """
    pdf_path_obj = Path(pdf_path).resolve()
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path_obj}")

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[Layer0] Parsing: %s", pdf_path_obj.name)

    try:
        from src.ingestion.parser import parse_and_chunk
        doc, chunks = parse_and_chunk(str(pdf_path_obj))
    except Exception as exc:
        logger.error("[Layer0] Parser failed for %s: %s", pdf_path_obj, exc, exc_info=True)
        raise RuntimeError(f"Parsing failed: {exc}") from exc

    doc_stem = pdf_path_obj.stem

    # ── Build markdown text from structured sections ──────────────────────────
    md_lines: list[str] = []
    for block in doc.get("sections", []):
        btype   = block.get("type", "text")
        content = block.get("content", "")
        section = block.get("section", "")

        if section:
            md_lines.append(f"\n## {section}\n")

        if btype == "table" and isinstance(content, dict):
            md_lines.append(table_to_markdown(content))
        elif btype == "kv" and isinstance(content, dict):
            for k, v in content.items():
                md_lines.append(f"**{k}**: {v}")
        else:
            md_lines.append(str(content))

    markdown_text = "\n\n".join(md_lines).strip()

    # ── Build JSON output (sections + chunks) ─────────────────────────────────
    json_data = {
        "source":   doc.get("source", pdf_path_obj.name),
        "sections": doc.get("sections", []),
        "chunks":   chunks,
        # Keep raw Docling JSON for triple extraction downstream
        "_docling": doc.get("_raw_json", {}),
    }

    # ── Write files ────────────────────────────────────────────────────────────
    md_path   = PARSED_DIR / f"{doc_stem}.md"
    json_path = PARSED_DIR / f"{doc_stem}.json"

    try:
        md_path.write_text(markdown_text, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot write {md_path}: {exc}") from exc

    try:
        json_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        raise RuntimeError(f"Cannot write {json_path}: {exc}") from exc

    logger.info(
        "[Layer0] Parsed %d sections → %d chunks | md=%s json=%s",
        len(doc["sections"]), len(chunks), md_path.name, json_path.name,
    )
    return md_path, json_path


# Keep old name as alias so existing code that calls parse_pdf_with_docling still works
parse_pdf_with_docling = parse_pdf


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Metadata Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata(json_path: Path) -> dict:
    """
    Read the parsed JSON and extract DOCUMENT-LEVEL metadata.

    Reads from the new structured JSON (source, sections, chunks).
    Falls back gracefully for all missing fields.

    Returns:
        Dict with: source, date_issued, tags, version, doc_type, severity
    """
    try:
        raw_json = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Cannot read JSON at %s: %s", json_path, exc)
        raise

    # Try to find metadata from KV chunks (new parser extracts these)
    kv_combined: dict = {}
    for chunk in raw_json.get("chunks", []):
        if chunk.get("metadata", {}).get("type") == "kv":
            kv_combined.update(chunk.get("metadata", {}))

    # Also check old Docling metadata location for backward compat
    docling_meta = raw_json.get("_docling", {}).get("description", {})

    source: str = json_path.stem + ".pdf"

    date_issued = (
        kv_combined.get("date_issued")
        or docling_meta.get("date_issued")
        or docling_meta.get("dateIssued")
        or None
    )

    raw_tags = kv_combined.get("tags") or docling_meta.get("tags") or []
    if isinstance(raw_tags, str):
        try:
            raw_tags = json.loads(raw_tags)
        except json.JSONDecodeError:
            raw_tags = [raw_tags]
    tags: list[str] = raw_tags if isinstance(raw_tags, list) else []

    version  = kv_combined.get("version")  or docling_meta.get("version")  or None
    doc_type = kv_combined.get("doc_type") or docling_meta.get("doc_type") or None
    severity = kv_combined.get("severity") or docling_meta.get("severity") or None

    metadata = {
        "source":      source,
        "date_issued": date_issued,
        "tags":        tags,
        "version":     version,
        "doc_type":    doc_type,
        "severity":    severity,
    }
    logger.info("[Layer0] Metadata: source=%s", source)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — SQLite Storage
# ─────────────────────────────────────────────────────────────────────────────

def store_document(md_path: Path, json_path: Path, metadata: dict) -> int:
    """Insert parsed document into SQLite `documents` table."""
    try:
        full_markdown_text = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Cannot read {md_path}: {exc}") from exc

    try:
        from src.core.config import BASE_DIR
        md_rel   = str(md_path.relative_to(BASE_DIR))
        json_rel = str(json_path.relative_to(BASE_DIR))
    except ValueError:
        md_rel   = str(md_path)
        json_rel = str(json_path)

    doc_id = insert_document(
        source             = metadata["source"],
        date_issued        = metadata.get("date_issued"),
        tags               = metadata.get("tags"),
        version            = metadata.get("version"),
        doc_type           = metadata.get("doc_type"),
        severity           = metadata.get("severity"),
        md_filepath        = md_rel,
        json_filepath      = json_rel,
        full_markdown_text = full_markdown_text,
    )
    logger.info("[Layer0] Stored → doc_id=%d, source=%s", doc_id, metadata["source"])
    return doc_id


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Full Layer 0 Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> int:
    """
    Full Layer 0 pipeline: parse → metadata → store → extract triples.

    Returns:
        doc_id (int): SQLite primary key of the stored document row.
    """
    logger.info("=== Layer 0: ingest_pdf started for '%s' ===", pdf_path)

    md_path, json_path = parse_pdf(pdf_path)
    metadata           = extract_metadata(json_path)
    doc_id             = store_document(md_path, json_path, metadata)

    # ── Extract entity triples from structured tables ─────────────────────────
    try:
        raw_json = json.loads(json_path.read_text(encoding="utf-8"))

        # New format: tables are in sections with type=="table"
        all_triples: list[dict] = []
        chunk_idx = 0

        for section in raw_json.get("sections", []):
            if section.get("type") != "table":
                continue
            content = section.get("content", {})
            if not isinstance(content, dict):
                continue

            rows = [content.get("headers", [])] + content.get("rows", [])
            table_dict = {
                "section_header": section.get("section", "General"),
                "rows": rows,
            }
            triples = extract_triples(table_dict, str(doc_id), chunk_idx)
            all_triples.extend(triples)
            chunk_idx += 1

        # Also try old Docling JSON format for backward compat
        docling_tables = raw_json.get("_docling", {}).get("tables", [])
        for tbl in docling_tables:
            section_header = tbl.get("section_header") or "General"
            data = tbl.get("data", {})
            table_cells = data.get("table_cells", []) if isinstance(data, dict) else []
            normalised_rows: list = []
            if table_cells:
                num_rows = data.get("num_rows", 0)
                num_cols = data.get("num_cols", 0)
                if num_rows > 0 and num_cols > 0:
                    grid = [[""] * num_cols for _ in range(num_rows)]
                    for cell in table_cells:
                        r = cell.get("start_row_offset_idx", 0)
                        c = cell.get("start_col_offset_idx", 0)
                        if r < num_rows and c < num_cols:
                            grid[r][c] = str(cell.get("text", "")).strip()
                    normalised_rows = grid
            if normalised_rows:
                table_dict = {"section_header": section_header, "rows": normalised_rows}
                triples = extract_triples(table_dict, str(doc_id), chunk_idx)
                all_triples.extend(triples)
                chunk_idx += 1

        if all_triples:
            save_triples(all_triples)
            logger.info("[Layer0] Stored %d triples for doc_id=%d", len(all_triples), doc_id)
        else:
            logger.info("[Layer0] No triples found for doc_id=%d", doc_id)
    except Exception as exc:
        logger.warning("[Layer0] Triple extraction failed for doc_id=%d: %s", doc_id, exc)

    logger.info("=== Layer 0: Ingested '%s' → doc_id=%d ===", metadata["source"], doc_id)
    return doc_id


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer 0 — Ingest a single PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    _configure_logging(args.log_level)
    init_db()

    try:
        doc_id = ingest_pdf(args.pdf)
        print(f"✅ Ingestion complete — doc_id={doc_id}")
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(2)
