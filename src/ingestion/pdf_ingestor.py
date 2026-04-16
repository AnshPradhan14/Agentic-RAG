"""
layer0_ingest.py — Layer 0: Data Ingestion & Processing

Responsibility:
    PDF → Docling (offline) → Markdown + JSON files on disk → SQLite `documents` table.

Pipeline for a single PDF:
    1. parse_pdf_with_docling()  — run Docling, write .md and .json to data/parsed/
    2. extract_metadata()        — read JSON, pull document-level metadata fields
    3. store_document()          — read full Markdown, INSERT into SQLite `documents`
    4. ingest_pdf()              — orchestrator that wires steps 1-3 together

Notes:
    - page_number and section_path are chunk-level attributes; do NOT extract here.
      They are populated by Layer 1 (layer1_indexer.py) using the saved JSON file.
    - data/parsed/ is auto-created if it does not exist.
    - The .md / .json files are a human-readable audit trail.  Even if deleted,
      SQLite still holds the full parsed content.

Usage (standalone):
    python layer0_ingest.py --pdf data/raw/my_document.pdf
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from src.core.config import PARSED_DIR, RAW_DIR
from src.core.database import init_db, insert_document

# ── Module logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — PDF Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_pdf_with_docling(pdf_path: str) -> tuple[Path, Path]:
    """Run Docling on a single PDF and persist Markdown + JSON to data/parsed/.

    Args:
        pdf_path: Absolute or relative path to the source PDF file.

    Returns:
        (md_path, json_path): Paths to the saved .md and .json files.

    Side effects:
        - Creates data/parsed/ if it does not exist.
        - Writes <doc_name>.md and <doc_name>.json into data/parsed/.
        - Logs output paths at INFO level.

    Raises:
        FileNotFoundError : If pdf_path does not exist.
        RuntimeError      : If Docling fails to produce both output files.
    """
    pdf_path_obj = Path(pdf_path).resolve()

    # ── Guard: source file must exist ─────────────────────────────────────────
    if not pdf_path_obj.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path_obj}\n"
            f"Place the file in {RAW_DIR} and retry."
        )

    logger.info("Starting Docling ingestion: %s", pdf_path_obj)

    # ── Ensure output directory exists ────────────────────────────────────────
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from docling.document_converter import DocumentConverter  # lazy import

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path_obj))

        markdown_text: str = result.document.export_to_markdown()
        json_data: dict = result.document.export_to_dict()

    except Exception as exc:  # noqa: BLE001
        logger.error("Docling failed for %s: %s", pdf_path_obj, exc)
        raise RuntimeError(
            f"Docling converter raised an error for '{pdf_path_obj}': {exc}"
        ) from exc

    doc_stem = pdf_path_obj.stem  # filename without extension

    # ── Write Markdown ─────────────────────────────────────────────────────────
    md_path = PARSED_DIR / f"{doc_stem}.md"
    try:
        md_path.write_text(markdown_text, encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not write Markdown file {md_path}: {exc}") from exc

    # ── Write JSON ─────────────────────────────────────────────────────────────
    json_path = PARSED_DIR / f"{doc_stem}.json"
    try:
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not write JSON file {json_path}: {exc}") from exc

    # ── Validate outputs ──────────────────────────────────────────────────────
    if not md_path.exists() or not json_path.exists():
        raise RuntimeError(
            f"Docling did not produce expected output files for '{pdf_path_obj}'. "
            f"Expected: {md_path} and {json_path}"
        )

    logger.info("Docling output saved → %s, %s", md_path, json_path)
    return md_path, json_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Metadata Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_metadata(json_path: Path) -> dict:
    """Read the Docling JSON output and extract DOCUMENT-LEVEL metadata fields.

    Only document-level fields are extracted here.  Chunk-level fields
    (page_number, section_path) are the responsibility of Layer 1.

    Args:
        json_path: Path to the saved .json file in data/parsed/.

    Returns:
        Dict with keys: source, date_issued, tags, version, doc_type, severity.
        All values use safe defaults — this function never raises on missing fields.

    Notes:
        - Uses .get() with safe defaults throughout; Docling JSON structure varies.
        - Logs extracted metadata at INFO, full dict at DEBUG.
        - Returns empty / None values gracefully — downstream layers handle blanks.
    """
    try:
        raw_json = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Could not read/parse JSON at %s: %s", json_path, exc)
        raise

    # ── Docling JSON top-level keys vary by version; inspect both common roots ─
    # Docling may nest metadata under "description" or directly at root level.
    doc_meta: dict = raw_json.get("description", raw_json)

    # ── Pull document-level fields with safe defaults ──────────────────────────
    source: str = json_path.stem + ".pdf"   # reconstruct source filename from stem

    # date_issued: look in common Docling metadata locations
    date_issued: str | None = (
        doc_meta.get("date_issued")
        or doc_meta.get("dateIssued")
        or raw_json.get("date_issued")
        or None
    )

    # tags: accept list or string; normalise to list[str]
    raw_tags = (
        doc_meta.get("tags")
        or raw_json.get("tags")
        or []
    )
    if isinstance(raw_tags, str):
        try:
            raw_tags = json.loads(raw_tags)
        except json.JSONDecodeError:
            raw_tags = [raw_tags]
    tags: list[str] = raw_tags if isinstance(raw_tags, list) else []

    version: str | None = (
        doc_meta.get("version")
        or raw_json.get("version")
        or None
    )

    doc_type: str | None = (
        doc_meta.get("doc_type")
        or doc_meta.get("docType")
        or raw_json.get("doc_type")
        or None
    )

    severity: str | None = (
        doc_meta.get("severity")
        or raw_json.get("severity")
        or None
    )

    metadata = {
        "source":      source,
        "date_issued": date_issued,
        "tags":        tags,
        "version":     version,
        "doc_type":    doc_type,
        "severity":    severity,
    }

    logger.info(
        "Metadata extracted: source=%s, tags=%s", metadata["source"], metadata["tags"]
    )
    logger.debug("Full metadata dict: %s", metadata)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — SQLite Storage
# ─────────────────────────────────────────────────────────────────────────────

def store_document(md_path: Path, json_path: Path, metadata: dict) -> int:
    """Insert a parsed document record into the SQLite `documents` table.

    Args:
        md_path   : Path to the .md file (stored as md_filepath in the DB).
        json_path : Path to the .json file (stored as json_filepath in the DB).
        metadata  : Dict returned by extract_metadata().

    Returns:
        doc_id (int): Auto-incremented primary key of the inserted row.

    Side effects:
        - Reads the full Markdown text from md_path.
        - Inserts one row into `documents` with all metadata + full_markdown_text.
        - Commits the transaction.

    Raises:
        OSError       : If the Markdown file cannot be read.
        sqlite3.Error : On DB failure (raised after rollback by database.insert_document).
    """
    # ── Read full Markdown content ─────────────────────────────────────────────
    try:
        full_markdown_text = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Cannot read Markdown file {md_path}: {exc}") from exc

    # ── Relative paths for portability ────────────────────────────────────────
    # Store relative-to-project-root paths so the DB is portable across machines.
    try:
        from src.core.config import BASE_DIR
        md_rel   = str(md_path.relative_to(BASE_DIR))
        json_rel = str(json_path.relative_to(BASE_DIR))
    except ValueError:
        # Paths outside BASE_DIR — store absolute as fallback
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

    logger.info("Stored to DB: doc_id=%d, source=%s", doc_id, metadata["source"])
    return doc_id


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Full Layer 0 Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str) -> int:
    """Full Layer 0 pipeline for a single PDF.

    Orchestrates parse → extract → store in a single call.  The returned
    doc_id is passed to Layer 1 (layer1_indexer.build_chunks) to create
    sentence-aligned chunks and FAISS embeddings.

    Args:
        pdf_path: Path to the raw PDF in data/raw/ (or any valid path).

    Returns:
        doc_id (int): SQLite primary key of the stored document row.

    Flow:
        1. parse_pdf_with_docling(pdf_path) → (md_path, json_path)
        2. extract_metadata(json_path)      → metadata dict
        3. store_document(md_path, json_path, metadata) → doc_id
        4. Log success
        5. Return doc_id
    """
    logger.info("=== Layer 0: ingest_pdf started for '%s' ===", pdf_path)

    md_path, json_path = parse_pdf_with_docling(pdf_path)
    metadata            = extract_metadata(json_path)
    doc_id              = store_document(md_path, json_path, metadata)

    logger.info(
        "=== Layer 0: Ingested '%s' → doc_id=%d ===",
        metadata["source"],
        doc_id,
    )
    return doc_id


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point (standalone usage)
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer 0 — Ingest a single PDF into the RAG pipeline."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the PDF file to ingest (e.g. data/raw/my_doc.pdf).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    _configure_logging(args.log_level)

    # Ensure schema exists before ingesting
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
