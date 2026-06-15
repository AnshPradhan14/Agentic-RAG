"""
pipeline.py
────────────────────────────────────────────────────────────────────────────
Phase 12 — Master Orchestrator for the PDF ingestion pipeline.

Entry point:
    run_ingestion(pdf_path, output_dir) -> dict

Stages
──────
 1.  Parse PDF with Docling          → raw_markdown
 2.  Filter Hindi/Devanagari         → filtered_markdown  (+count removed ¶s)
 3.  Determine batch strategy        → batch_pdf_paths
 4.  Parse each batch with Docling   → batch_markdowns[]
 5.  LLM restructure each batch      → structured_batches[]
 6.  Stitch + optional final pass    → full_structured_markdown
 7.  Extract section metadata (LLM)  → metadata_json
 8.  Chunk by sections               → chunks[]
 9.  Normalise each chunk content    → chunks[] (mutated)
10.  Save chunks to vector store     → (side-effect)
11.  Persist metadata JSON to disk   → {output_dir}/{stem}_metadata.json
12.  Persist structured MD to disk   → {output_dir}/{stem}_structured.md
13.  Persist chunks JSON to disk     → {output_dir}/{stem}_chunks.json
14.  Return summary dict

Logging
───────
  Every step logs INFO on start and completion.
  Recoverable issues log WARNING.
  Failures log ERROR and are appended to summary["errors"].
  Critical failures (Docling total failure, LLM total failure) re-raise after
  recording to ensure the caller always knows a document was not processed.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch_processor import (
    get_total_pages,
    split_pdf_into_batches,
    stitch_batches,
)
from .chunker import chunk_document
from .language_filter import filter_hindi
from .llm_restructurer import restructure_batches
from .metadata_builder import extract_metadata
from .normalizer import normalize_chunks
from .pdf_parser import parse_pdf_to_markdown
from .store import store_chunks
from src.core.database import document_exists
from src.core.config import PARSED_DIR, FINAL_TEXT_DIR

# ── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    """ISO-8601 UTC timestamp for log messages."""
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _count_hindi_paragraphs(raw: str, filtered: str) -> int:
    """
    Approximate count of paragraphs removed by the Hindi filter by comparing
    the paragraph count before and after filtering.
    """
    return max(0, len(raw.split("\n\n")) - len(filtered.split("\n\n")))


def _save_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


class _StepContext:
    """
    Context manager that wraps a pipeline step with timing, INFO logging,
    and structured error recording.

    Usage:
        with _StepContext("step_name", pdf_name, summary) as ctx:
            result = do_work()

    On exception:
        - Logs ERROR with step name + document name
        - Appends {"step": ..., "error": ..., "ts": ...} to summary["errors"]
        - Re-raises if critical=True (default), swallows if critical=False
    """

    def __init__(
        self,
        step: str,
        pdf_name: str,
        summary: dict[str, Any],
        critical: bool = True,
    ) -> None:
        self.step = step
        self.pdf_name = pdf_name
        self.summary = summary
        self.critical = critical
        self._start: float = 0.0

    def __enter__(self) -> "_StepContext":
        self._start = time.perf_counter()
        logger.info(f"[pipeline:{self.step}] START — {self.pdf_name} — {_now()}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._start
        if exc_type is None:
            logger.info(
                f"[pipeline:{self.step}] DONE — {self.pdf_name} "
                f"({elapsed:.2f}s) — {_now()}"
            )
            return False  # no exception

        # Record error
        error_record = {
            "step": self.step,
            "error": f"{exc_type.__name__}: {exc_val}",
            "ts": _now(),
        }
        self.summary["errors"].append(error_record)
        logger.error(
            f"[pipeline:{self.step}] FAILED — {self.pdf_name} — "
            f"{error_record['error']} ({elapsed:.2f}s)"
        )

        if self.critical:
            return False  # propagate exception
        else:
            logger.warning(
                f"[pipeline:{self.step}] Recoverable — continuing pipeline."
            )
            return True  # suppress exception


# ── Public API ────────────────────────────────────────────────────────────────

def run_ingestion(pdf_path: str, output_dir: str) -> dict[str, Any]:
    """
    Run the full ingestion pipeline for a single PDF.

    Parameters
    ──────────
    pdf_path : str
        Absolute or relative path to the source PDF file.
    output_dir : str
        Directory where output artefacts will be written.
        Created automatically if it does not exist.

    Returns
    ───────
    dict
        Summary dictionary with statistics and any recorded errors.
        Always returned — even on partial failure — so the caller can
        inspect what was or was not processed.

    Raises
    ──────
    FileNotFoundError
        If ``pdf_path`` does not exist.
    RuntimeError
        If a critical pipeline stage fails (Docling parse, LLM restructure,
        LLM metadata).  Non-critical stages (store, disk writes) log errors
        but do not raise.
    """
    pdf_path_obj = Path(pdf_path).resolve()
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pdf_name = pdf_path_obj.name        # "annual_report.pdf"
    
    if document_exists(pdf_name):
        logger.warning(f"[pipeline] Document '{pdf_name}' already exists in the database. Skipping ingestion.")
        return {
            "source_file": str(pdf_path_obj),
            "pdf_name": pdf_name,
            "status": "skipped_duplicate",
            "errors": []
        }

    pdf_stem = pdf_path_obj.stem        # "annual_report"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "source_file": str(pdf_path_obj),
        "pdf_name": pdf_name,
        "total_pages": 0,
        "batches_processed": 0,
        "total_sections": 0,
        "total_chunks": 0,
        "chunks_with_tables": 0,
        "hindi_paragraphs_removed": 0,
        "output_dir": str(output_path),
        "started_at": _now(),
        "finished_at": None,
        "errors": [],
    }

    # Intermediate values shared across steps
    raw_markdown: str = ""
    filtered_markdown: str = ""
    batch_pdf_paths: list[str] = []
    batch_markdowns: list[str] = []
    full_structured_markdown: str = ""
    metadata: dict[str, Any] = {}
    chunks: list[dict[str, Any]] = []

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — Count pages
    # ─────────────────────────────────────────────────────────────────────────
    with _StepContext("count_pages", pdf_name, summary, critical=True):
        total_pages = get_total_pages(str(pdf_path_obj))
        summary["total_pages"] = total_pages
        logger.info(f"[pipeline:count_pages] {pdf_name} has {total_pages} page(s).")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — Parse full PDF → raw markdown (Docling)
    # Skipped for large PDFs (>50 pages): those documents will be split into
    # 25-page batches in Step 4 and each batch is re-parsed individually in
    # Step 5. Parsing the full PDF here first would cause Docling to load all
    # pages into memory at once → std::bad_alloc on image-heavy PDFs.
    # ─────────────────────────────────────────────────────────────────────────
    _large_pdf = total_pages > 50
    if not _large_pdf:
        with _StepContext("parse_pdf", pdf_name, summary, critical=True):
            raw_markdown = parse_pdf_to_markdown(str(pdf_path_obj))
            logger.info(
                f"[pipeline:parse_pdf] Raw markdown extracted: "
                f"{len(raw_markdown):,} chars."
            )
    else:
        logger.info(
            f"[pipeline:parse_pdf] SKIPPED for large PDF ({total_pages} pages). "
            "Will parse per-batch in Step 5 to avoid OOM."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Filter Hindi / Devanagari
    # Recoverable: if filter blows up we continue with raw markdown.
    # Skipped for large PDFs — filtering is applied per-batch in Step 5.
    # ─────────────────────────────────────────────────────────────────────────
    if not _large_pdf:
        with _StepContext("filter_hindi", pdf_name, summary, critical=False):
            filtered_markdown = filter_hindi(raw_markdown)
            removed = _count_hindi_paragraphs(raw_markdown, filtered_markdown)
            summary["hindi_paragraphs_removed"] = removed
            logger.info(
                f"[pipeline:filter_hindi] Removed ~{removed} Hindi paragraph(s). "
                f"Filtered markdown: {len(filtered_markdown):,} chars."
            )
            if not filtered_markdown.strip():
                logger.warning(
                    f"[pipeline:filter_hindi] All content was filtered for "
                    f"'{pdf_name}'. Falling back to raw markdown."
                )
                filtered_markdown = raw_markdown

        if not filtered_markdown:
            filtered_markdown = raw_markdown  # guard if step was suppressed
    else:
        logger.info(
            f"[pipeline:filter_hindi] SKIPPED for large PDF — will filter per-batch."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — Split into 25-page batches (if > 50 pages)
    # ─────────────────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="rag_batches_") as tmp_dir:

        with _StepContext("split_batches", pdf_name, summary, critical=True):
            batch_pdf_paths = split_pdf_into_batches(str(pdf_path_obj), tmp_dir)
            logger.info(
                f"[pipeline:split_batches] {len(batch_pdf_paths)} batch(es) "
                f"for '{pdf_name}'."
            )

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5 — Parse each batch PDF → markdown
        # Individual batch failures are recoverable; we skip the failed batch.
        # For small PDFs (single batch), reuse filtered_markdown from Step 2/3.
        # For large PDFs, parse each 25-page batch here — this avoids loading
        # the entire PDF into Docling memory at once.
        # ─────────────────────────────────────────────────────────────────────
        if len(batch_pdf_paths) == 1 and batch_pdf_paths[0] == str(pdf_path_obj):
            # Small PDF (<=50 pages): reuse already-filtered markdown, skip re-parse
            batch_markdowns = [filtered_markdown]
        else:
            # Large PDF: parse each 25-page batch individually
            _total_hindi_removed = 0
            for i, batch_path in enumerate(batch_pdf_paths, start=1):
                batch_label = f"{pdf_name} — batch {i}/{len(batch_pdf_paths)}"
                with _StepContext("parse_batch", batch_label, summary, critical=False):
                    batch_raw = parse_pdf_to_markdown(batch_path)
                    batch_filtered = filter_hindi(batch_raw)
                    _total_hindi_removed += _count_hindi_paragraphs(batch_raw, batch_filtered)
                    if not batch_filtered.strip():
                        logger.warning(
                            f"[pipeline:parse_batch] Batch {i} is empty after "
                            f"Hindi filter — skipping."
                        )
                    else:
                        batch_markdowns.append(batch_filtered)
            summary["hindi_paragraphs_removed"] = _total_hindi_removed

        if not batch_markdowns:
            msg = f"All batches for '{pdf_name}' were empty after filtering."
            logger.error(f"[pipeline] {msg}")
            summary["errors"].append({"step": "parse_batch", "error": msg, "ts": _now()})
            summary["finished_at"] = _now()
            return summary

        summary["batches_processed"] = len(batch_markdowns)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 6 — LLM restructure all batches + stitch
        # Critical: we need structured markdown for metadata + chunking.
        # ─────────────────────────────────────────────────────────────────────
        with _StepContext("llm_restructure", pdf_name, summary, critical=True):
            full_structured_markdown = restructure_batches(
                batch_markdowns, pdf_name=pdf_name
            )
            logger.info(
                f"[pipeline:llm_restructure] Structured markdown: "
                f"{len(full_structured_markdown):,} chars."
            )

    # tmp_dir cleaned up here — batch PDFs removed

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — Extract section metadata via LLM
    # Critical: metadata drives chunking.  Falls back gracefully internally.
    # ─────────────────────────────────────────────────────────────────────────
    with _StepContext("extract_metadata", pdf_name, summary, critical=True):
        metadata = extract_metadata(full_structured_markdown, source_file=pdf_name)
        section_count = metadata.get("total_sections", len(metadata.get("sections", [])))
        summary["total_sections"] = section_count
        logger.info(
            f"[pipeline:extract_metadata] Extracted {section_count} section(s) "
            f"for '{pdf_name}'."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — Chunk by sections
    # ─────────────────────────────────────────────────────────────────────────
    with _StepContext("chunk_document", pdf_name, summary, critical=True):
        chunks = chunk_document(full_structured_markdown, metadata)
        summary["total_chunks"] = len(chunks)
        logger.info(
            f"[pipeline:chunk_document] Created {len(chunks)} chunk(s) "
            f"for '{pdf_name}'."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9 — Normalise chunk content
    # ─────────────────────────────────────────────────────────────────────────
    with _StepContext("normalize_chunks", pdf_name, summary, critical=False):
        chunks = normalize_chunks(chunks)
        table_chunks = sum(1 for c in chunks if c.get("has_table"))
        summary["chunks_with_tables"] = table_chunks
        logger.info(
            f"[pipeline:normalize_chunks] Normalised {len(chunks)} chunk(s). "
            f"{table_chunks} contain table(s)."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10 — Save to vector store
    # Recoverable: a store failure should not discard already-computed chunks.
    # ─────────────────────────────────────────────────────────────────────────
    with _StepContext("store_chunks", pdf_name, summary, critical=False):
        store_chunks(chunks, str(FINAL_TEXT_DIR), pdf_stem)
        logger.info(
            f"[pipeline:store_chunks] Saved {len(chunks)} chunk(s) to vector store."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEPS 11–13 — Persist artefacts to disk
    # Each write is independently recoverable.
    # ─────────────────────────────────────────────────────────────────────────
    raw_path       = PARSED_DIR / f"{pdf_stem}_raw.md"
    metadata_path  = FINAL_TEXT_DIR / f"{pdf_stem}_metadata.json"
    structured_path = FINAL_TEXT_DIR / f"{pdf_stem}_structured.md"
    chunks_path    = FINAL_TEXT_DIR / f"{pdf_stem}_chunks.json"

    # For large PDFs, Step 2 was skipped — reconstruct raw_markdown from batches
    if _large_pdf and not raw_markdown and batch_markdowns:
        raw_markdown = "\n\n".join(batch_markdowns)

    with _StepContext("save_raw_md", pdf_name, summary, critical=False):
        _save_text(raw_markdown, raw_path)
        logger.info(f"[pipeline:save_raw_md] Written → {raw_path}")

    with _StepContext("save_metadata_json", pdf_name, summary, critical=False):
        _save_json(metadata, metadata_path)
        logger.info(f"[pipeline:save_metadata_json] Written → {metadata_path}")

    with _StepContext("save_structured_md", pdf_name, summary, critical=False):
        _save_text(full_structured_markdown, structured_path)
        logger.info(f"[pipeline:save_structured_md] Written → {structured_path}")

    with _StepContext("save_chunks_json", pdf_name, summary, critical=False):
        _save_json(chunks, chunks_path)
        logger.info(f"[pipeline:save_chunks_json] Written → {chunks_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 14 — Finalise summary
    # ─────────────────────────────────────────────────────────────────────────
    summary["finished_at"] = _now()
    error_count = len(summary["errors"])
    logger.info(
        f"[pipeline] COMPLETE — '{pdf_name}' — "
        f"{summary['total_chunks']} chunks, "
        f"{summary['total_sections']} sections, "
        f"{summary['batches_processed']} batch(es), "
        f"{error_count} error(s). "
        f"Duration: started={summary['started_at']} finished={summary['finished_at']}"
    )
    return summary
