"""
pdf_parser.py
────────────────────────────────────────────────────────────────────────────
Phase 4 — Convert a PDF to raw Markdown using Docling.

Phase 14 Error Handling:
 - Page-level Docling errors → WARNING logged, page skipped, parsing continues.
 - If the full conversion returns empty output → ERROR logged, RuntimeError raised.
 - If Docling raises an unexpected exception → ERROR logged, re-raised to caller.
"""

from __future__ import annotations

import logging

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)


def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert entire PDF to raw markdown using Docling.

    Phase 14 behaviour:
      - Page-level parse errors: logged at WARNING, skipped by Docling (continues).
      - Empty result after conversion: RuntimeError raised with context.
      - Unhandled Docling exception: re-raised after ERROR log.

    Parameters
    ──────────
    pdf_path : str
        Absolute path to the PDF file.

    Returns
    ───────
    str
        Raw, unstructured Markdown string produced by Docling.

    Raises
    ──────
    RuntimeError
        If Docling returns completely empty output.
    Exception
        Any unhandled Docling exception is re-raised.
    """
    logger.info(f"[pdf_parser] Starting Docling conversion: {pdf_path}")

    try:
        converter = DocumentConverter()
        # raises_on_error=False means Docling keeps going on page-level failures
        # and surfaces them in result.errors rather than raising immediately.
        result = converter.convert(pdf_path, raises_on_error=False)

        # ── Phase 14: log page-level errors as warnings, continue ────────────
        if hasattr(result, "errors") and result.errors:
            for error in result.errors:
                page_no = getattr(error, "page_no", "unknown")
                msg = getattr(error, "error_message", str(error))
                logger.warning(
                    f"[pdf_parser] Page-level parse error on page {page_no} "
                    f"of '{pdf_path}': {msg} — skipping page, continuing."
                )

        markdown: str = result.document.export_to_markdown()

        # ── Phase 14: guard against totally empty output ──────────────────────
        if not markdown.strip():
            logger.error(
                f"[pdf_parser] Docling returned empty markdown for '{pdf_path}'. "
                "The PDF may be image-only or corrupted."
            )
            raise RuntimeError(
                f"Docling produced no extractable text from '{pdf_path}'. "
                "Ensure the PDF is text-based, not a pure scanned image."
            )

        logger.info(
            f"[pdf_parser] Conversion complete: {len(markdown):,} chars "
            f"extracted from '{pdf_path}'."
        )
        return markdown

    except RuntimeError:
        raise  # already logged above — propagate cleanly

    except Exception as exc:
        logger.error(
            f"[pdf_parser] Docling raised an unexpected error for '{pdf_path}': "
            f"{type(exc).__name__}: {exc}"
        )
        raise
