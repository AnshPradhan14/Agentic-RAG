"""
normalizer.py
────────────────────────────────────────────────────────────────────────────
Phase 11 — Text normalisation pipeline.

Applied to the ``content`` field of every chunk before storage.
Normalisation steps are applied IN ORDER — do not reorder them.

RULES (from spec):
  1.  Unicode normalisation  → NFKC
  2.  Strip zero-width characters  (U+200B, U+200C, U+200D, U+FEFF)
  3.  Normalise curly/typographic quotes  → straight ASCII quotes
  4.  Normalise dashes  (en-dash, em-dash) → ` - `
  5.  Collapse multiple spaces → single space  (newlines preserved)
  6.  Collapse 3+ consecutive newlines → 2 newlines
  7.  Strip leading/trailing whitespace per line
  8.  DO NOT lowercase — preserve casing for headings, acronyms, proper nouns

Public API
──────────
  normalize_text(text: str) -> str
      Apply the full pipeline to a single string.

  normalize_chunks(chunks: list[dict]) -> list[dict]
      Apply normalize_text() to the ``content`` field of every chunk in-place.
      Recalculates ``content_length`` after normalisation.
      Returns the same list (mutated) for convenience.
"""

from __future__ import annotations

import re
import unicodedata
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Compiled patterns (compiled once at import, not per-call) ────────────────

# Step 2: Zero-width / BOM characters
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")

# Step 3: Typographic / curly quotes → ASCII equivalents
#   Left/right double: " " → "
#   Left/right single: ' ' → '
#   Prime variants: ′ ″ (common in scanned PDFs)
_QUOTE_MAP: dict[str, str] = {
    "\u201c": '"',   # " LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',   # " RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',   # „ DOUBLE LOW-9 QUOTATION MARK
    "\u201f": '"',   # ‟ DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2018": "'",   # ' LEFT SINGLE QUOTATION MARK
    "\u2019": "'",   # ' RIGHT SINGLE QUOTATION MARK
    "\u201a": "'",   # ‚ SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",   # ‛ SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u2032": "'",   # ′ PRIME
    "\u2033": '"',   # ″ DOUBLE PRIME
    "\u0060": "'",   # ` GRAVE ACCENT (often misused as open quote in PDFs)
    "\u00ab": '"',   # « LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u00bb": '"',   # » RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
}
_QUOTE_RE = re.compile("|".join(re.escape(k) for k in _QUOTE_MAP))

# Step 4: Dashes — all common Unicode dash/hyphen-like characters
#   We replace with ` - ` (space–hyphen–space) to preserve readability.
#   We intentionally leave ASCII hyphen-minus (-) alone.
_DASH_MAP: dict[str, str] = {
    "\u2013": " - ",   # – EN DASH
    "\u2014": " - ",   # — EM DASH
    "\u2015": " - ",   # ― HORIZONTAL BAR
    "\u2212": " - ",   # − MINUS SIGN
    "\u2011": "-",     # ‑ NON-BREAKING HYPHEN → plain hyphen (no spaces)
    "\u00ad": "",      # ­ SOFT HYPHEN → remove entirely (PDF artifact)
}
_DASH_RE = re.compile("|".join(re.escape(k) for k in _DASH_MAP))

# Step 5: Collapse multiple horizontal spaces → single space.
#   \t is treated as a space; does NOT match \n or \r.
_MULTI_SPACE_RE = re.compile(r"[^\S\r\n]{2,}")

# Step 6: Collapse 3+ consecutive newlines → exactly 2 newlines.
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Step 7: Lines with only whitespace → empty lines; strip per-line
#   (We strip each line individually, not the whole string, to preserve
#    structure at start/end of content.)
_LINE_TRAILING_RE = re.compile(r"[^\S\n]+$", re.MULTILINE)  # trailing spaces per line
_LINE_LEADING_RE  = re.compile(r"^[^\S\n]+",  re.MULTILINE)  # leading spaces per line


# ── Individual normalisation steps ──────────────────────────────────────────

def _step1_unicode_nfkc(text: str) -> str:
    """
    NFKC decomposition + composition.
    Resolves compatibility characters (e.g. ﬁ ligature → fi, ² → 2,
    full-width ASCII → ASCII) that Docling sometimes emits.
    """
    return unicodedata.normalize("NFKC", text)


def _step2_strip_zero_width(text: str) -> str:
    """Remove zero-width joiners, non-joiners, spaces, and BOM."""
    return _ZERO_WIDTH_RE.sub("", text)


def _step3_normalise_quotes(text: str) -> str:
    """Replace typographic/curly/angle quotes with plain ASCII equivalents."""
    return _QUOTE_RE.sub(lambda m: _QUOTE_MAP[m.group()], text)


def _step4_normalise_dashes(text: str) -> str:
    """
    Replace em-dash, en-dash, and other Unicode dash variants.
    Soft hyphens (PDF line-break artifacts) are silently removed.
    Non-breaking hyphens become plain hyphens.
    """
    return _DASH_RE.sub(lambda m: _DASH_MAP[m.group()], text)


def _step5_collapse_spaces(text: str) -> str:
    """
    Collapse 2+ horizontal whitespace characters into a single space.
    Tabs are collapsed too.  Newlines are NOT touched here.
    """
    return _MULTI_SPACE_RE.sub(" ", text)


def _step6_collapse_newlines(text: str) -> str:
    """Collapse 3+ consecutive newlines into exactly two (one blank line)."""
    return _MULTI_NEWLINE_RE.sub("\n\n", text)


def _step7_strip_per_line(text: str) -> str:
    """
    Strip trailing and leading whitespace on every line independently.
    Preserves blank lines (empty after stripping) — does NOT remove them.
    Spec explicitly says strip leading/trailing whitespace per line,
    NOT to strip the whole string (that would lose intentional leading blank lines).
    """
    text = _LINE_TRAILING_RE.sub("", text)
    text = _LINE_LEADING_RE.sub("",  text)
    return text


# ── Public API ───────────────────────────────────────────────────────────────

_PIPELINE = (
    ("step1_unicode_nfkc",    _step1_unicode_nfkc),
    ("step2_strip_zero_width", _step2_strip_zero_width),
    ("step3_normalise_quotes", _step3_normalise_quotes),
    ("step4_normalise_dashes", _step4_normalise_dashes),
    ("step5_collapse_spaces",  _step5_collapse_spaces),
    ("step6_collapse_newlines", _step6_collapse_newlines),
    ("step7_strip_per_line",   _step7_strip_per_line),
)


def normalize_text(text: str) -> str:
    """
    Apply the full 7-step normalisation pipeline to a string.

    Casing is NEVER changed — proper nouns, acronyms, and heading
    capitalisation are preserved exactly as-is.

    Parameters
    ──────────
    text : str
        Raw content string (e.g. the ``content`` field of a chunk).

    Returns
    ───────
    str
        Normalised string, ready for embedding / storage.
    """
    if not text:
        return text

    for step_name, step_fn in _PIPELINE:
        try:
            text = step_fn(text)
        except Exception as exc:  # pragma: no cover
            # A normalisation step should never raise, but we guard anyway so
            # a single corrupt character cannot abort the entire pipeline.
            logger.warning(
                f"[normalizer] {step_name} raised {type(exc).__name__}: {exc}. "
                "Skipping step and continuing with unmodified text."
            )

    return text


def enrich_chunk_with_breadcrumb(chunk: dict[str, Any]) -> None:
    """
    Prepends the heading_path as a markdown comment to the chunk's content.
    """
    raw = chunk.get("content", "")
    heading_path = chunk.get("heading_path")
    if heading_path:
        chunk["content"] = f"<!-- context: {heading_path} -->\n\n{raw}"


def normalize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Apply normalize_text() to the ``content`` field of every chunk.

    Also recalculates ``content_length`` after normalisation.
    Mutates and returns the input list — no copy is made.

    Parameters
    ──────────
    chunks : list[dict]
        List of chunk dicts produced by chunker.chunk_document().

    Returns
    ───────
    list[dict]
        The same list, with each chunk's ``content`` and
        ``content_length`` updated in-place.
    """
    for chunk in chunks:
        enrich_chunk_with_breadcrumb(chunk)
        raw = chunk.get("content", "")
            
        normalised = normalize_text(raw)
        chunk["content"] = normalised
        chunk["content_length"] = len(normalised)

    logger.debug(
        f"[normalizer] Normalised {len(chunks)} chunk(s)."
    )
    return chunks
