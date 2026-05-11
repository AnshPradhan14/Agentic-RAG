"""
utils_cleaning.py — Advanced Text Cleaning Utilities for Bilingual RAG Parsing

Handles:
  - Bilingual Hindi|English splitting (pipe-delimiter and Unicode range)
  - OCR noise removal (spaced-out letters, garbage glyphs)
  - Key-value pair detection and normalization
  - Table cell normalization
  - Markdown table structure validation
  - LLM-ready text normalization

This module is STATELESS — all functions are pure (no I/O, no side-effects).
Import freely without worrying about circular dependencies.
"""

from __future__ import annotations

import html
import json
import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ── Unicode ranges ─────────────────────────────────────────────────────────────
# Devanagari block: U+0900–U+097F  (Hindi, Sanskrit, Marathi, Nepali …)
# Extended Devanagari: U+A8E0–U+A8FF, U+1CD0–U+1CFF
_DEVANAGARI_PATTERN = re.compile(
    r"[\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF]+"
)

# ── Common government document KV separators ──────────────────────────────────
_KV_SEPARATORS = re.compile(
    r"""
    (?P<key>[A-Za-z0-9][\w\s\(\)/\-]*)   # key: alphanumeric + common chars
    \s*                                   # optional whitespace
    (?::|–|-{1,2}|=>|=)                   # separator character(s)
    \s*                                   # optional whitespace
    (?P<value>.+)                         # value: rest of the string
    """,
    re.VERBOSE,
)

# ── Broken-word OCR patterns (spaced characters) ──────────────────────────────
_SPACED_CHARS_PATTERN = re.compile(r"\b([A-Za-z])(?: [A-Za-z]){2,}\b")


# =============================================================================
# 1. Bilingual Text Splitting and Hindi Removal
# =============================================================================

def split_bilingual(text: str) -> str:
    """
    Split bilingual text on the pipe '|' delimiter and keep only the English part.

    Government documents commonly encode bilingual content as:
        "अनुबंध|Contract"   → "Contract"
        "क्रम संख्या|Sr. No" → "Sr. No"

    If no pipe is found, the raw text is returned as-is for downstream cleaning.

    Args:
        text: Raw cell or field text, possibly bilingual.

    Returns:
        The English (right-hand) side after splitting, or the original text
        if no pipe delimiter is detected.
    """
    if "|" in text:
        parts = text.split("|")
        # Return the last non-empty part (rightmost = English in GOI docs)
        for part in reversed(parts):
            cleaned = part.strip()
            if cleaned:
                return cleaned
    return text


def remove_devanagari(text: str) -> str:
    """Remove all Devanagari Unicode characters from the text.

    Falls back to this when pipe-splitting is unavailable or insufficient.
    Removes Hindi characters in the range U+0900–U+097F plus related blocks.

    Args:
        text: Input string possibly containing Devanagari script.

    Returns:
        Text with all Devanagari characters removed.
    """
    return _DEVANAGARI_PATTERN.sub("", text)


def clean_bilingual_text(text: str) -> str:
    """
    Full bilingual cleaning pipeline for a single text value.

    Pipeline:
        1. Unescape HTML entities (e.g. &#124; → |)
        2. Split on '|' and keep English side
        3. Remove residual Devanagari characters via Unicode filter
        4. Collapse multiple spaces
        5. Strip

    Args:
        text: Raw text from OCR or PDF extraction, possibly bilingual.

    Returns:
        Clean English-only text, whitespace-normalized.
    """
    if not text:
        return ""

    # Step 1: HTML entity decoding
    text = html.unescape(text)

    # Step 2: Pipe-split (keep English / right side)
    text = split_bilingual(text)

    # Step 3: Remove any remaining Devanagari characters
    text = remove_devanagari(text)

    # Step 4: Normalize whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# =============================================================================
# 2. OCR Noise Removal
# =============================================================================

def fix_spaced_characters(text: str) -> str:
    """
    Collapse OCR-introduced character spacing artifacts.

    OCR engines frequently output "E l e c t r i c a l" instead of "Electrical".
    This function detects single-character sequences separated by single spaces
    and collapses them back into a single word.

    Only applies to sequences of 3 or more spaced characters to avoid
    false positives on legitimate single-letter abbreviations.

    Args:
        text: Input text possibly containing spaced OCR characters.

    Returns:
        Text with spaced characters collapsed.
    """
    def _join(m: re.Match) -> str:
        return m.group(0).replace(" ", "")

    # Match: letter + 2 or more (space + letter) sequences at word boundaries
    text = re.sub(r"\b[A-Za-z](?: [A-Za-z]){2,}\b", _join, text)
    return text


def remove_garbage_glyphs(text: str) -> str:
    """
    Remove non-printable ASCII characters and common OCR garbage glyphs.

    Keeps:
        - Standard ASCII printable characters (0x20–0x7E)
        - Common Unicode punctuation (em-dash, bullets, curly quotes)
    Removes:
        - Control characters
        - Private-use area codepoints
        - Replacement character (U+FFFD)

    Args:
        text: Raw text with potential garbage characters.

    Returns:
        Cleaned text with garbage glyphs removed.
    """
    # Replace known garbage glyphs with ASCII equivalents
    replacements = {
        "\u2013": "-",   # en-dash
        "\u2014": "-",   # em-dash
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201C": '"',   # left double quote
        "\u201D": '"',   # right double quote
        "\u2022": "*",   # bullet
        "\u2026": "...", # ellipsis
        "\uFFFD": "",    # replacement character (OCR failure marker)
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Remove remaining non-printable and control characters
    # (keep standard printable ASCII + printable Unicode letters/numbers)
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Keep: Letter, Number, Punctuation, Symbol, Space_Separator, line feeds/tabs
        if cat.startswith(("L", "N", "P", "S", "Z")) or ch in ("\n", "\t", " ", "\r"):
            cleaned_chars.append(ch)
    return "".join(cleaned_chars)


def normalize_whitespace(text: str, preserve_newlines: bool = True) -> str:
    """
    Normalize all whitespace in text.

    Args:
        text: Input text with potentially irregular whitespace.
        preserve_newlines: If True, collapse multiple newlines to a maximum of 2.
                           If False, replace all newlines with spaces.

    Returns:
        Whitespace-normalized text.
    """
    if preserve_newlines:
        # Collapse multiple spaces/tabs on a single line
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Collapse more than 2 consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)

    return text.strip()


# =============================================================================
# 3. Key-Value Pair Extraction
# =============================================================================

# Known government contract field name mappings → normalized snake_case keys
_KV_FIELD_MAP: dict[str, str] = {
    "contract no":         "contract_no",
    "contract number":     "contract_no",
    "gemc":                "contract_no",
    "order no":            "order_no",
    "order number":        "order_no",
    "po no":               "order_no",
    "purchase order":      "order_no",
    "dated":               "date_issued",
    "date":                "date_issued",
    "issue date":          "date_issued",
    "buyer":               "buyer_name",
    "buyer name":          "buyer_name",
    "seller":              "seller_name",
    "seller name":         "seller_name",
    "vendor":              "seller_name",
    "total value":         "total_value",
    "total amount":        "total_value",
    "grand total":         "total_value",
    "amount":              "total_value",
    "quantity":            "quantity",
    "qty":                 "quantity",
    "unit price":          "unit_price",
    "rate":                "unit_price",
    "delivery period":     "delivery_period",
    "delivery":            "delivery_period",
    "warranty":            "warranty_period",
    "warranty period":     "warranty_period",
    "payment terms":       "payment_terms",
    "payment":             "payment_terms",
    "consignee":           "consignee",
    "ship to":             "consignee",
    "billing address":     "billing_address",
    "organisation":        "organisation",
    "organization":        "organisation",
    "department":          "department",
    "ministry":            "ministry",
    "specification":       "specification",
}


def normalize_key(raw_key: str) -> str:
    """
    Normalize a raw KV key to a canonical snake_case form.

    Looks up the key in the known field map first. If not found,
    converts to lowercase snake_case by replacing spaces/hyphens with underscores
    and stripping non-alphanumeric characters.

    Args:
        raw_key: The raw field name/label from the document.

    Returns:
        Normalized snake_case key string.
    """
    cleaned = clean_bilingual_text(raw_key).lower().strip()
    # Remove trailing colons/punctuation
    cleaned = re.sub(r"[:\.\*\#]+$", "", cleaned).strip()

    # Check the known field map
    if cleaned in _KV_FIELD_MAP:
        return _KV_FIELD_MAP[cleaned]

    # Partial match: check if any known key is a substring
    for known_key, mapped in _KV_FIELD_MAP.items():
        if known_key in cleaned:
            return mapped

    # Fallback: convert to snake_case
    normalized = re.sub(r"[\s\-/]+", "_", cleaned)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def extract_kv_pairs(text: str) -> dict[str, str]:
    """
    Extract key-value pairs from a block of text.

    Detects patterns like:
        "Contract No: GEMC-511687756706424"
        "Total Value: INR 1,23,456"
        "Warranty Period – 2 years"

    Args:
        text: A block of text (paragraph or table cell content).

    Returns:
        Dict mapping normalized keys to extracted values.
        Empty dict if no KV pairs are found.
    """
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _KV_SEPARATORS.match(line)
        if m:
            raw_key = m.group("key").strip()
            raw_val = m.group("value").strip()
            # Only accept if key looks like a real label (no pure numbers)
            if raw_key and raw_val and not raw_key.isdigit():
                norm_key = normalize_key(raw_key)
                norm_val = clean_bilingual_text(raw_val)
                if norm_key and norm_val:
                    pairs[norm_key] = norm_val
    return pairs


# =============================================================================
# 4. Table Cell Cleaning
# =============================================================================

def clean_cell(cell_text: str) -> str:
    """
    Clean a single table cell value.

    Full pipeline:
        1. Bilingual split + Devanagari removal
        2. OCR glyph normalization
        3. Spaced-character fix
        4. Whitespace normalization (single-line)

    Args:
        cell_text: Raw cell text from OCR or PDF parser.

    Returns:
        Clean, English-only cell text.
    """
    if not isinstance(cell_text, str):
        cell_text = str(cell_text)

    text = clean_bilingual_text(cell_text)
    text = remove_garbage_glyphs(text)
    text = fix_spaced_characters(text)
    text = normalize_whitespace(text, preserve_newlines=False)
    return text


def clean_table(table: dict) -> dict:
    """
    Apply cell-level cleaning to an entire table dict.

    Operates on the standard internal table format:
        {
            "type": "table",
            "headers": [...],
            "rows": [[...], ...],
            "section": "...",
            ...
        }

    Args:
        table: Table dict with optional 'headers' and required 'rows'.

    Returns:
        A new table dict with all cell text cleaned. Does NOT mutate input.
    """
    cleaned = dict(table)  # shallow copy

    if "headers" in cleaned and isinstance(cleaned["headers"], list):
        cleaned["headers"] = [clean_cell(h) for h in cleaned["headers"]]

    if "rows" in cleaned and isinstance(cleaned["rows"], list):
        cleaned["rows"] = [
            [clean_cell(cell) for cell in row]
            for row in cleaned["rows"]
        ]

    return cleaned


# =============================================================================
# 5. Full Document Cleaning
# =============================================================================

def clean_text_block(text: str) -> str:
    """
    Full cleaning pipeline for a paragraph or heading block.

    Pipeline (in order):
        1. HTML entity decoding
        2. Bilingual split + Devanagari removal
        3. Garbage glyph removal
        4. Spaced-character OCR fix
        5. Markdown header artifact cleanup
        6. Whitespace normalization (preserving newlines)

    Args:
        text: Raw paragraph/heading text from the parser.

    Returns:
        Clean English-only text, paragraph structure preserved.
    """
    if not text:
        return ""

    # 1. HTML decoding
    text = html.unescape(text)

    # 2. Bilingual cleaning (pipe split + Devanagari removal)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(clean_bilingual_text(line))
    text = "\n".join(cleaned_lines)

    # 3. Garbage glyph removal
    text = remove_garbage_glyphs(text)

    # 4. Spaced-character OCR fix (e.g. "E l e c t r i c a l" → "Electrical")
    text = fix_spaced_characters(text)

    # 5. Markdown header artifacts (## "|Heading" → ## Heading)
    text = re.sub(r"^(#{1,6})\s*[\"'|]+\s*", r"\1 ", text, flags=re.MULTILINE)

    # 6. Whitespace normalization
    text = normalize_whitespace(text, preserve_newlines=True)

    return text


def clean_json_recursive(obj: Any) -> Any:
    """
    Recursively apply clean_text_block to all string values in a JSON structure.

    Useful for cleaning Docling JSON output or any arbitrary nested structure.

    Args:
        obj: A JSON-compatible Python object (dict, list, str, int, etc.).

    Returns:
        The same structure with all string leaves cleaned.
    """
    if isinstance(obj, str):
        return clean_bilingual_text(obj)
    if isinstance(obj, dict):
        return {k: clean_json_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json_recursive(item) for item in obj]
    return obj


# =============================================================================
# 6. Table → Markdown Conversion (Bonus)
# =============================================================================

def table_to_markdown(table: dict) -> str:
    """
    Convert a structured table dict to a GitHub Flavored Markdown table string.

    Input format:
        {
            "headers": ["Col A", "Col B", "Col C"],
            "rows": [["r1c1", "r1c2", "r1c3"], ...]
        }

    If headers are missing, uses the first row as headers.

    Args:
        table: Structured table dict (after cleaning).

    Returns:
        GFM Markdown table string.
    """
    headers = table.get("headers") or []
    rows = table.get("rows") or []

    if not headers and rows:
        headers = rows[0]
        rows = rows[1:]

    if not headers:
        return ""

    # Ensure all rows have the same column count as headers
    num_cols = len(headers)

    def _pad_row(row: list, n: int) -> list[str]:
        row = [str(c) for c in row]
        return row[:n] + [""] * max(0, n - len(row))

    header_line = "| " + " | ".join(_pad_row(headers, num_cols)) + " |"
    separator   = "| " + " | ".join(["---"] * num_cols) + " |"
    data_lines  = [
        "| " + " | ".join(_pad_row(row, num_cols)) + " |"
        for row in rows
    ]

    return "\n".join([header_line, separator] + data_lines)


# =============================================================================
# 7. Chunk Text Preparation (for embedding)
# =============================================================================

def prepare_chunk_text(block: dict) -> str:
    """
    Convert a structured block (text/table/kv) into a single flat string
    suitable for embedding.

    - text blocks → cleaned text as-is
    - table blocks → rendered as Markdown table
    - kv blocks → "Key: Value" lines

    Args:
        block: A block dict with 'type' and 'content' keys.

    Returns:
        A flat string ready for embedding. Never empty (returns "" on failure).
    """
    block_type = block.get("type", "text")
    content = block.get("content", "")

    try:
        if block_type == "text":
            return str(content).strip()

        elif block_type == "table":
            if isinstance(content, dict):
                return table_to_markdown(content)
            return str(content)

        elif block_type == "kv":
            if isinstance(content, dict):
                return "\n".join(f"{k}: {v}" for k, v in content.items())
            return str(content)

        else:
            return str(content).strip()

    except Exception as exc:  # noqa: BLE001
        logger.warning("prepare_chunk_text failed for block type '%s': %s", block_type, exc)
        return ""


# =============================================================================
# Self-test (run directly: python utils_cleaning.py)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("utils_cleaning.py — Self-Test")
    print("=" * 60)

    tests = [
        ("Bilingual split",        clean_bilingual_text("अनुबंध|Contract")),
        ("Devanagari removal",     remove_devanagari("यह एक परीक्षण है This is a test")),
        ("Spaced OCR chars",       fix_spaced_characters("E l e c t r i c a l E n g i n e e r")),
        ("KV extraction",          str(extract_kv_pairs("Contract No: GEMC-511687756706424\nTotal Value: INR 1,23,456"))),
        ("Table → MD",             table_to_markdown({"headers": ["Item", "Qty", "Price"], "rows": [["Battery", "10", "500"], ["Cable", "5", "200"]]})),
        ("Cell cleaning",          clean_cell("अनुबंध|Contract Number")),
        ("Key normalization",      normalize_key("Contract No.")),
    ]

    for label, result in tests:
        print(f"\n[{label}]")
        print(f"  → {result!r}")
