"""
chunker.py
────────────────────────────────────────────────────────────────────────────
Phase 10 — Section-aware chunker.

Takes a fully structured markdown string + validated metadata JSON produced
by metadata_builder.py and returns one chunk dict per section/subsection.

Core algorithm
──────────────
1.  Walk the structured markdown line-by-line and detect heading boundaries.
2.  Assign each heading line a (level, text, line_index) triple.
3.  Slice lines between consecutive headings to extract raw section content.
4.  Match each metadata section to its extracted content by normalised heading text.
5.  Enforce the 50-char minimum: short sections are merged upward into their
    immediate parent chunk.
6.  Build the final chunk dict with all 14 required fields.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# ── Tokeniser (shared instance, thread-safe after construction) ──────────────

_TOKENISER = tiktoken.get_encoding("cl100k_base")

# ── Regex helpers ────────────────────────────────────────────────────────────

# Matches ATX headings: optional leading whitespace, 1-4 hashes, then text
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,4})\s+(?P<text>.+)$")

# Table delimiter row: any line containing at least one |---| or | :---: |
_TABLE_DELIM_RE = re.compile(r"\|\s*:?-{2,}:?\s*\|")

# Minimum content length (characters) before merging into parent
_MIN_CONTENT_LENGTH = 50


# ── Internal dataclasses / type aliases ──────────────────────────────────────

class _HeadingSpan:
    """Represents a parsed heading and the body lines that follow it."""

    __slots__ = ("level", "text", "start_line", "body_lines")

    def __init__(self, level: int, text: str, start_line: int) -> None:
        self.level = level
        self.text = text
        self.start_line = start_line
        self.body_lines: list[str] = []

    @property
    def raw_content(self) -> str:
        """Heading line + body, joined.  May be empty if there is no body."""
        heading_prefix = "#" * self.level
        head = f"{heading_prefix} {self.text}"
        if self.body_lines:
            return head + "\n" + "\n".join(self.body_lines)
        return head

    @property
    def body_text(self) -> str:
        return "\n".join(self.body_lines).strip()


# ── Step 1 — Parse markdown into heading spans ───────────────────────────────

def _parse_heading_spans(markdown: str) -> list[_HeadingSpan]:
    """
    Walk the markdown line-by-line.  Every ATX heading (H1–H4) opens a new
    span.  Lines that belong to the *body* of the preceding heading are
    accumulated until the next same-or-higher-level heading is found.

    Section content = everything from its heading line up to (but not
    including) the next same-or-higher-level heading. (Spec §Phase 10.)
    """
    lines = markdown.splitlines()
    spans: list[_HeadingSpan] = []
    current: _HeadingSpan | None = None

    for line_idx, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group("hashes"))
            text = m.group("text").strip()
            new_span = _HeadingSpan(level, text, line_idx)

            if current is not None:
                # Trim trailing blank lines from body
                while current.body_lines and not current.body_lines[-1].strip():
                    current.body_lines.pop()
                spans.append(current)

            current = new_span
        else:
            if current is not None:
                current.body_lines.append(line)

    # Flush last span
    if current is not None:
        while current.body_lines and not current.body_lines[-1].strip():
            current.body_lines.pop()
        spans.append(current)

    return spans


# ── Step 2 — Build a lookup: normalised heading text → span ─────────────────

def _normalise_heading(text: str) -> str:
    """Lower-case and collapse whitespace for fuzzy heading matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _build_span_lookup(spans: list[_HeadingSpan]) -> dict[str, _HeadingSpan]:
    """
    Map normalised heading text → span.  On collision (two headings with
    identical text), the first occurrence wins.
    """
    lookup: dict[str, _HeadingSpan] = {}
    for span in spans:
        key = _normalise_heading(span.text)
        if key not in lookup:
            lookup[key] = span
    return lookup


# ── Step 3 — Resolve content for each metadata section ──────────────────────

def _resolve_content(section: dict[str, Any], lookup: dict[str, _HeadingSpan]) -> str:
    """
    Return the raw (pre-normalisation) content string for a metadata section.

    Falls back to the heading line only when no matching span is found
    (spec: "include only the heading line as content — do not create empty chunks").
    """
    key = _normalise_heading(section["heading"])
    span = lookup.get(key)

    if span is None:
        logger.debug(
            f"[chunker] No span found for heading '{section['heading']}'. "
            "Using heading-only content."
        )
        heading_prefix = "#" * section["level"]
        return f"{heading_prefix} {section['heading']}"

    return span.raw_content


# ── Step 4 — Token counting and table detection ──────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_TOKENISER.encode(text))


def _has_table(content: str) -> bool:
    return bool(_TABLE_DELIM_RE.search(content))


# ── Step 5 — Build chunk dict ────────────────────────────────────────────────

def _build_chunk(
    section: dict[str, Any],
    content: str,
    chunk_index: int,
    source_file: str,
    source_file_stem: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full 14-field chunk dict."""

    # Resolve parent and child headings from metadata sections list
    sections_by_id: dict[str, dict] = {s["id"]: s for s in metadata["sections"]}

    parent_heading: str | None = None
    if section.get("parent_id") and section["parent_id"] in sections_by_id:
        parent_heading = sections_by_id[section["parent_id"]]["heading"]

    child_headings: list[str] = [
        sections_by_id[child_id]["heading"]
        for child_id in section.get("children", [])
        if child_id in sections_by_id
    ]

    # heading_path_list  e.g. "Introduction > Background > Concepts" → [..., ...]
    heading_path: str = section.get("heading_path", section["heading"])
    heading_path_list: list[str] = [p.strip() for p in heading_path.split(">")]

    return {
        "chunk_id": f"{source_file_stem}__{section['id']}",
        "source_file": source_file,
        "page_range": section.get("approximate_page_range", [0, 0]),
        "heading_path": heading_path,
        "heading_path_list": heading_path_list,
        "section_level": section["level"],
        "section_heading": section["heading"],
        "chunk_index": chunk_index,
        "content": content,
        "content_length": len(content),
        "token_estimate": _count_tokens(content),
        "has_table": _has_table(content),
        "parent_heading": parent_heading,
        "child_headings": child_headings,
    }


# ── Step 6 — Merge short chunks into parent ──────────────────────────────────

def _merge_short_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Spec: "Minimum chunk content length: 50 characters — if section content
    is shorter, merge it into parent section chunk."

    We do a single forward pass.  When a chunk is below the threshold we
    append its content to the most-recent chunk that has the same or higher
    level (i.e. its nearest ancestor already in the output list) and discard
    the short chunk itself.  Indices are then renumbered.
    """
    merged: list[dict[str, Any]] = []

    for chunk in chunks:
        if chunk["content_length"] >= _MIN_CONTENT_LENGTH:
            merged.append(chunk)
            continue

        # Find nearest ancestor chunk already emitted
        parent_chunk: dict[str, Any] | None = None
        for candidate in reversed(merged):
            if candidate["section_level"] < chunk["section_level"]:
                parent_chunk = candidate
                break

        if parent_chunk is None and merged:
            # No ancestor found — merge into immediately preceding chunk
            parent_chunk = merged[-1]

        if parent_chunk is not None:
            logger.info(
                f"[chunker] Merging short chunk '{chunk['section_heading']}' "
                f"({chunk['content_length']} chars) into "
                f"'{parent_chunk['section_heading']}'."
            )
            separator = "\n\n" if parent_chunk["content"].strip() else ""
            parent_chunk["content"] += separator + chunk["content"]
            parent_chunk["content_length"] = len(parent_chunk["content"])
            parent_chunk["token_estimate"] = _count_tokens(parent_chunk["content"])
            parent_chunk["has_table"] = parent_chunk["has_table"] or chunk["has_table"]
            # Extend child_headings to reflect the absorbed section
            if chunk["section_heading"] not in parent_chunk["child_headings"]:
                parent_chunk["child_headings"].append(chunk["section_heading"])
        else:
            # No preceding chunk at all — keep as-is (first chunk edge case)
            merged.append(chunk)

    # Renumber chunk_index sequentially
    for i, chunk in enumerate(merged):
        chunk["chunk_index"] = i

    return merged


# ── Public API ───────────────────────────────────────────────────────────────

def chunk_document(
    structured_markdown: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert a structured markdown string + validated metadata dict into a
    list of chunk dicts (one per section, after short-chunk merging).

    Parameters
    ──────────
    structured_markdown
        The clean markdown string produced by llm_restructurer.py.
    metadata
        The validated metadata dict produced by metadata_builder.py.
        Must contain 'source_file' and 'sections' keys.

    Returns
    ───────
    List of chunk dicts matching the 14-field schema defined in Phase 10.
    """
    source_file: str = metadata.get("source_file", "unknown")
    source_file_stem: str = Path(source_file).stem

    sections: list[dict[str, Any]] = metadata.get("sections", [])
    if not sections:
        logger.warning(
            f"[chunker] Metadata for '{source_file}' has no sections. "
            "Returning empty chunk list."
        )
        return []

    # ── Parse markdown heading spans ─────────────────────────────────────────
    spans = _parse_heading_spans(structured_markdown)
    span_lookup = _build_span_lookup(spans)

    logger.info(
        f"[chunker] Parsed {len(spans)} heading spans from structured markdown "
        f"for '{source_file}'."
    )

    # ── Build one chunk per section ──────────────────────────────────────────
    raw_chunks: list[dict[str, Any]] = []
    for idx, section in enumerate(sections):
        content = _resolve_content(section, span_lookup)
        chunk = _build_chunk(
            section=section,
            content=content,
            chunk_index=idx,
            source_file=source_file,
            source_file_stem=source_file_stem,
            metadata=metadata,
        )
        raw_chunks.append(chunk)

    logger.info(
        f"[chunker] Built {len(raw_chunks)} raw chunks for '{source_file}' "
        "before short-chunk merging."
    )

    # ── Merge short chunks upward ────────────────────────────────────────────
    final_chunks = _merge_short_chunks(raw_chunks)

    logger.info(
        f"[chunker] Final chunk count for '{source_file}': {len(final_chunks)} "
        f"(merged {len(raw_chunks) - len(final_chunks)} short chunks)."
    )

    return final_chunks
