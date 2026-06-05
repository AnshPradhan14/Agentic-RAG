"""
metadata_builder.py
────────────────────────────────────────────────────────────────────────────
Phase 9 — Extract section hierarchy JSON from a structured markdown document.

Uses the unified LLM client (Groq or Ollama, driven by LLM_PROVIDER env var).

Phase 14 Error Handling:
 - API timeout → retried 3× with exponential backoff (inside LLMClient)
 - Invalid JSON → logged at ERROR, fallback single-section metadata returned
 - Validation failure → logged at ERROR, fallback returned
 - Empty LLM response → fallback returned

Temperature is always 0 — enforced inside LLMClient.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .llm_client import LLMClient, get_llm_client
from .llm_restructurer import load_prompt  # shared prompt loader

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_SECTION_KEYS: frozenset[str] = frozenset({
    "id",
    "heading",
    "level",
    "heading_path",
    "parent_id",
    "children",
    "approximate_page_range",
})

# ── Prompt ────────────────────────────────────────────────────────────────────

try:
    _METADATA_SYSTEM_PROMPT = load_prompt("metadata_extract_prompt.txt")
except FileNotFoundError as exc:
    raise RuntimeError(
        "metadata_extract_prompt.txt not found in ingestion/prompts/. "
        "Run Phase 1 setup first."
    ) from exc


# ── JSON safety extraction ────────────────────────────────────────────────────

def _strip_code_fences(raw: str) -> str:
    """
    Remove accidental ``` or ```json fences the model may emit despite being
    instructed not to.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        raw = raw[first_newline + 1:] if first_newline != -1 else raw[3:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    return raw.strip()


# ── Metadata validation ───────────────────────────────────────────────────────

def _validate_metadata(metadata: dict[str, Any], source_file: str) -> None:
    """
    Raise ValueError if the parsed metadata dict does not satisfy the schema.
    """
    if not isinstance(metadata.get("sections"), list):
        raise ValueError(
            f"metadata['sections'] is not a list for '{source_file}'. "
            f"Got: {type(metadata.get('sections'))}"
        )
    for i, section in enumerate(metadata["sections"]):
        missing = REQUIRED_SECTION_KEYS - set(section.keys())
        if missing:
            raise ValueError(
                f"Section {i} in '{source_file}' is missing keys: {sorted(missing)}"
            )
        page_range = section.get("approximate_page_range")
        if not isinstance(page_range, list) or len(page_range) != 2:
            raise ValueError(
                f"Section {i} ({section.get('id')}) has invalid "
                f"'approximate_page_range': {page_range!r}"
            )


# ── Fallback metadata ─────────────────────────────────────────────────────────

def _fallback_metadata(source_file: str) -> dict[str, Any]:
    """
    Single-root-section fallback used when LLM output is unparseable or invalid.
    Covers the entire document under one section — pipeline continues safely.
    Phase 14: "LLM returns invalid JSON → use fallback single-section metadata."
    """
    logger.warning(
        f"[metadata_builder] Using fallback single-section metadata for '{source_file}'."
    )
    return {
        "title": "Untitled",
        "total_sections": 1,
        "source_file": source_file,
        "sections": [
            {
                "id": "section_001",
                "heading": "Document Content",
                "level": 1,
                "heading_path": "Document Content",
                "parent_id": None,
                "children": [],
                "approximate_page_range": [1, 999],
            }
        ],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_metadata(
    structured_markdown: str,
    source_file: str,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """
    Call the configured LLM to extract a section hierarchy JSON from the
    structured markdown document.

    Returns a validated metadata dict with ``source_file`` injected.
    Falls back to a single-root-section dict on any failure so the pipeline
    can continue without aborting.

    Parameters
    ──────────
    structured_markdown : str
        Full structured markdown output from Phase 8.
    source_file : str
        PDF filename — injected into the returned dict and log messages.
    client : LLMClient | None
        Provide an existing client for test injection.

    Returns
    ───────
    dict
        Validated metadata dict, or fallback on failure.
    """
    if not structured_markdown.strip():
        logger.warning(
            f"[metadata_builder] Empty structured markdown for '{source_file}'. "
            "Returning fallback."
        )
        return _fallback_metadata(source_file)

    llm = client or get_llm_client()

    logger.info(f"[metadata_builder] Extracting section metadata for '{source_file}'.")

    # The metadata prompt requires ONLY the document as user content.
    raw_response = llm.complete(
        system_prompt=_METADATA_SYSTEM_PROMPT,
        user_prompt=structured_markdown,
    )

    # ── Phase 14: empty response guard ───────────────────────────────────────
    if not raw_response.strip():
        logger.error(
            f"[metadata_builder] LLM returned empty response for '{source_file}'. "
            "Using fallback metadata."
        )
        return _fallback_metadata(source_file)

    # ── JSON parsing ──────────────────────────────────────────────────────────
    cleaned = _strip_code_fences(raw_response)

    try:
        metadata: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(
            f"[metadata_builder] JSON decode failed for '{source_file}': {exc}. "
            f"Raw output (first 500 chars): {cleaned[:500]!r}. "
            "Using fallback metadata."
        )
        return _fallback_metadata(source_file)

    # ── Validation ────────────────────────────────────────────────────────────
    try:
        _validate_metadata(metadata, source_file)
    except ValueError as exc:
        logger.error(
            f"[metadata_builder] Validation failed for '{source_file}': {exc}. "
            "Using fallback metadata."
        )
        return _fallback_metadata(source_file)

    # ── Inject source_file and return ─────────────────────────────────────────
    metadata["source_file"] = source_file
    logger.info(
        f"[metadata_builder] Successfully extracted "
        f"{metadata.get('total_sections', '?')} section(s) for '{source_file}'."
    )
    return metadata
