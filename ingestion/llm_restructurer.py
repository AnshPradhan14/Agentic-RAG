"""
llm_restructurer.py
────────────────────────────────────────────────────────────────────────────
Phase 8 — LLM-based Markdown Restructuring.

Uses the unified LLM client (Groq or Ollama, driven by LLM_PROVIDER env var)
to clean and restructure raw PDF markdown into well-organised heading-hierarchy
markdown.

Phase 14 Error Handling (delegated to LLMClient):
 - API timeout → 3× retry with exponential backoff (2s, 4s, 8s)
 - Empty response → fall back to raw input markdown
 - API error → re-raised to pipeline for summary recording

Temperature is always 0 — enforced inside LLMClient.
"""

from __future__ import annotations

import logging
import os

import tiktoken

from .llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


# ── Prompt loader (shared by metadata_builder.py too) ────────────────────────

def load_prompt(filename: str) -> str:
    """Load a system prompt file from ingestion/prompts/."""
    prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
    path = os.path.join(prompt_dir, filename)
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


try:
    RESTRUCTURE_SYSTEM_PROMPT = load_prompt("restructure_prompt.txt")
except FileNotFoundError as exc:
    raise RuntimeError(
        "restructure_prompt.txt not found in ingestion/prompts/. "
        "Run Phase 1 setup first."
    ) from exc


# ── Token estimation ──────────────────────────────────────────────────────────

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def split_boilerplate(text: str) -> tuple[str, str]:
    """
    Split the markdown into structured header section and boilerplate terms.
    Returns (header_part, boilerplate_part).
    If no boundary is found, boilerplate_part is empty.
    """
    import re
    # Match "1. General Terms and Conditions" or "1. Special terms and conditions" at the start of a line
    pattern = re.compile(r'(?im)^1\.\s+(?:General|Special)\s+terms\s+and\s+conditions')
    match = pattern.search(text)
    if match:
        return text[:match.start()].strip(), text[match.start():].strip()
    return text, ""


def restructure_markdown(
    raw_markdown: str,
    source_label: str = "",
    client: LLMClient | None = None,
) -> str:
    """
    Restructure a raw markdown string into clean heading-hierarchy markdown.

    Parameters
    ──────────
    raw_markdown : str
        The raw, unstructured markdown extracted from a PDF batch.
    source_label : str
        Human-readable label used in log messages (e.g. "report.pdf — batch 2/4").
    client : LLMClient | None
        Provide an existing client for test injection.  When None, a fresh
        client is instantiated from the environment (LLM_PROVIDER).

    Returns
    ───────
    str
        Structured markdown.  Falls back to ``raw_markdown`` if the LLM
        returns empty content (Phase 14 rule).
    """
    llm = client or get_llm_client()
    label = f" [{source_label}]" if source_label else ""
    provider_name = type(llm).__name__.replace("Client", "").lstrip("_")
    model_name = getattr(llm, "_model", "unknown")

    # Split boilerplate to bypass LLM and save processing time
    header_part, boilerplate_part = split_boilerplate(raw_markdown)

    logger.info(
        f"[llm_restructurer] Restructuring{label} using provider={provider_name}, model={model_name} "
        f"— {_count_tokens(header_part)} input tokens (bypassing {_count_tokens(boilerplate_part)} boilerplate tokens)."
    )

    user_prompt = (
        "Restructure the following raw markdown extracted from a PDF:\n\n"
        + header_part
    )

    result = llm.complete(
        system_prompt=RESTRUCTURE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # Phase 14: "LLM returns empty content → use raw filtered markdown as-is"
    if not result.strip():
        logger.error(
            f"[llm_restructurer] Empty response{label}. "
            "Falling back to raw input markdown."
        )
        return raw_markdown

    # Force Organisation Details to be an H2 (local LLMs frequently ignore prompt rules for the very first heading)
    import re
    result = re.sub(r'(?im)^#\s+Organisation\s+Details', r'## Organisation Details', result)

    # Append the bypassed boilerplate exactly as it was
    if boilerplate_part:
        result = result.strip() + "\n\n" + boilerplate_part

    # Phase 14 Fix: Deterministically fix orphaned financial values from GeM tables
    result = _fix_financial_orphans(result)

    logger.info(
        f"[llm_restructurer] Restructure complete{label} — "
        f"{_count_tokens(result)} output tokens."
    )
    return result

def _fix_financial_orphans(text: str) -> str:
    """
    Finds headings for financial fields that have empty bodies, and searches
    for orphaned numeric values later in the document to attach to them.

    GeM contracts have a complex merged-cell pricing table that Docling
    consistently fails to parse. The monetary values (Total Value without
    Addons, Total Addon Value, Total Value Including Addons, Total Contract
    Value) end up as bare number lines scattered in the raw text.

    Assignment order matches the order values appear in a GeM contract:
      1. Total Value without Addons(INR)
      2. Total Addon Value(INR)
      3. Total Value Including Addons(INR)
      4. Amount of Contract            <- section wrapper, mirrors value from #3
      5. Total Contract Value Including All Duties and Taxes(INR)
    """
    import re
    lines = text.split("\n")

    # Collect standalone numeric lines (e.g. "233640", "0", "1,23,456")
    numeric_line_pattern = re.compile(r'^\s*(\d[\d,\.]*)\s*$')
    orphan_indices = [i for i, line in enumerate(lines) if numeric_line_pattern.match(line)]

    if not orphan_indices:
        return text

    # Value fields that each consume one orphan number, in document order
    value_fields = [
        "Total Value without Addons(INR)",
        "Total Addon Value(INR)",
        "Total Value Including Addons(INR)",
        "Total Contract Value Including All Duties and Taxes(INR)",
    ]
    # Wrapper headings that mirror another field's value (no orphan consumed)
    wrapper_mirrors = {
        "Amount of Contract": "Total Value Including Addons(INR)",
    }

    # Build heading -> line-index map
    heading_pat = re.compile(r'^##\s+(.+?)\s*$')
    heading_line_map: dict = {}
    for i, line in enumerate(lines):
        m = heading_pat.match(line)
        if m:
            heading_line_map[m.group(1)] = i

    assigned_values: dict = {}  # field label -> resolved value string

    def _has_value(idx: int) -> str | None:
        """Return the existing value string if heading already has a body, else None."""
        for j in range(idx + 1, len(lines)):
            stripped = lines[j].strip()
            if stripped:
                return None if stripped.startswith("#") else stripped
        return None

    # ── Pass 1: assign orphan numbers to value fields in order ───────────────
    orphan_queue = list(orphan_indices)
    for field in value_fields:
        idx = heading_line_map.get(field)
        if idx is None:
            continue
        existing = _has_value(idx)
        if existing:
            # Already has a value — record it for mirror fields
            assigned_values[field] = existing
            continue
        if not orphan_queue:
            break
        orphan_idx = orphan_queue.pop(0)
        val = lines[orphan_idx].strip()
        lines[orphan_idx] = ""  # remove orphan from its original location
        insert_at = idx + 1
        lines.insert(insert_at, f"**{field}:** {val}")
        assigned_values[field] = val
        # Shift all tracked indices that come after the insertion point
        orphan_queue = [x + 1 if x > idx else x for x in orphan_queue]
        heading_line_map = {k: (v + 1 if v > idx else v) for k, v in heading_line_map.items()}

    # ── Pass 2: fill wrapper/mirror headings ─────────────────────────────────
    for wrapper, parent_field in wrapper_mirrors.items():
        parent_val = assigned_values.get(parent_field)
        if parent_val is None:
            continue
        idx = heading_line_map.get(wrapper)
        if idx is None:
            continue
        if _has_value(idx) is None:  # empty heading
            lines.insert(idx + 1, f"**{wrapper}:** {parent_val}")

    return "\n".join(lines).replace("\n\n\n", "\n\n").strip()




# ── Batch orchestration ───────────────────────────────────────────────────────

BATCH_BREAK_MARKER = "\n\n---BATCH_BREAK---\n\n"

# Final stitch pass token threshold (Phase 8 spec)
FINAL_STITCH_TOKEN_LIMIT = 0


def restructure_batches(
    batch_markdowns: list[str],
    pdf_name: str = "",
) -> str:
    """
    Orchestrate restructuring across one or more batches.

    Steps:
      1. Build one shared LLM client (single provider init, not one per batch).
      2. Call restructure_markdown() once per batch.
      3. Stitch batches, remove BATCH_BREAK markers, deduplicate seam headings.
      4. If stitched result is < FINAL_STITCH_TOKEN_LIMIT, run ONE final pass
         to fix heading continuity across seam boundaries.

    Parameters
    ──────────
    batch_markdowns : list[str]
        Filtered markdown strings, one per batch (from batch_processor).
    pdf_name : str
        Document name used in log messages.

    Returns
    ───────
    str
        The fully stitched, structured markdown for the entire document.
    """
    if not batch_markdowns:
        logger.warning(f"[llm_restructurer] No batches provided for '{pdf_name}'.")
        return ""

    # Build the client ONCE — reused across all batches
    client = get_llm_client()

    restructured_batches: list[str] = [""] * len(batch_markdowns)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_single_batch(i: int, batch_md: str) -> tuple[int, str]:
        label = f"{pdf_name} — batch {i}/{len(batch_markdowns)}"
        if not batch_md.strip():
            logger.warning(f"[llm_restructurer] Batch {i} for '{pdf_name}' is empty — skipping.")
            return (i, "")
        structured = restructure_markdown(batch_md, source_label=label, client=client)
        return (i, structured)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [
            executor.submit(process_single_batch, i, batch_md)
            for i, batch_md in enumerate(batch_markdowns, start=1)
        ]
        for future in as_completed(futures):
            try:
                i, structured = future.result()
                restructured_batches[i - 1] = structured
            except Exception as exc:
                logger.error(f"[llm_restructurer] Batch failed: {exc}")
                raise

    # Remove empty batches that were skipped
    restructured_batches = [b for b in restructured_batches if b]

    if not restructured_batches:
        logger.error(
            f"[llm_restructurer] All batches empty for '{pdf_name}'. "
            "Returning empty string."
        )
        return ""

    # Stitch + clean seams
    stitched = BATCH_BREAK_MARKER.join(restructured_batches)
    stitched = _clean_stitched(stitched)

    logger.info(
        f"[llm_restructurer] Stitched {len(restructured_batches)} batch(es) "
        f"for '{pdf_name}' — {_count_tokens(stitched)} total tokens."
    )

    # Optional final continuity pass (only if document fits in context)
    if len(restructured_batches) > 1:
        token_count = _count_tokens(stitched)
        if token_count < FINAL_STITCH_TOKEN_LIMIT:
            logger.info(
                f"[llm_restructurer] Running final seam-continuity pass "
                f"({token_count} tokens < {FINAL_STITCH_TOKEN_LIMIT} limit)."
            )
            stitched = restructure_markdown(
                stitched,
                source_label=f"{pdf_name} — final stitch pass",
                client=client,
            )
        else:
            logger.info(
                f"[llm_restructurer] Skipping final pass — "
                f"{token_count} tokens > {FINAL_STITCH_TOKEN_LIMIT} limit."
            )

    return stitched


def _clean_stitched(text: str) -> str:
    """
    Remove BATCH_BREAK markers and deduplicate headings that appear identically
    on both sides of a batch seam boundary (Phase 14 spec).
    """
    cleaned = text.replace(BATCH_BREAK_MARKER, "\n\n")

    lines = cleaned.split("\n")
    deduped: list[str] = []
    prev_heading: str | None = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if stripped == prev_heading:
                logger.warning(
                    f"[llm_restructurer] Duplicate heading at batch seam — "
                    f"removing: {stripped!r}"
                )
                continue
            prev_heading = stripped
        elif stripped:
            prev_heading = None   # non-empty body text resets heading dedup context

        deduped.append(line)

    return "\n".join(deduped)
