"""
language_filter.py
──────────────────────────────────────────────────────────────────────────────
Strips Hindi/Devanagari text from Docling's raw markdown output.

Strategy — character-level stripping (not paragraph-level discard):

  1. Bilingual separator pattern:
       GEM contracts write every label as  "हिंदी लेबल|English Label"
       Docling escapes the pipe as &#124; inside table cells.
       We detect the pattern <hindi_part>|<english_part> and keep ONLY the
       English part after the separator.  This fixes the garbled bilingual
       table headers completely.

  2. Raw Devanagari stripping:
       Any remaining Devanagari characters (U+0900–U+097F) are removed.

  3. &#124; entity cleanup:
       After stripping, residual &#124; HTML entities (escaped pipes that
       had no English counterpart) are removed.

  4. Empty-cell / empty-row removal:
       Table rows that collapse to all-empty cells after stripping are dropped
       so the LLM doesn't see them as phantom rows.

  5. Collapsed whitespace:
       Extra spaces and blank lines produced by stripping are normalised.
"""

import regex

# ── Unicode ranges ────────────────────────────────────────────────────────────
_DEVANAGARI = regex.compile(r'[\u0900-\u097F]+')

# ── Bilingual separator: "हिंदी text|English text"  (or &#124; variant) ───────
# Captures everything before the first | (or &#124;) that follows Devanagari,
# and keeps only the part AFTER the separator.
_BILINGUAL_SEP = regex.compile(
    r'[\u0900-\u097F\s\"\'\`\.\,\:\;\!\?\-/\\()_\d]*'  # Hindi prefix (incl. punctuation)
    r'(?:\||\&\#124;)'                                   # separator
    r'([^\|\n\u0900-\u097F]*)',                          # English part (no Hindi, no newline)
    regex.UNICODE
)


def _strip_cell(cell: str) -> str:
    """
    Clean one table cell:
      1. Replace every "hindi|english" bilingual pair with just "english".
      2. Strip residual Devanagari characters.
      3. Remove &#124; entities.
      4. Remove Windows-1252 / latin1 control character artifacts (0x80–0x9F)
         and isolated non-printable characters Docling emits from bad glyph reads.
      5. Normalise whitespace.
    """
    # Step 1 — bilingual pair: keep English side
    def _keep_english(m: regex.Match) -> str:
        english = m.group(1).strip()
        return english if english else ''

    cell = _BILINGUAL_SEP.sub(_keep_english, cell)

    # Step 2 — strip any remaining Devanagari
    cell = _DEVANAGARI.sub('', cell)

    # Step 3 — remove leftover &#124; entities
    cell = cell.replace('&#124;', '').replace('&amp;#124;', '')

    # Step 4 — remove Windows-1252 control chars (0x80–0x9F) and isolated
    #          non-ASCII single chars that are encoding garbage from the PDF
    cell = regex.sub(r'[\x80-\x9f]', '', cell)
    # Remove lone non-ASCII alphabetic chars surrounded by spaces (e.g. " w ", " K ")
    cell = regex.sub(r'(?<!\w)[^\x00-\x7F](?!\w)', '', cell)

    # Step 5 — normalise whitespace
    cell = ' '.join(cell.split())
    return cell


def _is_separator_row(cells: list[str]) -> bool:
    """Return True if this is a markdown table separator row (e.g. |---|---|)."""
    return all(regex.fullmatch(r'-+', c.strip()) for c in cells if c.strip())


def _is_pseudo_table_line(line: str) -> bool:
    """
    Return True if this line looks like a plain key-value pair disguised as a table row
    (starts with |, has 1-2 cells, no separator pattern, contains a colon).
    """
    if not line.startswith('|'):
        return False
    parts = line.split('|')
    inner = parts[1:-1] if line.endswith('|') else parts[1:]
    if not inner:
        return False
    if len(inner) <= 2:
        if _is_separator_row(inner):
            return False
        return any(':' in cell for cell in inner)
    return False


def _strip_table_row(line: str) -> list[str] | None:
    """
    Strip Hindi from a table line and return the list of inner cells (no leading/trailing empty).
    Returns None if the row is a separator or becomes fully empty.
    """
    parts = line.split('|')
    inner = parts[1:-1] if len(parts) >= 2 else parts

    if _is_separator_row(inner):
        return None  # sentinel: separator row

    cleaned_inner = [_strip_cell(c) for c in inner]

    # Drop rows where every inner cell is empty
    if all(c.strip() == '' for c in cleaned_inner):
        return []  # sentinel: empty row

    return cleaned_inner


def _deduplicate_row(cells: list[str]) -> list[str]:
    """
    Collapse adjacent identical cells in a table row.
    Docling duplicates merged-cell content across every column slot it spans.
    E.g. ['GENERIC', 'GENERIC', 'Battery Capacity', 'Battery Capacity', '84 Ah', '84 Ah']
    becomes ['GENERIC', 'Battery Capacity', '84 Ah'].
    """
    if not cells:
        return cells
    result = [cells[0]]
    for cell in cells[1:]:
        if cell.strip() != result[-1].strip():
            result.append(cell)
    return result


def _deduplicate_table_block(table_lines: list[str]) -> list[str]:
    """
    Process a consecutive block of markdown table lines:
    1. Strip Hindi from every cell.
    2. Deduplicate adjacent identical cells in each row (fixes Docling merged-cell explosion).
    3. Align all rows to the same column count (the mode width of data rows).
    4. Rebuild proper GFM table syntax.
    """
    if not table_lines:
        return []

    # Pass 1: strip + dedup each row, track column counts
    processed_rows: list[list[str] | None] = []  # None = separator, [] = drop
    col_counts: list[int] = []

    for line in table_lines:
        inner = _strip_table_row(line)
        if inner is None:
            processed_rows.append(None)  # separator placeholder
        elif inner == []:
            pass  # drop empty row entirely
        else:
            deduped = _deduplicate_row(inner)
            processed_rows.append(deduped)
            col_counts.append(len(deduped))

    if not col_counts:
        return []

    # Determine target column count: maximum width among data rows to prevent truncation
    target_cols = max(col_counts)

    # Pass 2: rebuild rows, padding/trimming to target_cols
    output: list[str] = []
    separator_inserted = False

    for i, row in enumerate(processed_rows):
        if row is None:
            # Rebuild separator to match target_cols
            sep = '|' + '|'.join(['---'] * target_cols) + '|'
            output.append(sep)
            separator_inserted = True
            continue

        # Pad short rows with empty cells, trim extra
        padded = row[:target_cols]
        while len(padded) < target_cols:
            padded.append('')

        rebuilt = '|' + '|'.join(padded) + '|'
        output.append(rebuilt)

        # If no separator row existed, insert one after the first row (header)
        if i == 0 and not separator_inserted:
            sep = '|' + '|'.join(['---'] * target_cols) + '|'
            output.append(sep)
            separator_inserted = True

    return output


def _strip_line(line: str) -> str:
    """Strip Hindi from a non-table line, preserving markdown headings."""
    # Bilingual inline pairs
    def _keep_english(m: regex.Match) -> str:
        english = m.group(1).strip()
        return english if english else ''

    line = _BILINGUAL_SEP.sub(_keep_english, line)
    line = _DEVANAGARI.sub('', line)
    line = line.replace('&#124;', '').replace('&amp;#124;', '')
    line = regex.sub(r'[\x80-\x9f]', '', line)
    line = regex.sub(r'(?<!\w)[^\x00-\x7F](?!\w)', '', line)
    line = ' '.join(line.split())
    return line


def hoist_contract_details(text: str) -> str:
    """
    Find Contract No and Generated Date anywhere in the text,
    remove them from their disjointed original locations,
    and prepend them at the top under a '## Contract' heading.
    """
    import re
    
    contract_no = ""
    # match "Contract No: GEMC-..."
    m_forward = re.search(r'(?im)Contract No\s*:?\s*(GEMC-[\d]+)', text)
    if m_forward:
        contract_no = m_forward.group(1)
        text = text[:m_forward.start()] + text[m_forward.end():]
    else:
        # match "GEMC-... \n Contract No:"
        m_backward = re.search(r'(GEMC-[\d]+)[\s\n]*Contract No\s*:?', text, flags=re.IGNORECASE)
        if m_backward:
            contract_no = m_backward.group(1)
            text = text[:m_backward.start()] + text[m_backward.end():]
            
    gen_date = ""
    # match "Generated Date : 21-Jan-2026"
    m_date_forward = re.search(r'(?im)(?:Contract\s+)?Generated Date\s*:?\s*(\d{1,2}-[a-zA-Z]{3}-\d{4})', text)
    if m_date_forward:
        gen_date = m_date_forward.group(1)
        text = text[:m_date_forward.start()] + text[m_date_forward.end():]
    else:
        # backward pattern
        m_date_backward = re.search(r'(\d{1,2}-[a-zA-Z]{3}-\d{4})[\s\n]*(?:Contract\s+)?Generated Date\s*:?', text, flags=re.IGNORECASE)
        if m_date_backward:
            gen_date = m_date_backward.group(1)
            text = text[:m_date_backward.start()] + text[m_date_backward.end():]

    # Clean up empty Contract headings left behind
    text = re.sub(r'(?im)^#*\s*Contract\s*$', '', text)

    if contract_no or gen_date:
        hoisted = "## Contract\n"
        if contract_no:
            hoisted += f"- Contract No: {contract_no}\n"
        if gen_date:
            hoisted += f"- Generated Date: {gen_date}\n"
        hoisted += "\n"
        
        text = hoisted + text.lstrip()
        
    return text


def filter_hindi(raw_text: str) -> str:
    """
    Strip all Hindi/Devanagari content from Docling's raw markdown, and
    deduplicate adjacent identical table columns produced by Docling's
    merged-cell (colspan) expansion.

    Two-pass strategy:
      Pass 1 — line-by-line: strip Hindi from all non-table lines.
      Pass 2 — table-block-level: group consecutive table lines into blocks,
               strip Hindi + deduplicate columns for each block as a whole.
    """
    if not raw_text:
        return raw_text

    def deduplicate_consecutive_lines(text: str) -> str:
        lines = text.splitlines()
        deduped = []
        prev = None
        for line in lines:
            stripped = line.strip()
            if stripped and stripped == prev:
                continue
            if stripped:
                prev = stripped
            deduped.append(line)
        return '\n'.join(deduped)

    raw_text = deduplicate_consecutive_lines(raw_text)

    output_lines: list[str] = []
    table_buffer: list[str] = []

    def flush_table_buffer():
        """Process buffered table lines and append results to output_lines."""
        if table_buffer:
            cleaned_block = _deduplicate_table_block(table_buffer)
            output_lines.extend(cleaned_block)
            table_buffer.clear()

    for line in raw_text.splitlines():
        # Blank line: flush any in-progress table block, then keep blank
        if not line.strip():
            flush_table_buffer()
            output_lines.append('')
            continue

        s = line.strip()
        
        if _is_pseudo_table_line(s):
            # Treat as non-table line, strip the leading/trailing |
            flush_table_buffer()
            cleaned_s = s.strip('|').strip()
            cleaned = _strip_line(cleaned_s)
            if cleaned.strip():
                output_lines.append(cleaned)
            continue

        is_table = s.count('|') >= 2

        if is_table:
            # Accumulate into block — process as a unit when block ends
            table_buffer.append(line)
        else:
            # Non-table line: flush any in-progress table block first
            flush_table_buffer()
            cleaned = _strip_line(line)
            if cleaned.strip():
                output_lines.append(cleaned)

    # Flush any trailing table block
    flush_table_buffer()

    result = '\n'.join(output_lines)

    # Collapse 3+ consecutive blank lines to a single blank line
    result = regex.sub(r'\n{3,}', '\n\n', result)

    # Clean up and hoist Contract Details to the top
    result = hoist_contract_details(result.strip())

    return result
