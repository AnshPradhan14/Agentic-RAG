"""
cleaning_utils.py — Post-processing pipeline for unstructured OCR output
"""

import re

def clean_bilingual(text: str) -> str:
    """10. Clean Bilingual Text: Keep only English ('अनुबंध|Contract' -> 'Contract')"""
    if "|" in text:
        parts = text.split("|")
        # Keep the rightmost part that contains alphabet characters
        for part in reversed(parts):
            cleaned = part.strip()
            if cleaned and re.search(r'[A-Za-z]', cleaned):
                return cleaned
    
    # Fallback to removing Devanagari blocks
    text = re.sub(r"[\u0900-\u097F\uA8E0-\uA8FF\u1CD0-\u1CFF]+", "", text)
    return text.strip()

def remove_non_content_blocks(lines: list[str]) -> list[str]:
    """8. Remove Non-Content Blocks: <!-- image -->, empty lines, separators."""
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line in ["<!-- image -->", "<!-- table -->"]:
            continue
        if re.match(r'^[-*=_]{3,}$', line):
            continue
        cleaned.append(line)
    return cleaned

def remove_noise(lines: list[str]) -> list[str]:
    """3. OCR Noise Removal: Remove lines with >40% non-alphanumeric chars or garbage tokens."""
    cleaned = []
    for line in lines:
        chars = [c for c in line if not c.isspace()]
        if not chars:
            continue
        non_alpha = len([c for c in chars if not c.isalnum() and c not in "-.,:()"])
        if non_alpha / len(chars) > 0.4:
            continue
        
        # Garbage tokens (e.g. isolated random chars)
        if re.match(r'^[A-Z]"\s*$|^"f\s*$', line):
            continue
        
        cleaned.append(line)
    return cleaned

def deduplicate_headers(lines: list[str]) -> list[str]:
    """2. Duplicate Sections: Deduplicate consecutive identical headers."""
    cleaned = []
    last_header = None
    for line in lines:
        if line.startswith("##") or re.match(r'^#+\s', line) or re.match(r'^\d+\.\d+\s+[A-Za-z]', line):
            if line == last_header:
                continue
            last_header = line
        else:
            last_header = None
        cleaned.append(line)
    return cleaned

def is_valid_key(key: str) -> bool:
    """4. Incorrect Key Extraction: Reject purely numeric/random keys."""
    key = key.strip()
    # Must contain alphabetic words
    if not re.search(r'[A-Za-z]', key):
        return False
    # Reject things that look like dates acting as keys (e.g., 06_jan)
    if re.match(r'^\d{2}_[a-z]{3}$', key.lower()):
        return False
    return True

def merge_key_values(lines: list[str]) -> list[str]:
    """1. Broken Key-Value Pairs: Merge when key and value are split across lines."""
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Check if line looks like a key without a value (e.g., "Type :")
        m = re.match(r'^([A-Za-z\s]+)\s*:$', line)
        if m and i + 1 < len(lines):
            next_line = lines[i+1].strip()
            # Ensure next line is not another key or a header
            if not next_line.endswith(":") and not next_line.startswith("#"):
                merged.append(f"{m.group(1).strip()}: {next_line}")
                i += 2
                continue
        merged.append(line)
        i += 1
    return merged

def merge_multiline_values(lines: list[str]) -> list[str]:
    """5. Multi-line Value Merging: Merge values spanning multiple lines."""
    merged = []
    current_key = None
    current_values = []

    def flush_kv():
        if current_key:
            val_str = ", ".join(current_values).strip() if current_values else ""
            # Prevent trailing comma
            val_str = re.sub(r',\s*$', '', val_str)
            merged.append(f"{current_key}: {val_str}")

    for line in lines:
        # Check if it's a new key-value pair
        m = re.match(r'^([^:]+):\s*(.*)$', line)
        if m and is_valid_key(m.group(1)):
            flush_kv()
            key, val = m.group(1).strip(), m.group(2).strip()
            current_key = key
            current_values = [val] if val else []
        elif line.startswith("#") or re.match(r'^\d+\.\d+', line): # Header/section breaks KV
            flush_kv()
            current_key = None
            current_values = []
            merged.append(line)
        else:
            if current_key is not None:
                current_values.append(line.strip())
            else:
                merged.append(line)
    
    flush_kv()
    return merged

def clean_section_text(text: str) -> str:
    """Runs the full post-processing pipeline on unstructured text block."""
    lines = text.split('\n')
    lines = remove_non_content_blocks(lines)
    lines = remove_noise(lines)
    lines = [clean_bilingual(l) for l in lines]
    lines = deduplicate_headers(lines)
    lines = merge_key_values(lines)
    lines = merge_multiline_values(lines)
    return '\n'.join(lines)

def detect_tables(html_str: str) -> dict | None:
    """6. Table Parsing: Convert HTML table string from PP-Structure to structured JSON."""
    if not html_str:
        return None
    try:
        rows_html = re.findall(r"<tr[^>]*>(.*?)</tr>", html_str, re.DOTALL | re.IGNORECASE)
        grid = []
        for row_html in rows_html:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.DOTALL | re.IGNORECASE)
            # Remove internal HTML tags and clean bilingual text
            cleaned_cells = [clean_bilingual(re.sub(r"<[^>]+>", "", c).strip()) for c in cells]
            grid.append(cleaned_cells)

        if not grid:
            return None

        # Determine header vs data rows
        headers = grid[0]
        rows = grid[1:]
        return {
            "type": "table",
            "headers": headers,
            "rows": rows
        }
    except Exception as exc:
        return None

def structure_to_json(sections: list[dict]) -> dict:
    """9. Enforce Consistent Output Format."""
    final_sections = []
    
    for sec in sections:
        sec_name = sec.get("section", "General")
        sec_type = sec.get("type", "text")
        
        if sec_type == "table":
            final_sections.append({
                "name": sec_name,
                "type": "table",
                "data": {
                    "headers": sec.get("content", {}).get("headers", []),
                    "rows": sec.get("content", {}).get("rows", [])
                }
            })
        else:
            text_content = sec.get("content", "")
            data_dict = {}
            lines_not_kv = []
            
            for line in text_content.split('\n'):
                m = re.match(r'^([^:]+):\s*(.*)$', line)
                if m and is_valid_key(m.group(1)):
                    data_dict[m.group(1).strip()] = m.group(2).strip()
                else:
                    lines_not_kv.append(line)
            
            # If we found KVs, put them in data
            if data_dict:
                # Add any stray lines into a 'notes' or 'text' field if needed, or ignore
                if lines_not_kv and any(l.strip() for l in lines_not_kv):
                    data_dict["_text"] = "\n".join(l for l in lines_not_kv if l.strip())
                final_sections.append({
                    "name": sec_name,
                    "data": data_dict
                })
            elif text_content.strip():
                final_sections.append({
                    "name": sec_name,
                    "data": {"text": text_content.strip()}
                })
                
    return {"sections": final_sections}
