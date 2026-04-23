"""
table_relation_extractor.py — Change 2: Structured Triple Extraction

Extracts entity-attribute-value triples from tables parsed by Docling.
These triples enable a direct structured lookup path (triple_lookup tool)
for factual contract questions that avoids expensive semantic search.

Usage:
    from src.ingestion.table_relation_extractor import extract_triples
    triples = extract_triples(table_dict, doc_id="1", chunk_id=0)
"""

from typing import List, Dict


def extract_triples(table_dict: dict, doc_id: str, chunk_id: int) -> List[Dict]:
    """Extract entity-attribute-value triples from a single Docling-parsed table.

    Input: a single table parsed by Docling as a dict with keys:
      - section_header (str): nearest heading above the table
      - rows (list of lists): each row is [label_cell, value_cell, ...]

    Output: list of triple dicts, each with:
      - doc_id    (str)  : parent document identifier
      - chunk_id  (int)  : chunk index within the document
      - entity    (str)  : Subject/row context
      - attribute (str)  : Column header or field name
      - value     (str)  : Cell value

    Rows with fewer than 2 cells are silently skipped. Now handles wide data
    tables by mapping headers to cell values.
    """
    from typing import Dict, List
    triples = []
    entity_context = table_dict.get("section_header", "General")
    rows = table_dict.get("rows", [])
    
    if not rows:
        return triples

    headers = [str(c).strip() for c in rows[0]]
    data_rows = rows[1:]

    # Heuristic: 2-column tables are usually Key-Value pairs
    if len(headers) == 2:
        for row in rows:
            if len(row) >= 2:
                attribute = str(row[0]).strip()
                value = str(row[1]).strip()
                if attribute and value:
                    triples.append({
                        "doc_id":    doc_id,
                        "chunk_id":  chunk_id,
                        "entity":    entity_context,
                        "attribute": attribute,
                        "value":     value,
                    })
    else:
        # Standard data tables (>2 columns): Column 0 is the row entity.
        for row in data_rows:
            if not row or not str(row[0]).strip():
                continue
                
            row_entity = f"{entity_context} | {str(row[0]).strip()}" 
            
            for col_idx in range(1, min(len(row), len(headers))):
                attribute = headers[col_idx]
                value = str(row[col_idx]).strip()
                if attribute and value:
                    triples.append({
                        "doc_id":    doc_id,
                        "chunk_id":  chunk_id,
                        "entity":    row_entity,
                        "attribute": attribute,
                        "value":     value,
                    })
                    
    return triples
