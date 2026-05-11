"""
parser.py — PaddleOCR-based Document Parsing Pipeline

Pipeline:
    PDF → PaddleOCR (PP-Structure)
        → Layout-aware extraction (Tables, Text)
        → Post-processing via cleaning_utils
        → Structured JSON
"""

import json
import logging
import re
from pathlib import Path

# Important: Use the new post-processing pipeline
from src.ingestion.cleaning_utils import (
    clean_section_text,
    clean_bilingual,
    detect_tables,
    structure_to_json
)

logger = logging.getLogger(__name__)

def parse_document(pdf_path: str) -> dict:
    """
    Parse a PDF into structured JSON using PaddleOCR PP-Structure (CPU).
    """
    try:
        from paddleocr import PPStructure
        import numpy as np
        from PIL import Image
        import fitz  # PyMuPDF
    except ImportError as exc:
        logger.error(f"PaddleOCR import failed with error: {exc}")
        raise RuntimeError(f"PaddleOCR requirements missing or failing to import: {exc}") from exc

    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info(f"[Parser] Starting PP-Structure OCR on: {path.name}")
    
    # 6. Table Parsing & layout-aware OCR
    engine = PPStructure(
        table=True,
        ocr=True,
        show_log=False,
        use_gpu=False, # Must run on CPU
        lang="en",
    )

    doc = fitz.open(str(path))
    raw_sections = []

    current_section = "General"
    
    for page_num, page in enumerate(doc, start=1):
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_arr = np.array(img)

        result = engine(img_arr)
        
        for region in result:
            r_type = region.get("type", "").lower()
            res = region.get("res", [])
            
            if r_type == "table":
                html_tbl = res.get("html", "") if isinstance(res, dict) else ""
                parsed_tbl = detect_tables(html_tbl)
                if parsed_tbl:
                    raw_sections.append({
                        "type": "table",
                        "content": parsed_tbl,
                        "page": page_num,
                        "section": current_section
                    })
            elif r_type == "title":
                # 7. Section Classification
                texts = []
                if isinstance(res, list):
                    for item in res:
                        if isinstance(item, dict):
                            texts.append(item.get("text", ""))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            texts.append(str(item[1][0]))
                title_text = " ".join(texts).strip()
                title_text = clean_bilingual(title_text)
                if title_text:
                    current_section = title_text
            else:
                texts = []
                if isinstance(res, list):
                    for item in res:
                        if isinstance(item, dict):
                            texts.append(item.get("text", ""))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            texts.append(str(item[1][0]))
                
                text_content = " ".join(t for t in texts if t).strip()
                if text_content:
                    raw_sections.append({
                        "type": "text",
                        "content": text_content,
                        "page": page_num,
                        "section": current_section
                    })

    doc.close()
    
    # Post-processing pipeline
    section_texts = {}
    processed_sections = []
    
    for sec in raw_sections:
        if sec["type"] == "table":
            processed_sections.append(sec)
        else:
            sec_name = sec["section"]
            if sec_name not in section_texts:
                section_texts[sec_name] = []
            section_texts[sec_name].append(sec["content"])
            
    for sec_name, contents in section_texts.items():
        combined_text = "\n".join(contents)
        cleaned_text = clean_section_text(combined_text)
        if cleaned_text.strip():
            processed_sections.append({
                "type": "text",
                "content": cleaned_text,
                "section": sec_name
            })
            
    # 9. Enforce Consistent Output Format
    final_output = structure_to_json(processed_sections)
    return final_output

def chunk_document(doc: dict) -> list[dict]:
    """
    Convert the structured JSON back into flat chunks for embedding/RAG.
    """
    sections = doc.get("sections", [])
    chunks = []
    chunk_idx = 0
    
    for sec in sections:
        sec_name = sec.get("name", "General")
        sec_type = sec.get("type", "kv")
        
        if sec_type == "table":
            headers = sec.get("data", {}).get("headers", [])
            rows = sec.get("data", {}).get("rows", [])
            md_table = ""
            if headers:
                md_table += "| " + " | ".join(headers) + " |\n"
                md_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in rows:
                md_table += "| " + " | ".join(row) + " |\n"
            
            if md_table.strip():
                chunks.append({
                    "text": md_table.strip(),
                    "metadata": {
                        "type": "table",
                        "section": sec_name,
                        "chunk_index": chunk_idx
                    }
                })
                chunk_idx += 1
        else:
            data = sec.get("data", {})
            if "text" in data and len(data) == 1:
                text_val = data["text"]
                chunks.append({
                    "text": text_val,
                    "metadata": {
                        "type": "text",
                        "section": sec_name,
                        "chunk_index": chunk_idx
                    }
                })
                chunk_idx += 1
            else:
                kv_text = "\n".join([f"{k}: {v}" for k, v in data.items() if k != "_text"])
                if "_text" in data:
                    kv_text += "\n\n" + data["_text"]
                if kv_text.strip():
                    chunks.append({
                        "text": kv_text.strip(),
                        "metadata": {
                            "type": "kv",
                            "section": sec_name,
                            "chunk_index": chunk_idx
                        }
                    })
                    chunk_idx += 1
                    
    return chunks

def parse_and_chunk(pdf_path: str) -> tuple[dict, list[dict]]:
    doc = parse_document(pdf_path)
    chunks = chunk_document(doc)
    return doc, chunks

if __name__ == "__main__":
    import sys
    import json as _json

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.parser <path/to/file.pdf>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    pdf = sys.argv[1]
    doc, chunks = parse_and_chunk(pdf)

    out_path = Path(pdf).with_suffix(".parsed.json")
    out_path.write_text(_json.dumps({"document": doc, "chunks": chunks}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Parsed -> {len(chunks)} chunks saved to {out_path}")
