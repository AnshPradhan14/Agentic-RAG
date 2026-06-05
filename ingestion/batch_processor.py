import os
from pypdf import PdfReader, PdfWriter

def get_batch_ranges(total_pages: int, batch_size: int = 25, threshold: int = 50) -> list[tuple[int,int]]:
    """Returns list of (start_page, end_page) tuples. Pages are 1-indexed."""
    if total_pages <= threshold:
        return [(1, total_pages)]
    ranges = []
    start = 1
    while start <= total_pages:
        end = min(start + batch_size - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges

def get_total_pages(pdf_path: str) -> int:
    reader = PdfReader(pdf_path)
    return len(reader.pages)

def split_pdf_into_batches(pdf_path: str, output_dir: str) -> list[str]:
    """
    Splits the PDF into batch temp files. Returns list of paths to the batch PDFs.
    """
    total_pages = get_total_pages(pdf_path)
    ranges = get_batch_ranges(total_pages)
    
    # If only one batch, just return the original file
    if len(ranges) == 1 and ranges[0] == (1, total_pages):
        return [pdf_path]
        
    batch_paths = []
    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    
    reader = PdfReader(pdf_path)
    for i, (start, end) in enumerate(ranges):
        writer = PdfWriter()
        # Pages are 1-indexed in ranges, 0-indexed in pypdf
        for page_num in range(start - 1, end):
            writer.add_page(reader.pages[page_num])
            
        batch_filename = f"{pdf_stem}_batch_{i+1}.pdf"
        batch_path = os.path.join(output_dir, batch_filename)
        with open(batch_path, "wb") as f_out:
            writer.write(f_out)
            
        batch_paths.append(batch_path)
        
    return batch_paths

def stitch_batches(restructured_batches: list[str]) -> str:
    """
    Stitch them in order with a \n\n---BATCH_BREAK---\n\n separator.
    Then remove ---BATCH_BREAK--- markers to ensure continuity.
    """
    stitched_text = "\n\n---BATCH_BREAK---\n\n".join(restructured_batches)
    stitched_text = stitched_text.replace("\n\n---BATCH_BREAK---\n\n", "\n\n")
    return stitched_text
