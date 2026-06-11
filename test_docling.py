import sys
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.datamodel.base_models import InputFormat

try:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    
    # Try setting tableformer mode
    try:
        pipeline_options.table_structure_options.mode = "accurate"
        print("Set mode to accurate")
    except Exception as e:
        print("Failed to set mode:", e)

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    print("DocumentConverter initialized successfully.")
except Exception as e:
    print(f"Error: {e}")
