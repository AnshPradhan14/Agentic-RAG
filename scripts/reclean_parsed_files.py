import json
import logging
from pathlib import Path
from src.ingestion.markdown_cleaner import MarkdownCleaner
from src.core.config import PARSED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reclean_all():
    """Reads all .md and .json files in data/parsed and applies MarkdownCleaner."""
    if not PARSED_DIR.exists():
        logger.error(f"Directory {PARSED_DIR} does not exist.")
        return

    md_files = list(PARSED_DIR.glob("*.md"))
    json_files = list(PARSED_DIR.glob("*.json"))

    logger.info(f"Found {len(md_files)} markdown and {len(json_files)} json files.")

    for md_file in md_files:
        logger.info(f"Cleaning {md_file.name}...")
        try:
            content = md_file.read_text(encoding="utf-8")
            cleaned = MarkdownCleaner.clean_text(content)
            md_file.write_text(cleaned, encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to clean {md_file.name}: {e}")

    for json_file in json_files:
        logger.info(f"Cleaning {json_file.name}...")
        try:
            with open(json_file, 'r', encoding="utf-8") as f:
                data = json.load(f)
            cleaned_data = MarkdownCleaner.clean_json(data)
            with open(json_file, 'w', encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to clean {json_file.name}: {e}")

    print("\n✅ All files cleaned. You may need to re-run ingestion to update the database.")
    print("Run: python main.py ingest --pdf data/raw/<filename>.pdf to fully re-index.")

if __name__ == "__main__":
    reclean_all()
