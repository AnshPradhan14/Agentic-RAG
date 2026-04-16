"""
setup_once.py — One-time setup script for the A-RAG Pipeline

Run this ONCE after installing requirements.txt:
    python scripts/setup_once.py

What it does:
    1. Downloads NLTK 'punkt' and 'punkt_tab' tokenizer data.
    2. Verifies that Ollama is reachable.
    3. Creates the data/ and index/ directory structure.
    4. Initialises the SQLite database schema.

After this script succeeds, the pipeline is ready for:
    python run.py ingest --pdf data/raw/<your_file>.pdf
    python run.py ask    --query "Your question here"
"""

import subprocess
import sys


def step(msg: str) -> None:
    print(f"\n{'─'*60}\n🔧 {msg}\n{'─'*60}")


def success(msg: str) -> None:
    print(f"   ✅ {msg}")


def warn(msg: str) -> None:
    print(f"   ⚠️  {msg}")


def main() -> None:
    print("\n🚀 A-RAG Pipeline — One-Time Setup")
    print("=" * 60)

    # ── Step 1: NLTK data ─────────────────────────────────────────────────────
    step("Downloading NLTK tokenizer data (punkt, punkt_tab) ...")
    import nltk
    nltk.download("punkt",     quiet=False)
    nltk.download("punkt_tab", quiet=False)
    success("NLTK data downloaded.")

    # ── Step 2: Check Groq API Key ───────────────────────────────────────────
    step("Checking for Groq API configuration ...")
    import os
    from dotenv import load_dotenv
    load_dotenv()
    if os.environ.get("GROQ_API_KEY"):
        success("Groq API key found in .env file.")
    else:
        warn(
            "GROQ_API_KEY NOT found. Please add it to your .env file.\n"
            "   Get a key here: https://console.groq.com/keys"
        )

    # ── Step 3: Directory structure ───────────────────────────────────────────
    step("Creating project directory structure ...")
    from src.core.config import INDEX_DIR, PARSED_DIR, RAW_DIR
    for d in [RAW_DIR, PARSED_DIR, INDEX_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        success(f"Directory ready: {d}")

    # ── Step 4: SQLite schema ─────────────────────────────────────────────────
    step("Initialising SQLite database schema ...")
    from src.core.database import init_db
    init_db()
    from src.core.config import DB_PATH
    success(f"Database schema created at: {DB_PATH}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🎉 Setup complete!  Next steps:")
    print()
    print("  1. Copy your PDF(s) into:  data/raw/")
    print("  2. Ingest a PDF:")
    print("       python run.py ingest --pdf data/raw/<your_file>.pdf")
    print("  3. Ask a question (using Groq Cloud):")
    print('       python run.py ask --query "Your question here"')
    print()


if __name__ == "__main__":
    main()
