"""
main.py — Entry Point: CLI for the A-RAG Pipeline

Provides two subcommands:
    ingest  --pdf <path>           Run Layer 0 + Layer 1 on a single PDF.
    ask     --query "<question>"   Run the A-RAG agent (via Groq) and print the answer.

On every startup, init_db() is called to ensure the schema exists.

Usage:
    # Ingest a PDF
    python main.py ingest --pdf data/raw/my_document.pdf

    # Ask a question
    python main.py ask --query "What is the UPS bypass procedure?"

    # Ingest without rebuilding embeddings (useful for batch + single embed step)
    python main.py ingest --pdf data/raw/my_doc.pdf --skip-embeddings

    # Set log verbosity
    python main.py --log-level DEBUG ask --query "explain safety risks"
"""

import argparse
import logging
import sys

from src.core.config import RAW_DIR

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> int:
    """Handler for the 'ingest' subcommand.

    Wires in the new Docling-based pipeline (ingestion/pipeline.py).

    Returns:
        Exit code (0 = success, non-zero = failure).
    """
    from pathlib import Path
    from src.core.config import PARSED_DIR
    from ingestion.pipeline import run_ingestion

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ PDF file not found at path: {args.pdf}", file=sys.stderr)
        return 1

    print(f"⏳ Starting ingestion for {pdf_path.name}...")
    try:
        summary = run_ingestion(str(pdf_path), str(PARSED_DIR))
        
        if summary.get("status") == "skipped_duplicate":
            print(f"⚠️ Skipped: Document '{pdf_path.name}' has already been ingested.")
            return 0

        errors = summary.get("errors", [])
        if errors and not summary.get("total_chunks"):
            print(f"❌ Ingestion failed with errors: {errors}", file=sys.stderr)
            return 1

        print("─" * 60)
        print("✅ Ingestion Complete!")
        print(f"   Source file:       {summary.get('source_file')}")
        print(f"   Total Pages:       {summary.get('total_pages')}")
        print(f"   Batches Processed: {summary.get('batches_processed')}")
        print(f"   Total Sections:    {summary.get('total_sections')}")
        print(f"   Total Chunks:      {summary.get('total_chunks')}")
        print(f"   Chunks w/ Tables:  {summary.get('chunks_with_tables')}")
        print(f"   Hindi Paras Del:   {summary.get('hindi_paragraphs_removed')}")
        if errors:
            print(f"   Warnings/Errors:   {len(errors)} occurred (check logs)")
        print("─" * 60)
        return 0
    except Exception as exc:
        print(f"❌ Ingestion failed with exception: {exc}", file=sys.stderr)
        logger.exception("Ingestion failed")
        return 1


def cmd_ask(args: argparse.Namespace) -> int:
    """Handler for the 'ask' subcommand.

    Runs the full A-RAG agent loop and prints the final answer to stdout.

    Returns:
        Exit code (0 = success, non-zero = failure).
    """
    from src.core.database import init_db
    from src.agents.rag_agent import build_llm, run_agent

    init_db()

    print(f"\n🔍 Query: {args.query}\n")
    print("⏳ Running agent ... (this may take 30–120 seconds on CPU)\n")

    try:
        llm = build_llm()
        answer = run_agent(
            query          = args.query,
            max_iterations = args.max_iterations,
            llm_with_tools = llm,
        )
        print("─" * 60)
        print("🤖 Answer:\n")
        print(answer)
        print("─" * 60)
        return 0

    except FileNotFoundError as exc:
        print(
            f"\n❌ FAISS index not found.\n"
            f"   Please ingest at least one PDF first:\n"
            f"   python main.py ingest --pdf data/raw/<your_file>.pdf\n\n"
            f"   Details: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        if "api_key" in str(exc).lower():
            print(
                "\n❌ Groq API Key Error.\n"
                "   Make sure GROQ_API_KEY is set in your .env file.\n",
                file=sys.stderr,
            )
        else:
            logger.exception("Unexpected error during agent run: %s", exc)
            print(f"❌ Unexpected error: {exc}", file=sys.stderr)
        return 3


# ─────────────────────────────────────────────────────────────────────────────
# CLI definition
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="A-RAG Pipeline — Agentic Retrieval-Augmented Generation (offline indexing, Groq Cloud LLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py ingest --pdf data/raw/document.pdf\n"
            "  python main.py ask --query \"What is the UPS bypass procedure?\"\n"
            "  python main.py --log-level DEBUG ask --query \"safety risks\"\n"
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO). Output goes to stderr.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── ingest subcommand ─────────────────────────────────────────────────────
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest a PDF: parse → chunk → embed → store.",
    )
    ingest_parser.add_argument(
        "--pdf",
        required=True,
        metavar="PATH",
        help=f"Path to the PDF file (e.g. {RAW_DIR / 'my_doc.pdf'}).",
    )
    ingest_parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        default=False,
        help="Skip FAISS embedding step (useful for batch ingestion).",
    )

    # ── ask subcommand ────────────────────────────────────────────────────────
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question; the agent searches documents and returns an answer.",
    )
    ask_parser.add_argument(
        "--query",
        required=True,
        metavar="QUESTION",
        help='Natural language question (e.g. "What are the safety risks?").',
    )
    ask_parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        metavar="N",
        help="Maximum number of agent iterations before forcing an answer (default: 10).",
    )

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    _configure_logging(args.log_level)

    handler = {"ingest": cmd_ingest, "ask": cmd_ask}.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    main()
