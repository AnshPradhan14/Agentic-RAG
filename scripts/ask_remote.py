"""
ask_remote.py — Lightweight CLI client for the A-RAG FastAPI Server

Usage (after starting `python run.py server` in another terminal):
    python scripts/ask_remote.py "What is the contract number?"
    python scripts/ask_remote.py "Who is the seller company?"
    python scripts/ask_remote.py "What operating systems are supported?"
"""

import sys
import time
import urllib.request
import json


SERVER_URL = "http://127.0.0.1:8000"


def check_server() -> bool:
    """Returns True if the server is running and healthy."""
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def ask(query: str, max_iterations: int = 10) -> dict:
    """Send a query to the server and return the response dict."""
    payload = json.dumps({"query": query, "max_iterations": max_iterations}).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"Your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    # ── Check server is running ────────────────────────────────────────────────
    if not check_server():
        print("\n❌ Server is not running!")
        print("   Start it first in a separate terminal:")
        print("   python server.py\n")
        sys.exit(1)

    # ── Send query ─────────────────────────────────────────────────────────────
    print(f"\n🔍 Query: {query}\n")
    print("⏳ Asking A-RAG agent...\n")

    t0 = time.time()
    try:
        result = ask(query)
        elapsed = round(time.time() - t0, 2)
        print("─" * 60)
        print("🤖 Answer:\n")
        print(result["answer"])
        print("─" * 60)
        print(f"⏱  Answered in {elapsed}s  (server processing: {result['time_seconds']}s)")

    except Exception as exc:
        print(f"❌ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
