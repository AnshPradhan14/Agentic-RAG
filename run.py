#!/usr/bin/env python3
"""
run.py — Root-level entry point for the A-RAG Pipeline.

Provides a convenient proxy to the internal CLI and server.

Usage:
    python run.py ingest --pdf data/raw/my_doc.pdf
    python run.py ask --query "What are the safety risks?"
    python run.py server
    python run.py setup
"""

import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    project_root = Path(__file__).parent.resolve()

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    # Force mirror and ignore SSL errors (common on restricted Windows networks)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["PYTHONHTTPSVERIFY"] = "0"
    env["CURL_CA_BUNDLE"] = ""
    env["HF_HUB_OFFLINE"] = "0"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    if cmd == "server":
        print("🚀 Starting FastAPI Server...")
        subprocess.run([sys.executable, str(project_root / "src" / "api" / "server.py")] + args, env=env)

    elif cmd == "setup":
        print("🔧 Running One-Time Setup...")
        subprocess.run([sys.executable, str(project_root / "scripts" / "setup_once.py")] + args, env=env)

    elif cmd in ["ingest", "ask"]:
        # Forward to src/main.py
        subprocess.run([sys.executable, str(project_root / "src" / "main.py"), cmd] + args, env=env)

    else:
        # Fallback: try forwarding directly to src/main.py
        subprocess.run([sys.executable, str(project_root / "src" / "main.py"), cmd] + args, env=env)

if __name__ == "__main__":
    main()
