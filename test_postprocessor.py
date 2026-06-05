"""
test_postprocessor.py -- Validate MarkdownPostProcessor against real parsed files.

Run from project root:
    python test_postprocessor.py

Shows a before/after diff for each parsed .md file.
No ingestion or DB changes — read-only.
"""

import sys
import difflib
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path

# Make sure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.markdown_postprocessor import MarkdownPostProcessor

PARSED_DIR = Path("data/parsed")

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RED   = "\033[91m"
GREEN = "\033[92m"
CYAN  = "\033[96m"
RESET = "\033[0m"
BOLD  = "\033[1m"


def show_diff(before: str, after: str, filename: str, context: int = 3) -> None:
    before_lines = before.splitlines(keepends=True)
    after_lines  = after.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile=f"BEFORE {filename}",
        tofile=f"AFTER  {filename}",
        n=context,
    ))

    if not diff:
        print(f"  {CYAN}(no changes){RESET}")
        return

    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            print(f"{BOLD}{line}{RESET}", end="")
        elif line.startswith("+"):
            print(f"{GREEN}{line}{RESET}", end="")
        elif line.startswith("-"):
            print(f"{RED}{line}{RESET}", end="")
        else:
            print(line, end="")


def run_spot_checks(before: str, after: str) -> list[str]:
    """Return a list of specific fix confirmations."""
    checks = []

    # Check 1: Image placeholders removed
    if "<!-- image -->" in before and "<!-- image -->" not in after:
        checks.append("✅ <!-- image --> placeholders removed")
    elif "<!-- image -->" in before:
        checks.append("❌ <!-- image --> placeholders still present")

    # Check 2: Header artifacts cleaned
    bad_headers = ["## fg", "## w \"|", "## D\" \"|", "## bc"]
    for bh in bad_headers:
        if bh in before and bh not in after:
            checks.append(f"✅ Header artifact cleaned: '{bh}'")

    # Check 3: Fused words fixed
    fused_examples = ["DepartmentofAtomic", "evaluatedbybuyer", "verifiedbybuyer",
                       "DesignationofFinancial", "DesignationofAdministrative",
                       "contractisguided", "governedbythe"]
    for fw in fused_examples:
        if fw.lower() in before.lower() and fw.lower() not in after.lower():
            checks.append(f"✅ Fused word fixed: '{fw}'")

    # Check 4: KV tables reconstructed
    if "| Field | Value |" in after:
        count = after.count("| Field | Value |")
        checks.append(f"✅ KV tables reconstructed: {count} table(s) created")

    # Check 5: Fewer consecutive blank lines
    if "\n\n\n" in before and "\n\n\n" not in after:
        checks.append("✅ Excess blank lines normalized")

    return checks


def main():
    md_files = sorted(PARSED_DIR.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {PARSED_DIR}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  MarkdownPostProcessor — Validation Run")
    print(f"  Testing on {len(md_files)} file(s) in {PARSED_DIR}/")
    print(f"{'='*70}\n")

    total_delta = 0
    for md_path in md_files:
        print(f"\n{BOLD}{'-'*70}{RESET}")
        print(f"{BOLD}FILE: {md_path.name}{RESET}")
        print(f"{'-'*70}")

        before = md_path.read_text(encoding="utf-8")
        after  = MarkdownPostProcessor.postprocess(before)

        delta = len(after) - len(before)
        total_delta += delta
        print(f"  Size: {len(before):,} chars → {len(after):,} chars  (delta={delta:+,})")

        # Spot checks
        checks = run_spot_checks(before, after)
        if checks:
            print("\n  Spot checks:")
            for c in checks:
                print(f"    {c}")

        # Show first 80 diff lines (context=2)
        print("\n  Diff (first 80 lines, context=2):")
        before_lines = before.splitlines(keepends=True)
        after_lines  = after.splitlines(keepends=True)
        diff = list(difflib.unified_diff(before_lines, after_lines,
                                          fromfile=f"BEFORE", tofile=f"AFTER",
                                          n=2))
        for line in diff[:80]:
            if line.startswith("+++") or line.startswith("---"):
                print(f"  {BOLD}{line}{RESET}", end="")
            elif line.startswith("+"):
                print(f"  {GREEN}{line}{RESET}", end="")
            elif line.startswith("-"):
                print(f"  {RED}{line}{RESET}", end="")
            else:
                print(f"  {line}", end="")

        if len(diff) > 80:
            print(f"\n  ... {len(diff) - 80} more diff lines (truncated)")

    print(f"\n{'='*70}")
    print(f"  Done. Total char delta across all files: {total_delta:+,}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
