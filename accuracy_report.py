"""Standalone accuracy report CLI.

Reads one or more JSONL event logs and prints accuracy summaries.

Usage
-----
    # Single log:
    python accuracy_report.py logs/events_EXAM-2026-S0001.jsonl

    # All logs in the default logs/ directory:
    python accuracy_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exam_monitor.accuracy_report import compute_accuracy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="JSONL log file(s). Defaults to all files in logs/")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.logs:
        paths = [Path(p) for p in args.logs]
    else:
        log_dir = Path("logs")
        if not log_dir.exists():
            print("No logs/ directory found. Pass a log file path explicitly.", file=sys.stderr)
            return 1
        paths = sorted(log_dir.glob("events_*.jsonl"))
        if not paths:
            print("No event log files found in logs/", file=sys.stderr)
            return 1

    ok = True
    for p in paths:
        if not p.exists():
            print(f"[skip] {p} – file not found", file=sys.stderr)
            ok = False
            continue
        try:
            report = compute_accuracy(p)
            report.print()
        except Exception as exc:
            print(f"[error] {p}: {exc}", file=sys.stderr)
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
