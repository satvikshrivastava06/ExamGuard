"""Standalone accuracy report and monitoring summary CLI.

Reads one or more JSONL event logs and prints accuracy summaries,
measures camera availability & detection coverage, identifies sessions with
insufficient monitoring data, and exports structured summaries (JSON / CSV).

Usage
-----
    # Single log human-readable report:
    python accuracy_report.py logs/events_EXAM-2026-S0001.jsonl

    # All logs in logs/ directory:
    python accuracy_report.py

    # Export structured monitoring summaries to JSON or CSV:
    python accuracy_report.py --json output/summaries.json
    python accuracy_report.py --csv output/summaries.csv

    # Identify sessions with insufficient monitoring data:
    python accuracy_report.py --check-sufficiency
    python accuracy_report.py --insufficient-only

    # Custom sufficiency thresholds:
    python accuracy_report.py --check-sufficiency --min-duration 60 --min-availability 0.85 --min-coverage 0.75
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exam_monitor.accuracy_report import (
    AccuracyReport,
    compute_accuracy,
    export_summaries_csv,
    export_summaries_json,
)
from exam_monitor.data_sufficiency import (
    DataSufficiencyCriteria,
    evaluate_session_sufficiency,
)
from exam_monitor.database import (
    SummaryStore,
    compute_accuracy_from_db,
    fetch_detection_confidences,
)
from exam_monitor.mysql_cli import add_mysql_args, connect_mysql_from_args


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="JSONL log file(s). Defaults to all files in logs/")
    ap.add_argument("--json", dest="json_out", help="Export structured monitoring summaries to this JSON file")
    ap.add_argument("--csv", dest="csv_out", help="Export structured monitoring summaries to this CSV file")
    ap.add_argument("--check-sufficiency", action="store_true", help="Audit and display data sufficiency evaluation for each session")
    ap.add_argument("--insufficient-only", action="store_true", help="Display only sessions flagged as having insufficient monitoring data")
    ap.add_argument("--min-duration", type=float, default=30.0, help="Minimum required session duration in seconds (default: 30.0)")
    ap.add_argument("--min-availability", type=float, default=0.80, help="Minimum camera availability ratio (default: 0.80)")
    ap.add_argument("--min-coverage", type=float, default=0.70, help="Minimum detection coverage ratio (default: 0.70)")
    ap.add_argument("--max-interruptions", type=int, default=3, help="Maximum allowed interruptions (default: 3)")
    ap.add_argument("--quiet", action="store_true", help="Do not print reports to stdout (useful when exporting to JSON/CSV)")
    add_mysql_args(ap)
    args = ap.parse_args()

    criteria = DataSufficiencyCriteria(
        min_duration_s=args.min_duration,
        min_camera_availability=args.min_availability,
        min_detection_coverage=args.min_coverage,
        max_interruptions=args.max_interruptions,
    )

    reports: list[AccuracyReport] = []
    ok = True

    if args.mysql:
        db_mgr = connect_mysql_from_args(args)
        conn = db_mgr.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM sessions ORDER BY start_time DESC")
            session_ids = [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()
        if not session_ids:
            print("No sessions found in MySQL database.", file=sys.stderr)
            return 1
        summary_store = SummaryStore(db_mgr)
        for sid in session_ids:
            try:
                report = compute_accuracy_from_db(sid, db_mgr)
                frame_confs = fetch_detection_confidences(sid, db_mgr)
                if len(frame_confs) > len(report.confidence_samples):
                    report.confidence_samples = frame_confs
                summary_store.save_summary(report, criteria=criteria)
                reports.append(report)
            except Exception as exc:
                print(f"[error] MySQL session {sid}: {exc}", file=sys.stderr)
                ok = False
    else:
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

        for p in paths:
            if not p.exists():
                print(f"[skip] {p} – file not found", file=sys.stderr)
                ok = False
                continue
            try:
                report = compute_accuracy(p)
                reports.append(report)
            except Exception as exc:
                print(f"[error] {p}: {exc}", file=sys.stderr)
                ok = False

    if not reports:
        print("No valid session reports could be computed.", file=sys.stderr)
        return 1

    # Print reports if not quiet
    if not args.quiet:
        for r in reports:
            suff = r.sufficiency(criteria)
            if args.insufficient_only and suff.is_sufficient:
                continue
            r.print(criteria)

    # If --insufficient-only was passed, provide quick summary
    if args.insufficient_only or args.check_sufficiency:
        insufficient = [r for r in reports if not r.is_sufficient(criteria)]
        print(f"\nSufficiency Audit Summary: {len(insufficient)}/{len(reports)} sessions have insufficient monitoring data.")
        for r in insufficient:
            eval_res = r.sufficiency(criteria)
            reasons_str = ", ".join(r_val.value for r_val in eval_res.reasons)
            print(f"  * {r.session_id}: {reasons_str}")

    # Export structured summaries if requested
    if args.json_out:
        export_summaries_json(reports, args.json_out, criteria=criteria)
        print(f"[export] Structured JSON summary saved to: {args.json_out}")

    if args.csv_out:
        export_summaries_csv(reports, args.csv_out, criteria=criteria)
        print(f"[export] Structured CSV summary saved to: {args.csv_out}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
