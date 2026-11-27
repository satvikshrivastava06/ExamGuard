"""CLI tool to compare face detection quality across sample conditions.

Evaluates and compares face detection quality, accuracy, silent miss rates,
safe unavailable flags, confidence, and composite quality scores across varied
sample conditions (lighting, pose angles, occlusion, distance, blur, etc.).

Usage:
------
    # Compare conditions for default Haar detector against 'control' baseline:
    python compare_conditions.py

    # Specify custom directory and output path:
    python compare_conditions.py --dir sample_images/robustness --baseline control --json output/comparison.json

    # Export formatted Markdown and CSV tables:
    python compare_conditions.py --markdown output/comparison.md --csv output/comparison.csv

    # Head-to-head multi-detector comparison (Haar vs YuNet):
    python compare_conditions.py --detectors haar yunet --markdown output/multi_detector.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from exam_monitor.condition_comparison import (
    ConditionComparisonReport,
    compare_conditions_across_detectors,
    compare_image_results,
)
from exam_monitor.config import Config
from exam_monitor.detector import build_detector
from evaluate_images import evaluate_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="sample_images/robustness", help="Root directory containing categorized test images")
    ap.add_argument("--detector", default="haar", choices=["haar", "yunet"], help="Detector model to evaluate (default: haar)")
    ap.add_argument("--detectors", nargs="+", help="Multiple detectors to compare side-by-side (e.g. --detectors haar yunet)")
    ap.add_argument("--baseline", default="control", help="Baseline condition name for relative quality drop comparison (default: control)")
    ap.add_argument("--json", dest="json_out", help="Export structured comparison report to JSON file")
    ap.add_argument("--markdown", dest="md_out", help="Export formatted Markdown comparison report to file")
    ap.add_argument("--csv", dest="csv_out", help="Export tabular CSV report to file")
    ap.add_argument("--process-width", type=int, default=Config.process_width, help="Pre-processing resize width")
    ap.add_argument("--silent-miss-threshold", type=float, default=0.20, help="Vulnerability silent miss threshold (default: 0.20)")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"Error: directory not found: {root}. Generate assets first: python make_test_assets.py", file=sys.stderr)
        return 1

    cfg = Config(process_width=args.process_width)

    # Multi-detector comparison
    if args.detectors and len(args.detectors) > 1:
        print(f"Comparing detectors {args.detectors} across sample conditions in {root}...\n")
        multi_rep = compare_conditions_across_detectors(root, args.detectors, cfg)
        md_text = multi_rep.to_markdown()
        print(md_text)

        if args.md_out:
            multi_rep.export_markdown(args.md_out)
            print(f"\n[export] Markdown report saved to: {args.md_out}")
        if args.json_out:
            multi_rep.export_json(args.json_out)
            print(f"[export] JSON report saved to: {args.json_out}")
        return 0

    # Single detector condition comparison
    print(f"Evaluating detector '{args.detector}' across sample conditions in {root}...\n")
    try:
        det = build_detector(args.detector, cfg)
    except Exception as exc:
        print(f"Failed to build detector '{args.detector}': {exc}", file=sys.stderr)
        return 1

    results = evaluate_dir(root, det, cfg)
    if not results:
        print(f"No test images found under {root}.", file=sys.stderr)
        return 1

    report = compare_image_results(results, detector_name=args.detector, baseline_condition=args.baseline)

    # Print summary to console
    print(report.to_markdown())

    # Exports
    if args.json_out:
        report.export_json(args.json_out)
        print(f"\n[export] JSON report saved to: {args.json_out}")

    if args.md_out:
        report.export_markdown(args.md_out)
        print(f"[export] Markdown report saved to: {args.md_out}")

    if args.csv_out:
        report.export_csv(args.csv_out)
        print(f"[export] CSV report saved to: {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
