"""Run the detector over still images and report counts, confidences, latency.

Expected face count is read from the filename suffix `_<N>face`, e.g. `desk_2face.png`.
Annotated copies go to output/annotated/. Exit code 1 if any count mismatches.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

from exam_monitor.camera import IMAGE_EXTS
from exam_monitor.config import Config
from exam_monitor.detector import build_detector
from exam_monitor.metrics import LatencyTracker
from exam_monitor.overlay import draw_detections
from exam_monitor.preprocess import describe

import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="sample_images")
    ap.add_argument("--out", default="output/annotated")
    ap.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    ap.add_argument("--repeat", type=int, default=20, help="timing repetitions per image")
    args = ap.parse_args()

    cfg = Config()
    detector = build_detector(args.detector, cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"No images in {args.images}")

    lat, failures = LatencyTracker(), 0
    print(f"{'image':28s} {'size':>9s} {'exp':>3s} {'got':>3s} {'max_conf':>8s} {'p50 ms':>7s}  result")
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"{p.name:28s} unreadable")
            failures += 1
            continue
        detector.detect(img)  # warm-up
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            dets = detector.detect(img)
            lat.add(p.name, (time.perf_counter() - t0) * 1e3)
        m = re.search(r"_(\d+)face", p.stem)
        expected = int(m.group(1)) if m else None
        verdict = "n/a" if expected is None else ("PASS" if expected == len(dets) else "FAIL")
        failures += verdict == "FAIL"
        d = describe(img)
        print(f"{p.name:28s} {d['width']}x{d['height']:<4d} {expected if expected is not None else '-':>3} "
              f"{len(dets):>3d} {max((x.confidence for x in dets), default=0):8.2f} "
              f"{lat.stats(p.name)['p50']:7.1f}  {verdict}")
        draw_detections(img, dets)
        cv2.imwrite(str(out_dir / p.name), img)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
