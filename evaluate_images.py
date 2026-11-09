"""Score face detection on the labelled robustness images (lighting, angle, occlusion, ...).

For every image we ask: what would the monitor CONCLUDE? Not just "did the detector fire".
  detected          the face was found                                  good
  silent_miss       a face is there, the monitor says FACE_ABSENT       BAD: a false absence nobody is warned about
  flagged           the monitor says DETECTION_UNAVAILABLE (too dark)   acceptable: it admits it cannot judge
  false_positive    nothing is there, but a face was reported           BAD
  correct_negative  nothing is there and none was reported             good

Expected face count comes from the file name suffix `_<N>face`.
Usage:  python evaluate_images.py [--dir sample_images/robustness] [--out output]
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from exam_monitor.camera import IMAGE_EXTS
from exam_monitor.config import Config
from exam_monitor.detector import FaceDetector, build_detector
from exam_monitor.events import EventType
from exam_monitor.pipeline import analyze_frame
from exam_monitor.preprocess import resize_to_width, to_gray


@dataclass(frozen=True)
class ImageResult:
    category: str
    name: str
    expected: int
    faces: int
    observation: str
    max_confidence: float
    luma: float
    outcome: str


def classify(expected: int, faces: int, obs: EventType) -> str:
    if obs == EventType.DETECTION_UNAVAILABLE:
        return "flagged"
    if expected == 0:
        return "false_positive" if faces > 0 else "correct_negative"
    return "detected" if faces >= 1 else "silent_miss"


def evaluate_image(path: Path, root: Path, detector: FaceDetector, cfg: Config) -> ImageResult:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    m = re.search(r"_(\d+)face$", path.stem)
    expected = int(m.group(1)) if m else 1
    obs, dets, _ = analyze_frame(img, detector, cfg)
    small, _ = resize_to_width(img, 64)
    rel = path.relative_to(root)
    return ImageResult(
        category=rel.parts[0] if len(rel.parts) > 1 else "(root)",
        name=re.sub(r"_\d+face$", "", path.stem), expected=expected, faces=len(dets),
        observation=obs.value, max_confidence=round(max((d.confidence for d in dets), default=0.0), 2),
        luma=round(float(to_gray(small).mean()), 1), outcome=classify(expected, len(dets), obs))


def evaluate_dir(root: Path, detector: FaceDetector, cfg: Config) -> list[ImageResult]:
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return [evaluate_image(p, root, detector, cfg) for p in paths]


def summarise(results: list[ImageResult]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        out[r.category][r.outcome] += 1
        out[r.category]["total"] += 1
    return out


def markdown(results: list[ImageResult], detector_name: str) -> str:
    lines = [f"# Robustness report ({detector_name} detector, synthetic variants)", "",
             "| category | images | detected | silent miss | flagged unavailable | false positive | correct negative |",
             "|---|---|---|---|---|---|---|"]
    for cat, c in sorted(summarise(results).items()):
        lines.append(f"| {cat} | {c['total']} | {c['detected']} | {c['silent_miss']} | {c['flagged']} | "
                     f"{c['false_positive']} | {c['correct_negative']} |")
    lines += ["", "| category | image | expected | faces | monitor says | mean luma | outcome |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r.category} | {r.name} | {r.expected} | {r.faces} | {r.observation} | {r.luma} | {r.outcome} |")
    lines += ["", "**silent miss** is the dangerous outcome: a present person reported as absent with no warning.",
              "**flagged** is a safe failure: the monitor says it cannot judge. All images are synthetic."]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="sample_images/robustness")
    ap.add_argument("--out", default="output")
    ap.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    ap.add_argument("--process-width", type=int, default=Config.process_width,
                    help="width the frame is shrunk to before detection (bigger = sees smaller faces, slower)")
    a = ap.parse_args()

    cfg = Config(process_width=a.process_width)
    detector = build_detector(a.detector, cfg)
    root = Path(a.dir)
    results = evaluate_dir(root, detector, cfg)
    if not results:
        raise SystemExit(f"no images under {root}. Run: python make_test_assets.py")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "robustness_images.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(ImageResult.__dataclass_fields__.keys())
        w.writerows([[getattr(r, k) for k in ImageResult.__dataclass_fields__] for r in results])
    (out / "robustness_report.md").write_text(markdown(results, a.detector))

    print(f"{'category':12s} {'image':20s} {'exp':>3s} {'got':>3s} {'luma':>6s}  {'monitor says':22s} outcome")
    for r in results:
        print(f"{r.category:12s} {r.name:20s} {r.expected:>3d} {r.faces:>3d} {r.luma:6.1f}  {r.observation:22s} {r.outcome}")
    print("\nper category:")
    for cat, c in sorted(summarise(results).items()):
        print(f"  {cat:12s} {c['total']:>2d} images | detected {c['detected']:>2d} | silent_miss {c['silent_miss']:>2d} | "
              f"flagged {c['flagged']:>2d} | false_pos {c['false_positive']:>2d} | correct_neg {c['correct_negative']:>2d}")
    print(f"\nreport: {out / 'robustness_report.md'}   csv: {out / 'robustness_images.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
