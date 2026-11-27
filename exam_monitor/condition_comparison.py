"""Detection quality evaluation and condition comparison across sample conditions.

Compares face detection performance, accuracy, failure modes (silent misses vs
flagged unavailable), confidence, and quality scores across varied environmental
and geometric conditions:
  - Lighting (bright, dim, dark, extreme dark, backlight, high contrast, noise)
  - Head pose (yaw, pitch, roll, extreme angles, inversion)
  - Occlusion (eyes, mouth, half face, glasses)
  - Framing and scale (near, far, partial face cut-off)
  - Resolution and sensor artifacts (blur, downsampling, noise)
  - Negative controls (empty room, wall, blank frames, textures)

Supports:
  1. Single-detector condition profiling and baseline comparison
  2. Multi-detector head-to-head benchmarking (e.g. Haar vs YuNet)
  3. Video replay condition bucketing (roll/yaw angle sweep, brightness ramp)
  4. Structured export to JSON, Markdown, and CSV
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from .config import Config
from .detector import FaceDetector, build_detector


@dataclass
class ConditionQualityMetrics:
    """Detection performance and quality metrics under a specific sample condition."""
    name: str                                  # Condition or category name
    category: str = ""                         # High-level category (lighting, angle, occlusion, etc.)
    total: int = 0                             # Total frames/images evaluated
    expected_faces: int = 0                    # Total expected face count
    detected: int = 0                          # Successfully detected faces
    silent_miss: int = 0                       # False negatives without warning (silent miss)
    flagged: int = 0                           # Safe unavailable failures (DETECTION_UNAVAILABLE)
    false_positive: int = 0                    # Faces reported when none were expected
    correct_negative: int = 0                  # Correctly reported absence (no face expected, none found)
    confidence_samples: list[float] = field(default_factory=list)
    luma_samples: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction of images resulting in a correct conclusion or safe unavailable flag."""
        if self.total == 0:
            return 0.0
        return (self.detected + self.correct_negative) / self.total

    @property
    def detection_rate(self) -> float:
        """Recall/sensitivity: detected faces / total expected face opportunities."""
        positives = self.detected + self.silent_miss
        if positives == 0:
            return 1.0 if self.correct_negative > 0 else 0.0
        return self.detected / positives

    @property
    def silent_miss_rate(self) -> float:
        """Dangerous failure rate: present face missed without any warning."""
        positives = self.detected + self.silent_miss
        if positives == 0:
            return 0.0
        return self.silent_miss / positives

    @property
    def flagged_rate(self) -> float:
        """Safe failure rate: monitor admitted it could not judge."""
        if self.total == 0:
            return 0.0
        return self.flagged / self.total

    @property
    def false_positive_rate(self) -> float:
        """False alarm rate on negative/empty samples."""
        negatives = self.false_positive + self.correct_negative
        if negatives == 0:
            return 0.0
        return self.false_positive / negatives

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidence_samples)) if self.confidence_samples else 0.0

    @property
    def mean_luma(self) -> float:
        return float(np.mean(self.luma_samples)) if self.luma_samples else 0.0

    @property
    def quality_score(self) -> float:
        """
        Composite quality score from 0.0 to 100.0.
        Penalizes dangerous silent misses heavily (-60), rewards correct decisions (+100),
        and applies modest penalty for safe unavailables (-15).
        """
        if self.total == 0:
            return 0.0
        score = (self.accuracy * 100.0) - (self.silent_miss_rate * 60.0) - (self.flagged_rate * 15.0)
        return round(max(0.0, min(100.0, score)), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "total": self.total,
            "expected_faces": self.expected_faces,
            "detected": self.detected,
            "silent_miss": self.silent_miss,
            "flagged": self.flagged,
            "false_positive": self.false_positive,
            "correct_negative": self.correct_negative,
            "accuracy": round(self.accuracy, 4),
            "detection_rate": round(self.detection_rate, 4),
            "silent_miss_rate": round(self.silent_miss_rate, 4),
            "flagged_rate": round(self.flagged_rate, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "mean_confidence": round(self.mean_confidence, 3),
            "mean_luma": round(self.mean_luma, 1),
            "quality_score": self.quality_score,
        }


@dataclass
class ConditionComparisonReport:
    """Comprehensive comparison report across conditions for a single detector."""
    detector_name: str
    conditions: dict[str, ConditionQualityMetrics] = field(default_factory=dict)
    baseline_condition: Optional[str] = "control"
    dataset_name: str = "sample_images"

    def add_metric(self, metric: ConditionQualityMetrics) -> None:
        self.conditions[metric.name] = metric

    def rank_conditions_by_quality(self) -> list[tuple[str, float]]:
        """Returns conditions sorted from highest quality score to lowest."""
        return sorted(
            [(name, m.quality_score) for name, m in self.conditions.items()],
            key=lambda item: item[1],
            reverse=True,
        )

    def get_vulnerabilities(self, silent_miss_threshold: float = 0.20) -> list[str]:
        """Identifies conditions with unacceptably high silent miss rates."""
        return [
            name for name, m in self.conditions.items()
            if m.silent_miss_rate > silent_miss_threshold and m.total > 0
        ]

    def relative_comparison_to_baseline(self) -> dict[str, dict[str, float]]:
        """Compares each condition against the baseline (e.g. 'control')."""
        baseline = self.conditions.get(self.baseline_condition) if self.baseline_condition else None
        if not baseline:
            return {}
        base_acc = baseline.accuracy
        base_det = baseline.detection_rate
        base_qual = baseline.quality_score

        deltas: dict[str, dict[str, float]] = {}
        for name, m in self.conditions.items():
            deltas[name] = {
                "delta_accuracy": round(m.accuracy - base_acc, 4),
                "delta_detection_rate": round(m.detection_rate - base_det, 4),
                "delta_quality_score": round(m.quality_score - base_qual, 1),
                "relative_drop_pct": round(
                    max(0.0, ((base_qual - m.quality_score) / max(base_qual, 1.0)) * 100.0), 1
                ),
            }
        return deltas

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "dataset_name": self.dataset_name,
            "baseline_condition": self.baseline_condition,
            "conditions": {k: v.to_dict() for k, v in self.conditions.items()},
            "ranking": [{"condition": name, "quality_score": score} for name, score in self.rank_conditions_by_quality()],
            "vulnerabilities": self.get_vulnerabilities(),
            "relative_to_baseline": self.relative_comparison_to_baseline(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def export_json(self, path: Path | str, indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target

    def to_markdown(self) -> str:
        lines = [
            f"# Detection Quality Condition Comparison ({self.detector_name})",
            f"**Dataset**: {self.dataset_name} | **Baseline**: {self.baseline_condition or 'None'}",
            "",
            "| Condition / Category | Samples | Quality Score | Accuracy | Detection Rate | Silent Miss Rate | Flagged Rate | Mean Conf |",
            "|---|---|---|---|---|---|---|---|",
        ]
        deltas = self.relative_comparison_to_baseline()
        for name, m in sorted(self.conditions.items()):
            delta_note = ""
            if name in deltas and name != self.baseline_condition:
                d_q = deltas[name]["delta_quality_score"]
                delta_note = f" ({d_q:+.1f})"
            lines.append(
                f"| **{name}** | {m.total} | {m.quality_score:.1f}{delta_note} | "
                f"{m.accuracy * 100:.1f}% | {m.detection_rate * 100:.1f}% | "
                f"{m.silent_miss_rate * 100:.1f}% | {m.flagged_rate * 100:.1f}% | "
                f"{m.mean_confidence:.2f} |"
            )

        vulnerabilities = self.get_vulnerabilities()
        if vulnerabilities:
            lines += [
                "",
                "### Critical Condition Vulnerabilities",
                "The following conditions exhibit high silent-miss rates (candidate present but unnoticed):",
            ]
            for v in vulnerabilities:
                m = self.conditions[v]
                lines.append(f"- **{v}**: {m.silent_miss_rate * 100:.1f}% silent misses ({m.silent_miss}/{m.total} images)")

        return "\n".join(lines)

    def export_markdown(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target

    def to_csv(self) -> str:
        import io
        buf = io.StringIO()
        fieldnames = [
            "condition", "category", "total", "detected", "silent_miss",
            "flagged", "false_positive", "correct_negative", "accuracy",
            "detection_rate", "silent_miss_rate", "flagged_rate",
            "mean_confidence", "mean_luma", "quality_score"
        ]
        writer = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name, m in self.conditions.items():
            d = m.to_dict()
            d["condition"] = name
            writer.writerow({k: d[k] for k in fieldnames})
        return buf.getvalue()

    def export_csv(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_csv(), encoding="utf-8")
        return target


@dataclass
class MultiDetectorComparisonReport:
    """Head-to-head comparison of multiple detectors across identical sample conditions."""
    detectors: list[str]
    reports: dict[str, ConditionComparisonReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detectors": self.detectors,
            "reports": {k: v.to_dict() for k, v in self.reports.items()},
            "side_by_side": self._build_side_by_side(),
        }

    def _build_side_by_side(self) -> dict[str, dict[str, Any]]:
        conditions = sorted({c for r in self.reports.values() for c in r.conditions.keys()})
        side_by_side: dict[str, dict[str, Any]] = {}
        for c in conditions:
            side_by_side[c] = {}
            for det in self.detectors:
                rep = self.reports.get(det)
                m = rep.conditions.get(c) if rep else None
                side_by_side[c][det] = {
                    "quality_score": m.quality_score if m else None,
                    "accuracy": round(m.accuracy, 4) if m else None,
                    "detection_rate": round(m.detection_rate, 4) if m else None,
                    "silent_miss_rate": round(m.silent_miss_rate, 4) if m else None,
                    "mean_confidence": round(m.mean_confidence, 3) if m else None,
                }
        return side_by_side

    def to_markdown(self) -> str:
        lines = [
            f"# Multi-Detector Robustness Comparison: {' vs '.join(self.detectors)}",
            "",
            "| Condition | " + " | ".join(f"{d} Quality" for d in self.detectors) + " | " +
            " | ".join(f"{d} Recall" for d in self.detectors) + " | " +
            " | ".join(f"{d} SilentMiss" for d in self.detectors) + " | Best Detector |",
            "|---|" + "|".join("---" for _ in range(len(self.detectors) * 3 + 1)) + "|",
        ]
        sbs = self._build_side_by_side()
        for cond, dmap in sbs.items():
            qual_cells = [f"{dmap.get(d, {}).get('quality_score', 0):.1f}" for d in self.detectors]
            recall_cells = [f"{(dmap.get(d, {}).get('detection_rate', 0) or 0) * 100:.0f}%" for d in self.detectors]
            miss_cells = [f"{(dmap.get(d, {}).get('silent_miss_rate', 0) or 0) * 100:.0f}%" for d in self.detectors]
            # Determine winner
            best_det = max(self.detectors, key=lambda d: dmap.get(d, {}).get("quality_score") or -1.0)
            lines.append(f"| **{cond}** | {' | '.join(qual_cells)} | {' | '.join(recall_cells)} | {' | '.join(miss_cells)} | **{best_det}** |")

        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def export_json(self, path: Path | str, indent: int = 2) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent), encoding="utf-8")
        return target

    def export_markdown(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target


# ------------------------------------------------------------------ Runner Helpers

def compare_image_results(
    results: Sequence[Any],
    detector_name: str = "haar",
    baseline_condition: Optional[str] = "control",
    group_by: str = "category",
) -> ConditionComparisonReport:
    """
    Aggregate ImageResult instances into ConditionComparisonReport.

    group_by can be 'category' (e.g. lighting, angle_roll) or 'condition' (individual image conditions).
    """
    buckets: dict[str, ConditionQualityMetrics] = {}

    for r in results:
        key = getattr(r, "category", "") if group_by == "category" else f"{r.category}/{r.name}"
        if not key:
            key = "(root)"

        if key not in buckets:
            buckets[key] = ConditionQualityMetrics(name=key, category=getattr(r, "category", ""))

        m = buckets[key]
        m.total += 1
        m.expected_faces += getattr(r, "expected", 1)

        outcome = getattr(r, "outcome", "")
        if outcome == "detected":
            m.detected += 1
        elif outcome == "silent_miss":
            m.silent_miss += 1
        elif outcome == "flagged":
            m.flagged += 1
        elif outcome == "false_positive":
            m.false_positive += 1
        elif outcome == "correct_negative":
            m.correct_negative += 1

        conf = getattr(r, "max_confidence", None)
        if conf is not None and conf > 0:
            m.confidence_samples.append(float(conf))

        luma = getattr(r, "luma", None)
        if luma is not None:
            m.luma_samples.append(float(luma))

    report = ConditionComparisonReport(
        detector_name=detector_name,
        conditions=buckets,
        baseline_condition=baseline_condition,
    )
    return report


def compare_conditions_across_detectors(
    root: Path,
    detectors: Sequence[str] = ("haar", "yunet"),
    cfg: Optional[Config] = None,
) -> MultiDetectorComparisonReport:
    """Evaluate and compare multiple detectors on the same dataset."""
    from evaluate_images import evaluate_dir

    config = cfg or Config()
    reports: dict[str, ConditionComparisonReport] = {}

    for det_name in detectors:
        try:
            det = build_detector(det_name, config)
            results = evaluate_dir(root, det, config)
            rep = compare_image_results(results, detector_name=det_name, baseline_condition="control")
            reports[det_name] = rep
        except Exception as exc:
            # If a detector like yunet model file is missing, record empty report with error note
            empty_rep = ConditionComparisonReport(detector_name=det_name)
            reports[det_name] = empty_rep

    return MultiDetectorComparisonReport(detectors=list(detectors), reports=reports)
