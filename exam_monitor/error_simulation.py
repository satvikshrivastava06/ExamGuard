"""Simulated detection error injection and score-input testing utilities.

Provides deterministic fault-injection wrappers and simulation runners so the
monitoring pipeline, ``ConditionQualityMetrics.quality_score``,
``AccuracyReport``, and ``UncertaintyAssessment`` can be tested against
controlled detection errors:

  * ``FALSE_NEGATIVE``: drops real detections (silent miss)
  * ``FALSE_POSITIVE``: injects spurious bounding boxes
  * ``DETECTOR_EXCEPTION``: raises a runtime error inside ``detect()``
  * ``BLIND_FRAME``: replaces frames with all-black frames -> ``DETECTION_UNAVAILABLE``
  * ``CONFIDENCE_NOISE``: degrades/perturbs detection confidence scores
  * ``CAMERA_DROPOUT``: simulates missing webcam frames -> ``CAMERA_UNAVAILABLE``
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .condition_comparison import ConditionQualityMetrics
from .config import Config
from .detector import Detection, FaceDetector
from .events import EventType, IntegrityStateMachine, ListSink
from .pipeline import analyze_frame
from .uncertainty import UncertaintyAssessment


class SimulatedErrorType(str, Enum):
    FALSE_NEGATIVE = "FALSE_NEGATIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    DETECTOR_EXCEPTION = "DETECTOR_EXCEPTION"
    BLIND_FRAME = "BLIND_FRAME"
    CONFIDENCE_NOISE = "CONFIDENCE_NOISE"
    CAMERA_DROPOUT = "CAMERA_DROPOUT"


@dataclass(frozen=True)
class ErrorInjectionConfig:
    """Probabilities and deterministic frame-index overrides for error injection."""
    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0
    exception_rate: float = 0.0
    blind_frame_rate: float = 0.0
    camera_dropout_rate: float = 0.0
    confidence_noise_std: float = 0.0
    seed: int = 42
    # Optional explicit frame indices to force specific errors deterministically
    force_errors_at: dict[int, SimulatedErrorType] = field(default_factory=dict)


class StaticTruthDetector:
    """Deterministic detector returning a fixed detection per expected face."""

    name = "static_truth"

    def __init__(self, expected_faces: int = 1, base_confidence: float = 0.92):
        self.expected_faces = expected_faces
        self.base_confidence = base_confidence
        self.last_timing: dict = {"preprocess_ms": 0.1, "inference_ms": 0.2}

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        return [
            Detection((40 + i * 80, 40, 60, 60), self.base_confidence)
            for i in range(self.expected_faces)
        ]


class SimulatedErrorDetector:
    """Wraps a :class:`FaceDetector` and injects simulated detection errors."""

    def __init__(self, base: FaceDetector, err_cfg: ErrorInjectionConfig):
        self._base = base
        self._err_cfg = err_cfg
        self._rng = random.Random(err_cfg.seed)
        self._call_idx = 0
        self.name = f"sim({base.name})"
        self.last_timing: dict = {}
        self.injected_errors: list[tuple[int, SimulatedErrorType]] = []

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        idx = self._call_idx
        self._call_idx += 1

        forced = self._err_cfg.force_errors_at.get(idx)
        if forced == SimulatedErrorType.DETECTOR_EXCEPTION or (
            forced is None and self._rng.random() < self._err_cfg.exception_rate
        ):
            self.injected_errors.append((idx, SimulatedErrorType.DETECTOR_EXCEPTION))
            raise RuntimeError(f"Simulated detector failure at frame {idx}")

        dets = list(self._base.detect(bgr))
        self.last_timing = dict(getattr(self._base, "last_timing", {}))

        if forced == SimulatedErrorType.FALSE_NEGATIVE or (
            forced is None and dets and self._rng.random() < self._err_cfg.false_negative_rate
        ):
            self.injected_errors.append((idx, SimulatedErrorType.FALSE_NEGATIVE))
            return []

        if forced == SimulatedErrorType.FALSE_POSITIVE or (
            forced is None and self._rng.random() < self._err_cfg.false_positive_rate
        ):
            self.injected_errors.append((idx, SimulatedErrorType.FALSE_POSITIVE))
            dets.append(Detection((10, 10, 45, 45), 0.35))

        if self._err_cfg.confidence_noise_std > 0.0 and dets:
            noisy: list[Detection] = []
            for d in dets:
                delta = self._rng.gauss(0.0, self._err_cfg.confidence_noise_std)
                noisy_conf = max(0.01, min(1.0, d.confidence + delta))
                noisy.append(Detection(d.box, noisy_conf))
            self.injected_errors.append((idx, SimulatedErrorType.CONFIDENCE_NOISE))
            dets = noisy

        return dets


@dataclass
class SimulationResult:
    """Combined output of running score inputs through simulated detection errors."""
    metrics: ConditionQualityMetrics
    report: "AccuracyReport"
    uncertainty: UncertaintyAssessment
    injected_errors: list[tuple[int, SimulatedErrorType]]


def simulate_score_inputs(
    expected_faces_per_step: Sequence[int],
    err_cfg: ErrorInjectionConfig,
    *,
    step_s: float = 0.5,
    cfg: Optional[Config] = None,
    session_id: str = "SIM-ERROR-SESSION",
    condition_name: str = "simulated_errors",
) -> SimulationResult:
    """Run synthetic frames through ``analyze_frame``, ``IntegrityStateMachine``,
    ``ConditionQualityMetrics``, ``AccuracyReport``, and ``UncertaintyAssessment``.
    """
    from .accuracy_report import compute_accuracy_from_events

    monitor_cfg = cfg or Config()
    rng = random.Random(err_cfg.seed + 997)
    sink = ListSink()
    sm = IntegrityStateMachine(monitor_cfg, session_id, sink)
    sm.start(0.0)

    metrics = ConditionQualityMetrics(name=condition_name, category="simulation")
    good_frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    dark_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    injected: list[tuple[int, SimulatedErrorType]] = []

    for idx, expected_n in enumerate(expected_faces_per_step):
        now = (idx + 1) * step_s
        forced = err_cfg.force_errors_at.get(idx)

        if forced == SimulatedErrorType.CAMERA_DROPOUT or (
            forced is None and rng.random() < err_cfg.camera_dropout_rate
        ):
            injected.append((idx, SimulatedErrorType.CAMERA_DROPOUT))
            sm.observe(
                EventType.CAMERA_UNAVAILABLE,
                now,
                frame_id=idx,
                details={"problem": "UNAVAILABLE", "why": "simulated_dropout"},
            )
            metrics.total += 1
            metrics.expected_faces += expected_n
            metrics.flagged += 1
            continue

        frame = good_frame
        if forced == SimulatedErrorType.BLIND_FRAME or (
            forced is None and rng.random() < err_cfg.blind_frame_rate
        ):
            injected.append((idx, SimulatedErrorType.BLIND_FRAME))
            frame = dark_frame

        base_det = StaticTruthDetector(expected_faces=expected_n)
        # Apply per-step error config without advancing RNG unevenly across steps
        step_err = ErrorInjectionConfig(
            false_negative_rate=err_cfg.false_negative_rate,
            false_positive_rate=err_cfg.false_positive_rate,
            exception_rate=err_cfg.exception_rate,
            confidence_noise_std=err_cfg.confidence_noise_std,
            seed=err_cfg.seed + idx,
            force_errors_at={0: forced} if forced is not None else {},
        )
        sim_det = SimulatedErrorDetector(base_det, step_err)
        obs, dets, details = analyze_frame(frame, sim_det, monitor_cfg)
        for _, err_type in sim_det.injected_errors:
            injected.append((idx, err_type))

        sm.observe(obs, now, frame_id=idx, details=details)

        # Score the frame outcome against ground truth
        metrics.total += 1
        metrics.expected_faces += expected_n
        if obs == EventType.DETECTION_UNAVAILABLE:
            metrics.flagged += 1
        elif expected_n == 0:
            if len(dets) == 0:
                metrics.correct_negative += 1
            else:
                metrics.false_positive += 1
        else:
            if len(dets) == expected_n:
                metrics.detected += 1
                metrics.confidence_samples.append(max(d.confidence for d in dets))
            elif len(dets) == 0:
                metrics.silent_miss += 1
            else:
                metrics.false_positive += 1

    end_t = (len(expected_faces_per_step) + 1) * step_s
    sm.end(end_t)

    event_dicts = [
        {
            "type": ev.type.value,
            "session_id": ev.session_id,
            "ts_utc": ev.ts_utc,
            "ts_mono": ev.ts_mono,
            "frame_id": ev.frame_id,
            "details": ev.details,
        }
        for ev in sink.events
    ]
    report = compute_accuracy_from_events(session_id, event_dicts)
    uncertainty = report.uncertainty()
    return SimulationResult(
        metrics=metrics,
        report=report,
        uncertainty=uncertainty,
        injected_errors=injected,
    )
