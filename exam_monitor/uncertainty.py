"""Uncertainty indicators and misconduct-evaluation guards for exam monitoring.

Computes structured uncertainty metrics from an :class:`AccuracyReport` and its
:class:`SufficiencyAssessment` so downstream reviewers and automated scoring
pipelines can distinguish:

  * **Confirmed observations**: ``FACE_PRESENT``, ``FACE_ABSENT``, ``MULTIPLE_FACES``
    recorded while the camera was streaming and the detector was active.
  * **Uncertain / unobserved time**: ``DETECTION_UNAVAILABLE`` (dark/blown-out
    frames or detector exceptions), ``CAMERA_UNAVAILABLE``, and
    ``CAMERA_PERMISSION_DENIED``.

Invariant
---------
Missing webcam data (``CAMERA_UNAVAILABLE`` or ``CAMERA_PERMISSION_DENIED``) and
detector-blind intervals (``DETECTION_UNAVAILABLE``) are **never** treated as
``FACE_ABSENT`` or confirmed misconduct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from .data_sufficiency import DataSufficiencyCriteria, SufficiencyAssessment

if TYPE_CHECKING:
    from .accuracy_report import AccuracyReport


class UncertaintyLevel(str, Enum):
    """Overall uncertainty tier for a monitoring session."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class UncertaintyIndicator:
    """A single explainable uncertainty signal detected in a session."""
    code: str
    severity: UncertaintyLevel
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "value": round(self.value, 4) if isinstance(self.value, float) else self.value,
            "threshold": round(self.threshold, 4) if isinstance(self.threshold, float) else self.threshold,
        }


@dataclass(frozen=True)
class MisconductEvaluation:
    """Explicit separation between confirmed observations and missing/uncertain data."""
    confirmed_absence_s: float
    confirmed_multiple_faces_s: float
    unobserved_or_uncertain_s: float
    missing_webcam_s: float
    missing_webcam_treated_as_misconduct: bool
    reliable_for_misconduct_decision: bool
    verdict_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmed_absence_s": round(self.confirmed_absence_s, 2),
            "confirmed_multiple_faces_s": round(self.confirmed_multiple_faces_s, 2),
            "unobserved_or_uncertain_s": round(self.unobserved_or_uncertain_s, 2),
            "missing_webcam_s": round(self.missing_webcam_s, 2),
            "missing_webcam_treated_as_misconduct": self.missing_webcam_treated_as_misconduct,
            "reliable_for_misconduct_decision": self.reliable_for_misconduct_decision,
            "verdict_note": self.verdict_note,
        }


@dataclass
class UncertaintyAssessment:
    """Full uncertainty report for a single monitored session."""
    session_id: str
    uncertainty_level: UncertaintyLevel
    uncertainty_ratio: float
    uncertain_s: float
    confirmed_s: float
    indicators: list[UncertaintyIndicator] = field(default_factory=list)
    misconduct_evaluation: Optional[MisconductEvaluation] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "uncertainty_level": self.uncertainty_level.value,
            "uncertainty_ratio": round(self.uncertainty_ratio, 4),
            "uncertain_s": round(self.uncertain_s, 2),
            "confirmed_s": round(self.confirmed_s, 2),
            "indicator_count": len(self.indicators),
            "indicators": [ind.to_dict() for ind in self.indicators],
            "misconduct_evaluation": (
                self.misconduct_evaluation.to_dict()
                if self.misconduct_evaluation is not None
                else None
            ),
        }

    def summary_text(self) -> str:
        codes = ", ".join(ind.code for ind in self.indicators) or "NONE"
        return (
            f"Session {self.session_id}: uncertainty={self.uncertainty_level.value} "
            f"({self.uncertainty_ratio * 100:.1f}% uncertain), indicators=[{codes}]"
        )


def evaluate_uncertainty(
    report: "AccuracyReport",
    criteria: Optional[DataSufficiencyCriteria] = None,
    sufficiency: Optional[SufficiencyAssessment] = None,
) -> UncertaintyAssessment:
    """Compute uncertainty indicators and misconduct-guard metadata for *report*."""
    crit = criteria or DataSufficiencyCriteria()
    suff = sufficiency or report.sufficiency(crit)

    total_s = report.total_monitored_s
    missing_webcam_s = report.camera_outage_s
    uncertain_s = report.detection_unavailable_s + missing_webcam_s
    confirmed_s = report.detection_coverage_s

    if total_s <= 0:
        uncertainty_ratio = 1.0
    else:
        # Clamp to [0.0, 1.0] to guard against minor floating-point drift.
        uncertainty_ratio = min(1.0, max(0.0, uncertain_s / total_s))

    indicators: list[UncertaintyIndicator] = []

    if total_s <= 0:
        indicators.append(
            UncertaintyIndicator(
                code="NO_MONITORED_DURATION",
                severity=UncertaintyLevel.CRITICAL,
                message="Session has zero monitored duration; no reliable visual evidence exists.",
                value=0.0,
                threshold=crit.min_duration_s,
            )
        )

    if missing_webcam_s > 0 or (total_s > 0 and report.camera_availability_ratio == 0.0):
        severity = (
            UncertaintyLevel.CRITICAL
            if report.camera_availability_ratio == 0.0
            else (
                UncertaintyLevel.HIGH
                if report.camera_availability_ratio < crit.min_camera_availability
                else UncertaintyLevel.MODERATE
            )
        )
        indicators.append(
            UncertaintyIndicator(
                code="MISSING_WEBCAM_DATA",
                severity=severity,
                message=(
                    f"Webcam data missing for {missing_webcam_s:.1f}s "
                    f"(availability {report.camera_availability_ratio * 100:.1f}%); "
                    "missing webcam intervals are excluded from absence/misconduct."
                ),
                value=report.camera_availability_ratio,
                threshold=crit.min_camera_availability,
            )
        )

    if report.detection_unavailable_s > 0 or report.n_detection_failure > 0:
        severity = (
            UncertaintyLevel.HIGH
            if report.detection_unavailable_ratio > crit.max_detection_failure_ratio
            else UncertaintyLevel.MODERATE
        )
        indicators.append(
            UncertaintyIndicator(
                code="DETECTOR_BLIND_TIME",
                severity=severity,
                message=(
                    f"Detector could not judge for {report.detection_unavailable_s:.1f}s "
                    f"across {report.n_detection_failure} event(s) (e.g. dark/blown-out frame or error)."
                ),
                value=report.detection_unavailable_ratio,
                threshold=crit.max_detection_failure_ratio,
            )
        )

    if report.n_interruptions > 0:
        severity = (
            UncertaintyLevel.HIGH
            if report.n_interruptions > crit.max_interruptions
            else UncertaintyLevel.MODERATE
        )
        indicators.append(
            UncertaintyIndicator(
                code="SESSION_INTERRUPTIONS",
                severity=severity,
                message=(
                    f"{report.n_interruptions} monitoring loop interruption(s) recorded; "
                    "stall gaps are excluded from active presence and absence."
                ),
                value=float(report.n_interruptions),
                threshold=float(crit.max_interruptions),
            )
        )

    if not report.has_session_end:
        indicators.append(
            UncertaintyIndicator(
                code="INCOMPLETE_SESSION",
                severity=UncertaintyLevel.HIGH,
                message="Session log lacks a clean SESSION_ENDED event.",
            )
        )

    if report.face_present_s > 0 and not report.confidence_samples:
        indicators.append(
            UncertaintyIndicator(
                code="NO_CONFIDENCE_SAMPLES",
                severity=UncertaintyLevel.MODERATE,
                message="Face presence was recorded without detector confidence samples.",
            )
        )
    elif report.confidence_samples and report.avg_confidence is not None and report.avg_confidence < 0.5:
        # YuNet outputs 0..1 scores; Haar outputs cascade weights (typically >= 2.0).
        # Values < 0.5 indicate low-confidence detections under normalized scoring.
        indicators.append(
            UncertaintyIndicator(
                code="LOW_DETECTION_CONFIDENCE",
                severity=UncertaintyLevel.MODERATE,
                message=f"Mean detector confidence is low ({report.avg_confidence:.3f}).",
                value=report.avg_confidence,
                threshold=0.5,
            )
        )

    if not suff.is_sufficient:
        indicators.append(
            UncertaintyIndicator(
                code="INSUFFICIENT_MONITORING_DATA",
                severity=UncertaintyLevel.HIGH,
                message=(
                    "Session failed data sufficiency audit: "
                    + ", ".join(r.value for r in suff.reasons)
                ),
            )
        )

    # Determine overall uncertainty tier
    if (
        total_s <= 0
        or uncertainty_ratio > 0.50
        or report.camera_availability_ratio == 0.0
    ):
        level = UncertaintyLevel.CRITICAL
    elif uncertainty_ratio > 0.20 or not suff.is_sufficient:
        level = UncertaintyLevel.HIGH
    elif uncertainty_ratio > 0.05 or len(indicators) > 0:
        level = UncertaintyLevel.MODERATE
    else:
        level = UncertaintyLevel.LOW

    reliable_for_decision = (
        suff.is_sufficient
        and level in (UncertaintyLevel.LOW, UncertaintyLevel.MODERATE)
        and confirmed_s > 0
    )

    if total_s <= 0 or confirmed_s <= 0:
        verdict_note = (
            "No confirmed visual detection intervals; missing webcam or blind-frame data "
            "is NOT treated as confirmed misconduct."
        )
    elif missing_webcam_s > 0:
        verdict_note = (
            f"Missing webcam data ({missing_webcam_s:.1f}s) is classified as uncertain/unobserved "
            "and is NOT treated as confirmed misconduct."
        )
    elif not reliable_for_decision:
        verdict_note = (
            "Monitoring uncertainty is too high or data is insufficient for automated misconduct conclusions."
        )
    else:
        verdict_note = (
            "Monitoring coverage is sufficient; confirmed absence/multiple-face intervals reflect "
            "active camera and detector observations only."
        )

    misconduct_eval = MisconductEvaluation(
        confirmed_absence_s=report.face_absent_s,
        confirmed_multiple_faces_s=report.multiple_faces_s,
        unobserved_or_uncertain_s=uncertain_s,
        missing_webcam_s=missing_webcam_s,
        missing_webcam_treated_as_misconduct=False,
        reliable_for_misconduct_decision=reliable_for_decision,
        verdict_note=verdict_note,
    )

    return UncertaintyAssessment(
        session_id=report.session_id,
        uncertainty_level=level,
        uncertainty_ratio=uncertainty_ratio,
        uncertain_s=uncertain_s,
        confirmed_s=confirmed_s,
        indicators=indicators,
        misconduct_evaluation=misconduct_eval,
    )
