"""Data sufficiency evaluation for exam monitoring sessions.

Identifies whether an exam session has sufficient monitoring data for
reliable proctoring decisions, or flags sessions with insufficient data
along with specific, explainable reasons:
  - Session too short
  - Camera availability too low (hardware/driver failure or OS permission blocked)
  - Detection coverage too low (frames too dark, blown out, or detector failure)
  - Excessive interruptions or stalls
  - Abrupt termination (missing SESSION_ENDED)
  - Insufficient recorded events or no presence observed
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .accuracy_report import AccuracyReport


class InsufficientDataReason(str, Enum):
    """Categorized root causes for why a session has insufficient monitoring data."""
    NO_DATA = "NO_DATA"                                     # Empty or unreadable log
    SESSION_TOO_SHORT = "SESSION_TOO_SHORT"                 # Monitored duration < min_duration_s
    INSUFFICIENT_EVENTS = "INSUFFICIENT_EVENTS"             # Too few events emitted
    LOW_CAMERA_AVAILABILITY = "LOW_CAMERA_AVAILABILITY"     # Camera availability ratio < threshold
    LOW_DETECTION_COVERAGE = "LOW_DETECTION_COVERAGE"       # Detection coverage ratio < threshold
    EXCESSIVE_DETECTION_FAILURES = "EXCESSIVE_DETECTION_FAILURES"  # High ratio in DETECTION_UNAVAILABLE
    EXCESSIVE_INTERRUPTIONS = "EXCESSIVE_INTERRUPTIONS"     # Too many SESSION_INTERRUPTED gaps
    MISSING_SESSION_END = "MISSING_SESSION_END"             # Premature termination / crash without SESSION_ENDED
    ZERO_PRESENCE_DETECTED = "ZERO_PRESENCE_DETECTED"       # Candidate was never seen during session


@dataclass(frozen=True)
class DataSufficiencyCriteria:
    """Configurable quality and integrity thresholds for monitoring data sufficiency."""
    min_duration_s: float = 30.0                      # Minimum required session monitoring duration (seconds)
    min_camera_availability: float = 0.80             # Minimum camera uptime ratio (0.0 - 1.0)
    min_detection_coverage: float = 0.70              # Minimum valid face detection coverage ratio (0.0 - 1.0)
    min_events: int = 3                               # Minimum events in log (SESSION_STARTED + state + ENDED)
    max_interruptions: int = 3                        # Maximum tolerated SESSION_INTERRUPTED events
    max_detection_failure_ratio: float = 0.20         # Maximum fraction of time in DETECTION_UNAVAILABLE
    require_session_end: bool = True                  # Must have a clean SESSION_ENDED event
    require_presence: bool = False                    # If True, candidate must have been present at least once


@dataclass
class SufficiencyAssessment:
    """Detailed sufficiency verdict for a single session."""
    session_id: str
    is_sufficient: bool
    reasons: list[InsufficientDataReason] = field(default_factory=list)
    details: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "is_sufficient": self.is_sufficient,
            "reasons": [r.value for r in self.reasons],
            "details": self.details,
            "metrics": self.metrics,
        }

    def summary_text(self) -> str:
        if self.is_sufficient:
            return f"Session {self.session_id}: SUFFICIENT monitoring data"
        reasons_str = ", ".join(r.value for r in self.reasons)
        return f"Session {self.session_id}: INSUFFICIENT monitoring data ({reasons_str})"


def evaluate_session_sufficiency(
    report: "AccuracyReport",
    criteria: Optional[DataSufficiencyCriteria] = None,
) -> SufficiencyAssessment:
    """Evaluate whether an :class:`AccuracyReport` meets data sufficiency criteria."""
    crit = criteria or DataSufficiencyCriteria()
    reasons: list[InsufficientDataReason] = []
    details: dict[str, str] = {}
    metrics: dict[str, Any] = {
        "total_monitored_s": round(report.total_monitored_s, 2),
        "camera_availability_ratio": round(report.camera_availability_ratio, 4),
        "detection_coverage_ratio": round(report.detection_coverage_ratio, 4),
        "detection_unavailable_ratio": round(report.detection_unavailable_ratio, 4),
        "n_events": report.n_events,
        "n_interruptions": report.n_interruptions,
        "has_session_end": report.has_session_end,
        "face_present_s": round(report.face_present_s, 2),
    }

    # 1. Zero or negligible events
    if report.n_events < crit.min_events:
        reasons.append(InsufficientDataReason.INSUFFICIENT_EVENTS)
        details[InsufficientDataReason.INSUFFICIENT_EVENTS.value] = (
            f"Only {report.n_events} events recorded (threshold: >={crit.min_events})"
        )

    # 2. Session duration too short
    if report.total_monitored_s < crit.min_duration_s:
        reasons.append(InsufficientDataReason.SESSION_TOO_SHORT)
        details[InsufficientDataReason.SESSION_TOO_SHORT.value] = (
            f"Monitored for {report.total_monitored_s:.1f}s (threshold: >={crit.min_duration_s:.1f}s)"
        )

    # 3. Camera availability
    if report.total_monitored_s > 0 and report.camera_availability_ratio < crit.min_camera_availability:
        reasons.append(InsufficientDataReason.LOW_CAMERA_AVAILABILITY)
        details[InsufficientDataReason.LOW_CAMERA_AVAILABILITY.value] = (
            f"Camera availability ratio is {report.camera_availability_ratio * 100:.1f}% "
            f"(threshold: >={crit.min_camera_availability * 100:.1f}%, "
            f"outage: {report.camera_outage_s:.1f}s)"
        )

    # 4. Detection coverage
    if report.total_monitored_s > 0 and report.detection_coverage_ratio < crit.min_detection_coverage:
        reasons.append(InsufficientDataReason.LOW_DETECTION_COVERAGE)
        details[InsufficientDataReason.LOW_DETECTION_COVERAGE.value] = (
            f"Detection coverage ratio is {report.detection_coverage_ratio * 100:.1f}% "
            f"(threshold: >={crit.min_detection_coverage * 100:.1f}%)"
        )

    # 5. Excessive detection unavailable / failure
    if report.total_monitored_s > 0 and report.detection_unavailable_ratio > crit.max_detection_failure_ratio:
        reasons.append(InsufficientDataReason.EXCESSIVE_DETECTION_FAILURES)
        details[InsufficientDataReason.EXCESSIVE_DETECTION_FAILURES.value] = (
            f"Detection unavailable ratio is {report.detection_unavailable_ratio * 100:.1f}% "
            f"(threshold: <={crit.max_detection_failure_ratio * 100:.1f}%, "
            f"duration: {report.detection_unavailable_s:.1f}s)"
        )

    # 6. Excessive interruptions
    if report.n_interruptions > crit.max_interruptions:
        reasons.append(InsufficientDataReason.EXCESSIVE_INTERRUPTIONS)
        details[InsufficientDataReason.EXCESSIVE_INTERRUPTIONS.value] = (
            f"{report.n_interruptions} interruptions detected (threshold: <={crit.max_interruptions})"
        )

    # 7. Missing session end
    if crit.require_session_end and not report.has_session_end:
        reasons.append(InsufficientDataReason.MISSING_SESSION_END)
        details[InsufficientDataReason.MISSING_SESSION_END.value] = (
            "Session log has no clean SESSION_ENDED event (likely crashed or prematurely terminated)"
        )

    # 8. Zero presence when required
    if crit.require_presence and report.total_monitored_s >= crit.min_duration_s and report.face_present_s <= 0:
        reasons.append(InsufficientDataReason.ZERO_PRESENCE_DETECTED)
        details[InsufficientDataReason.ZERO_PRESENCE_DETECTED.value] = (
            "Candidate face was never detected throughout the monitored session"
        )

    return SufficiencyAssessment(
        session_id=report.session_id,
        is_sufficient=len(reasons) == 0,
        reasons=reasons,
        details=details,
        metrics=metrics,
    )


def identify_insufficient_sessions(
    reports: list["AccuracyReport"],
    criteria: Optional[DataSufficiencyCriteria] = None,
) -> list[SufficiencyAssessment]:
    """Filter and return only assessments for sessions with insufficient data."""
    crit = criteria or DataSufficiencyCriteria()
    assessments = [evaluate_session_sufficiency(r, crit) for r in reports]
    return [a for a in assessments if not a.is_sufficient]


def filter_sessions_by_sufficiency(
    reports: list["AccuracyReport"],
    criteria: Optional[DataSufficiencyCriteria] = None,
) -> tuple[list["AccuracyReport"], list[SufficiencyAssessment]]:
    """Partition reports into (sufficient_reports, insufficient_assessments)."""
    crit = criteria or DataSufficiencyCriteria()
    sufficient: list["AccuracyReport"] = []
    insufficient: list[SufficiencyAssessment] = []
    for r in reports:
        assessment = evaluate_session_sufficiency(r, crit)
        if assessment.is_sufficient:
            sufficient.append(r)
        else:
            insufficient.append(assessment)
    return sufficient, insufficient
