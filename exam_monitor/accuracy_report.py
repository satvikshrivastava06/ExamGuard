"""Accuracy report and structured monitoring summaries computed from session event logs.

Reads ``logs/events_<session_id>.jsonl`` (or any supplied path) and produces
structured summaries and human-readable accuracy metrics:

  Camera availability  – fraction of monitored time the camera was streaming
  Detection coverage   – fraction of monitored time face detection actively operated
  Face-presence ratio  – fraction of monitored time a face was detected
  Confidence stats     – per-frame max_confidence (avg / p50 / p95 / max)
  Absence events       – count, total duration, longest single absence
  Outages & Incidents  – camera outages, detection failures, interruptions
  Data sufficiency     – automated assessment of whether monitoring data is sufficient
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .data_sufficiency import (
    DataSufficiencyCriteria,
    SufficiencyAssessment,
    evaluate_session_sufficiency,
)
from .uncertainty import (
    UncertaintyAssessment,
    evaluate_uncertainty,
)


UNCERTAIN_STATES = frozenset({
    "DETECTION_UNAVAILABLE",
    "DETECTION_FAILURE",
    "CAMERA_UNAVAILABLE",
    "CAMERA_PERMISSION_DENIED",
})

MISCONDUCT_SIGNAL_STATES = frozenset({
    "FACE_ABSENT",
    "MULTIPLE_FACES",
})


@dataclass(frozen=True)
class DetectionInterval:
    """A single contiguous monitoring state interval within a session."""
    state: str
    start_mono: float
    end_mono: float
    duration_s: float
    start_utc: Optional[str] = None
    end_utc: Optional[str] = None
    is_uncertain: bool = False
    is_misconduct_signal: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "start_mono": round(self.start_mono, 3),
            "end_mono": round(self.end_mono, 3),
            "duration_s": round(self.duration_s, 3),
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "is_uncertain": self.is_uncertain,
            "is_misconduct_signal": self.is_misconduct_signal,
            "details": self.details,
        }


@dataclass
class AccuracyReport:
    session_id: str

    # Timing (seconds)
    total_monitored_s: float = 0.0
    face_present_s: float = 0.0
    face_absent_s: float = 0.0
    multiple_faces_s: float = 0.0
    detection_unavailable_s: float = 0.0
    camera_available_s: float = 0.0
    camera_unavailable_s: float = 0.0
    camera_permission_denied_s: float = 0.0

    # Counts
    n_events: int = 0
    n_face_present_events: int = 0
    n_face_absent_events: int = 0
    n_multiple_faces_events: int = 0
    n_detection_failure: int = 0
    n_camera_unavailable: int = 0
    n_camera_permission_denied: int = 0
    n_interruptions: int = 0
    n_prolonged_absences: int = 0

    # Lifecycle metadata
    has_session_start: bool = False
    has_session_end: bool = False
    start_time_utc: Optional[str] = None
    end_time_utc: Optional[str] = None
    final_state: Optional[str] = None

    # Absence details
    absence_durations: list[float] = field(default_factory=list)

    # Confidence samples (max_confidence from each event that carries it)
    confidence_samples: list[float] = field(default_factory=list)

    # Chronological state intervals
    intervals: list[DetectionInterval] = field(default_factory=list)

    # ------------------------------------------------------------------ Properties
    @property
    def presence_ratio(self) -> float:
        """Fraction of monitored time that a face was present (0.0–1.0)."""
        if self.total_monitored_s <= 0:
            return 0.0
        return min(1.0, max(0.0, self.face_present_s / self.total_monitored_s))

    @property
    def camera_outage_s(self) -> float:
        """Total duration camera was down or blocked."""
        return self.camera_unavailable_s + self.camera_permission_denied_s

    @property
    def camera_availability_ratio(self) -> float:
        """Fraction of monitored time the camera hardware was streaming and usable (0.0–1.0)."""
        if self.total_monitored_s <= 0:
            return 0.0
        return min(1.0, max(0.0, self.camera_available_s / self.total_monitored_s))

    @property
    def detection_coverage_s(self) -> float:
        """Total monitored time with active, decisive face detection."""
        return self.face_present_s + self.face_absent_s + self.multiple_faces_s

    @property
    def detection_coverage_ratio(self) -> float:
        """Fraction of monitored time where face detection was actively operating (0.0–1.0)."""
        if self.total_monitored_s <= 0:
            return 0.0
        return min(1.0, max(0.0, self.detection_coverage_s / self.total_monitored_s))

    @property
    def detection_unavailable_ratio(self) -> float:
        """Fraction of monitored time where detector could not judge (dark/blown out/error)."""
        if self.total_monitored_s <= 0:
            return 0.0
        return min(1.0, max(0.0, self.detection_unavailable_s / self.total_monitored_s))

    @property
    def effective_monitoring_ratio(self) -> float:
        """Fraction of monitored time with both camera available and detection functioning."""
        if self.total_monitored_s <= 0:
            return 0.0
        return min(1.0, max(0.0, self.detection_coverage_s / self.total_monitored_s))

    @property
    def longest_absence_s(self) -> float:
        return max(self.absence_durations, default=0.0)

    @property
    def avg_absence_s(self) -> float:
        return statistics.mean(self.absence_durations) if self.absence_durations else 0.0

    @property
    def total_absence_duration_s(self) -> float:
        return sum(self.absence_durations)

    @property
    def avg_confidence(self) -> Optional[float]:
        return statistics.mean(self.confidence_samples) if self.confidence_samples else None

    @property
    def p50_confidence(self) -> Optional[float]:
        return statistics.median(self.confidence_samples) if self.confidence_samples else None

    @property
    def p95_confidence(self) -> Optional[float]:
        if not self.confidence_samples:
            return None
        s = sorted(self.confidence_samples)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def max_confidence(self) -> Optional[float]:
        return max(self.confidence_samples) if self.confidence_samples else None

    # ------------------------------------------------------------------ Sufficiency & Uncertainty
    def sufficiency(self, criteria: Optional[DataSufficiencyCriteria] = None) -> SufficiencyAssessment:
        """Evaluate data sufficiency of this session against configured criteria."""
        return evaluate_session_sufficiency(self, criteria)

    def is_sufficient(self, criteria: Optional[DataSufficiencyCriteria] = None) -> bool:
        """Convenience property: True if session monitoring data is sufficient."""
        return self.sufficiency(criteria).is_sufficient

    def uncertainty(
        self,
        criteria: Optional[DataSufficiencyCriteria] = None,
        sufficiency: Optional[SufficiencyAssessment] = None,
    ) -> UncertaintyAssessment:
        """Evaluate uncertainty indicators and misconduct-guard metadata."""
        return evaluate_uncertainty(self, criteria=criteria, sufficiency=sufficiency)

    def aggregate_intervals(self) -> dict[str, Any]:
        """Aggregate detection intervals by state and merge adjacent same-state segments."""
        by_state: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[DetectionInterval]] = {}
        for iv in self.intervals:
            grouped.setdefault(iv.state, []).append(iv)

        for state, items in grouped.items():
            durations = [i.duration_s for i in items]
            total = sum(durations)
            ratio = min(1.0, max(0.0, total / self.total_monitored_s)) if self.total_monitored_s > 0 else 0.0
            by_state[state] = {
                "count": len(items),
                "total_s": round(total, 2),
                "min_s": round(min(durations), 2),
                "max_s": round(max(durations), 2),
                "avg_s": round(statistics.mean(durations), 2),
                "ratio": round(ratio, 4),
                "is_uncertain": state in UNCERTAIN_STATES,
                "is_misconduct_signal": state in MISCONDUCT_SIGNAL_STATES,
            }

        merged: list[DetectionInterval] = []
        for iv in self.intervals:
            if merged and merged[-1].state == iv.state:
                prev = merged[-1]
                merged[-1] = DetectionInterval(
                    state=prev.state,
                    start_mono=prev.start_mono,
                    end_mono=iv.end_mono,
                    duration_s=prev.duration_s + iv.duration_s,
                    start_utc=prev.start_utc,
                    end_utc=iv.end_utc,
                    is_uncertain=prev.is_uncertain,
                    is_misconduct_signal=prev.is_misconduct_signal,
                    details=prev.details,
                )
            else:
                merged.append(iv)

        return {
            "total_intervals": len(self.intervals),
            "merged_interval_count": len(merged),
            "uncertain_s": round(self.detection_unavailable_s + self.camera_outage_s, 2),
            "confirmed_s": round(self.detection_coverage_s, 2),
            "by_state": by_state,
            "intervals": [iv.to_dict() for iv in self.intervals],
            "merged_intervals": [iv.to_dict() for iv in merged],
        }

    # ------------------------------------------------------------------ Structured Exports
    def to_dict(self, criteria: Optional[DataSufficiencyCriteria] = None) -> dict[str, Any]:
        """Export structured summary as a nested dictionary."""
        suff = self.sufficiency(criteria)
        unc = self.uncertainty(criteria=criteria, sufficiency=suff)
        agg = self.aggregate_intervals()
        unc = self.uncertainty(criteria=criteria, sufficiency=suff)
        agg = self.aggregate_intervals()
        return {
            "session_id": self.session_id,
            "summary": {
                "total_monitored_s": round(self.total_monitored_s, 2),
                "presence_ratio": round(self.presence_ratio, 4),
                "camera_availability_ratio": round(self.camera_availability_ratio, 4),
                "detection_coverage_ratio": round(self.detection_coverage_ratio, 4),
                "effective_monitoring_ratio": round(self.effective_monitoring_ratio, 4),
                "is_sufficient": suff.is_sufficient,
                "uncertainty_level": unc.uncertainty_level.value,
                "uncertainty_ratio": round(unc.uncertainty_ratio, 4),
                "reliable_for_misconduct_decision": (
                    unc.misconduct_evaluation.reliable_for_misconduct_decision
                    if unc.misconduct_evaluation is not None
                    else False
                ),
            },
            "timing": {
                "total_monitored_s": round(self.total_monitored_s, 2),
                "face_present_s": round(self.face_present_s, 2),
                "face_absent_s": round(self.face_absent_s, 2),
                "multiple_faces_s": round(self.multiple_faces_s, 2),
                "detection_unavailable_s": round(self.detection_unavailable_s, 2),
                "camera_available_s": round(self.camera_available_s, 2),
                "camera_unavailable_s": round(self.camera_unavailable_s, 2),
                "camera_permission_denied_s": round(self.camera_permission_denied_s, 2),
                "camera_outage_s": round(self.camera_outage_s, 2),
            },
            "ratios": {
                "presence_ratio": round(self.presence_ratio, 4),
                "camera_availability_ratio": round(self.camera_availability_ratio, 4),
                "detection_coverage_ratio": round(self.detection_coverage_ratio, 4),
                "detection_unavailable_ratio": round(self.detection_unavailable_ratio, 4),
                "effective_monitoring_ratio": round(self.effective_monitoring_ratio, 4),
            },
            "counts": {
                "n_events": self.n_events,
                "n_face_present_events": self.n_face_present_events,
                "n_face_absent_events": self.n_face_absent_events,
                "n_multiple_faces_events": self.n_multiple_faces_events,
                "n_detection_failure": self.n_detection_failure,
                "n_camera_unavailable": self.n_camera_unavailable,
                "n_camera_permission_denied": self.n_camera_permission_denied,
                "n_interruptions": self.n_interruptions,
                "n_prolonged_absences": self.n_prolonged_absences,
            },
            "absence": {
                "event_count": self.n_face_absent_events,
                "total_s": round(self.face_absent_s, 2),
                "longest_s": round(self.longest_absence_s, 2),
                "avg_s": round(self.avg_absence_s, 2),
                "durations": self.absence_durations,
            },
            "confidence": {
                "avg": round(self.avg_confidence, 3) if self.avg_confidence is not None else None,
                "p50": round(self.p50_confidence, 3) if self.p50_confidence is not None else None,
                "p95": round(self.p95_confidence, 3) if self.p95_confidence is not None else None,
                "max": round(self.max_confidence, 3) if self.max_confidence is not None else None,
                "samples_count": len(self.confidence_samples),
            },
            "lifecycle": {
                "has_session_start": self.has_session_start,
                "has_session_end": self.has_session_end,
                "start_time_utc": self.start_time_utc,
                "end_time_utc": self.end_time_utc,
                "final_state": self.final_state,
            },
            "sufficiency": suff.to_dict(),
            "detection_intervals": agg,
            "uncertainty": unc.to_dict(),
        }

    def to_json(self, indent: int = 2, criteria: Optional[DataSufficiencyCriteria] = None) -> str:
        """Export structured summary as a JSON string."""
        return json.dumps(self.to_dict(criteria), indent=indent)

    def export_json(self, path: Path | str, indent: int = 2, criteria: Optional[DataSufficiencyCriteria] = None) -> Path:
        """Export structured summary to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=indent, criteria=criteria), encoding="utf-8")
        return target

    def to_csv_row(self, criteria: Optional[DataSufficiencyCriteria] = None) -> dict[str, Any]:
        """Export structured summary as a flat dictionary for tabular/CSV serialization."""
        suff = self.sufficiency(criteria)
        unc = self.uncertainty(criteria=criteria, sufficiency=suff)
        agg = self.aggregate_intervals()
        unc = self.uncertainty(criteria=criteria, sufficiency=suff)
        return {
            "session_id": self.session_id,
            "total_monitored_s": round(self.total_monitored_s, 2),
            "face_present_s": round(self.face_present_s, 2),
            "face_absent_s": round(self.face_absent_s, 2),
            "multiple_faces_s": round(self.multiple_faces_s, 2),
            "detection_unavailable_s": round(self.detection_unavailable_s, 2),
            "camera_available_s": round(self.camera_available_s, 2),
            "camera_unavailable_s": round(self.camera_unavailable_s, 2),
            "camera_permission_denied_s": round(self.camera_permission_denied_s, 2),
            "presence_ratio": round(self.presence_ratio, 4),
            "camera_availability_ratio": round(self.camera_availability_ratio, 4),
            "detection_coverage_ratio": round(self.detection_coverage_ratio, 4),
            "effective_monitoring_ratio": round(self.effective_monitoring_ratio, 4),
            "is_sufficient": suff.is_sufficient,
            "sufficiency_reasons": ";".join(r.value for r in suff.reasons),
            "uncertainty_level": unc.uncertainty_level.value,
            "uncertainty_ratio": round(unc.uncertainty_ratio, 4),
            "uncertainty_indicators": ";".join(i.code for i in unc.indicators),
            "reliable_for_misconduct_decision": (
                unc.misconduct_evaluation.reliable_for_misconduct_decision
                if unc.misconduct_evaluation is not None
                else False
            ),
            "missing_webcam_treated_as_misconduct": (
                unc.misconduct_evaluation.missing_webcam_treated_as_misconduct
                if unc.misconduct_evaluation is not None
                else False
            ),
            "n_events": self.n_events,
            "n_face_present_events": self.n_face_present_events,
            "n_face_absent_events": self.n_face_absent_events,
            "n_multiple_faces_events": self.n_multiple_faces_events,
            "n_detection_failure": self.n_detection_failure,
            "n_camera_unavailable": self.n_camera_unavailable,
            "n_camera_permission_denied": self.n_camera_permission_denied,
            "n_interruptions": self.n_interruptions,
            "longest_absence_s": round(self.longest_absence_s, 2),
            "avg_confidence": round(self.avg_confidence, 3) if self.avg_confidence is not None else "",
            "p50_confidence": round(self.p50_confidence, 3) if self.p50_confidence is not None else "",
            "p95_confidence": round(self.p95_confidence, 3) if self.p95_confidence is not None else "",
            "has_session_start": self.has_session_start,
            "has_session_end": self.has_session_end,
            "final_state": self.final_state or "",
        }

    def export_csv(self, path: Path | str, criteria: Optional[DataSufficiencyCriteria] = None) -> Path:
        """Export single report as a CSV file."""
        return export_summaries_csv([self], path, criteria=criteria)

    def print(self, criteria: Optional[DataSufficiencyCriteria] = None) -> None:
        """Print a human-readable summary to stdout."""
        import sys
        out = sys.stdout

        bar_width = 30

        def _bar(ratio: float) -> str:
            filled = int(max(0.0, min(1.0, ratio)) * bar_width)
            return "#" * filled + "-" * (bar_width - filled)

        def p(s: str = "") -> None:
            out.buffer.write((s + "\n").encode("utf-8"))
            out.buffer.flush()

        suff = self.sufficiency(criteria)
        unc = self.uncertainty(criteria=criteria, sufficiency=suff)
        agg = self.aggregate_intervals()

        p()
        p("=" * 65)
        p("  EXAM SESSION MONITORING REPORT")
        p(f"  Session : {self.session_id}")
        p("=" * 65)

        p(f"\n  Face Presence")
        p(f"    [{_bar(self.presence_ratio)}] {self.presence_ratio * 100:.1f}%")
        p(f"    Present  : {_fmt_dur(self.face_present_s)}")
        p(f"    Absent   : {_fmt_dur(self.face_absent_s)}")
        p(f"    Multiple : {_fmt_dur(self.multiple_faces_s)}")
        p(f"    Monitored: {_fmt_dur(self.total_monitored_s)}")

        p(f"\n  Camera Availability & Detection Coverage")
        p(f"    Camera Availability : [{_bar(self.camera_availability_ratio)}] {self.camera_availability_ratio * 100:.1f}%")
        p(f"      Available   : {_fmt_dur(self.camera_available_s)}")
        p(f"      Unavailable : {_fmt_dur(self.camera_unavailable_s)}")
        p(f"      Permission  : {_fmt_dur(self.camera_permission_denied_s)}")
        p(f"    Detection Coverage  : [{_bar(self.detection_coverage_ratio)}] {self.detection_coverage_ratio * 100:.1f}%")
        p(f"      Active / Decisive   : {_fmt_dur(self.detection_coverage_s)}")
        p(f"      Unavailable / Blind : {_fmt_dur(self.detection_unavailable_s)}")

        p(f"\n  Absence Events ({self.n_face_absent_events})")
        if self.absence_durations:
            p(f"    Longest  : {self.longest_absence_s:.1f}s")
            p(f"    Avg      : {self.avg_absence_s:.1f}s")
        else:
            p("    None recorded")

        p(f"\n  Detector Confidence")
        if self.confidence_samples:
            p(f"    Avg  : {self.avg_confidence:.3f}")
            p(f"    p50  : {self.p50_confidence:.3f}")
            p(f"    p95  : {self.p95_confidence:.3f}")
            p(f"    Max  : {self.max_confidence:.3f}")
            p(f"    n    : {len(self.confidence_samples)}")
        else:
            p("    No confidence data in log")

        p(f"\n  Incidents & Outages")
        p(f"    Camera outages     : {self.n_camera_unavailable + self.n_camera_permission_denied}")
        p(f"    Detection failures : {self.n_detection_failure}")
        p(f"    Interruptions      : {self.n_interruptions}")
        p(f"    Total events       : {self.n_events}")

        p(f"\n  Data Sufficiency Audit")
        if suff.is_sufficient:
            p("    Verdict: [PASSED] Monitoring data is SUFFICIENT for reliable evaluation.")
        else:
            p("    Verdict: [FAILED] Monitoring data is INSUFFICIENT.")
            for r in suff.reasons:
                detail = suff.details.get(r.value, "")
                p(f"      - {r.value}: {detail}")

        p(f"\n  Aggregated Detection Intervals ({agg['total_intervals']} raw, {agg['merged_interval_count']} merged)")
        by_state = agg["by_state"]
        if by_state:
            for st, st_stats in by_state.items():
                tag = " [UNCERTAIN]" if st_stats["is_uncertain"] else ""
                p(
                    f"    {st:<26} count={st_stats['count']:>3}  "
                    f"total={st_stats['total_s']:>6.1f}s  "
                    f"avg={st_stats['avg_s']:>5.1f}s  "
                    f"max={st_stats['max_s']:>5.1f}s{tag}"
                )
        else:
            p("    No state intervals recorded")

        p(f"\n  Uncertainty Indicators & Misconduct Guard")
        p(f"    Uncertainty Level : {unc.uncertainty_level.value} ({unc.uncertainty_ratio * 100:.1f}% uncertain)")
        if unc.indicators:
            for ind in unc.indicators:
                p(f"      - [{ind.severity.value}] {ind.code}: {ind.message}")
        else:
            p("      - None (high confidence, complete monitoring coverage)")
        if unc.misconduct_evaluation is not None:
            me = unc.misconduct_evaluation
            p(f"    Reliable for Misconduct Decision : {me.reliable_for_misconduct_decision}")
            p(f"    Missing Webcam as Misconduct     : {me.missing_webcam_treated_as_misconduct}")
            p(f"    Note: {me.verdict_note}")
        p("=" * 65)
        p()


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def compute_accuracy_from_events(session_id: str, events: list[dict[str, Any]]) -> AccuracyReport:
    """Compute an :class:`AccuracyReport` from an in-memory list of event dictionaries."""
    if not events:
        raise ValueError(f"No events found for session {session_id}")

    report = AccuracyReport(session_id=session_id)
    report.n_events = len(events)

    # Walk through events in chronological order (ts_mono is our reliable clock)
    # timeline: [(ts_mono, event_type, details, ts_utc), ...]
    timeline = sorted(
        [(e["ts_mono"], e["type"], e.get("details", {}), e.get("ts_utc", "")) for e in events],
        key=lambda x: x[0],
    )

    # Track state intervals
    current_state: Optional[str] = None
    state_entered_at: float = 0.0
    state_entered_utc: Optional[str] = None
    state_details: dict[str, Any] = {}
    session_start: Optional[float] = None
    session_end: Optional[float] = None

    def _close_interval(ts: float, end_utc: Optional[str] = None) -> None:
        nonlocal current_state, state_entered_at
        if current_state is None or session_start is None:
            return
        duration = ts - state_entered_at
        if duration <= 0:
            return
        normalized_state = (
            "DETECTION_UNAVAILABLE"
            if current_state in ("DETECTION_UNAVAILABLE", "DETECTION_FAILURE")
            else current_state
        )
        if normalized_state == "FACE_PRESENT":
            report.face_present_s += duration
            report.camera_available_s += duration
        elif normalized_state == "FACE_ABSENT":
            report.face_absent_s += duration
            report.camera_available_s += duration
        elif normalized_state == "MULTIPLE_FACES":
            report.multiple_faces_s += duration
            report.camera_available_s += duration
        elif normalized_state == "DETECTION_UNAVAILABLE":
            report.detection_unavailable_s += duration
            report.camera_available_s += duration
        elif normalized_state == "CAMERA_UNAVAILABLE":
            report.camera_unavailable_s += duration
        elif normalized_state == "CAMERA_PERMISSION_DENIED":
            report.camera_permission_denied_s += duration

        report.intervals.append(
            DetectionInterval(
                state=normalized_state,
                start_mono=state_entered_at,
                end_mono=ts,
                duration_s=duration,
                start_utc=state_entered_utc,
                end_utc=end_utc,
                is_uncertain=normalized_state in UNCERTAIN_STATES,
                is_misconduct_signal=normalized_state in MISCONDUCT_SIGNAL_STATES,
                details=dict(state_details),
            )
        )

    for ts_mono, etype, details, ts_utc in timeline:
        if etype == "SESSION_STARTED":
            if session_start is None:
                session_start = ts_mono
                report.has_session_start = True
                report.start_time_utc = ts_utc
        elif etype == "SESSION_ENDED":
            _close_interval(ts_mono, ts_utc)
            session_end = ts_mono
            report.has_session_end = True
            report.end_time_utc = ts_utc
            report.final_state = details.get("final_state")
            current_state = None
        elif etype == "SESSION_INTERRUPTED":
            report.n_interruptions += 1
            # Gap doesn't count toward active face or absence – close the interval
            _close_interval(ts_mono, ts_utc)
            # Re-open a new interval from after the gap
            if current_state is not None:
                state_entered_at = ts_mono
                state_entered_utc = ts_utc
        elif etype == "FACE_PRESENT":
            _close_interval(ts_mono, ts_utc)
            current_state = "FACE_PRESENT"
            state_entered_at = ts_mono
            state_entered_utc = ts_utc
            state_details = dict(details or {})
            report.n_face_present_events += 1
            conf = details.get("max_confidence")
            if conf is not None and conf > 0:
                report.confidence_samples.append(float(conf))
        elif etype == "FACE_ABSENT":
            _close_interval(ts_mono, ts_utc)
            current_state = "FACE_ABSENT"
            state_entered_at = ts_mono
            state_entered_utc = ts_utc
            state_details = dict(details or {})
            report.n_face_absent_events += 1
            prev_dur = details.get("previous_duration_s")
            if isinstance(prev_dur, (int, float)) and prev_dur > 0:
                report.absence_durations.append(float(prev_dur))
        elif etype == "MULTIPLE_FACES":
            _close_interval(ts_mono, ts_utc)
            current_state = "MULTIPLE_FACES"
            state_entered_at = ts_mono
            state_entered_utc = ts_utc
            state_details = dict(details or {})
            report.n_multiple_faces_events += 1
        elif etype in ("DETECTION_UNAVAILABLE", "DETECTION_FAILURE"):
            _close_interval(ts_mono, ts_utc)
            current_state = "DETECTION_UNAVAILABLE"
            state_entered_at = ts_mono
            state_entered_utc = ts_utc
            state_details = dict(details or {})
            report.n_detection_failure += 1
        elif etype == "CAMERA_UNAVAILABLE":
            _close_interval(ts_mono, ts_utc)
            current_state = "CAMERA_UNAVAILABLE"
            state_entered_at = ts_mono
            state_entered_utc = ts_utc
            state_details = dict(details or {})
            report.n_camera_unavailable += 1
        elif etype == "CAMERA_PERMISSION_DENIED":
            _close_interval(ts_mono, ts_utc)
            current_state = "CAMERA_PERMISSION_DENIED"
            state_entered_at = ts_mono
            state_entered_utc = ts_utc
            state_details = dict(details or {})
            report.n_camera_permission_denied += 1
        elif etype == "PROLONGED_ABSENCE":
            report.n_prolonged_absences += 1
            dur = details.get("duration_s")
            if isinstance(dur, (int, float)) and dur > 0 and dur not in report.absence_durations:
                report.absence_durations.append(float(dur))

    # If log ended without SESSION_ENDED (crash / still running), close at last event
    if session_end is None and timeline:
        session_end = timeline[-1][0]
        _close_interval(session_end, timeline[-1][3])

    if session_start is not None and session_end is not None:
        report.total_monitored_s = session_end - session_start

    return report


def compute_accuracy(log_path: Path) -> AccuracyReport:
    """Parse a JSONL event log and return an :class:`AccuracyReport`."""
    events = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not events:
        raise ValueError(f"No events found in {log_path}")

    session_id = events[0].get("session_id", log_path.stem)
    return compute_accuracy_from_events(session_id, events)


# ------------------------------------------------------------------ Batch Summary Helpers

def summarize_sessions(log_paths: list[Path | str]) -> list[AccuracyReport]:
    """Compute AccuracyReport summaries for multiple log paths."""
    reports: list[AccuracyReport] = []
    for p in log_paths:
        path = Path(p)
        if path.is_file():
            try:
                reports.append(compute_accuracy(path))
            except Exception:
                pass
    return reports


def export_summaries_json(
    reports: list[AccuracyReport],
    path: Path | str,
    indent: int = 2,
    criteria: Optional[DataSufficiencyCriteria] = None,
) -> Path:
    """Export multiple AccuracyReport summaries to a single JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "count": len(reports),
        "summaries": [r.to_dict(criteria) for r in reports],
    }
    target.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return target


def export_summaries_csv(
    reports: list[AccuracyReport],
    path: Path | str,
    criteria: Optional[DataSufficiencyCriteria] = None,
) -> Path:
    """Export multiple AccuracyReport summaries to a tabular CSV file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.to_csv_row(criteria) for r in reports]
    fieldnames = list(rows[0].keys()) if rows else [
        "session_id", "total_monitored_s", "face_present_s", "face_absent_s",
        "multiple_faces_s", "detection_unavailable_s", "camera_available_s",
        "camera_unavailable_s", "camera_permission_denied_s", "presence_ratio",
        "camera_availability_ratio", "detection_coverage_ratio", "effective_monitoring_ratio",
        "is_sufficient", "sufficiency_reasons", "uncertainty_level", "uncertainty_ratio",
        "uncertainty_indicators", "reliable_for_misconduct_decision",
        "missing_webcam_treated_as_misconduct", "n_events", "n_face_present_events",
        "n_face_absent_events", "n_multiple_faces_events", "n_detection_failure",
        "n_camera_unavailable", "n_camera_permission_denied", "n_interruptions",
        "longest_absence_s", "avg_confidence", "p50_confidence", "p95_confidence",
        "has_session_start", "has_session_end", "final_state"
    ]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target
