"""Accuracy report computed from a session's JSONL event log.

Reads ``logs/events_<session_id>.jsonl`` (or any path you supply) and
produces a human-readable accuracy summary:

  Face-presence ratio   – fraction of monitored time a face was detected
  Confidence stats      – per-frame max_confidence (avg / p50 / p95 / max)
  Absence events        – count, total duration, longest single absence
  Interruptions         – SESSION_INTERRUPTED count (stalls / sleep gaps)
  Detection failures    – DETECTION_FAILURE count
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AccuracyReport:
    session_id: str

    # Timing
    total_monitored_s: float = 0.0
    face_present_s: float = 0.0
    face_absent_s: float = 0.0

    # Counts
    n_events: int = 0
    n_face_present_events: int = 0
    n_face_absent_events: int = 0
    n_detection_failure: int = 0
    n_interruptions: int = 0

    # Absence details
    absence_durations: list[float] = field(default_factory=list)

    # Confidence samples (max_confidence from each event that carries it)
    confidence_samples: list[float] = field(default_factory=list)

    @property
    def presence_ratio(self) -> float:
        """Fraction of monitored time that a face was present (0.0–1.0)."""
        if self.total_monitored_s <= 0:
            return 0.0
        return self.face_present_s / self.total_monitored_s

    @property
    def longest_absence_s(self) -> float:
        return max(self.absence_durations, default=0.0)

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

    def print(self) -> None:
        import sys
        out = sys.stdout

        bar_width = 40
        filled = int(self.presence_ratio * bar_width)
        # Use ASCII-safe characters so Windows cmd/PowerShell never chokes on encoding
        bar = "#" * filled + "-" * (bar_width - filled)

        def p(s: str = "") -> None:
            out.buffer.write((s + "\n").encode("utf-8"))
            out.buffer.flush()

        p()
        p("=" * 60)
        p("  EXAM SESSION ACCURACY REPORT")
        p(f"  Session : {self.session_id}")
        p("=" * 60)

        p(f"\n  Face Presence")
        p(f"    [{bar}] {self.presence_ratio * 100:.1f}%")
        p(f"    Present  : {_fmt_dur(self.face_present_s)}")
        p(f"    Absent   : {_fmt_dur(self.face_absent_s)}")
        p(f"    Monitored: {_fmt_dur(self.total_monitored_s)}")

        p(f"\n  Absence Events ({self.n_face_absent_events})")
        if self.absence_durations:
            p(f"    Longest  : {self.longest_absence_s:.1f}s")
            p(f"    Avg      : {statistics.mean(self.absence_durations):.1f}s")
        else:
            p("    None recorded")

        p(f"\n  Detector Confidence")
        if self.confidence_samples:
            p(f"    Avg  : {self.avg_confidence:.3f}")
            p(f"    p50  : {self.p50_confidence:.3f}")
            p(f"    p95  : {self.p95_confidence:.3f}")
            p(f"    Max  : {max(self.confidence_samples):.3f}")
            p(f"    n    : {len(self.confidence_samples)}")
        else:
            p("    No confidence data in log")

        p(f"\n  Incidents")
        p(f"    Detection failures : {self.n_detection_failure}")
        p(f"    Interruptions      : {self.n_interruptions}")
        p(f"    Total events       : {self.n_events}")
        p("=" * 60)
        p()


def _fmt_dur(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


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
    report = AccuracyReport(session_id=session_id)
    report.n_events = len(events)

    # Walk through events in chronological order (ts_mono is our reliable clock)
    # Build a timeline: [(ts_mono, event_type, details), ...]
    timeline = sorted(
        [(e["ts_mono"], e["type"], e.get("details", {})) for e in events],
        key=lambda x: x[0],
    )

    # Track state intervals
    current_state: Optional[str] = None
    state_entered_at: float = 0.0
    session_start: Optional[float] = None
    session_end: Optional[float] = None

    def _close_interval(ts: float) -> None:
        nonlocal current_state, state_entered_at
        if current_state is None or session_start is None:
            return
        duration = ts - state_entered_at
        if duration <= 0:
            return
        if current_state == "FACE_PRESENT":
            report.face_present_s += duration
        elif current_state in ("FACE_ABSENT", "MULTIPLE_FACES"):
            report.face_absent_s += duration

    for ts_mono, etype, details in timeline:
        if etype == "SESSION_STARTED":
            if session_start is None:
                session_start = ts_mono
        elif etype == "SESSION_ENDED":
            _close_interval(ts_mono)
            session_end = ts_mono
            current_state = None
        elif etype == "SESSION_INTERRUPTED":
            report.n_interruptions += 1
            # Gap doesn't count toward face_absent – we just close the interval
            _close_interval(ts_mono)
            # Re-open a new interval from after the gap
            if current_state is not None:
                state_entered_at = ts_mono
        elif etype == "FACE_PRESENT":
            _close_interval(ts_mono)
            current_state = "FACE_PRESENT"
            state_entered_at = ts_mono
            report.n_face_present_events += 1
            conf = details.get("max_confidence")
            if conf is not None and conf > 0:
                report.confidence_samples.append(float(conf))
        elif etype == "FACE_ABSENT":
            _close_interval(ts_mono)
            if current_state == "FACE_PRESENT":
                # Record the absence duration from when face went away
                pass
            current_state = "FACE_ABSENT"
            state_entered_at = ts_mono
            report.n_face_absent_events += 1
            prev_dur = details.get("previous_duration_s")
            if isinstance(prev_dur, (int, float)) and prev_dur > 0:
                report.absence_durations.append(float(prev_dur))
        elif etype == "DETECTION_FAILURE":
            report.n_detection_failure += 1
        elif etype == "MULTIPLE_FACES":
            _close_interval(ts_mono)
            current_state = "MULTIPLE_FACES"
            state_entered_at = ts_mono

    # If log ended without SESSION_ENDED (crash / still running), use last ts
    if session_end is None and timeline:
        session_end = timeline[-1][0]
        _close_interval(session_end)

    if session_start is not None and session_end is not None:
        report.total_monitored_s = session_end - session_start

    return report
