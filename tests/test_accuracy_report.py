"""Unit tests for exam_monitor.accuracy_report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from exam_monitor.accuracy_report import compute_accuracy, AccuracyReport


def _write_log(events: list[dict]) -> Path:
    """Write a list of event dicts to a temp JSONL file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for ev in events:
        tmp.write(json.dumps(ev) + "\n")
    tmp.close()
    return Path(tmp.name)


def _event(etype: str, ts_mono: float, session_id: str = "TEST-SESSION", details: dict | None = None) -> dict:
    return {
        "type": etype,
        "session_id": session_id,
        "ts_local": "2026-11-03T18:00:00.000+05:30",
        "ts_mono": ts_mono,
        "frame_id": None,
        "details": details or {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestPresenceRatio:
    def test_perfect_presence(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 0.5, details={"n_faces": 1, "max_confidence": 5.0, "detector": "haar",
                                                   "previous_state": None, "previous_duration_s": None}),
            _event("SESSION_ENDED", 60.0, details={"final_state": "FACE_PRESENT"}),
        ])
        r = compute_accuracy(log)
        # face_present from 0.5 to 60.0 = 59.5 s; total = 60.0 s
        assert r.total_monitored_s == pytest.approx(60.0)
        assert r.face_present_s == pytest.approx(59.5, abs=0.01)
        assert r.presence_ratio == pytest.approx(59.5 / 60.0, abs=0.01)

    def test_zero_presence(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_ABSENT", 1.0, details={"n_faces": 0, "max_confidence": 0.0, "detector": "haar",
                                                  "previous_state": None, "previous_duration_s": None}),
            _event("SESSION_ENDED", 30.0, details={"final_state": "FACE_ABSENT"}),
        ])
        r = compute_accuracy(log)
        assert r.face_present_s == pytest.approx(0.0)
        assert r.presence_ratio == pytest.approx(0.0)
        assert r.face_absent_s == pytest.approx(29.0, abs=0.01)

    def test_alternating_present_absent(self):
        # 0→10 absent, 10→20 present, 20→30 absent, 30→40 present
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_ABSENT", 0.5, details={"previous_state": None, "previous_duration_s": None,
                                                 "n_faces": 0, "max_confidence": 0.0, "detector": "haar"}),
            _event("FACE_PRESENT", 10.0, details={"previous_state": "FACE_ABSENT", "previous_duration_s": 9.5,
                                                   "n_faces": 1, "max_confidence": 7.0, "detector": "haar"}),
            _event("FACE_ABSENT", 20.0, details={"previous_state": "FACE_PRESENT", "previous_duration_s": 10.0,
                                                  "n_faces": 0, "max_confidence": 0.0, "detector": "haar"}),
            _event("FACE_PRESENT", 30.0, details={"previous_state": "FACE_ABSENT", "previous_duration_s": 10.0,
                                                   "n_faces": 1, "max_confidence": 5.0, "detector": "haar"}),
            _event("SESSION_ENDED", 40.0, details={"final_state": "FACE_PRESENT"}),
        ])
        r = compute_accuracy(log)
        assert r.face_present_s == pytest.approx(20.0, abs=0.1)
        assert r.face_absent_s == pytest.approx(19.5, abs=0.1)


class TestConfidence:
    def test_confidence_collected_from_face_present_events(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"n_faces": 1, "max_confidence": 4.0, "detector": "haar",
                                                   "previous_state": None, "previous_duration_s": None}),
            _event("FACE_PRESENT", 5.0, details={"n_faces": 1, "max_confidence": 8.0, "detector": "haar",
                                                   "previous_state": None, "previous_duration_s": None}),
            _event("SESSION_ENDED", 10.0),
        ])
        r = compute_accuracy(log)
        assert r.confidence_samples == pytest.approx([4.0, 8.0])
        assert r.avg_confidence == pytest.approx(6.0)
        assert r.p50_confidence == pytest.approx(6.0, abs=1.0)

    def test_zero_confidence_not_collected(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_ABSENT", 1.0, details={"n_faces": 0, "max_confidence": 0.0, "detector": "haar",
                                                  "previous_state": None, "previous_duration_s": None}),
            _event("SESSION_ENDED", 5.0),
        ])
        r = compute_accuracy(log)
        assert r.confidence_samples == []
        assert r.avg_confidence is None


class TestIncidents:
    def test_interruption_count(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("SESSION_INTERRUPTED", 5.0, details={"gap_s": 3.0}),
            _event("SESSION_INTERRUPTED", 15.0, details={"gap_s": 2.0}),
            _event("SESSION_ENDED", 20.0),
        ])
        r = compute_accuracy(log)
        assert r.n_interruptions == 2

    def test_detection_failure_count(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("DETECTION_FAILURE", 2.0, details={"reason": "too_dark"}),
            _event("DETECTION_FAILURE", 4.0, details={"reason": "too_dark"}),
            _event("SESSION_ENDED", 10.0),
        ])
        r = compute_accuracy(log)
        assert r.n_detection_failure == 2

    def test_absence_duration_from_details(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"n_faces": 1, "max_confidence": 5.0, "detector": "haar",
                                                   "previous_state": None, "previous_duration_s": None}),
            _event("FACE_ABSENT", 11.0, details={"n_faces": 0, "max_confidence": 0.0, "detector": "haar",
                                                  "previous_state": "FACE_PRESENT", "previous_duration_s": 10.0}),
            _event("SESSION_ENDED", 25.0),
        ])
        r = compute_accuracy(log)
        assert 10.0 in r.absence_durations
        assert r.longest_absence_s == pytest.approx(10.0)


class TestEdgeCases:
    def test_empty_log_raises(self):
        log = _write_log([])
        with pytest.raises(ValueError, match="No events"):
            compute_accuracy(log)

    def test_no_session_ended_uses_last_event(self):
        """Log truncated mid-session (crash) should still produce a report."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"n_faces": 1, "max_confidence": 6.0, "detector": "haar",
                                                   "previous_state": None, "previous_duration_s": None}),
            # No SESSION_ENDED
        ])
        r = compute_accuracy(log)
        assert r.total_monitored_s == pytest.approx(1.0, abs=0.1)

    def test_multiple_sessions_appended(self):
        """JSONL files accumulate across re-starts; report should span all entries."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"n_faces": 1, "max_confidence": 5.0, "detector": "haar",
                                                   "previous_state": None, "previous_duration_s": None}),
            _event("SESSION_ENDED", 10.0),
            # Second run
            _event("SESSION_STARTED", 100.0),
            _event("FACE_PRESENT", 101.0, details={"n_faces": 1, "max_confidence": 7.0, "detector": "haar",
                                                    "previous_state": None, "previous_duration_s": None}),
            _event("SESSION_ENDED", 120.0),
        ])
        r = compute_accuracy(log)
        assert r.n_events == 6
        # total span: 0 → 120 = 120s; face_present = (10-1) + (120-101) = 9+19 = 28s
        assert r.face_present_s == pytest.approx(28.0, abs=0.5)
