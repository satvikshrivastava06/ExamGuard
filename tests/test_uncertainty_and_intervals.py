"""Tests for aggregated detection intervals and uncertainty indicators."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from exam_monitor import (
    UncertaintyLevel,
    compute_accuracy,
    compute_accuracy_from_db,
    compute_accuracy_from_events,
)


def _ev(etype: str, ts_mono: float, session_id: str = "INT-001", details: dict | None = None) -> dict:
    return {
        "type": etype,
        "session_id": session_id,
        "ts_utc": f"2026-11-27T10:00:{int(ts_mono):02d}.000+00:00",
        "ts_mono": ts_mono,
        "frame_id": int(ts_mono * 30),
        "details": details or {},
    }


class TestAggregatedDetectionIntervals:
    def test_intervals_and_by_state_aggregation(self):
        events = [
            _ev("SESSION_STARTED", 0.0),
            _ev("FACE_PRESENT", 1.0, details={"max_confidence": 0.91}),
            _ev("FACE_ABSENT", 21.0),
            _ev("FACE_PRESENT", 26.0, details={"max_confidence": 0.88}),
            _ev("DETECTION_UNAVAILABLE", 46.0, details={"reason": "too_dark_or_lens_covered"}),
            _ev("FACE_PRESENT", 50.0, details={"max_confidence": 0.93}),
            _ev("SESSION_ENDED", 60.0, details={"final_state": "FACE_PRESENT"}),
        ]
        report = compute_accuracy_from_events("INT-001", events)
        agg = report.aggregate_intervals()

        assert agg["total_intervals"] == 5
        assert agg["merged_interval_count"] == 5
        assert agg["confirmed_s"] == pytest.approx(55.0)
        assert agg["uncertain_s"] == pytest.approx(4.0)

        by_state = agg["by_state"]
        assert by_state["FACE_PRESENT"]["count"] == 3
        assert by_state["FACE_PRESENT"]["total_s"] == pytest.approx(50.0)
        assert by_state["FACE_PRESENT"]["min_s"] == pytest.approx(10.0)
        assert by_state["FACE_PRESENT"]["max_s"] == pytest.approx(20.0)
        assert by_state["FACE_PRESENT"]["is_uncertain"] is False
        assert by_state["FACE_PRESENT"]["is_misconduct_signal"] is False

        assert by_state["FACE_ABSENT"]["count"] == 1
        assert by_state["FACE_ABSENT"]["total_s"] == pytest.approx(5.0)
        assert by_state["FACE_ABSENT"]["is_misconduct_signal"] is True

        assert by_state["DETECTION_UNAVAILABLE"]["count"] == 1
        assert by_state["DETECTION_UNAVAILABLE"]["total_s"] == pytest.approx(4.0)
        assert by_state["DETECTION_UNAVAILABLE"]["is_uncertain"] is True
        assert by_state["DETECTION_UNAVAILABLE"]["is_misconduct_signal"] is False

    def test_merged_intervals_across_interruption(self):
        events = [
            _ev("SESSION_STARTED", 0.0),
            _ev("FACE_PRESENT", 1.0, details={"max_confidence": 0.9}),
            _ev("SESSION_INTERRUPTED", 15.0, details={"gap_s": 3.0}),
            _ev("SESSION_ENDED", 40.0, details={"final_state": "FACE_PRESENT"}),
        ]
        report = compute_accuracy_from_events("INT-002", events)
        agg = report.aggregate_intervals()

        assert agg["total_intervals"] == 2
        assert agg["merged_interval_count"] == 1
        assert agg["merged_intervals"][0]["duration_s"] == pytest.approx(39.0)

    def test_db_compute_accuracy_shares_intervals_and_uncertainty(self):
        rows = [
            ("SESSION_STARTED", 0.0, "2026-11-27 10:00:00", None, json.dumps({})),
            ("FACE_PRESENT", 1.0, "2026-11-27 10:00:01", 30, json.dumps({"max_confidence": 0.92})),
            ("SESSION_ENDED", 60.0, "2026-11-27 10:01:00", None, json.dumps({"final_state": "FACE_PRESENT"})),
        ]
        cursor = MagicMock()
        cursor.fetchall.return_value = rows
        conn = MagicMock()
        conn.cursor.return_value = cursor
        db = MagicMock()
        db.get_connection.return_value = conn

        report = compute_accuracy_from_db("DB-001", db)
        assert len(report.intervals) == 1
        assert report.uncertainty().uncertainty_level == UncertaintyLevel.LOW


class TestUncertaintyIndicators:
    def test_clean_session_has_low_uncertainty(self):
        events = [
            _ev("SESSION_STARTED", 0.0),
            _ev("FACE_PRESENT", 1.0, details={"max_confidence": 0.94}),
            _ev("SESSION_ENDED", 60.0, details={"final_state": "FACE_PRESENT"}),
        ]
        report = compute_accuracy_from_events("UNC-LOW", events)
        unc = report.uncertainty()

        assert unc.uncertainty_level == UncertaintyLevel.LOW
        assert unc.uncertainty_ratio == pytest.approx(0.0)
        assert unc.indicators == []
        assert unc.misconduct_evaluation is not None
        assert unc.misconduct_evaluation.reliable_for_misconduct_decision is True
        assert unc.misconduct_evaluation.missing_webcam_treated_as_misconduct is False

    def test_blind_frames_and_low_confidence_raise_indicators(self):
        events = [
            _ev("SESSION_STARTED", 0.0),
            _ev("FACE_PRESENT", 1.0, details={"max_confidence": 0.30}),
            _ev("DETECTION_UNAVAILABLE", 20.0, details={"reason": "too_dark_or_lens_covered"}),
            _ev("FACE_PRESENT", 35.0, details={"max_confidence": 0.35}),
            _ev("SESSION_ENDED", 60.0, details={"final_state": "FACE_PRESENT"}),
        ]
        report = compute_accuracy_from_events("UNC-MOD", events)
        unc = report.uncertainty()
        codes = {ind.code for ind in unc.indicators}

        assert "DETECTOR_BLIND_TIME" in codes
        assert "LOW_DETECTION_CONFIDENCE" in codes
        assert unc.uncertainty_level in (UncertaintyLevel.HIGH, UncertaintyLevel.MODERATE)
        assert unc.uncertain_s == pytest.approx(15.0)

    def test_to_dict_and_csv_include_intervals_and_uncertainty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            for ev in [
                _ev("SESSION_STARTED", 0.0),
                _ev("FACE_PRESENT", 1.0, details={"max_confidence": 0.85}),
                _ev("SESSION_ENDED", 40.0, details={"final_state": "FACE_PRESENT"}),
            ]:
                tmp.write(json.dumps(ev) + "\n")
            path = Path(tmp.name)

        report = compute_accuracy(path)
        d = report.to_dict()
        row = report.to_csv_row()

        assert "detection_intervals" in d
        assert "uncertainty" in d
        assert d["summary"]["uncertainty_level"] == "LOW"
        assert row["uncertainty_level"] == "LOW"
        assert row["missing_webcam_treated_as_misconduct"] is False
