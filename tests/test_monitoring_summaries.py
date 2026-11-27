"""Unit tests for camera availability, detection coverage, structured summary export,
and session data sufficiency identification.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from exam_monitor.accuracy_report import (
    AccuracyReport,
    compute_accuracy,
    export_summaries_csv,
    export_summaries_json,
    summarize_sessions,
)
from exam_monitor.data_sufficiency import (
    DataSufficiencyCriteria,
    InsufficientDataReason,
    SufficiencyAssessment,
    evaluate_session_sufficiency,
    filter_sessions_by_sufficiency,
    identify_insufficient_sessions,
)
from exam_monitor.database import compute_accuracy_from_db, export_db_summaries


def _write_log(events: list[dict]) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for ev in events:
        tmp.write(json.dumps(ev) + "\n")
    tmp.close()
    return Path(tmp.name)


def _event(etype: str, ts_mono: float, session_id: str = "TEST-SESSION-001", details: dict | None = None) -> dict:
    return {
        "type": etype,
        "session_id": session_id,
        "ts_utc": "2026-11-20T10:00:00.000+00:00",
        "ts_mono": ts_mono,
        "frame_id": int(ts_mono * 30),
        "details": details or {},
    }


# ====================================================================== Camera Availability & Detection Coverage

class TestCameraAvailabilityAndDetectionCoverage:

    def test_full_camera_availability_and_coverage(self):
        """Standard session: camera available 100%, detection actively covering 100% of state time."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"max_confidence": 0.95}),
            _event("FACE_ABSENT", 30.0),
            _event("FACE_PRESENT", 40.0, details={"max_confidence": 0.88}),
            _event("SESSION_ENDED", 100.0),
        ])
        report = compute_accuracy(log)

        assert report.total_monitored_s == pytest.approx(100.0)
        # Face was present 1.0-30.0 (29s) + 40.0-100.0 (60s) = 89s
        assert report.face_present_s == pytest.approx(89.0)
        # Face was absent 30.0-40.0 (10s)
        assert report.face_absent_s == pytest.approx(10.0)

        # Camera was available for all intervals from 1.0 to 100.0 (99s)
        assert report.camera_available_s == pytest.approx(99.0)
        assert report.camera_unavailable_s == pytest.approx(0.0)
        assert report.camera_permission_denied_s == pytest.approx(0.0)
        assert report.camera_outage_s == pytest.approx(0.0)
        assert report.camera_availability_ratio == pytest.approx(99.0 / 100.0)

        # Detection coverage = face_present + face_absent = 89 + 10 = 99s
        assert report.detection_coverage_s == pytest.approx(99.0)
        assert report.detection_coverage_ratio == pytest.approx(99.0 / 100.0)
        assert report.effective_monitoring_ratio == pytest.approx(99.0 / 100.0)

    def test_camera_unavailable_outage(self):
        """Camera was unplugged or failed mid-session."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"max_confidence": 0.90}),
            _event("CAMERA_UNAVAILABLE", 20.0, details={"reason": "device_busy"}),
            _event("FACE_PRESENT", 50.0, details={"max_confidence": 0.91}),
            _event("SESSION_ENDED", 100.0),
        ])
        report = compute_accuracy(log)

        # Camera available: 1.0 to 20.0 (19s) + 50.0 to 100.0 (50s) = 69s
        assert report.camera_available_s == pytest.approx(69.0)
        # Camera outage: 20.0 to 50.0 (30s)
        assert report.camera_unavailable_s == pytest.approx(30.0)
        assert report.camera_outage_s == pytest.approx(30.0)
        assert report.camera_availability_ratio == pytest.approx(69.0 / 100.0)
        assert report.n_camera_unavailable == 1

        # Detection coverage was only active when camera was streaming
        assert report.detection_coverage_s == pytest.approx(69.0)
        assert report.detection_coverage_ratio == pytest.approx(69.0 / 100.0)

    def test_camera_permission_denied(self):
        """OS blocked camera permission."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("CAMERA_PERMISSION_DENIED", 0.5, details={"platform": "windows"}),
            _event("SESSION_ENDED", 50.0),
        ])
        report = compute_accuracy(log)

        assert report.camera_permission_denied_s == pytest.approx(49.5)
        assert report.camera_available_s == pytest.approx(0.0)
        assert report.camera_availability_ratio == pytest.approx(0.0)
        assert report.detection_coverage_s == pytest.approx(0.0)
        assert report.detection_coverage_ratio == pytest.approx(0.0)
        assert report.n_camera_permission_denied == 1

    def test_detection_unavailable_vs_camera_available(self):
        """Camera was streaming, but room went completely dark (DETECTION_UNAVAILABLE)."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"max_confidence": 0.85}),
            _event("DETECTION_UNAVAILABLE", 20.0, details={"reason": "too_dark"}),
            _event("FACE_PRESENT", 40.0, details={"max_confidence": 0.87}),
            _event("SESSION_ENDED", 60.0),
        ])
        report = compute_accuracy(log)

        # Camera was functional throughout (1.0 to 60.0 = 59s)
        assert report.camera_available_s == pytest.approx(59.0)
        assert report.camera_availability_ratio == pytest.approx(59.0 / 60.0)

        # But detection was blind from 20.0 to 40.0 (20s)
        assert report.detection_unavailable_s == pytest.approx(20.0)
        assert report.detection_unavailable_ratio == pytest.approx(20.0 / 60.0)

        # Active detection coverage = present (19s + 20s) = 39s
        assert report.detection_coverage_s == pytest.approx(39.0)
        assert report.detection_coverage_ratio == pytest.approx(39.0 / 60.0)

    def test_multiple_faces_tracked_in_coverage(self):
        """Multiple faces detected still counts as valid face detection coverage."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0),
            _event("MULTIPLE_FACES", 20.0, details={"n_faces": 2}),
            _event("SESSION_ENDED", 40.0),
        ])
        report = compute_accuracy(log)

        assert report.multiple_faces_s == pytest.approx(20.0)
        assert report.n_multiple_faces_events == 1
        assert report.detection_coverage_s == pytest.approx(39.0)
        assert report.detection_coverage_ratio == pytest.approx(39.0 / 40.0)


# ====================================================================== Structured Summary Exports

class TestStructuredSummaryExports:

    def test_to_dict_structure(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"max_confidence": 0.92}),
            _event("SESSION_ENDED", 50.0),
        ])
        report = compute_accuracy(log)
        d = report.to_dict()

        assert d["session_id"] == "TEST-SESSION-001"
        assert "summary" in d
        assert "timing" in d
        assert "ratios" in d
        assert "counts" in d
        assert "absence" in d
        assert "confidence" in d
        assert "lifecycle" in d
        assert "sufficiency" in d

        assert d["summary"]["total_monitored_s"] == 50.0
        assert d["summary"]["presence_ratio"] == pytest.approx(49.0 / 50.0, abs=1e-3)
        assert d["summary"]["camera_availability_ratio"] == pytest.approx(49.0 / 50.0, abs=1e-3)
        assert d["summary"]["detection_coverage_ratio"] == pytest.approx(49.0 / 50.0, abs=1e-3)
        assert d["summary"]["is_sufficient"] is True

        assert d["lifecycle"]["has_session_start"] is True
        assert d["lifecycle"]["has_session_end"] is True

    def test_to_json_and_export_json(self, tmp_path):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"max_confidence": 0.90}),
            _event("SESSION_ENDED", 60.0),
        ])
        report = compute_accuracy(log)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "TEST-SESSION-001"

        out_file = tmp_path / "summary.json"
        saved = report.export_json(out_file)
        assert saved.exists()
        loaded = json.loads(saved.read_text(encoding="utf-8"))
        assert loaded["session_id"] == "TEST-SESSION-001"

    def test_to_csv_row_and_export_csv(self, tmp_path):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"max_confidence": 0.88}),
            _event("SESSION_ENDED", 60.0),
        ])
        report = compute_accuracy(log)
        row = report.to_csv_row()
        assert row["session_id"] == "TEST-SESSION-001"
        assert "camera_availability_ratio" in row
        assert "detection_coverage_ratio" in row
        assert "is_sufficient" in row

        out_file = tmp_path / "summary.csv"
        saved = report.export_csv(out_file)
        assert saved.exists()
        lines = saved.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2  # header + row
        assert "TEST-SESSION-001" in lines[1]

    def test_batch_export_json_and_csv(self, tmp_path):
        log1 = _write_log([
            _event("SESSION_STARTED", 0.0, session_id="SESSION-01"),
            _event("FACE_PRESENT", 1.0, session_id="SESSION-01"),
            _event("SESSION_ENDED", 45.0, session_id="SESSION-01"),
        ])
        log2 = _write_log([
            _event("SESSION_STARTED", 0.0, session_id="SESSION-02"),
            _event("FACE_PRESENT", 1.0, session_id="SESSION-02"),
            _event("SESSION_ENDED", 65.0, session_id="SESSION-02"),
        ])
        reports = summarize_sessions([log1, log2])
        assert len(reports) == 2

        # Export batch JSON
        json_path = tmp_path / "batch.json"
        export_summaries_json(reports, json_path)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["count"] == 2
        assert len(data["summaries"]) == 2

        # Export batch CSV
        csv_path = tmp_path / "batch.csv"
        export_summaries_csv(reports, csv_path)
        csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_lines) == 3  # header + 2 rows


# ====================================================================== Data Sufficiency Identification

class TestDataSufficiencyIdentification:

    def test_sufficient_session_passes(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0, details={"max_confidence": 0.90}),
            _event("FACE_ABSENT", 40.0),
            _event("FACE_PRESENT", 45.0, details={"max_confidence": 0.92}),
            _event("SESSION_ENDED", 60.0),
        ])
        report = compute_accuracy(log)
        assessment = report.sufficiency()
        assert assessment.is_sufficient is True
        assert len(assessment.reasons) == 0
        assert report.is_sufficient() is True

    def test_identify_session_too_short(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0),
            _event("SESSION_ENDED", 15.0),  # 15s < 30s threshold
        ])
        report = compute_accuracy(log)
        assessment = report.sufficiency(DataSufficiencyCriteria(min_duration_s=30.0))
        assert assessment.is_sufficient is False
        assert InsufficientDataReason.SESSION_TOO_SHORT in assessment.reasons

    def test_identify_low_camera_availability(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0),
            _event("CAMERA_UNAVAILABLE", 10.0),
            _event("SESSION_ENDED", 100.0),  # Camera down 10s to 100s = 90% outage
        ])
        report = compute_accuracy(log)
        assessment = report.sufficiency(DataSufficiencyCriteria(min_camera_availability=0.80))
        assert assessment.is_sufficient is False
        assert InsufficientDataReason.LOW_CAMERA_AVAILABILITY in assessment.reasons

    def test_identify_low_detection_coverage_from_dark_frames(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0),
            _event("DETECTION_UNAVAILABLE", 10.0, details={"reason": "too_dark"}),
            _event("SESSION_ENDED", 100.0),  # Detection blind for 90s
        ])
        report = compute_accuracy(log)
        assessment = report.sufficiency(DataSufficiencyCriteria(min_detection_coverage=0.70))
        assert assessment.is_sufficient is False
        assert InsufficientDataReason.LOW_DETECTION_COVERAGE in assessment.reasons
        assert InsufficientDataReason.EXCESSIVE_DETECTION_FAILURES in assessment.reasons

    def test_identify_missing_session_ended(self):
        """Process was aborted or crashed without clean shutdown."""
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0),
            _event("FACE_PRESENT", 40.0),
            # Missing SESSION_ENDED
        ])
        report = compute_accuracy(log)
        assessment = report.sufficiency(DataSufficiencyCriteria(require_session_end=True))
        assert assessment.is_sufficient is False
        assert InsufficientDataReason.MISSING_SESSION_END in assessment.reasons

    def test_identify_excessive_interruptions(self):
        log = _write_log([
            _event("SESSION_STARTED", 0.0),
            _event("FACE_PRESENT", 1.0),
            _event("SESSION_INTERRUPTED", 10.0),
            _event("SESSION_INTERRUPTED", 20.0),
            _event("SESSION_INTERRUPTED", 30.0),
            _event("SESSION_INTERRUPTED", 40.0),
            _event("SESSION_ENDED", 100.0),
        ])
        report = compute_accuracy(log)
        assessment = report.sufficiency(DataSufficiencyCriteria(max_interruptions=3))
        assert assessment.is_sufficient is False
        assert InsufficientDataReason.EXCESSIVE_INTERRUPTIONS in assessment.reasons

    def test_filter_and_identify_insufficient_sessions(self):
        good_log = _write_log([
            _event("SESSION_STARTED", 0.0, session_id="GOOD-01"),
            _event("FACE_PRESENT", 1.0, session_id="GOOD-01"),
            _event("SESSION_ENDED", 50.0, session_id="GOOD-01"),
        ])
        short_log = _write_log([
            _event("SESSION_STARTED", 0.0, session_id="BAD-SHORT"),
            _event("FACE_PRESENT", 1.0, session_id="BAD-SHORT"),
            _event("SESSION_ENDED", 10.0, session_id="BAD-SHORT"),
        ])
        reports = summarize_sessions([good_log, short_log])

        insufficient_assessments = identify_insufficient_sessions(reports)
        assert len(insufficient_assessments) == 1
        assert insufficient_assessments[0].session_id == "BAD-SHORT"

        sufficient, insufficient = filter_sessions_by_sufficiency(reports)
        assert len(sufficient) == 1
        assert sufficient[0].session_id == "GOOD-01"
        assert len(insufficient) == 1
        assert insufficient[0].session_id == "BAD-SHORT"


# ====================================================================== Database Integration

class TestDatabaseIntegrationSummaries:

    def test_compute_accuracy_from_db_mock(self):
        db_mock = MagicMock()
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        db_mock.get_connection.return_value = conn_mock
        conn_mock.cursor.return_value = cursor_mock

        # Return mock events: (type, ts_mono, ts_utc, frame_id, details)
        cursor_mock.fetchall.return_value = [
            ("SESSION_STARTED", 0.0, "2026-11-20T10:00:00.000+00:00", 0, "{}"),
            ("FACE_PRESENT", 1.0, "2026-11-20T10:00:01.000+00:00", 30, '{"max_confidence": 0.94}'),
            ("SESSION_ENDED", 60.0, "2026-11-20T10:01:00.000+00:00", 1800, '{"final_state": "FACE_PRESENT"}'),
        ]

        report = compute_accuracy_from_db("MOCK-DB-SESSION-01", db_mock)
        assert report.session_id == "MOCK-DB-SESSION-01"
        assert report.total_monitored_s == pytest.approx(60.0)
        assert report.face_present_s == pytest.approx(59.0)
        assert report.camera_availability_ratio == pytest.approx(59.0 / 60.0)
        assert report.detection_coverage_ratio == pytest.approx(59.0 / 60.0)
        assert report.is_sufficient() is True

    def test_export_db_summaries_mock(self, tmp_path):
        db_mock = MagicMock()
        conn_mock = MagicMock()
        cursor_mock = MagicMock()
        db_mock.get_connection.return_value = conn_mock
        conn_mock.cursor.return_value = cursor_mock

        # 1st call for DISTINCT session_id
        cursor_mock.fetchall.side_effect = [
            [("SESSION-DB-01",)],
            [  # 2nd call for fetch_session_events
                ("SESSION_STARTED", 0.0, "2026-11-20T10:00:00.000+00:00", 0, "{}"),
                ("FACE_PRESENT", 1.0, "2026-11-20T10:00:01.000+00:00", 30, '{"max_confidence": 0.89}'),
                ("SESSION_ENDED", 40.0, "2026-11-20T10:00:40.000+00:00", 1200, "{}"),
            ]
        ]

        json_out = tmp_path / "db_export.json"
        saved = export_db_summaries(db_mock, json_out, fmt="json")
        assert saved.exists()
        data = json.loads(saved.read_text(encoding="utf-8"))
        assert data["count"] == 1
        assert data["summaries"][0]["session_id"] == "SESSION-DB-01"
