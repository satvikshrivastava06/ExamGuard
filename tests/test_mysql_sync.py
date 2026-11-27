"""Unit tests for MySQL sync helpers, datetime normalization, buffered detections, and summary storage."""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

from exam_monitor.accuracy_report import AccuracyReport
from exam_monitor.config import Config
from exam_monitor.database import (
    BufferedDetectionStore,
    DatabaseSink,
    DetectionStore,
    SummaryStore,
    _to_mysql_datetime,
)
from exam_monitor.detector import Detection
from exam_monitor.events import Event, EventType, ListSink
from exam_monitor.monitoring_service import MonitoringService
from exam_monitor.session_binding import MonitoringRegistry


def test_to_mysql_datetime_normalizes_iso_utc():
    assert _to_mysql_datetime("2026-11-27T13:25:00.123+00:00") == "2026-11-27 13:25:00.123000"
    assert _to_mysql_datetime("2026-11-27T13:25:00.500Z") == "2026-11-27 13:25:00.500000"
    assert _to_mysql_datetime("2026-11-03T18:13:11.658+05:30") == "2026-11-03 12:43:11.658000"
    assert _to_mysql_datetime(None) is None


def test_database_sink_writes_normalized_datetime():
    db_mock = MagicMock()
    conn_mock = MagicMock()
    cursor_mock = MagicMock()
    db_mock.get_connection.return_value = conn_mock
    conn_mock.cursor.return_value = cursor_mock

    sink = DatabaseSink(db_mock)
    ev = Event(
        type=EventType.SESSION_STARTED,
        session_id="TEST-MYSQL-01",
        ts_utc="2026-11-27T13:25:00.123+00:00",
        ts_mono=10.0,
        frame_id=None,
        details={},
    )
    sink.write(ev)

    assert cursor_mock.execute.call_count == 2
    first_params = cursor_mock.execute.call_args_list[0][0][1]
    second_params = cursor_mock.execute.call_args_list[1][0][1]
    assert first_params == ("TEST-MYSQL-01", "2026-11-27 13:25:00.123000")
    assert second_params[2] == "2026-11-27 13:25:00.123000"
    conn_mock.commit.assert_called_once()


def test_summary_store_upserts_confidence_and_sufficiency():
    db_mock = MagicMock()
    conn_mock = MagicMock()
    cursor_mock = MagicMock()
    db_mock.get_connection.return_value = conn_mock
    conn_mock.cursor.return_value = cursor_mock

    report = AccuracyReport(
        session_id="TEST-SUMMARY-01",
        total_monitored_s=60.0,
        face_present_s=55.0,
        camera_available_s=59.0,
        n_events=3,
        has_session_start=True,
        has_session_end=True,
        start_time_utc="2026-11-27T10:00:00.000+00:00",
        end_time_utc="2026-11-27T10:01:00.000+00:00",
        final_state="FACE_PRESENT",
        confidence_samples=[0.85, 0.90, 0.95],
    )
    SummaryStore(db_mock).save_summary(report)

    assert cursor_mock.execute.call_count == 2
    summary_params = cursor_mock.execute.call_args_list[1][0][1]
    assert summary_params[0] == "TEST-SUMMARY-01"
    assert summary_params[10] == 0.9  # avg_confidence
    assert summary_params[14] == 3    # confidence_samples
    assert summary_params[20] is True # is_sufficient
    parsed = json.loads(summary_params[22])
    assert parsed["confidence"]["avg"] == 0.9
    conn_mock.commit.assert_called_once()


def test_buffered_detection_store_flushes_on_close():
    db_mock = MagicMock()
    conn_mock = MagicMock()
    cursor_mock = MagicMock()
    db_mock.get_connection.return_value = conn_mock
    conn_mock.cursor.return_value = cursor_mock

    writer = BufferedDetectionStore(db_mock, batch_size=10)
    writer.enqueue("TEST-DET-01", 1, (Detection(box=(10, 20, 30, 40), confidence=0.91),))
    writer.enqueue("TEST-DET-01", 2, (Detection(box=(12, 22, 30, 40), confidence=0.95),))
    writer.close(timeout=2.0)

    assert cursor_mock.executemany.called
    inserted_rows = cursor_mock.executemany.call_args[0][1]
    assert len(inserted_rows) == 2
    assert inserted_rows[0][-1] == 0.91
    assert inserted_rows[1][-1] == 0.95


def test_monitoring_service_on_detections_callback():
    class _FakeDetector:
        name = "fake"

        def detect(self, frame):
            return [Detection(box=(10, 10, 50, 50), confidence=0.93)]

    cfg = Config(process_every_n=1)
    registry = MonitoringRegistry(cfg, lambda sid: ListSink())
    received = []
    svc = MonitoringService(cfg, registry, detector_factory=lambda name, c: _FakeDetector())
    svc.start_monitoring(
        "TEST-SVC-DET-01",
        source="sample_images/astronaut_1face.png",
        detector="fake",
        publish_frames=False,
        on_detections=lambda sid, fid, dets: received.append((sid, fid, dets)),
    )
    time.sleep(0.3)
    svc.stop_monitoring("TEST-SVC-DET-01")

    assert len(received) > 0
    sid, _, dets = received[0]
    assert sid == "TEST-SVC-DET-01"
    assert dets[0].confidence == 0.93
