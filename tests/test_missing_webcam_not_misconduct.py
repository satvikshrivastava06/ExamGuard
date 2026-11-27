"""Verify that missing webcam data is NEVER treated as confirmed misconduct."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from exam_monitor import (
    InsufficientDataReason,
    UncertaintyLevel,
    compute_accuracy_from_events,
)
from exam_monitor.config import Config
from exam_monitor.absence import AbsenceTracker
from exam_monitor.camera_probe import OsCheck, PermissionState
from exam_monitor.events import EventType, IntegrityStateMachine, ListSink
from exam_monitor.monitoring_service import MonitoringService
from exam_monitor.session_binding import MonitoringRegistry


def _ev(etype: str, ts_mono: float, session_id: str = "NO-CAM-01", details: dict | None = None) -> dict:
    return {
        "type": etype,
        "session_id": session_id,
        "ts_utc": "2026-11-27T10:00:00.000+00:00",
        "ts_mono": ts_mono,
        "frame_id": None,
        "details": details or {},
    }


class _FailingCamera:
    """Mock camera that never delivers a frame (simulates missing/disconnected webcam)."""

    is_webcam = True
    file_fps = 0.0

    def __init__(self, source=0, cfg=None, loop=False):
        self.source = source

    def read(self):
        return False, None

    def maybe_reopen(self, now: float) -> None:
        pass

    def release(self) -> None:
        pass


class TestMissingWebcamNotMisconduct:
    def test_100_percent_camera_unavailable_is_not_absence_or_misconduct(self):
        events = [
            _ev("SESSION_STARTED", 0.0),
            _ev("CAMERA_UNAVAILABLE", 2.0, details={"problem": "NO_DEVICE"}),
            _ev("SESSION_ENDED", 60.0, details={"final_state": "CAMERA_UNAVAILABLE"}),
        ]
        report = compute_accuracy_from_events("NO-CAM-01", events)
        suff = report.sufficiency()
        unc = report.uncertainty(sufficiency=suff)

        assert report.face_absent_s == pytest.approx(0.0)
        assert report.multiple_faces_s == pytest.approx(0.0)
        assert report.camera_unavailable_s == pytest.approx(58.0)
        assert report.camera_availability_ratio == pytest.approx(0.0)

        assert suff.is_sufficient is False
        assert InsufficientDataReason.LOW_CAMERA_AVAILABILITY in suff.reasons

        assert unc.uncertainty_level == UncertaintyLevel.CRITICAL
        assert any(i.code == "MISSING_WEBCAM_DATA" for i in unc.indicators)
        assert unc.misconduct_evaluation is not None
        assert unc.misconduct_evaluation.confirmed_absence_s == pytest.approx(0.0)
        assert unc.misconduct_evaluation.missing_webcam_treated_as_misconduct is False
        assert unc.misconduct_evaluation.reliable_for_misconduct_decision is False

        agg = report.aggregate_intervals()
        assert agg["by_state"]["CAMERA_UNAVAILABLE"]["is_uncertain"] is True
        assert agg["by_state"]["CAMERA_UNAVAILABLE"]["is_misconduct_signal"] is False

    def test_permission_denied_is_not_absence_or_misconduct(self):
        events = [
            _ev("SESSION_STARTED", 0.0),
            _ev("CAMERA_PERMISSION_DENIED", 0.5, details={"problem": "PERMISSION_DENIED"}),
            _ev("SESSION_ENDED", 45.0, details={"final_state": "CAMERA_PERMISSION_DENIED"}),
        ]
        report = compute_accuracy_from_events("PERM-DENIED-01", events)
        unc = report.uncertainty()

        assert report.face_absent_s == pytest.approx(0.0)
        assert report.camera_permission_denied_s == pytest.approx(44.5)
        assert unc.uncertainty_level == UncertaintyLevel.CRITICAL
        assert unc.misconduct_evaluation is not None
        assert unc.misconduct_evaluation.confirmed_absence_s == pytest.approx(0.0)
        assert unc.misconduct_evaluation.missing_webcam_treated_as_misconduct is False
        assert unc.misconduct_evaluation.reliable_for_misconduct_decision is False

    def test_absence_tracker_stops_at_camera_drop(self):
        cfg = Config(hold_face_absent_s=2.0, prolonged_absence_s=10.0)
        sink = ListSink()
        sm = IntegrityStateMachine(cfg, "DROP-MID-ABSENCE", sink)
        tracker = AbsenceTracker(
            cfg.prolonged_absence_s,
            wall_clock=lambda: datetime(2026, 11, 27, 10, 0, 0, tzinfo=timezone.utc),
        )

        sm.start(0.0)
        # 5 seconds of confirmed absence, then 30 seconds of camera unavailable
        for t in (1.0, 2.0, 3.0, 4.0, 5.0):
            evs = sm.observe(EventType.FACE_ABSENT, t, frame_id=int(t))
            for ev in evs:
                if ev.type == EventType.FACE_ABSENT:
                    tracker.enter(sm.last_transition_started, t)

        assert tracker.active is True

        for t in (6.0, 7.0, 8.0, 35.0):
            evs = sm.observe(EventType.CAMERA_UNAVAILABLE, t, frame_id=int(t))
            for ev in evs:
                if ev.type == EventType.CAMERA_UNAVAILABLE:
                    ep = tracker.leave(sm.last_transition_started, "camera_unavailable")
                    assert ep is None  # 5s < 10s prolonged threshold; outage did NOT inflate absence

        assert tracker.active is False
        assert all(ev.type != EventType.PROLONGED_ABSENCE for ev in sink.events)

    def test_live_service_with_failing_camera_emits_no_misconduct(self):
        cfg = Config(
            hold_camera_unavailable_s=0.05,
            hold_camera_permission_denied_s=0.05,
            camera_diag_interval_s=0.05,
        )
        sink = ListSink()
        registry = MonitoringRegistry(cfg, lambda sid: sink)
        svc = MonitoringService(
            cfg,
            registry=registry,
            detector_factory=lambda name, c: object(),  # never called because cam.read() returns False
        )
        with (
            patch("exam_monitor.monitoring_service.Camera", _FailingCamera),
            patch(
                "exam_monitor.monitoring_service.inspect_os",
                return_value=OsCheck(PermissionState.GRANTED, False, "no device"),
            ),
        ):
            svc.start_monitoring("NO-WEBCAM-LIVE")
            import time
            time.sleep(0.25)
            svc.stop_monitoring("NO-WEBCAM-LIVE")

        types = [ev.type for ev in sink.events]
        assert EventType.CAMERA_UNAVAILABLE in types
        assert EventType.FACE_ABSENT not in types
        assert EventType.MULTIPLE_FACES not in types
        assert EventType.PROLONGED_ABSENCE not in types

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
        report = compute_accuracy_from_events("NO-WEBCAM-LIVE", event_dicts)
        unc = report.uncertainty()
        assert unc.misconduct_evaluation is not None
        assert unc.misconduct_evaluation.confirmed_absence_s == pytest.approx(0.0)
        assert unc.misconduct_evaluation.missing_webcam_treated_as_misconduct is False
        assert unc.misconduct_evaluation.reliable_for_misconduct_decision is False
