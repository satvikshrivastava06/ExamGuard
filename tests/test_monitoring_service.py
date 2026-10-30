import threading
import time
from pathlib import Path

import pytest

from exam_monitor.config import Config
from exam_monitor.events import EventType as E, ListSink
from exam_monitor.monitoring_service import MonitoringService, MonitoringStartError
from exam_monitor.session_binding import (
    InvalidSessionId, MonitoringRegistry, SessionAlreadyMonitored, SessionNotMonitored, UnknownSession,
)

ROOT = Path(__file__).resolve().parent.parent
FACE = str(ROOT / "sample_images" / "astronaut_1face.png")
NOFACE = str(ROOT / "sample_images" / "coffee_0face.png")
SID_A, SID_B = "exam-sess-0001", "exam-sess-0002"

FAST = Config(hold_face_present_s=0.1, hold_face_absent_s=0.2, hold_multiple_faces_s=0.1,
              hold_detection_failure_s=0.1, hold_camera_unavailable_s=0.2, stall_s=3.0)


def make(validator=None, detector_factory=None):
    sinks: dict[str, ListSink] = {}

    def factory(sid):
        sinks[sid] = ListSink()
        return sinks[sid]

    reg = MonitoringRegistry(FAST, factory, validator)
    kwargs = {"detector_factory": detector_factory} if detector_factory else {}
    return MonitoringService(FAST, reg, **kwargs), reg, sinks


def wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def monitor_threads():
    return [t for t in threading.enumerate() if t.name.startswith("monitor-")]


@pytest.fixture
def svc():
    service, reg, sinks = make()
    yield service, reg, sinks
    service.stop_all()


def test_start_binds_status_to_session_and_reaches_face_present(svc):
    service, reg, sinks = svc
    st = service.start_monitoring(SID_A, FACE)
    assert st.session_id == SID_A and st.active
    assert wait_for(lambda: service.get_status(SID_A).state == "FACE_PRESENT")
    assert {e.session_id for e in sinks[SID_A].events} == {SID_A}
    assert service.is_running(SID_A)


def test_stop_emits_session_ended_last_and_frees_everything(svc):
    service, reg, sinks = svc
    service.start_monitoring(SID_A, FACE)
    assert wait_for(lambda: service.get_status(SID_A).state == "FACE_PRESENT")
    res = service.stop_monitoring(SID_A)
    assert res.worker_stopped_cleanly and res.worker_error is None
    assert res.status.active is False and res.status.state == "FACE_PRESENT"
    n = len(sinks[SID_A].events)
    assert sinks[SID_A].events[-1].type == E.SESSION_ENDED
    time.sleep(0.3)
    assert len(sinks[SID_A].events) == n                  # nothing written after SESSION_ENDED
    assert monitor_threads() == []
    with pytest.raises(SessionNotMonitored):
        service.get_status(SID_A)
    with pytest.raises(SessionNotMonitored):
        service.stop_monitoring(SID_A)                    # second stop
    service.start_monitoring(SID_A, FACE)                 # ID reusable after stop


def test_double_start_rejected_and_only_one_worker(svc):
    service, *_ = svc
    service.start_monitoring(SID_A, FACE)
    with pytest.raises(SessionAlreadyMonitored):
        service.start_monitoring(SID_A, FACE)
    assert len(monitor_threads()) == 1


@pytest.mark.parametrize("bad", ["", "../../etc/passwd", "short", None])
def test_invalid_session_id_starts_nothing(svc, bad):
    service, reg, sinks = svc
    with pytest.raises(InvalidSessionId):
        service.start_monitoring(bad, FACE)
    assert sinks == {} and monitor_threads() == []


def test_validator_rejection_starts_nothing():
    service, reg, sinks = make(validator=lambda sid: False)
    with pytest.raises(UnknownSession):
        service.start_monitoring(SID_A, FACE)
    assert sinks == {} and monitor_threads() == []


def test_bad_detector_leaves_no_session_behind(svc):
    service, reg, sinks = svc
    with pytest.raises(MonitoringStartError):
        service.start_monitoring(SID_A, FACE, detector="nope")
    assert sinks == {} and monitor_threads() == []        # failed before bind
    with pytest.raises(SessionNotMonitored):
        service.get_status(SID_A)


def test_worker_setup_failure_rolls_back_bound_session(svc, monkeypatch):
    service, reg, sinks = svc
    import exam_monitor.monitoring_service as ms

    def boom(*a, **k):
        raise RuntimeError("camera exploded")

    monkeypatch.setattr(ms, "Camera", boom)
    with pytest.raises(MonitoringStartError):
        service.start_monitoring(SID_A, FACE)
    assert sinks[SID_A].events[-1].type == E.SESSION_ENDED   # audit trail shows the failed attempt
    with pytest.raises(SessionNotMonitored):
        service.get_status(SID_A)
    monkeypatch.undo()
    service.start_monitoring(SID_A, FACE)                    # and the ID is free again


def test_two_sessions_are_independent(svc):
    service, reg, sinks = svc
    service.start_monitoring(SID_A, FACE)
    service.start_monitoring(SID_B, NOFACE)
    assert wait_for(lambda: service.get_status(SID_A).state == "FACE_PRESENT"
                    and service.get_status(SID_B).state == "FACE_ABSENT")
    service.stop_monitoring(SID_A)
    assert service.get_status(SID_B).active                  # B unaffected by A stopping
    assert {e.session_id for e in sinks[SID_A].events} == {SID_A}
    assert {e.session_id for e in sinks[SID_B].events} == {SID_B}


def test_unavailable_camera_surfaces_as_event_and_stop_is_prompt(svc):
    service, reg, sinks = svc
    service.start_monitoring(SID_A, "does_not_exist.mp4")
    assert wait_for(lambda: service.get_status(SID_A).state == "CAMERA_UNAVAILABLE")
    t0 = time.monotonic()
    res = service.stop_monitoring(SID_A)
    assert time.monotonic() - t0 < 2.0 and res.worker_stopped_cleanly


def test_detector_exception_becomes_detection_failure_and_worker_survives():
    class Exploding:
        name = "exploding"
        last_timing: dict = {}

        def detect(self, bgr):
            raise ValueError("model crashed")

    service, reg, sinks = make(detector_factory=lambda name, cfg: Exploding())
    try:
        service.start_monitoring(SID_A, FACE)
        assert wait_for(lambda: service.get_status(SID_A).state == "DETECTION_FAILURE")
        assert service.is_running(SID_A)
    finally:
        service.stop_all()


def test_snapshot_frames_only_when_requested_and_read_only(svc):
    service, *_ = svc
    service.start_monitoring(SID_A, FACE, publish_frames=False)
    service.start_monitoring(SID_B, FACE, publish_frames=True)
    assert wait_for(lambda: service.snapshot(SID_A).frame_id >= 0 and service.snapshot(SID_B).frame is not None)
    assert service.snapshot(SID_A).frame is None
    frame = service.snapshot(SID_B).frame
    with pytest.raises(ValueError):
        frame[0, 0, 0] = 1                                    # writeable=False


def test_stop_all_stops_everything(svc):
    service, *_ = svc
    service.start_monitoring(SID_A, FACE)
    service.start_monitoring(SID_B, FACE)
    results = service.stop_all()
    assert set(results) == {SID_A, SID_B} and monitor_threads() == []


def test_hung_worker_does_not_block_release(svc, monkeypatch):
    """Simulate a camera read that never returns: stop must give up after the timeout, still
    emit SESSION_ENDED, and report worker_stopped_cleanly=False."""
    service, reg, sinks = svc
    import exam_monitor.camera as camera_mod
    release_hang = threading.Event()
    orig_read = camera_mod.Camera.read

    def hanging_read(self):
        release_hang.wait(5)
        return orig_read(self)

    monkeypatch.setattr(camera_mod.Camera, "read", hanging_read)
    service.start_monitoring(SID_A, FACE)
    time.sleep(0.1)
    res = service.stop_monitoring(SID_A, timeout=0.3)
    assert res.worker_stopped_cleanly is False
    assert sinks[SID_A].events[-1].type == E.SESSION_ENDED
    release_hang.set()                                        # let the stuck thread finish
    assert wait_for(lambda: monitor_threads() == [])          # and it exits on its own


# ====================================================================== camera problems (Sprint 2: webcam unavailable / permission denied)
import numpy as np

import exam_monitor.monitoring_service as ms
from exam_monitor.camera_probe import OsCheck, PermissionState as S
from exam_monitor.monitoring_service import CameraCheckFailed

DIAG = Config(hold_face_present_s=0.1, hold_face_absent_s=0.2, hold_multiple_faces_s=0.1,
              hold_detection_failure_s=0.1, hold_camera_unavailable_s=0.2,
              hold_camera_permission_denied_s=0.1, camera_diag_interval_s=0.05, stall_s=3.0)
NO_SUCH_CAMERA = 99


def make_diag():
    sinks: dict[str, ListSink] = {}

    def factory(sid):
        sinks[sid] = ListSink()
        return sinks[sid]

    reg_ = MonitoringRegistry(DIAG, factory)
    return MonitoringService(DIAG, reg_), sinks


def events_of(sinks, sid, etype):
    return [e for e in sinks[sid].events if e.type == etype]


def test_no_device_reports_camera_unavailable_with_diagnosis(monkeypatch):
    monkeypatch.setattr(ms, "inspect_os", lambda src, *a, **k: OsCheck(S.UNKNOWN, False, "/dev/video99 does not exist"))
    service, sinks = make_diag()
    try:
        service.start_monitoring(SID_A, NO_SUCH_CAMERA)
        assert wait_for(lambda: service.get_status(SID_A).state == "CAMERA_UNAVAILABLE")
        d = events_of(sinks, SID_A, E.CAMERA_UNAVAILABLE)[0].details
        assert d["problem"] == "no_device" and d["hint"] and d["detail"]
    finally:
        service.stop_all()


def test_permission_denied_reports_distinct_event(monkeypatch):
    monkeypatch.setattr(ms, "inspect_os", lambda src, *a, **k: OsCheck(S.DENIED, True, "no read/write access to /dev/video99"))
    service, sinks = make_diag()
    try:
        service.start_monitoring(SID_A, NO_SUCH_CAMERA)
        assert wait_for(lambda: service.get_status(SID_A).state == "CAMERA_PERMISSION_DENIED")
        d = events_of(sinks, SID_A, E.CAMERA_PERMISSION_DENIED)[0].details
        assert d["problem"] == "permission_denied" and d["permission"] == "denied"
        assert "video" in d["hint"] or "Settings" in d["hint"]
        assert events_of(sinks, SID_A, E.CAMERA_UNAVAILABLE) == []
    finally:
        service.stop_all()


def test_state_follows_os_permission_when_it_changes(monkeypatch):
    state = {"v": OsCheck(S.DENIED, True, "denied")}
    monkeypatch.setattr(ms, "inspect_os", lambda src, *a, **k: state["v"])
    service, sinks = make_diag()
    try:
        service.start_monitoring(SID_A, NO_SUCH_CAMERA)
        assert wait_for(lambda: service.get_status(SID_A).state == "CAMERA_PERMISSION_DENIED")
        state["v"] = OsCheck(S.GRANTED, True, "granted")      # user fixed the setting; camera still missing
        assert wait_for(lambda: service.get_status(SID_A).state == "CAMERA_UNAVAILABLE")
        assert service.is_running(SID_A)                      # worker survived both
    finally:
        service.stop_all()


class BlackWebcam:
    """Stands in for macOS-style denial: opens fine, streams only black."""
    is_webcam = True
    consecutive_failures = 0
    is_open = True

    def __init__(self, *a, **k):
        pass

    def read(self):
        return True, np.zeros((480, 640, 3), np.uint8)

    def maybe_reopen(self, now):
        return False

    def release(self):
        pass


def test_black_frames_become_permission_denied_only_when_os_blocks(monkeypatch):
    monkeypatch.setattr(ms, "Camera", BlackWebcam)
    monkeypatch.setattr(ms, "inspect_os", lambda src, *a, **k: OsCheck(S.NOT_DETERMINED, None, "not granted yet"))
    service, sinks = make_diag()
    try:
        service.start_monitoring(SID_A, 0)
        assert wait_for(lambda: service.get_status(SID_A).state == "CAMERA_PERMISSION_DENIED")
        assert events_of(sinks, SID_A, E.DETECTION_FAILURE) == []
    finally:
        service.stop_all()


def test_black_frames_with_permission_granted_stay_detection_failure(monkeypatch):
    monkeypatch.setattr(ms, "Camera", BlackWebcam)
    monkeypatch.setattr(ms, "inspect_os", lambda src, *a, **k: OsCheck(S.GRANTED, None, "ok"))
    service, sinks = make_diag()
    try:
        service.start_monitoring(SID_A, 0)
        assert wait_for(lambda: service.get_status(SID_A).state == "DETECTION_FAILURE")   # lens cover / dark room
        assert events_of(sinks, SID_A, E.CAMERA_PERMISSION_DENIED) == []
    finally:
        service.stop_all()


def test_require_camera_blocks_start_and_binds_nothing(svc):
    service, reg, sinks = svc
    with pytest.raises(CameraCheckFailed) as ei:
        service.start_monitoring(SID_A, "does_not_exist.mp4", require_camera=True)
    assert ei.value.check.problem.value == "no_device" and ei.value.check.hint
    assert sinks == {} and monitor_threads() == []             # no session record, no thread
    with pytest.raises(SessionNotMonitored):
        service.get_status(SID_A)


def test_require_camera_passes_with_working_source(svc):
    service, reg, sinks = svc
    st = service.start_monitoring(SID_A, FACE, require_camera=True)
    assert st.active and wait_for(lambda: service.get_status(SID_A).state == "FACE_PRESENT")


def test_default_start_does_not_require_camera(svc):
    service, reg, sinks = svc
    assert service.start_monitoring(SID_A, "does_not_exist.mp4").active     # session exists; events will report the problem


def test_preflight_camera_does_not_create_a_session(svc):
    service, reg, sinks = svc
    assert service.preflight_camera(FACE).ok
    assert not service.preflight_camera("does_not_exist.mp4").ok
    assert sinks == {} and monitor_threads() == []
