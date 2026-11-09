"""Resource cleanup: camera released, log closed, threads ended, nothing leaks across sessions."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

import exam_monitor.monitoring_service as ms
from exam_monitor.camera import Camera
from exam_monitor.config import Config
from exam_monitor.events import EventType as E, JsonlSink, ListSink
from exam_monitor.monitoring_service import MonitoringService
from exam_monitor.session_binding import MonitoringRegistry

ROOT = Path(__file__).resolve().parent.parent
FACE = str(ROOT / "sample_images" / "astronaut_1face.png")
SID = "exam-sess-0001"
FAST = Config(hold_face_present_s=0.1, stall_s=3.0)


def wait_for(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def monitor_threads():
    return [t for t in threading.enumerate() if t.name.startswith("monitor-")]


def make(tmp_path, cfg=FAST, detector_factory=None):
    sinks = []

    def factory(sid):
        s = JsonlSink(tmp_path / f"events_{sid}.jsonl")
        sinks.append(s)
        return s

    reg = MonitoringRegistry(cfg, factory)
    kw = {"detector_factory": detector_factory} if detector_factory else {}
    return MonitoringService(cfg, reg, **kw), sinks


class TrackedCamera(Camera):
    released = 0

    def release(self):
        TrackedCamera.released += 1
        super().release()


def test_stop_closes_log_ends_thread_and_releases_camera(tmp_path, monkeypatch):
    TrackedCamera.released = 0
    monkeypatch.setattr(ms, "Camera", TrackedCamera)
    service, sinks = make(tmp_path)
    service.start_monitoring(SID, FACE)
    assert wait_for(lambda: service.get_status(SID).state == "FACE_PRESENT")
    service.stop_monitoring(SID)
    assert sinks[0].closed
    assert monitor_threads() == []
    assert TrackedCamera.released >= 1
    last = json.loads((tmp_path / f"events_{SID}.jsonl").read_text().splitlines()[-1])
    assert last["type"] == "SESSION_ENDED"


def test_context_manager_cleans_up_even_when_body_raises(tmp_path):
    service, sinks = make(tmp_path)
    with pytest.raises(RuntimeError):
        with service:
            service.start_monitoring(SID, FACE)
            raise RuntimeError("boom in the caller")
    assert sinks[0].closed and monitor_threads() == []


def test_worker_dying_on_a_sink_error_still_releases_the_camera(tmp_path, monkeypatch):
    TrackedCamera.released = 0
    monkeypatch.setattr(ms, "Camera", TrackedCamera)

    class FlakySink(ListSink):
        def write(self, event):
            if event.type == E.FACE_PRESENT:
                raise OSError("disk full")
            super().write(event)

    reg = MonitoringRegistry(FAST, lambda sid: FlakySink())
    service = MonitoringService(FAST, reg)
    service.start_monitoring(SID, FACE)
    assert wait_for(lambda: not service.is_running(SID))          # worker died by itself
    res = service.stop_monitoring(SID)
    assert "disk full" in (res.worker_error or "")
    assert TrackedCamera.released >= 1 and monitor_threads() == []


def test_detector_close_is_called(tmp_path):
    closed = threading.Event()

    class Det:
        name, last_timing = "fake", {}

        def detect(self, bgr):
            return []

        def close(self):
            closed.set()

    service, _ = make(tmp_path, detector_factory=lambda n, c: Det())
    service.start_monitoring(SID, FACE)
    time.sleep(0.2)
    service.stop_monitoring(SID)
    assert closed.is_set()


def test_a_failing_detector_close_does_not_break_stop(tmp_path):
    class Det:
        name, last_timing = "fake", {}

        def detect(self, bgr):
            return []

        def close(self):
            raise RuntimeError("close exploded")

    service, _ = make(tmp_path, detector_factory=lambda n, c: Det())
    service.start_monitoring(SID, FACE)
    time.sleep(0.2)
    res = service.stop_monitoring(SID)
    assert res.worker_stopped_cleanly and monitor_threads() == []


def test_jsonl_sink_and_camera_context_managers(tmp_path):
    with JsonlSink(tmp_path / "a.jsonl") as sink:
        assert not sink.closed
    assert sink.closed
    sink.close()                                   # idempotent
    with Camera(FACE, Config()) as cam:
        assert cam.is_open
    assert not cam.is_open


@pytest.mark.skipif(not hasattr(psutil.Process, "num_fds"), reason="fd counting is POSIX-only")
def test_repeated_sessions_do_not_leak_threads_or_file_descriptors(tmp_path):
    service, sinks = make(tmp_path)
    proc = psutil.Process()
    # warm-up cycle so one-off allocations (imports, thread pools) do not count as a leak
    service.start_monitoring("warmup-session-00", FACE)
    time.sleep(0.15)
    service.stop_monitoring("warmup-session-00")
    fds_before, threads_before = proc.num_fds(), threading.active_count()

    for i in range(25):
        sid = f"cycle-session-{i:02d}"
        service.start_monitoring(sid, FACE, publish_frames=True)
        time.sleep(0.08)
        service.stop_monitoring(sid)

    assert all(s.closed for s in sinks)
    assert threading.active_count() == threads_before
    assert proc.num_fds() <= fds_before + 1        # no growth per cycle (25 cycles would show as +25)


def test_normal_interpreter_exit_without_stop_still_ends_the_session(tmp_path):
    """atexit safety net: the caller forgot stop_monitoring()."""
    script = f"""
import time
from exam_monitor.config import Config
from exam_monitor.events import JsonlSink
from exam_monitor.monitoring_service import MonitoringService
from exam_monitor.session_binding import MonitoringRegistry
cfg = Config(hold_face_present_s=0.1)
reg = MonitoringRegistry(cfg, lambda sid: JsonlSink(r"{tmp_path}/events_{SID}.jsonl"))
svc = MonitoringService(cfg, reg)
svc.start_monitoring("{SID}", r"{FACE}")
time.sleep(0.8)
# no stop_monitoring(): the interpreter just exits
"""
    r = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(l) for l in (tmp_path / f"events_{SID}.jsonl").read_text().splitlines()]
    assert lines[0]["type"] == "SESSION_STARTED" and lines[-1]["type"] == "SESSION_ENDED"


@pytest.mark.skipif(sys.platform.startswith("win"), reason="SIGTERM semantics differ on Windows")
def test_sigterm_ends_session_and_closes_log(tmp_path):
    p = subprocess.Popen(
        [sys.executable, "preview.py", "--session-id", SID, "--source", FACE, "--no-gui",
         "--log-dir", str(tmp_path)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    log = tmp_path / f"events_{SID}.jsonl"
    assert wait_for(lambda: log.exists() and "FACE_PRESENT" in log.read_text(), timeout=20)
    p.send_signal(signal.SIGTERM)
    out, err = p.communicate(timeout=20)
    assert p.returncode == 0, err
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert lines[-1]["type"] == "SESSION_ENDED"
    assert "final status" in out


def test_ctrl_c_equivalent_sigint_ends_session(tmp_path):
    if sys.platform.startswith("win"):
        pytest.skip("SIGINT delivery to a child differs on Windows")
    p = subprocess.Popen(
        [sys.executable, "preview.py", "--session-id", SID, "--source", FACE, "--no-gui",
         "--log-dir", str(tmp_path)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    log = tmp_path / f"events_{SID}.jsonl"
    assert wait_for(lambda: log.exists() and "FACE_PRESENT" in log.read_text(), timeout=20)
    p.send_signal(signal.SIGINT)
    p.communicate(timeout=20)
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert lines[-1]["type"] == "SESSION_ENDED"
