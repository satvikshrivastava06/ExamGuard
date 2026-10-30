"""start_monitoring / stop_monitoring, wired to the session registry.

Lifecycle for one session ID:

    start_monitoring(sid)                      stop_monitoring(sid)
      validate id (format)                       pop worker from the table
      build detector      (no side effects)      signal worker, join (bounded)
      registry.bind(sid)  -> SESSION_STARTED     registry.release(sid) -> SESSION_ENDED
      open camera                                (worker closes the camera itself)
      start worker thread

Rules this module enforces:
  * The worker thread is the ONLY thing that touches the camera. cv2.VideoCapture is not
    thread-safe and read() blocks, so nobody else may call it.
  * Every observation goes through the session's SessionMonitor, so every event carries the
    right session ID (see session_binding.py).
  * SESSION_ENDED is the last event of a session: the worker is joined BEFORE release().
  * A failed start leaves no live session behind.

Camera problems never crash the worker. While the camera is down it reports
CAMERA_PERMISSION_DENIED (the OS says the app is blocked) or CAMERA_UNAVAILABLE (anything else),
with the diagnosis and a user-facing hint in the event details. Callers that want to fail fast
use preflight_camera() or start_monitoring(require_camera=True).
"""
from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from .camera import Camera
from .camera_probe import (
    CameraCheck, CameraProblem, OsCheck, PermissionState, check_camera, hint_for, inspect_os,
)
from .config import Config
from .detector import Detection, FaceDetector, build_detector
from .events import EventType
from .metrics import FpsMeter, LatencyTracker, ResourceMonitor
from .pipeline import analyze_frame
from .session_binding import (
    MonitoringRegistry, MonitoringStatus, SessionBindingError, SessionMonitor,
    SessionNotMonitored, validate_session_id,
)

log = logging.getLogger(__name__)

Source = Union[int, str]
DetectorFactory = Callable[[str, Config], FaceDetector]


class MonitoringStartError(SessionBindingError):
    """start_monitoring failed after validation (bad detector name, missing model, ...)."""


class CameraCheckFailed(MonitoringStartError):
    """require_camera=True and the preflight found a problem. Nothing was started or bound."""

    def __init__(self, check: CameraCheck):
        super().__init__(f"{check.problem.value}: {check.detail}. {check.hint}")
        self.check = check


@dataclass(frozen=True)
class LiveSnapshot:
    """What the worker last produced. Read-only; safe to hand to a UI thread."""
    frame_id: int
    detections: tuple[Detection, ...]
    fps: float
    analyze_p50_ms: float
    analyze_p95_ms: float
    proc_cpu_pct: float
    rss_mb: float
    frame: Optional[np.ndarray] = None      # only when publish_frames=True; flagged non-writeable


@dataclass(frozen=True)
class StopResult:
    status: MonitoringStatus                # final status, active=False
    worker_stopped_cleanly: bool            # False if the worker did not exit within the timeout
    worker_error: Optional[str]             # set if the worker died on an unexpected error


class _Worker(threading.Thread):
    def __init__(self, monitor: SessionMonitor, cam: Camera, detector: FaceDetector,
                 cfg: Config, publish_frames: bool, source: Source):
        super().__init__(name=f"monitor-{monitor.session_id}", daemon=True)
        self._monitor, self._cam, self._detector = monitor, cam, detector
        self._cfg, self._publish, self._source = cfg, publish_frames, source
        self._os = OsCheck(PermissionState.UNKNOWN, None, "not checked yet")
        self._os_at = float("-inf")
        self._stop_evt = threading.Event()
        self._snap_lock = threading.Lock()
        self._snap = LiveSnapshot(-1, (), 0.0, 0.0, 0.0, 0.0, 0.0)
        self.error: Optional[str] = None

    def request_stop(self) -> None:
        self._stop_evt.set()

    def snapshot(self) -> LiveSnapshot:
        with self._snap_lock:
            return self._snap

    def _observe(self, obs: EventType, now: float, frame_id: int, details: Optional[dict] = None) -> bool:
        """False means the monitor was released under us: stop quietly."""
        try:
            self._monitor.observe(obs, now, frame_id, details)
            return True
        except SessionNotMonitored:
            return False

    def _refresh_os(self, now: float) -> None:
        """Re-read OS camera state, at most every camera_diag_interval_s. Cheap and never opens the camera."""
        if not self._cam.is_webcam or now - self._os_at < self._cfg.camera_diag_interval_s:
            return
        try:
            self._os = inspect_os(self._source)
        except Exception as exc:                      # a broken probe must not stop monitoring
            self._os = OsCheck(PermissionState.UNKNOWN, None, f"probe failed: {type(exc).__name__}")
        self._os_at = now

    def _camera_problem(self, now: float, why: str) -> tuple[EventType, dict]:
        """Classify a camera that is not delivering usable frames."""
        self._refresh_os(now)
        plat = sys.platform
        if self._os.blocked:
            problem, etype = CameraProblem.PERMISSION_DENIED, EventType.CAMERA_PERMISSION_DENIED
        else:
            problem = CameraProblem.NO_DEVICE if self._os.device_present is False else CameraProblem.UNAVAILABLE
            etype = EventType.CAMERA_UNAVAILABLE
        return etype, {"problem": problem.value, "permission": self._os.permission.value,
                       "detail": self._os.detail, "why": why, "hint": hint_for(problem, plat, self._os)}

    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:          # anything that escapes _loop is a bug or a sink failure
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("monitor worker for %s died", self._monitor.session_id)
        finally:
            self._cam.release()           # the thread that used the camera closes it

    def _loop(self) -> None:
        cfg, cam = self._cfg, self._cam
        lat, fps, res = LatencyTracker(), FpsMeter(), ResourceMonitor()
        resources, last_res = res.sample(), time.monotonic()
        period = 0.0 if cam.is_webcam else 1.0 / cfg.capture_fps   # real-time pacing for files/stills
        frame_id, dets = 0, ()

        while not self._stop_evt.is_set():
            t_loop = time.perf_counter()
            ok, frame = cam.read()
            now = time.monotonic()

            if not ok:
                etype, info = self._camera_problem(now, "read_failed")
                if not self._observe(etype, now, frame_id, info):
                    return
                cam.maybe_reopen(now)
                dets = ()
                self._stop_evt.wait(0.03)              # wakes immediately on stop
                frame_id += 1
                continue

            if frame_id % cfg.process_every_n == 0:
                t0 = time.perf_counter()
                obs, found, details = analyze_frame(frame, self._detector, cfg)
                if (obs == EventType.DETECTION_FAILURE and cam.is_webcam
                        and details.get("reason") == "too_dark_or_lens_covered"):
                    etype, info = self._camera_problem(now, "dark_frames")
                    if etype == EventType.CAMERA_PERMISSION_DENIED:    # blocked camera, not a dark room
                        obs, found, details = etype, [], info
                lat.add("analyze_ms", (time.perf_counter() - t0) * 1e3)
                dets = tuple(found)
                if not self._observe(obs, now, frame_id, details):
                    return

            fps.tick()
            if now - last_res >= 1.0:
                resources, last_res = res.sample(), now
            self._publish_snapshot(frame_id, dets, fps.fps, lat, resources, frame)
            frame_id += 1

            if period:
                self._stop_evt.wait(max(0.0, period - (time.perf_counter() - t_loop)))

    def _publish_snapshot(self, frame_id, dets, fps, lat: LatencyTracker, resources, frame) -> None:
        stats = lat.stats("analyze_ms")
        shown = None
        if self._publish:
            shown = frame.copy()                      # own copy: the camera may reuse its buffer
            shown.flags.writeable = False             # readers must copy before drawing
        snap = LiveSnapshot(frame_id, dets, fps, stats["p50"], stats["p95"],
                            resources["proc_cpu_pct"], resources["rss_mb"], shown)
        with self._snap_lock:
            self._snap = snap


class MonitoringService:
    def __init__(self, cfg: Config, registry: MonitoringRegistry,
                 detector_factory: DetectorFactory = build_detector):
        self._cfg, self._registry, self._detector_factory = cfg, registry, detector_factory
        self._lock = threading.Lock()           # guards _workers; held only for short sections
        self._workers: dict[str, _Worker] = {}

    # ------------------------------------------------------------------ start
    def preflight_camera(self, source: Source = 0, **kwargs) -> CameraCheck:
        """Open the camera briefly and report why it does or does not work. For UIs to call BEFORE
        starting a session. Opens the device, so do not call it for a source a worker is using."""
        return check_camera(source, self._cfg, **kwargs)

    def start_monitoring(self, session_id: str, source: Source = 0, detector: str = "haar",
                         publish_frames: bool = False, require_camera: bool = False) -> MonitoringStatus:
        """Bind monitoring to `session_id` and start the worker. Raises:
        InvalidSessionId / UnknownSession / SessionAlreadyMonitored (from the registry),
        CameraCheckFailed (require_camera=True and the camera is unusable; nothing was bound),
        MonitoringStartError (detector or worker setup failed; nothing is left running).

        With require_camera=False (default) a dead camera does NOT block the start: the session is
        bound and the problem is recorded as CAMERA_UNAVAILABLE / CAMERA_PERMISSION_DENIED events,
        so the exam record shows it and the camera can be fixed mid-session."""
        sid = validate_session_id(session_id)
        try:
            det = self._detector_factory(detector, self._cfg)   # cheap, no side effects: fail first
        except Exception as exc:
            raise MonitoringStartError(f"cannot build detector {detector!r}: {exc}") from exc

        if require_camera:
            check = self.preflight_camera(source)
            if not check.ok:
                raise CameraCheckFailed(check)

        # Held across bind -> insert so a concurrent stop() cannot slip between them.
        with self._lock:
            monitor = self._registry.bind(sid)                  # validator, duplicate check, SESSION_STARTED
            try:
                cam = Camera(source, self._cfg, loop=True)      # never raises; a dead device shows up as events
                worker = _Worker(monitor, cam, det, self._cfg, publish_frames, source)
                self._workers[sid] = worker
                worker.start()
            except Exception as exc:
                self._workers.pop(sid, None)
                self._registry.release(sid)                     # do not leave a session bound with no worker
                raise MonitoringStartError(f"cannot start worker for {sid!r}: {exc}") from exc
        return monitor.status()

    # ------------------------------------------------------------------ stop
    def stop_monitoring(self, session_id: str, timeout: float = 5.0) -> StopResult:
        """Stop the worker, emit SESSION_ENDED, free the session ID. Raises SessionNotMonitored
        if there is nothing to stop (including a second call)."""
        sid = validate_session_id(session_id)
        with self._lock:
            worker = self._workers.pop(sid, None)
        if worker is None:
            raise SessionNotMonitored(f"no live monitor for session {sid!r}")

        worker.request_stop()
        if worker is not threading.current_thread():
            worker.join(timeout)
        stopped = not worker.is_alive()          # join() returns None; is_alive() tells if it timed out
        if not stopped:
            # Usually a hung camera read. Release anyway: the worker's next observe() raises
            # SessionNotMonitored and it exits, closing the camera from its own thread.
            log.warning("worker for %s did not stop within %.1fs", sid, timeout)
        final = self._registry.release(sid)      # after join, so SESSION_ENDED is the last event
        return StopResult(final, stopped, worker.error)

    def stop_all(self, timeout: float = 5.0) -> dict[str, StopResult]:
        with self._lock:
            sids = list(self._workers)
        results = {}
        for sid in sids:
            try:
                results[sid] = self.stop_monitoring(sid, timeout)
            except SessionBindingError:
                pass                              # raced with another stop; already gone
        return results

    # ------------------------------------------------------------------ read side
    def get_status(self, session_id: str) -> MonitoringStatus:
        return self._registry.get_status(session_id)

    def is_running(self, session_id: str) -> bool:
        sid = validate_session_id(session_id)
        with self._lock:
            worker = self._workers.get(sid)
        return worker is not None and worker.is_alive()

    def snapshot(self, session_id: str) -> LiveSnapshot:
        sid = validate_session_id(session_id)
        with self._lock:
            worker = self._workers.get(sid)
        if worker is None:
            raise SessionNotMonitored(f"no live monitor for session {sid!r}")
        return worker.snapshot()
