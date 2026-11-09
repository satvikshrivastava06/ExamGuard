"""Run a recorded video through the REAL monitoring pipeline, deterministically.

Why: live testing is slow (a 50 s scenario takes 50 s) and not repeatable (timing jitter). Here the
clock is the video's own clock: frame i happens at t = i / fps, however fast the computer is.
So a 52-second video replays in a few seconds, and the same video always gives the same events.

It uses exactly the production pieces: FrameSampler -> analyze_frame -> SessionMonitor
(state machine + prolonged-absence tracker + session guard). Only the clock and the sink differ.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from .config import Config
from .detector import FaceDetector, HaarFaceDetector
from .events import Event, ListSink
from .pipeline import analyze_frame
from .preprocess import resize_to_width, to_gray
from .sampler import FrameSampler
from .session_binding import MonitoringRegistry, MonitoringStatus


class VirtualClock:
    """A clock you move by hand. Used as the registry clock so SESSION_STARTED/ENDED get video time."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@dataclass(frozen=True)
class FrameRecord:
    """One ANALYSED frame (frames skipped by the sampler have no record)."""
    t: float
    frame_index: int
    observation: str      # FACE_PRESENT / FACE_ABSENT / MULTIPLE_FACES / DETECTION_UNAVAILABLE
    n_faces: int
    luma: float           # mean grey level 0..255, a measure of how bright the frame was


@dataclass
class ReplayResult:
    events: list[Event]
    records: list[FrameRecord]
    fps: float
    frames_total: int
    frames_analysed: int
    duration_s: float
    wall_s: float
    final_status: MonitoringStatus


def replay_video(path: "Path | str", cfg: Optional[Config] = None, detector: Optional[FaceDetector] = None,
                 session_id: str = "replay-session-01") -> ReplayResult:
    cfg = cfg or Config(sample_target_fps=4.0)
    detector = detector or HaarFaceDetector(cfg)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    clock, sink = VirtualClock(), ListSink()
    registry = MonitoringRegistry(cfg, lambda sid: sink, clock=clock)
    monitor = registry.bind(session_id)
    sampler = FrameSampler.from_config(cfg)
    records: list[FrameRecord] = []
    started = time.perf_counter()
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            t = i / fps
            clock.t = t
            if sampler.should_process(i, t):
                obs, dets, details = analyze_frame(frame, detector, cfg)
                monitor.observe(obs, t, i, details)
                small, _ = resize_to_width(frame, 64)
                records.append(FrameRecord(t, i, obs.value, len(dets), float(to_gray(small).mean())))
            i += 1
    finally:
        cap.release()
        clock.t = i / fps                       # session ends when the video ends
        final = registry.release(session_id)    # flushes an open absence, writes SESSION_ENDED
    return ReplayResult(sink.events, records, fps, i, len(records), i / fps, time.perf_counter() - started, final)


def confirmed_state_at(events: list[Event], t: float) -> Optional[str]:
    """The monitor's confirmed state at video time t (None before the first state event)."""
    from .events import STATE_TYPES
    state = None
    for e in events:
        if e.ts_mono > t:
            break
        if e.type in STATE_TYPES:
            state = e.type.value
    return state
