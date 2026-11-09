"""Frame sampling: decide which captured frames are worth analysing.

Two modes:
  every_n     analyse frame 0, N, 2N, ...   Simple, but the analysis rate changes with camera FPS.
  target_fps  analyse at most this many frames per second of wall time. Independent of camera FPS,
              so a 60 fps camera and a 15 fps camera cost the same.

Capture and display still run at full camera speed; only the expensive detection is sampled.
"""
from __future__ import annotations

from typing import Optional

from .config import Config


class FrameSampler:
    def __init__(self, every_n: int = 1, target_fps: float = 0.0):
        if every_n < 1:
            raise ValueError("every_n must be >= 1")
        if target_fps < 0:
            raise ValueError("target_fps must be >= 0")
        self.every_n, self.target_fps = every_n, target_fps
        self._next_due: Optional[float] = None
        self.seen = 0
        self.selected = 0

    @classmethod
    def from_config(cls, cfg: Config) -> "FrameSampler":
        return cls(every_n=cfg.process_every_n, target_fps=cfg.sample_target_fps)

    def should_process(self, frame_id: int, now: float) -> bool:
        self.seen += 1
        take = self._time_based(now) if self.target_fps > 0 else frame_id % self.every_n == 0
        self.selected += take
        return take

    def _time_based(self, now: float) -> bool:
        period = 1.0 / self.target_fps
        if self._next_due is None:
            self._next_due = now + period
            return True
        if now < self._next_due:
            return False
        # Advance on a fixed schedule so the long-run rate matches target_fps, but if we fell
        # behind (a stall) resync to "now" instead of analysing a burst of frames to catch up.
        self._next_due += period
        if self._next_due <= now:
            self._next_due = now + period
        return True
