from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .config import Config

Source = Union[int, str]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _backends(is_webcam: bool) -> list[Optional[int]]:
    if not is_webcam:
        return [None]
    if sys.platform.startswith("win"):
        # DSHOW opens fast and honours resolution requests; MSMF can take seconds to open.
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


class Camera:
    """Webcam (int index), video file, or still image behind one read() interface.

    read() never raises: it returns (False, None) on any failure and counts consecutive
    failures so the caller can debounce and trigger maybe_reopen().
    """

    def __init__(self, source: Source, cfg: Config, loop: bool = False):
        self.source = source
        self.cfg = cfg
        self.loop = loop
        self.consecutive_failures = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._still: Optional[np.ndarray] = None
        self._last_open_attempt = 0.0
        self.open()

    @property
    def is_webcam(self) -> bool:
        return isinstance(self.source, int)

    @property
    def is_open(self) -> bool:
        return self._cap is not None or self._still is not None

    def open(self) -> bool:
        self.release()
        self._last_open_attempt = time.monotonic()
        if isinstance(self.source, str) and Path(self.source).suffix.lower() in IMAGE_EXTS:
            img = cv2.imread(self.source)
            if img is None or img.size == 0:
                return False
            self._still = img
            return True
        for backend in _backends(self.is_webcam):
            cap = (cv2.VideoCapture(self.source) if backend is None
                   else cv2.VideoCapture(self.source, backend))
            if cap.isOpened():
                if self.is_webcam:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.capture_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.capture_height)
                    cap.set(cv2.CAP_PROP_FPS, self.cfg.capture_fps)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort; not every backend honours it
                self._cap = cap
                return True
            cap.release()
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._still is not None:
            return True, self._still.copy()
        if self._cap is None:
            self.consecutive_failures += 1
            return False, None
        try:
            ok, frame = self._cap.read()
            if not ok and self.loop and not self.is_webcam:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
        except cv2.error:
            ok, frame = False, None
        if not ok or frame is None or frame.size == 0:
            self.consecutive_failures += 1
            return False, None
        self.consecutive_failures = 0
        return True, frame

    def maybe_reopen(self, now: float) -> bool:
        """Rate-limited recovery after repeated read failures (unplugged/claimed device)."""
        if (self.consecutive_failures >= self.cfg.reopen_after_failures
                and now - self._last_open_attempt >= self.cfg.reopen_interval_s):
            return self.open()
        return False

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._still = None
