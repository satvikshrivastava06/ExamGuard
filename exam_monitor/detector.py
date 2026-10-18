from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import cv2
import numpy as np

from .config import Config
from .preprocess import clahe, resize_to_width, to_gray


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]   # x, y, w, h in ORIGINAL frame pixels
    confidence: float                # Haar: cascade weight (unbounded). YuNet: 0..1 score.


class FaceDetector(Protocol):
    name: str
    last_timing: dict

    def detect(self, bgr: np.ndarray) -> list[Detection]: ...


class HaarFaceDetector:
    """Viola-Jones cascade. CPU-cheap, frontal-only, weak under pose/occlusion/lighting change."""

    name = "haar"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {path}")
        self.last_timing: dict = {}

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        t0 = perf_counter()
        small, scale = resize_to_width(bgr, self.cfg.process_width)
        gray = clahe(to_gray(small))        # contrast normalisation; the cascade is brightness-sensitive
        t1 = perf_counter()
        min_side = max(20, int(self.cfg.process_width * self.cfg.haar_min_face_frac))
        rects, _, weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=self.cfg.haar_scale_factor,
            minNeighbors=self.cfg.haar_min_neighbors,
            minSize=(min_side, min_side),
            outputRejectLevels=True,
        )
        t2 = perf_counter()
        out = [
            Detection(tuple(int(round(v / scale)) for v in (x, y, w, h)), float(wt))
            for (x, y, w, h), wt in zip(rects, weights)
            if wt >= self.cfg.haar_min_weight
        ]
        self.last_timing = {"preprocess_ms": (t1 - t0) * 1e3, "inference_ms": (t2 - t1) * 1e3}
        return out


class YuNetDetector:
    """DNN detector (OpenCV FaceDetectorYN). Handles pose/scale better, outputs a real score + landmarks."""

    name = "yunet"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        model = Path(cfg.dnn_model_path)
        if not model.is_file():
            raise FileNotFoundError(f"{model} missing. Run: python download_model.py")
        self._net = cv2.FaceDetectorYN.create(str(model), "", (320, 320), cfg.dnn_score_threshold, 0.3, 50)
        self.last_timing: dict = {}

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        t0 = perf_counter()
        small, scale = resize_to_width(bgr, self.cfg.process_width)
        h, w = small.shape[:2]
        self._net.setInputSize((w, h))
        t1 = perf_counter()
        _, faces = self._net.detect(small)
        t2 = perf_counter()
        self.last_timing = {"preprocess_ms": (t1 - t0) * 1e3, "inference_ms": (t2 - t1) * 1e3}
        if faces is None:
            return []
        return [Detection(tuple(int(round(v / scale)) for v in f[:4]), float(f[14])) for f in faces]


def build_detector(name: str, cfg: Config) -> FaceDetector:
    if name == "haar":
        return HaarFaceDetector(cfg)
    if name == "yunet":
        return YuNetDetector(cfg)
    raise ValueError(f"Unknown detector '{name}' (expected 'haar' or 'yunet')")
