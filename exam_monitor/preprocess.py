"""Image-processing primitives. All functions assume OpenCV's BGR channel order."""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Config


def describe(img: np.ndarray) -> dict:
    """Pixels and channels: shape is (H, W, C) in OpenCV, dtype uint8 = 0..255 per channel."""
    channels = 1 if img.ndim == 2 else img.shape[2]
    per_channel = img.reshape(-1, channels).mean(axis=0).round(1).tolist()
    return {
        "height": img.shape[0], "width": img.shape[1], "channels": channels,
        "dtype": str(img.dtype), "min": int(img.min()), "max": int(img.max()),
        "mean_bgr" if channels == 3 else "mean": per_channel,
    }


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    # OpenCV decodes and captures BGR. matplotlib, PIL and most DNN weights expect RGB.
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_to_width(img: np.ndarray, width: int) -> Tuple[np.ndarray, float]:
    """Aspect-preserving resize. Returns (image, scale) so boxes can be mapped back."""
    h, w = img.shape[:2]
    if w == width:
        return img, 1.0
    scale = width / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR  # AREA avoids aliasing on downscale
    return cv2.resize(img, (width, max(1, round(h * scale))), interpolation=interp), scale


def normalize_01(img: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in [0, 1]. Needed for DNN input, NOT for Haar (which requires uint8)."""
    return img.astype(np.float32) / 255.0


def standardize(img: np.ndarray, mean: Tuple[float, ...], std: Tuple[float, ...]) -> np.ndarray:
    """(x/255 - mean) / std per channel. mean/std must match the channel order of the image."""
    x = normalize_01(img)
    return (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)


# --- Noise reduction / filtering (cost grows quickly down this list) ---
def gaussian_denoise(img: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)          # sensor noise; also blurs edges


def median_denoise(img: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.medianBlur(img, k)                     # salt-and-pepper noise; edge preserving


def bilateral_denoise(img: np.ndarray, d: int = 7, sigma_color: float = 50, sigma_space: float = 50) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)  # edge-aware, slow


def nlm_denoise(gray: np.ndarray, h: float = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, None, h, 7, 21)  # best quality, far too slow for real time


# --- Contrast / thresholding ---
def equalize(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def otsu_threshold(gray: np.ndarray) -> Tuple[float, np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)       # Otsu is noise sensitive; blur first
    thr, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr), mask


def adaptive_threshold(gray: np.ndarray, block: int = 31, c: int = 5) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, c)


# --- Frame usability ---
def frame_issue(bgr: np.ndarray, cfg: Config) -> Optional[str]:
    """Cheap guard run before detection. Returns a reason string if the frame is unusable."""
    small, _ = resize_to_width(bgr, 64)
    luma = float(to_gray(small).mean())
    if luma < cfg.dark_mean_luma:
        return "too_dark_or_lens_covered"
    if luma > cfg.bright_mean_luma:
        return "overexposed"
    return None
