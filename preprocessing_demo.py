"""Render every preprocessing stage of one image into a labelled contact sheet."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from exam_monitor import preprocess as P


def tile(img: np.ndarray, label: str, size=(256, 256)) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (0, 0), (size[0], 20), (0, 0, 0), -1)
    cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="sample_images/astronaut_1face.png")
    ap.add_argument("--out", default="output/preprocessing_sheet.png")
    a = ap.parse_args()
    bgr = cv2.imread(a.image)
    if bgr is None:
        raise SystemExit(f"cannot read {a.image}")
    print("original:", P.describe(bgr))

    gray = P.to_gray(bgr)
    rng = np.random.default_rng(0)
    noisy = np.clip(gray.astype(np.int16) + rng.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
    small, _ = P.resize_to_width(bgr, 160)
    thr, otsu = P.otsu_threshold(gray)
    print(f"otsu threshold={thr:.0f}; normalized float32 range="
          f"{P.normalize_01(gray).min():.2f}..{P.normalize_01(gray).max():.2f}")

    tiles = [
        tile(bgr, "BGR (as decoded)"),
        tile(P.bgr_to_rgb(bgr), "BGR shown as RGB (wrong colours)"),
        tile(gray, "grayscale"),
        tile(small, "resized 160w (INTER_AREA)"),
        tile(P.normalize_01(gray), "normalized [0,1] float32"),
        tile(noisy, "gaussian noise sigma=20"),
        tile(P.gaussian_denoise(noisy), "gaussian 5x5"),
        tile(P.median_denoise(noisy), "median 5"),
        tile(P.bilateral_denoise(noisy), "bilateral"),
        tile(P.equalize(gray), "hist equalization"),
        tile(P.clahe(gray), "CLAHE"),
        tile(otsu, f"otsu thr={thr:.0f}"),
        tile(P.adaptive_threshold(gray), "adaptive threshold"),
    ]
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
