from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection

GREEN, RED, AMBER, WHITE = (0, 200, 0), (0, 0, 220), (0, 180, 255), (255, 255, 255)


def draw_detections(img: np.ndarray, dets: list[Detection], multi_face_alert: bool = False) -> None:
    color = RED if multi_face_alert else GREEN
    for d in dets:
        x, y, w, h = d.box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{d.confidence:.2f}", (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    pad, lh = 6, 18
    cv2.rectangle(img, (0, 0), (img.shape[1], pad * 2 + lh * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, pad + lh * (i + 1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def blank_frame(width: int, height: int, message: str) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    cv2.putText(img, message, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, AMBER, 2, cv2.LINE_AA)
    return img
