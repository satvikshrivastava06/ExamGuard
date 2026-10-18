from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Acquisition ---
    capture_width: int = 640
    capture_height: int = 480
    capture_fps: int = 30

    # --- Real-time processing ---
    process_width: int = 320      # detector input width; boxes are mapped back to full res
    process_every_n: int = 2      # frame skipping: analyse 1 of every N frames

    # --- Haar cascade ---
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    haar_min_face_frac: float = 0.07   # min face width as a fraction of process_width (~22 px at 320)
    haar_min_weight: float = 2.0       # cascade level weight; NOT a probability, calibrate per camera

    # --- DNN (YuNet) ---
    dnn_score_threshold: float = 0.7   # 0..1, closer to a real confidence
    dnn_model_path: str = "models/face_detection_yunet_2023mar.onnx"

    # --- Frame usability (detection-failure guard) ---
    dark_mean_luma: float = 20.0       # mean grey level below this = covered lens / no light
    bright_mean_luma: float = 240.0    # above this = blown out

    # --- Event debounce: a condition must persist this long before an event fires ---
    hold_face_present_s: float = 0.5
    hold_face_absent_s: float = 3.0
    hold_multiple_faces_s: float = 1.5
    hold_detection_failure_s: float = 1.0
    hold_camera_unavailable_s: float = 2.0
    stall_s: float = 2.0               # gap between observations that counts as an interruption

    # --- Camera recovery ---
    reopen_after_failures: int = 5
    reopen_interval_s: float = 1.0
