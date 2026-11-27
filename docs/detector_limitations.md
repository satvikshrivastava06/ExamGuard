# Face Detector Limitations & Fairness Guarantees

This document describes the technical limitations of the face detectors used in `exam_monitor_cv`, how those limitations affect monitoring outputs, and the safeguards wired into the pipeline so detector blind spots or missing webcam data are never misclassified as confirmed misconduct.

---

## 1. Supported Detectors

The pipeline (`exam_monitor/detector.py`) supports two CPU-runnable face detectors:

| Detector | Class | Underlying Model | Confidence Semantics |
|---|---|---|---|
| `haar` | `HaarFaceDetector` | OpenCV `haarcascade_frontalface_default.xml` (Viola-Jones cascade) | Unbounded cascade level weight (`haar_min_weight = 2.0` default) — **not** a calibrated probability. |
| `yunet` | `YuNetDetector` | OpenCV `FaceDetectorYN` (`face_detection_yunet_2023mar.onnx`) | Bounded `0.0 .. 1.0` detection score (`dnn_score_threshold = 0.7` default). |

Both detectors operate on frames resized to `process_width = 320` pixels (`exam_monitor/config.py`) before mapping bounding boxes back to original resolution.

---

## 2. Known Detector Limitations

### 2.1 Lighting & Exposure Extremes
- **Dark / covered lens / unlit rooms**: Before running inference, `frame_issue()` (`exam_monitor/preprocess.py`) checks mean grayscale luma on a 64px thumbnail. Frames with `luma < dark_mean_luma` (`20.0`) or `luma > bright_mean_luma` (`240.0`) are rejected before detection and emitted as `DETECTION_UNAVAILABLE` (`too_dark_or_lens_covered` or `overexposed`).
- **Backlighting & uneven contrast**: `HaarFaceDetector` applies CLAHE (`clipLimit=2.0`, `tileGridSize=(8, 8)`), which improves moderate contrast issues but still degrades under strong backlighting or harsh shadows across the face.
- **Borderline dim frames**: Frames just above `dark_mean_luma` (e.g. luma `21–40`) pass the preflight check yet can cause **silent misses** (false negatives), especially with `haar`.

### 2.2 Head Pose (Yaw, Pitch, Roll)
- **`haar` is frontal-only**: `HaarFaceDetector` is trained on upright frontal faces. Turning the head (yaw), looking down at notes/keyboard (pitch), or tilting (roll) quickly drops recall.
- **`yunet` handles moderate pose better**, but extreme yaw/pitch/roll or inverted orientations still produce missed detections or reduced confidence.

### 2.3 Occlusion & Accessories
- Partial occlusion (hand on chin/face, mask, heavy glare on glasses, hair covering eyes) can break Haar feature responses and lower YuNet scores below `dnn_score_threshold`.
- `exam_monitor` performs **face detection only** — it does **not** perform facial recognition, identity verification, or gaze/pose estimation.

### 2.4 Scale, Resolution, and Motion Blur
- `HaarFaceDetector` enforces `minSize = max(20, int(process_width * haar_min_face_frac))` (`~22 px` at `process_width=320`). Candidates seated very far from the webcam or using low-resolution/heavily compressed streams may fall below effective detection size.
- Rapid movement or low-light shutter blur reduces edge sharpness needed by both detectors.

### 2.5 False Positives & Multiple Faces
- Textured backgrounds, posters, reflections, or high-contrast patterns can occasionally trigger spurious bounding boxes (false `MULTIPLE_FACES` or false `FACE_PRESENT` when the seat is empty).
- Confidence scores are not directly comparable across detectors (`haar` cascade weights can exceed `1.0`, whereas `yunet` scores are in `[0.0, 1.0]`).

### 2.6 Sampling & Latency Tradeoffs
- Real-time monitoring samples frames via `FrameSampler` (`process_every_n = 2` or `sample_target_fps`). Very brief events shorter than the sampling interval or shorter than the state-machine debounce windows (`hold_*`) are intentionally filtered out.

---

## 3. Fairness & Misconduct Safeguards in the System

To prevent detector limitations or hardware failures from producing false misconduct flags, the system enforces the following architectural guarantees:

1. **Missing Webcam Data != Confirmed Misconduct**
   - When `Camera.read()` fails (`exam_monitor/monitoring_service.py`) or OS camera permission is blocked (`exam_monitor/camera_probe.py`), the worker emits `CAMERA_UNAVAILABLE` or `CAMERA_PERMISSION_DENIED` — **never** `FACE_ABSENT` or `PROLONGED_ABSENCE`.
   - On dark webcam frames (`too_dark_or_lens_covered`), `_Worker._loop` re-checks OS camera permissions so an OS-blocked black feed is classified as `CAMERA_PERMISSION_DENIED` rather than `DETECTION_UNAVAILABLE` or `FACE_ABSENT`.
   - `AbsenceTracker.leave(..., "camera_unavailable")` (`exam_monitor/absence.py`) closes any in-progress absence at the exact moment the camera drops, so outage time is never added to absence duration.

2. **Debounced State Transitions (`IntegrityStateMachine`)**
   - Transient single-frame misses or spurious boxes do not fire events. Observations must persist for `hold_face_absent_s` (`3.0s`), `hold_multiple_faces_s` (`1.5s`), `hold_detection_failure_s` (`1.0s`), etc. (`exam_monitor/events.py`).
   - Monitoring loop stalls (`gap > stall_s = 2.0s`) emit `SESSION_INTERRUPTED` and reset pending debounce candidates.

3. **Aggregated Detection Intervals & Uncertainty Indicators**
   - `AccuracyReport.aggregate_intervals()` (`exam_monitor/accuracy_report.py`) partitions the session timeline into explicit `DetectionInterval`s, tagging `DETECTION_UNAVAILABLE`, `CAMERA_UNAVAILABLE`, and `CAMERA_PERMISSION_DENIED` with `is_uncertain=True` and `is_misconduct_signal=False`.
   - `evaluate_uncertainty()` (`exam_monitor/uncertainty.py`) computes `uncertainty_ratio`, `uncertainty_level` (`LOW` / `MODERATE` / `HIGH` / `CRITICAL`), explainable `UncertaintyIndicator`s (`MISSING_WEBCAM_DATA`, `DETECTOR_BLIND_TIME`, `LOW_DETECTION_CONFIDENCE`, `SESSION_INTERRUPTIONS`, `INCOMPLETE_SESSION`, `INSUFFICIENT_MONITORING_DATA`), and a `MisconductEvaluation` guard with `missing_webcam_treated_as_misconduct = False` and `reliable_for_misconduct_decision`.

4. **Data Sufficiency Audit**
   - `evaluate_session_sufficiency()` (`exam_monitor/data_sufficiency.py`) flags sessions with low camera availability, low detection coverage, excessive detection failures, or missing `SESSION_ENDED` events so unreliable sessions are not used for automated decisions.
