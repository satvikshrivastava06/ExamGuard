# Exam Monitor: Face-Monitoring Core (Team 2, Sprint 1)

Webcam capture -> face detection -> debounced monitoring events, with latency/FPS/resource measurement.
Detection only. No recognition, no identity, no frame storage.

## Setup (Windows / macOS / Linux)
```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
| Goal | Command |
|---|---|
| Live webcam preview (q quits, m switches detector). **Session ID is required** | `python preview.py --session-id EXAM-2026-S0001` |
| Fail fast if the camera is unusable | `python preview.py --session-id <id> --require-camera` |
| Different camera (0 is often a virtual cam / IR cam) | `python preview.py --session-id <id> --source 1` |
| Sample-image detection test (exit 1 on mismatch) | `python test_faces.py` |
| Latency / FPS / CPU / RSS, webcam | `python benchmark.py --source 0 --frames 300 --json output/bench.json` |
| Same, no webcam | `python benchmark.py --source sample_images/astronaut_1face.png` |
| Preprocessing contact sheet | `python preprocessing_demo.py` |
| DNN detector | `python download_model.py` then `python preview.py --detector yunet` |
| Unit tests (event state machine) | `python -m pytest -q` |

Event log: `logs/events_<session>.jsonl` (metadata only).

Sample-image convention: `name_<N>face.png` = expected face count N. Add your own images to
`sample_images/` (with consent). `two_astronauts_2face.png` is synthetic (one image tiled twice).

Design and event contract: `docs/face_monitoring_architecture_and_events.md`.

## Sprint 1 main topics -> where
| Topic | Where | Command |
|---|---|---|
| Set up OpenCV and webcam access | `exam_monitor/camera.py`, `requirements.txt` | `python check_camera.py` |
| Capture frames, local preview | `preview.py` (HUD: FPS, latency, state) | `python preview.py --session-id <id>` |
| Face detection: sample images and webcam | `test_faces.py`, `sample_images/`, `exam_monitor/detector.py` | `python test_faces.py` (images); preview (webcam) |
| Frame-processing latency | `benchmark.py`, `exam_monitor/metrics.py` | `python benchmark.py --source 0 --frames 300` |

## Sprint 2 Team 2 -> where
| Item | Where |
|---|---|
| Monitoring component that accepts a session ID | `exam_monitor/session_binding.py`, `monitoring_service.py` |
| Start/stop monitoring | `MonitoringService.start_monitoring / stop_monitoring` |
| Associate status with the correct session | `MonitoringRegistry`, `get_status(session_id)` |
| Webcam unavailable / permission denied | `exam_monitor/camera_probe.py`, `check_camera.py`, `--require-camera` |

Docs: `docs/session_association.md`, `docs/monitoring_start_stop.md`, `docs/camera_unavailable_permission.md` (binding) and `docs/monitoring_start_stop.md` (start/stop service).
Preview keys: q stops monitoring and quits (the old `m` detector switch was removed).

## Sprint 3 (Team 2)
| Goal | Command |
|---|---|
| Generate the labelled test set (images + videos with ground truth) | `python make_test_assets.py` |
| Score lighting / angle / occlusion images | `python evaluate_images.py` (report in `output/robustness_report.md`) |
| Replay every sample video against its ground truth | `python evaluate_video.py --all` |
| Your own clip | `python evaluate_video.py --video my.mp4` with a `my.truth.json` beside it |
| Time-based sampling (5 analyses/second) and a 15 s absence threshold | `python preview.py --session-id <id> --sample-fps 5 --absence-threshold 15` |
| All tests / skip the slow video tests | `python -m pytest -q` / `python -m pytest -q -m "not slow"` |

Docs: `docs/sprint3_explained.md`. `sample_images/robustness/` and `sample_videos/` are generated; do not commit them.
