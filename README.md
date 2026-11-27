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

## Recent Updates: Data Persistence, Summaries & Condition Comparison
| Feature | Description | Where |
|---|---|---|
| **Structured Monitoring Summaries** | Export comprehensive session metrics, timing, ratios, and incident counts to JSON and CSV. | `exam_monitor/accuracy_report.py`, `accuracy_report.py` |
| **Camera Availability & Coverage** | Precise measurement of camera stream uptime, permission denials, and active face detection coverage. | `exam_monitor/accuracy_report.py` |
| **Condition Comparison** | Compare detection quality, silent miss rates, and composite quality across environmental/pose conditions and detectors. | `exam_monitor/condition_comparison.py`, `compare_conditions.py`, `evaluate_images.py` |
| **Data Sufficiency Identification** | Identify sessions with insufficient monitoring data (short duration, outages, blind frames, crashes) with specific root-cause reasons. | `exam_monitor/data_sufficiency.py`, `accuracy_report.py` |
| **MySQL Integration** | Persistent storage for session metadata, debounced events, and raw per-frame detections, with direct summary exports. | `exam_monitor/database.py` |
| **Event Publishing** | A decoupled publish-subscribe system (`EventBus`) with asynchronous buffered delivery to prevent pipeline blocking. | `exam_monitor/event_publisher.py` |
| **Health Monitoring** | Real-time diagnostics tracking FPS, latency, CPU, and component status (Healthy/Degraded/Unhealthy). | `exam_monitor/health.py` |
| **Aggregated Intervals & Uncertainty** | Per-state aggregated detection intervals, uncertainty levels/indicators, and misconduct-guard verification. | `exam_monitor/uncertainty.py`, `exam_monitor/accuracy_report.py` |
| **Error Simulation & Limitations** | Deterministic error injection (`SimulatedErrorDetector`) for score-input testing and documented detector limitations. | `exam_monitor/error_simulation.py`, `docs/detector_limitations.md` |

### New Command Shortcuts
| Goal | Command |
|---|---|
| Structured summary & sufficiency audit | `python accuracy_report.py --check-sufficiency` |
| Export summaries to JSON & CSV | `python accuracy_report.py --json output/summaries.json --csv output/summaries.csv` |
| List only insufficient monitoring sessions | `python accuracy_report.py --insufficient-only` |
| Compare detection quality across conditions | `python compare_conditions.py --baseline control --markdown output/comparison.md` |
| Compare multiple detectors across conditions | `python compare_conditions.py --detectors haar yunet` |


