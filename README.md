# Exam Monitor: Face-Monitoring Core
**Team 2 · Sprint 1**

Webcam capture → face detection → debounced monitoring events, with latency, FPS, and resource measurement.

**Scope:** Detection only. No recognition, identity verification, or frame storage.

## Setup

### Windows, macOS, and Linux

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

| Goal | Command |
|---|---|
| Live webcam preview (`q` quits; `m` switches detector). **A session ID is required.** | `python preview.py --session-id EXAM-2026-S0001` |
| Use a different camera (`0` may be a virtual or IR camera) | `python preview.py --session-id <id> --source 1` |
| Diagnose camera availability and permissions before an exam | `python check_camera.py --source 0` |
| Test face detection on sample images (exits with code `1` on a mismatch) | `python test_faces.py` |
| Measure latency, FPS, CPU, and RSS using a webcam | `python benchmark.py --source 0 --frames 300 --json output/bench.json` |
| Run the same benchmark without a webcam | `python benchmark.py --source sample_images/astronaut_1face.png` |
| Generate a preprocessing contact sheet | `python preprocessing_demo.py` |
| Use the DNN detector | `python download_model.py`, then `python preview.py --detector yunet` |
| Run the unit test suite | `python -m pytest -q` |

## Event Log

Logs are written to:

```text
logs/events_<session>.jsonl
```

The log contains metadata only; frames are not stored.

## Sample Images

The filename convention is:

```text
name_<N>face.png
```

Here, `N` is the expected number of faces in the image. Add your own images to `sample_images/` only with consent.

`two_astronauts_2face.png` is synthetic: it contains the same image tiled twice.

## Related Documentation

- **Architecture and event contract:** `docs/face_monitoring_architecture_and_events.md`
- **Sprint 2 — session association:** `docs/session_association.md`
- **Sprint 2 — monitoring lifecycle:** `docs/monitoring_start_stop.md`
- **Sprint 2 — camera availability and permissions:** `docs/camera_unavailable_permission.md`
