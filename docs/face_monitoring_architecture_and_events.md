# Face-Monitoring Architecture and Event Specification
Smart Exam Monitoring & Integrity Analytics Platform | Team 2 | Sprint 1 output

## 1. Scope
**In:** webcam capture, face *detection*, debounced monitoring events, performance measurement.
**Out (Sprint 1):** face recognition / identity verification, gaze or head-pose, audio, screen capture, cheating verdicts.

**Principle: events are signals for a human reviewer, never conclusions.** The system reports "no face detected for 3 s",
not "student left".

## 2. Architecture

```
 Camera (webcam | video | image)         exam_monitor/camera.py
    |  BGR uint8 frame, never raises; failure counter + rate-limited reopen
    v
 Frame skipper (analyse 1 of N)          preview.py / benchmark.py   (Config.process_every_n)
    v
 Frame usability guard                   preprocess.frame_issue()    too dark / overexposed -> DETECTION_FAILURE
    v
 Detector (swappable interface)          detector.py
    |  Haar: resize -> gray -> CLAHE -> detectMultiScale3     YuNet: resize -> DNN
    |  -> list[Detection(box in original px, confidence)]
    v
 Per-frame observation                   pipeline.analyze_frame()
    |  FACE_PRESENT | FACE_ABSENT | MULTIPLE_FACES | DETECTION_FAILURE   (+ CAMERA_UNAVAILABLE from camera layer)
    v
 Debounced state machine                 events.IntegrityStateMachine
    v
 Event sink (JSONL, metadata only)       events.JsonlSink       -> later: DB / API / reviewer UI
```
Metrics (`metrics.py`) wrap every stage: per-stage p50/p95/max, FPS, process CPU%, RSS.
Preview overlay (`overlay.py`) is display-only and not part of the event contract.

Design decisions
- **Detector behind a `detect(bgr) -> list[Detection]` interface.** Haar and YuNet are interchangeable; Sprint 2 can add a second detector for cross-checking.
- **Detect on a downscaled copy** (320 px wide) and map boxes back to original pixels. Cost is roughly proportional to pixel count.
- **Frame skipping by index** (`process_every_n=2`). Last detections are reused for drawing; the state machine only sees analysed frames.
- **Events are emitted on state entry, not per frame.** A raw observation must persist for a hold time first.
- **Camera layer returns `(False, None)` instead of raising.** The loop turns it into `CAMERA_UNAVAILABLE`.
- **No frames, crops or embeddings are ever persisted.** Only event metadata.

## 3. Event specification

Monitoring state is exactly one of the five states below (or `None` before the first confirmed state).

| Event | Raw condition (per analysed frame) | Hold before event | Payload `details` | Meaning for reviewer |
|---|---|---|---|---|
| `FACE_PRESENT` | exactly 1 detection above threshold | 0.5 s | `n_faces, max_confidence, detector` | Normal. Also serves as the recovery event. |
| `FACE_ABSENT` | 0 detections | 3.0 s | same | Candidate: left seat, looking away, occluded, **or detector miss**. |
| `MULTIPLE_FACES` | 2+ detections | 1.5 s | same | Candidate: second person, photo/poster, reflection, **or duplicate box**. |
| `DETECTION_FAILURE` | frame unusable (`too_dark_or_lens_covered`, `overexposed`) or detector exception | 1.0 s | `reason`, optional `error` | Monitoring is blind. Not evidence of anything. |
| `CAMERA_UNAVAILABLE` | `read()` fails continuously | 2.0 s | `previous_state` | Device unplugged, claimed by another app, or permission revoked. |
| `SESSION_INTERRUPTED` | gap between observations > 2.0 s | immediate (point event) | `gap_s` | Monitor itself stalled (sleep, hang). The gap is excluded from debounce. |
| `SESSION_STARTED` / `SESSION_ENDED` | lifecycle | immediate | `final_state` on end | Bookends a session. |

All state events also carry `previous_state` and `previous_duration_s` so a reviewer sees how long the prior state lasted.

Rules
1. A state event fires once on entry. Staying in the state produces nothing.
2. A pending change is cancelled if the raw observation flips back before its hold time.
3. After a stall, any pending change is cancelled (an outage must not be counted as absence).
4. Hold times live in `Config`, not in code paths. Tune them per exam type.

Envelope (JSON Lines, one per line)
```json
{"type":"FACE_ABSENT","session_id":"49bc65c7","ts_utc":"2026-10-18T15:59:30.786+00:00",
 "ts_mono":200.933664237,"frame_id":78,
 "details":{"n_faces":0,"max_confidence":0.0,"detector":"haar","previous_state":"FACE_PRESENT","previous_duration_s":412.6}}
```
`ts_utc` for correlation across systems; `ts_mono` for durations (immune to clock changes).

Not yet specified (Sprint 2): event severity levels, reviewer workflow, retention period, tamper-evidence of the log, clock sync with the server.

## 4. Concept coverage map

| Topic | Decision / implementation |
|---|---|
| Image acquisition, pixels, channels | `camera.py`; frames are `uint8`, shape `(H, W, 3)`; `preprocess.describe()` prints it. |
| RGB vs BGR | OpenCV is **BGR**. Convert with `bgr_to_rgb()` only at boundaries that expect RGB (display libs, some DNNs). `preprocessing_demo.py` shows the wrong-colour result of skipping it. |
| Grayscale | Haar needs single-channel `uint8`. Colour adds no information for it. |
| Resizing | `INTER_AREA` for downscale (no aliasing), `INTER_LINEAR` for upscale. Scale factor is returned so boxes map back. |
| Normalisation | `[0,1]` float32 / mean-std for DNN inputs only. **Haar requires uint8; never feed it normalised floats.** |
| Filtering / noise reduction | Gaussian (sensor noise), median (impulse noise), bilateral (edge-preserving, slow), NLM (best, far too slow for real time). Not in the hot path by default; CLAHE is. |
| Thresholding | Otsu and adaptive are provided for the demo and for later brightness/shadow analysis. Not used for detection. |
| Video stream, capture, frame extraction | `Camera.read()`; DSHOW/AVFoundation/V4L2 backends; buffer size 1 to reduce stale frames (best effort). |
| FPS, real-time, frame skipping | `FpsMeter`; `process_every_n`. Display runs at camera rate, analysis at 1/N. |
| Processing latency | `capture_wait` (blocking on camera) is **excluded** from the processing budget; budget against `analyze_ms`. |
| Resource utilisation | `ResourceMonitor`: process CPU% (per core, can exceed 100), system CPU%, RSS, threads. |
| Detection vs recognition | Detection answers "is there a face and where". Recognition answers "whose". We do the first only. |
| Pipeline, boxes, confidence, thresholds | `Detection(box, confidence)`. Threshold is per detector, see below. |
| Haar cascade | Integral-image Haar-like features, AdaBoost-selected, evaluated as a rejection cascade over an image pyramid. Fast on CPU; frontal-only; brittle to pose, occlusion, lighting. |
| DNN detection | YuNet via `cv2.FaceDetectorYN`: handles pose/scale better, returns a bounded score and 5 landmarks. Needs a model file; heavier. |

**Confidence is not comparable across detectors.** Haar's value is a cascade weight (unbounded, uncalibrated; threshold
`haar_min_weight=2.0` is a starting point to calibrate per camera). YuNet's is a 0-1 score (`0.7`).

## 5. Limitations and mitigations

| Factor | Failure mode | Sprint 1 mitigation | Residual risk / required test |
|---|---|---|---|
| Poor lighting | Miss -> false `FACE_ABSENT`; backlight (window behind) silhouettes the face | CLAHE; mean-luma guard -> `DETECTION_FAILURE` only for near-black/blown-out | Dim-but-not-dark frames still read as absence. Needs a lighting test matrix. |
| Camera angle | Frontal Haar misses faces turned or tilted (looking down at notes) | none for Haar; YuNet tolerates more | Cannot separate "looking down" from "absent". Do not infer behaviour from absence. |
| Occlusion | Hand, mask, hair, glasses glare, head covering -> miss | none | Always route to human review; never auto-penalise. |
| Low resolution / distance | Face under min size is dropped. **Seen in testing:** at 320 px width the 0.10 minimum (32 px) rejected 2 faces that the 0.07 minimum (22 px) found. | Default lowered to 0.07 | Lower minimum raises false-positive risk. Re-check on real backgrounds. |
| False positives | Posters, photos, screens, patterned backgrounds -> spurious `MULTIPLE_FACES` | `minNeighbors`, weight threshold, 1.5 s hold | Not measured on real rooms. A printed photo is also a "face". |
| False negatives | Any of the above -> `FACE_ABSENT` | 3 s hold | `FACE_ABSENT` cannot distinguish a miss from real absence. **This is the most important limitation.** |
| Skin tone | Detector accuracy can differ across groups; auto-exposure struggles with dark skin under backlight or dim light | none, **not measured** | Must evaluate miss rate per group on a consented, representative dataset before any use that affects a student. Do not assume parity. |
| Accessibility | Wheelchair/posture, tremor or involuntary movement, visual impairment (not facing screen), facial differences, religious head coverings, needing a scribe or reader, assistive devices in frame | none | Needs an accommodation path agreed *before* the exam and human review of every flag. Events must not auto-fail anyone. |
| Privacy | Continuous video of a home; bystanders (family) appear and trigger `MULTIPLE_FACES`; camera data is personal data | local processing, no frame/crop/embedding persistence, metadata-only log | Needs: informed consent and stated purpose, retention limit, access control on logs, review of obligations under India's DPDP Act 2023 with legal/compliance, disclosure to students of exactly what is recorded. |

## 6. Measured baseline

Sandbox, shared CPU, Haar, 320 px, analysing every frame, still-image source (no camera cost):

| Metric | Value |
|---|---|
| `analyze_ms` p50 / p95 | 88 / 106 ms |
| of which preprocess / inference (p50) | 1.9 / 85.9 ms |
| Achievable analysis rate at p95 | ~9.5 fps |
| Process CPU | ~100% of one core |
| RSS | ~72 MB |

Treat these as **not representative of target hardware**: the sandbox CPU is slow and shared, and Haar cost depends on
image content (a large face filling the frame is a worst case for the cascade). Re-run on the team's laptops and on a
low-end student machine: `python benchmark.py --source 0 --frames 300 --json output/bench.json`.
Not measured at all: real webcam capture, YuNet, GUI draw/`imshow` cost.

Suggested budget (to confirm): analysis p95 <= 66 ms at `process_every_n=2` (i.e. <= 15 analysed fps) and
process CPU <= 25% of one core on the minimum supported machine.

## 7. Known issues and Sprint 2 backlog
1. `Camera.maybe_reopen()` runs on the main loop. Re-opening a dead device blocks (observed ~1 s per attempt on a missing device in the sandbox), which can trip `SESSION_INTERRUPTED`. Move capture and reopen to a worker thread with a latest-frame slot.
2. `FACE_ABSENT` vs detector miss is indistinguishable. Add temporal confidence trend and/or a second detector agreeing before escalation.
3. Blur and low-resolution are not detected as `DETECTION_FAILURE`; only brightness and exceptions are.
4. YuNet path implemented but **not run** (model download blocked in the sandbox). Verify with `download_model.py`.
5. `preview.py` GUI path (`imshow`, key handling, detector switch) and real webcam capture were not run; everything else was exercised headless.
6. Build a consented evaluation set (lighting x angle x occlusion x skin tone x accessibility cases) and report per-group miss rates.
7. Decide thresholds with data: `hold_face_absent_s`, `haar_min_weight`, `dnn_score_threshold`.
8. Event severity, retention, log integrity, and the reviewer UI.

## 8. Sprint 1 acceptance checklist
- [x] OpenCV installed; capture abstraction for webcam, video, image
- [x] Frame capture + local preview with HUD (code written; webcam run pending on team hardware)
- [x] Face detection on sample images (3/3 expected counts; sample set is tiny and unrepresentative)
- [ ] Face detection on live webcam (run `preview.py` on a laptop)
- [x] Latency / FPS / CPU / RSS measurement (sandbox numbers only)
- [x] Event model implemented and unit-tested (7 tests, fake-clock)
- [x] Architecture and event specification (this document)
- [ ] Baseline benchmark on target hardware
