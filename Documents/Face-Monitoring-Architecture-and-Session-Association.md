# Face-Monitoring Architecture and Event Specification

**Project:** Smart Exam Monitoring & Integrity Analytics Platform  
**Team:** 2  
**Deliverable:** Sprint 1

## 1. Scope

### Included in Sprint 1

- Webcam, video, and image capture
- Face detection
- Debounced monitoring events
- Performance measurement

### Out of scope for Sprint 1

- Face recognition or identity verification
- Gaze estimation or head-pose analysis
- Audio monitoring
- Screen capture
- Automated cheating verdicts

> **Core principle:** Events are signals for a human reviewer, not conclusions. The system reports “no face detected for 3 seconds,” not “the student left.”

## 2. Architecture

```text
Camera (webcam / video / image)              exam_monitor/camera.py
    |  BGR uint8 frame; never raises
    |  Failure counter and rate-limited reopen
    v
Frame skipper (analyse 1 of N)                preview.py / benchmark.py
    |  Config.process_every_n
    v
Frame usability guard                        preprocess.frame_issue()
    |  Too dark / overexposed -> DETECTION_FAILURE
    v
Detector (swappable interface)                detector.py
    |  Haar: resize -> grayscale -> CLAHE -> detectMultiScale3
    |  YuNet: resize -> DNN
    |  Returns list[Detection(box in original pixels, confidence)]
    v
Per-frame observation                         pipeline.analyze_frame()
    |  FACE_PRESENT / FACE_ABSENT / MULTIPLE_FACES / DETECTION_FAILURE
    |  CAMERA_UNAVAILABLE is emitted by the camera layer
    v
Debounced state machine                       events.IntegrityStateMachine
    v
Event sink (JSONL; metadata only)              events.JsonlSink
    |  Future integrations: database / API / reviewer UI
```

`metrics.py` measures every stage, including per-stage p50, p95, and maximum latency, FPS, process CPU usage, and RSS.

The preview overlay in `overlay.py` is for display only; it is not part of the event contract.

### Design decisions

- **Swappable detector interface:** Detectors use `detect(bgr) -> list[Detection]`. Haar and YuNet can be substituted, and Sprint 2 can introduce a second detector for cross-checking.
- **Downscaled detection:** Detection runs on a copy scaled to 320 pixels wide. Bounding boxes are mapped back to the original image dimensions. Processing cost is roughly proportional to pixel count.
- **Frame skipping:** `process_every_n=2` analyses every second frame. The most recent detections are reused for drawing, but the state machine receives only analysed frames.
- **Debounced events:** Events are emitted when the confirmed state changes, not on every frame. A raw observation must persist for its configured hold time before a state change is confirmed.
- **Graceful camera failures:** The camera layer returns `(False, None)` rather than raising an exception. The main loop converts this condition into `CAMERA_UNAVAILABLE`.
- **Metadata-only persistence:** Frames, face crops, and embeddings are never persisted. Only event metadata is written.

## 3. Event specification

The monitoring state is one of five states, or `None` before the first state has been confirmed.

| Event | Raw condition on an analysed frame | Hold time | `details` payload | Meaning for the reviewer |
|---|---|---:|---|---|
| `FACE_PRESENT` | Exactly one detection above the confidence threshold | 0.5 s | `n_faces`, `max_confidence`, `detector` | Normal state. Also serves as the recovery event. |
| `FACE_ABSENT` | No detections | 3.0 s | Same as above | A candidate signal: the student may have left the seat, looked away, or been occluded—or the detector may have missed the face. |
| `MULTIPLE_FACES` | Two or more detections | 1.5 s | Same as above | A candidate signal: another person, a photo or poster, a reflection, or a duplicate bounding box. |
| `DETECTION_FAILURE` | Unusable frame (`too_dark_or_lens_covered` or `overexposed`) or detector exception | 1.0 s | `reason` and optional `error` | Monitoring is temporarily blind. This is not evidence of misconduct. |
| `CAMERA_UNAVAILABLE` | `read()` continues to fail | 2.0 s | `previous_state` | The device may be unplugged, in use by another application, or blocked by revoked permissions. |
| `SESSION_INTERRUPTED` | Gap between observations exceeds 2.0 s | Immediate point event | `gap_s` | The monitor itself stalled, for example because of sleep or a hang. The gap is excluded from debounce timing. |
| `SESSION_STARTED` / `SESSION_ENDED` | Session lifecycle event | Immediate | `final_state` on session end | Marks the beginning and end of a session. |

Every state event also includes `previous_state` and `previous_duration_s`, allowing a reviewer to see how long the preceding state lasted.

### Event rules

1. A state event is emitted once when the system enters that state. Remaining in the same state produces no additional event.
2. A pending state change is cancelled if the raw observation changes back before the hold time has elapsed.
3. After a stall, any pending state change is cancelled. An outage must not be counted as face absence.
4. Hold times are configured in `Config`, rather than hard-coded in processing paths, so they can be tuned for different exam types.

### Example event envelope

Events use JSON Lines (one JSON object per line):

```json
{
  "type": "FACE_ABSENT",
  "session_id": "49bc65c7",
  "ts_utc": "2026-10-18T15:59:30.786+00:00",
  "ts_mono": 200.933664237,
  "frame_id": 78,
  "details": {
    "n_faces": 0,
    "max_confidence": 0.0,
    "detector": "haar",
    "previous_state": "FACE_PRESENT",
    "previous_duration_s": 412.6
  }
}
```

- `ts_utc` supports correlation across systems.
- `ts_mono` is used for duration calculations and is not affected by system clock changes.

The following items are not yet specified and are deferred to Sprint 2: event severity levels, reviewer workflow, log-retention period, tamper-evidence for the log, and clock synchronisation with the server.

## 4. Concept coverage map

| Topic | Decision or implementation |
|---|---|
| Image acquisition, pixels, and channels | `camera.py`. Frames use `uint8` with shape `(H, W, 3)`. `preprocess.describe()` prints frame information. |
| RGB vs. BGR | OpenCV uses **BGR**. Call `bgr_to_rgb()` only at boundaries that expect RGB, such as some display libraries and DNNs. `preprocessing_demo.py` demonstrates the colour error caused by skipping conversion. |
| Grayscale | Haar detection requires single-channel `uint8` grayscale images. Colour adds no information for this detector. |
| Resizing | Use `INTER_AREA` when downscaling and `INTER_LINEAR` when upscaling. The scale factor is retained so boxes can be mapped back to the original frame. |
| Normalisation | `[0,1]` `float32` scaling or mean-standard-deviation normalisation is for DNN inputs only. **Haar requires `uint8`; do not pass normalised floating-point images to it.** |
| Filtering and noise reduction | Gaussian filtering targets sensor noise; median filtering targets impulse noise; bilateral filtering preserves edges but is slow; non-local means (NLM) can be effective but is too slow for real-time use. These are not enabled in the default hot path. CLAHE is used. |
| Thresholding | Otsu and adaptive thresholding are available for demonstrations and possible later brightness or shadow analysis. They are not used for face detection. |
| Video capture and frame extraction | `Camera.read()` supports DSHOW, AVFoundation, and V4L2 backends. Buffer size 1 is requested to reduce stale frames, on a best-effort basis. |
| FPS, real-time processing, and frame skipping | `FpsMeter` and `process_every_n` provide measurement and frame skipping. The display runs at the camera frame rate, while analysis runs at approximately `1/N` of that rate. |
| Processing latency | `capture_wait`, which blocks while waiting for the camera, is **excluded** from the processing budget. The budget is measured against `analyze_ms`. |
| Resource utilisation | `ResourceMonitor` records process CPU usage (which can exceed 100% across cores), system CPU usage, RSS, and thread count. |
| Detection vs. recognition | Detection answers “Is there a face, and where is it?” Recognition answers “Whose face is it?” Sprint 1 implements detection only. |
| Pipeline, boxes, confidence, and thresholds | A detection is represented by `Detection(box, confidence)`. Thresholds are detector-specific. |
| Haar cascade | Uses integral-image Haar-like features selected with AdaBoost and evaluated through a rejection cascade over an image pyramid. It is fast on CPU, but is mainly suited to frontal faces and is sensitive to pose, occlusion, and lighting. |
| DNN detection | YuNet is accessed through `cv2.FaceDetectorYN`. It handles pose and scale variation better, returns a bounded score and five landmarks, and requires a model file. It is more computationally demanding. |

> **Confidence scores are not directly comparable across detectors.** Haar returns a cascade weight that is unbounded and uncalibrated. `haar_min_weight=2.0` is only a starting point and must be calibrated for each camera. YuNet returns a score between 0 and 1; `0.7` is the current threshold.

## 5. Limitations and mitigations

| Factor | Failure mode | Sprint 1 mitigation | Residual risk or required test |
|---|---|---|---|
| Poor lighting | Missed detections can cause false `FACE_ABSENT` events. Backlighting, such as a window behind the student, can turn the face into a silhouette. | CLAHE; a mean-luminance guard emits `DETECTION_FAILURE` only for nearly black or severely overexposed frames. | Dim but usable-looking frames may still be interpreted as absence. A lighting test matrix is needed. |
| Camera angle | Frontal Haar detection can miss turned or tilted faces, including a student looking down at notes. | No mitigation for Haar; YuNet tolerates a wider range of poses. | The system cannot distinguish “looking down” from “absent.” Do not infer behaviour from absence alone. |
| Occlusion | A hand, mask, hair, glare on glasses, or head covering can cause a missed detection. | None in Sprint 1. | Route flags to human review; never penalise a student automatically. |
| Low resolution or distance | Faces below the minimum size are dropped. In testing at 320 px width, a 0.10 minimum (32 px) rejected two faces that a 0.07 minimum (about 22 px) detected. | The default minimum was lowered to 0.07. | A lower minimum can increase false positives. Re-test against real backgrounds. |
| False positives | Posters, photos, screens, and patterned backgrounds can trigger spurious `MULTIPLE_FACES` events. | `minNeighbors`, the weight threshold, and a 1.5-second hold time. | Performance has not been measured in real rooms. A printed photograph may also be detected as a face. |
| False negatives | Any of the conditions above can cause `FACE_ABSENT`. | A 3-second hold time reduces transient events. | `FACE_ABSENT` cannot distinguish a detector miss from actual absence. **This is the most important limitation.** |
| Skin tone | Detector performance may vary across groups. Auto-exposure can struggle with darker skin under backlighting or dim lighting. | None; performance has not been measured. | Before using the system in a way that affects students, measure miss rates on a consented, representative dataset for relevant groups. Do not assume parity. |
| Accessibility | Posture or wheelchair use, tremors or involuntary movement, visual impairment, facial differences, religious head coverings, the use of a scribe or reader, and assistive devices in frame may affect detections. | None in Sprint 1. | Agree on an accommodation process before the exam and require human review of every flag. Events must never automatically fail a student. |
| Privacy | Continuous video may capture a student's home or bystanders, who may trigger `MULTIPLE_FACES`. Camera data is personal data. | Local processing; no saved frames, crops, or embeddings; metadata-only logs. | Define informed consent and purpose, a retention limit, log access controls, and disclosure of exactly what is recorded. Review obligations under India's Digital Personal Data Protection Act, 2023 with legal or compliance staff. |

## 6. Measured baseline

The following measurements were taken in a sandbox using a shared CPU, Haar detection, a 320-pixel-wide image, and analysis of every frame. The source was a still image, so camera capture cost was not included.

| Metric | Measured value |
|---|---:|
| `analyze_ms` p50 / p95 | 88 / 106 ms |
| Preprocessing / inference latency (p50) | 1.9 / 85.9 ms |
| Achievable analysis rate at p95 | About 9.5 FPS |
| Process CPU usage | About 100% of one core |
| RSS | About 72 MB |

These measurements are **not representative of the target hardware**. The sandbox CPU is slow and shared, and Haar's cost depends on image content. A large face filling the frame is a worst-case scenario for the cascade.

Re-run the benchmark on team laptops and on a low-end student machine:

```bash
python benchmark.py --source 0 --frames 300 --json output/bench.json
```

The following have not been measured: real webcam capture, YuNet performance, and GUI drawing or `imshow()` overhead.

### Suggested performance budget (to be confirmed)

- Analysis p95 latency: **66 ms or less** with `process_every_n=2` (equivalent to at least 15 analysed FPS)
- Process CPU usage: **25% or less of one core** on the minimum supported machine

## 7. Known issues and Sprint 2 backlog

1. **Camera reopen can block the main loop.** `Camera.maybe_reopen()` runs on the main loop. Reopening a dead device took about one second per attempt in the sandbox and can trigger `SESSION_INTERRUPTED`. Move capture and reopening to a worker thread with a latest-frame slot.
2. **Face absence and detector misses are indistinguishable.** Add a temporal confidence trend and/or require agreement from a second detector before escalation.
3. **Blur and low resolution are not detected as `DETECTION_FAILURE`.** The current checks cover brightness issues and detector exceptions only.
4. **YuNet has not been run.** Its path is implemented, but downloading the model was blocked in the sandbox. Verify it with `download_model.py`.
5. **The GUI path and live webcam remain untested.** `preview.py`, including `imshow()`, key handling, detector switching, and real webcam capture, was not run. The other checks were exercised headlessly.
6. **Build a representative evaluation set.** Include lighting, camera angle, occlusion, skin tone, and accessibility cases, then report miss rates for relevant groups.
7. **Set thresholds using data.** Calibrate `hold_face_absent_s`, `haar_min_weight`, and `dnn_score_threshold`.
8. **Define operational policies.** Specify event severity, log retention, log integrity, and the reviewer interface.

## 8. Sprint 1 acceptance checklist

- [x] OpenCV installed; capture abstraction supports webcam, video, and image sources.
- [x] Frame capture and local preview with HUD implemented. Live webcam testing remains pending on team hardware.
- [x] Face detection tested on sample images: 3/3 expected counts. The sample set is small and unrepresentative.
- [ ] Face detection tested on a live webcam. Run `preview.py` on a laptop.
- [x] Latency, FPS, CPU, and RSS measurement implemented; current figures are from the sandbox only.
- [x] Event model implemented and unit-tested: 7 tests using a fake clock.
- [x] Architecture and event specification documented here.
- [ ] Baseline benchmark completed on target hardware.

---

# Sprint 2: Associate Monitoring Status with the Correct Exam Session

**Team:** 2  
**Deliverable:** Sprint 2 — session association

> **Scope note:** This item covers session association only. It does not implement start/stop monitoring functions, webcam-unavailable or permission-denied handling, backend session lookup, or API endpoints.

## 1. Implementation

**Module:** `exam_monitor/session_binding.py`

| Component | Role |
|---|---|
| `validate_session_id` | Applies an allow-list check: 8–64 characters from `[A-Za-z0-9_-]`, validated with `fullmatch`. Because the ID is used in a log filename, this also helps prevent path traversal. |
| `MonitoringStatus` | Provides a frozen (immutable) snapshot containing `session_id`, `active`, the confirmed `state`, `state_since_utc`, `last_event_type`, `last_observation_utc`, and `event_count`. It is safe to return from an API handler. |
| `_BoundSink` | Wraps the event sink and raises `SessionMismatch` if an event's `session_id` differs from the bound session. The association is checked on every write. |
| `SessionMonitor` | Manages one monitor for one session. It owns a state machine and a lock. `observe()` feeds in observations; `status()` returns the current snapshot. |
| `MonitoringRegistry` | Provides `bind(id)`, `get_status(id)`, `list_active()`, and `release(id)`. It rejects invalid IDs, IDs refused by the validator, and attempts to create a second live monitor for the same ID. |

`bind()` emits `SESSION_STARTED`, and `release()` emits `SESSION_ENDED`, so both lifecycle events carry the session ID.

## 2. Assumptions to Confirm with Team 1

1. **Session ID format:** IDs are strings matching `[A-Za-z0-9_-]{8,64}`. UUIDs with hyphens fit this format. Update `_SESSION_ID_RE` if the backend uses a different format.
2. **What makes a session valid:** Ultimately, the session should exist, have `ACTIVE` status, and belong to the logged-in candidate. Verifying this requires a backend lookup, which can be supplied through `validator=` when creating `MonitoringRegistry`. Until that is connected, only the ID format is checked.
3. **Rebinding after release:** Rebinding an ID after `release()` is allowed, which may be needed to resume after a crash. Change this behavior if a finished session must never be monitored again.
4. **Process scope:** The duplicate-monitor guarantee applies only within one process and one registry. It does not hold across multiple processes or machines; the backend would need to enforce that guarantee.

## 3. References

### Retrieved and consulted during implementation

- Python `dataclasses` and `frozen=True`: [Python documentation](https://docs.python.org/3.8/library/dataclasses.html)
- Path traversal and validating against known-good input rather than sanitising: [OWASP Path Traversal](https://owasp.org/www-community/attacks/Path_Traversal)
- Lock versus `RLock` re-entrancy: [CPython documentation commit](https://github.com/python/cpython/commit/7eb59cd95b09ede8c7524e8f71fbea5623715f7f)
- PEP 544, structural subtyping and `typing.Protocol` (used for `EventSink`): [PEP 544 text mirror](https://peps-dir-aaturner.readthedocs.io/en/latest/_sources/pep-0544.txt)

### References 

- [Python `threading` documentation](https://docs.python.org/3/library/threading.html)
- [Python `re` documentation](https://docs.python.org/3/library/re.html) — `fullmatch` and the trailing-newline behavior of `$` were checked by running the tests.
- [PEP 544](https://peps.python.org/pep-0544/)

## 4. Originality Note

The code was written for this project. The references document the language features and security practices used; no code was copied from a repository.
