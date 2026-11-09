# Sprint 3 explained (Team 2): frame capture, face states, prolonged absence, sampling, cleanup, testing

## 1. What Sprint 3 asked, and where it lives
| Requirement | Where | One-line answer |
|---|---|---|
| Frame capture + face detection with OpenCV | `camera.py`, `detector.py` (Sprints 1-2) | already built; one new property, `Camera.file_fps` |
| Track face present / absent / detection unavailable | `events.py`, `pipeline.py` | three states, debounced; "detection failure" renamed to `DETECTION_UNAVAILABLE` |
| Configurable threshold before recording a prolonged absence | `absence.py`, `session_binding.py`, `config.py` | `prolonged_absence_s` (default 10 s) |
| Frame sampling | `sampler.py` | every-N-frames **or** time-based (N analyses per second) |
| Resource cleanup | `monitoring_service.py`, `events.py`, `camera.py`, `preview.py` | context managers, atexit + SIGTERM safety, leak tests |
| Test lighting, camera angles, sample videos | `make_test_assets.py`, `evaluate_images.py`, `evaluate_video.py`, `replay.py`, `tests/test_robustness.py` | 45 labelled images, 6 ground-truth videos, deterministic replay |

## 2. How one frame travels (say this out loud)
```
camera frame -> FrameSampler (analyse this one?) -> analyze_frame (dark? run detector)
   -> ONE observation: FACE_PRESENT | FACE_ABSENT | MULTIPLE_FACES | DETECTION_UNAVAILABLE
   -> state machine (must persist for a few seconds)  -> confirmed state event
   -> AbsenceTracker (follows confirmed FACE_ABSENT)  -> PROLONGED_ABSENCE record if >= threshold
   -> every event stamped with the session ID and checked by the session guard -> log file
```
**Two thresholds, two jobs.** `hold_face_absent_s` = 3 s answers "is the face absent *now*?" (a state).
`prolonged_absence_s` = 10 s answers "was it long enough to *record*?" (a record). A 6 s absence is a state but not a record.

## 3. File by file

### NEW `exam_monitor/absence.py`: the prolonged-absence rule
- **Idea:** one definition of absence for the whole system: *a confirmed FACE_ABSENT state*. It is recorded only if it lasted at least the threshold.
- **Code:** `AbsenceTracker` has three methods. `enter(started, now)` is called when FACE_ABSENT is confirmed; `started` is when the absent frames *began* (not when confirmation came 3 s later). `leave(ended, reason)` closes the episode and returns an `AbsenceEpisode` only if `duration >= threshold_s`. `ongoing_s(now)` feeds the live status.
- **Why it follows confirmed states, not raw frames:** my first version watched raw frames with a "grace period". The angle-sweep video showed it stitching a flickering detector (present, absent, present...) into one fake 10 s absence. Following the state machine fixes that, and `test_flickering_detector_never_becomes_a_fake_absence` guards it.
- **Say:** "Blind time is never counted as absence. If the camera goes dark or the app freezes, the episode ends at the last moment we could see."

### NEW `exam_monitor/sampler.py`: frame sampling
- **Idea:** capture and display run at full camera speed; only the expensive detection is sampled.
- **Code:** `FrameSampler.should_process(frame_id, now)`. Mode A (`every_n`): `frame_id % n == 0`. Mode B (`target_fps`): keeps a `_next_due` time and advances it by a fixed period, so the long-run rate matches the target whatever the camera FPS. If the program stalls it re-syncs to "now" instead of analysing a burst of frames to catch up.
- **Say:** "Every-N changes with camera speed; time-based gives the same CPU cost on a 15 fps and a 60 fps camera."

### NEW `exam_monitor/replay.py`: run a video through the real pipeline
- **Idea:** a 54 s video replays in about 10 s and gives identical results every time, because the clock is the video's own (`t = frame_index / fps`).
- **Code:** `VirtualClock` (a clock you set by hand), `replay_video()` (same sampler, `analyze_frame`, and session monitor as production), `confirmed_state_at(events, t)`.

### NEW `make_test_assets.py`: builds the labelled test set
- **Idea:** every test image has a known right answer, in its file name (`roll_+30_1face.png`).
- **Code:** small image operations (`brightness`, `gamma`, `rotate`, `perspective_squeeze`, `occlude`, `far_away`, ...) assembled by `build_image_variants()`. It finds the real face box first so occlusions land on the eyes or mouth. `write_videos()` builds 6 videos plus `.truth.json` files saying what the monitor should conclude, second by second.
- **Be honest:** all of it is synthetic. A perspective squeeze is not a real head turn.

### NEW `evaluate_images.py`: scores images by what the *monitor concludes*
- Five outcomes: `detected`, `silent_miss` (face there, monitor says absent: the dangerous one), `flagged` (monitor says "cannot judge": a safe failure), `false_positive`, `correct_negative`.

### NEW `evaluate_video.py`: scores videos against ground truth
- Segment scoring: inside each truth segment, after allowing the hold time plus 1 s, the confirmed state must match at least 90% of the time. Also checks the recorded absences against expected durations, and prints detection rate per brightness or per head angle. `present_throughout` videos treat any recorded absence as false.

### NEW tests (`tests/`)
`test_absence.py` (18), `test_sampler.py` (11), `test_cleanup.py` (10), `test_robustness.py` (51, including 14 expected failures). `pytest.ini` registers the `slow` marker for video tests.

### REPLACED `exam_monitor/config.py`
New settings: `sample_target_fps` (0 = use `process_every_n`), `prolonged_absence_s` (10), and from earlier sprints the camera diagnosis settings.

### REPLACED `exam_monitor/events.py`
- `DETECTION_UNAVAILABLE` is the new name; `DETECTION_FAILURE` stays as an alias of the same member, so old code keeps working. New point event `PROLONGED_ABSENCE`.
- `IntegrityStateMachine.record()` writes a point event through the same sink; `last_transition_started` remembers when the observations behind the latest state began.
- `JsonlSink` now accepts `str` or `Path`, has `.closed`, and works as a context manager.

### REPLACED `exam_monitor/session_binding.py`
`SessionMonitor` owns an `AbsenceTracker`. `_follow_absence()` turns confirmed state events into `enter`/`leave` calls. A stall ends the episode at the last observation, and tracking resumes after it. On release, an open episode is flushed **before** `SESSION_ENDED`. `MonitoringStatus` gains `absence_ongoing_s`, `prolonged_absence_count`, `prolonged_absence_total_s`.

### REPLACED `exam_monitor/monitoring_service.py`
- Worker uses `FrameSampler`. Snapshot reports `frames_read` and `frames_analysed`.
- **Cleanup:** the worker's `finally` releases the camera, calls `detector.close()` if it exists (a failing close is logged, never fatal) and drops the last frame copy. `MonitoringService` is a context manager (`with ... as svc:`), registers a weak-reference `atexit` hook, and has `close()`.
- **Bug fixed:** video files were paced at the config camera rate (30 fps), so a 10 fps clip played 3x too fast and all time thresholds were wrong. Now `period = 1 / (cam.file_fps or cfg.capture_fps)`.

### REPLACED `exam_monitor/camera.py` and `exam_monitor/pipeline.py`
`Camera`: context manager plus `file_fps`. `pipeline.py`: uses the new `DETECTION_UNAVAILABLE` name.

### REPLACED `preview.py`
New options `--sample-fps` and `--absence-threshold`. SIGTERM becomes a normal exit; the final report now prints from `finally`, so it appears on every exit path. HUD shows sampling counts, current absence and recorded count.

### REPLACED `tests/test_events.py`, `tests/test_monitoring_service.py`
Only the renamed state string changed, plus a video-pacing regression test.

## 4. Test results (synthetic data, Haar detector, 320 px detection width)
**Images (45).**
| Category | Result |
|---|---|
| Brightness 0.5x, 0.3x, gamma, bright, overexposed, low contrast, noise (sigma 25 and 50), side light, blur, low resolution (320/160/96 px), mirrored, yaw and pitch squeeze, roll 15 degrees | detected |
| Very dark (mean grey < 20), blown-out blank frame | **flagged "unavailable"**, never "absent" (safe failure) |
| Roll 30 and 45 degrees, upside down | silent miss |
| Face smaller than about 20 px (shrunk to 0.3x and 0.2x) | silent miss |
| Face cut off by the frame edge (30%, 50%) | silent miss |
| Eyes, mouth or half the face covered | silent miss |
| Backlight (dark face, bright background) | silent miss |
| 6 negative controls (empty room, wall, noise, stripes, dark, white) | no false positives |

**Videos (6)**, replayed on the video clock at 4 analyses per second.
| Video | Result |
|---|---|
| presence/absence/dark timeline | pass: 6 s absence = state only; 18 s absence recorded as 18.0 s; dark = unavailable |
| 0.3 s face flash inside a 26 s absence | pass: ONE record (26.0 s), not two |
| two people | pass: `MULTIPLE_FACES` |
| fading light | pass: below mean grey 20 it says "unavailable", never "absent" |
| fake head turn (yaw) | pass: no false record (after the flicker fix) |
| head roll 0 to 45 degrees and back | **fail (known limit):** two false prolonged absences (10.0 s and 14.3 s) while the person never left |

**Detection width trade-off** (images are 512 px, so 640 upsamples them; real 640x480 webcams would not be upsampled):
| width | far faces found | roll images found | false positives | analyse p50, sandbox |
|---|---|---|---|---|
| 320 | 1/3 | 2/6 | 0 | 95 ms |
| 480 | 1/3 | 3/6 | 0 | 138 ms |
| 640 | 3/3 | 4/6 | 1 | 198 ms |

## 5. Bugs found while building (worth mentioning: they show the tests work)
1. Raw-frame grace period turned a flickering detector into a fake 10 s absence. Redesigned around confirmed states.
2. Video files played 3x too fast in live mode. Fixed, regression-tested.
3. `JsonlSink` crashed on a plain string path. Now accepts both.
4. `preview.py` skipped its final report on SIGTERM. Moved into `finally`.
5. `absence_ongoing_s` stayed non-zero after the session closed. Reset on close.
Each fix has a test, and each test was checked by breaking the code on purpose to confirm it fails.

## 6. Limitations to state plainly
- A person who is present but missed (tilted head, small, covered, backlit) looks identical to someone who left. The prolonged-absence record says "no face detected for N seconds", never "left".
- Recommended next step if tilted heads matter: retry on rotated copies only when no face is found, or switch to the YuNet DNN detector. Neither is implemented or measured here.
- Nothing was tested on a real webcam, real lighting, or real people. Synthetic variants show how the pipeline reacts, not how students will behave.
- SIGKILL, a power cut, or a hard crash cannot be caught: an absence still open at that moment is lost (it is only written when it ends).
- Windows: SIGTERM semantics differ; the signal tests are skipped there, and the file-descriptor leak test needs Linux or macOS.
- Sandbox timings are from a slow shared CPU.

## 7. Sources (retrieved and read while building this sprint)
- OpenCV 4.10 geometric transformations (`getRotationMatrix2D`, `warpAffine`, `getPerspectiveTransform`, `warpPerspective`): https://docs.opencv.org/4.10.0/da/d6e/tutorial_py_geometric_transformations.html
- OpenCV: `INTER_AREA` preferred for shrinking, `INTER_LINEAR` for enlarging: https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
- Haar cascades in OpenCV are not robust to rotation and need a frontal, upright face: https://hub.packtpub.com/tracking-faces-haar-cascades (book excerpt, secondary source)
- Frontal Haar fails on tilted heads; CLAHE helps with uneven light (community Q&A, 2014): https://kiwix.lema.org/stackoverflow.com_en_all_2023-11/questions/26981241/haar-classifier-robustness-train-my-own-or-alternative-exists
- Python `atexit`: handlers do not run when the process is killed by a signal Python does not handle: https://docs.python.org/library/atexit
- Turning SIGTERM into a normal exit so cleanup runs (community recipe): https://code.activestate.com:443/recipes/533117-cleaning-up-when-killed
- Earlier sprints' sources (threading, OpenCV `VideoCapture` not thread-safe, camera permission): see `docs/monitoring_start_stop.md` and `docs/camera_unavailable_permission.md`.

Referenced but **not opened** this session (verify before citing): the official pages for `signal`, `weakref`, `cv2.VideoWriter`, and `psutil.Process.num_fds`.
The implementation is original to this project. Nothing was copied from these pages.
