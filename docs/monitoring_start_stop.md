# Sprint 2 (Team 2): start/stop monitoring, wired to the exam session

Builds on `session_association.md`. Code: `exam_monitor/monitoring_service.py`.
Camera diagnosis distinguishes permission-denied from no-device where the operating system exposes that information; see `camera_unavailable_permission.md`. Backend session lookup and API endpoints are not built.

## API
```python
registry = MonitoringRegistry(cfg, sink_factory, validator=None)   # session_binding.py
service  = MonitoringService(cfg, registry)

status = service.start_monitoring("EXAM-2026-S0001", source=0, detector="haar", publish_frames=False)
service.get_status(sid)      # MonitoringStatus: session_id, active, state, event_count, ...
service.snapshot(sid)        # LiveSnapshot: detections, fps, latency, optional read-only frame
service.is_running(sid)
result = service.stop_monitoring(sid, timeout=5.0)   # StopResult(status, worker_stopped_cleanly, worker_error)
service.stop_all()
```

## Lifecycle
```
start_monitoring(sid)                           stop_monitoring(sid)
 1 validate_session_id (format, no side effects)  1 pop worker from table (under lock)
 2 build detector      (fail before any session)  2 worker.request_stop(); join(timeout)
 3 registry.bind(sid)  -> SESSION_STARTED         3 registry.release(sid) -> SESSION_ENDED   (after join)
 4 open Camera                                    4 worker closes the camera in its own thread
 5 start worker thread
```
Failure rules
| Failure | Result |
|---|---|
| Bad / hostile session ID | `InvalidSessionId`; no sink, no thread |
| Validator rejects | `UnknownSession`; nothing created |
| Detector can't be built | `MonitoringStartError` before bind; no session record |
| Worker setup fails after bind | session released (`SESSION_ENDED` logged), `MonitoringStartError`; ID reusable |
| Second start for same ID | `SessionAlreadyMonitored`; the first worker is untouched |
| Second stop | `SessionNotMonitored` |
| Camera missing / denied / unplugged | worker keeps running; `CAMERA_UNAVAILABLE` after its hold time |
| Detector exception / dark frame | `DETECTION_FAILURE`; worker survives |
| Worker hung in `cap.read()` | `stop` gives up after `timeout`, still emits `SESSION_ENDED`, returns `worker_stopped_cleanly=False`; the worker exits and closes the camera when the read returns |
| Sink (disk) error in worker | worker stops, `StopResult.worker_error` is set, `is_running()` goes False |

## Design rules and why
1. **One worker thread per session owns the camera.** OpenCV documents `VideoCapture` as synchronous and not thread-safe, so nothing else may call it.
2. **GUI stays on the main thread.** The worker publishes immutable `LiveSnapshot`s (frame copy marked non-writeable); `preview.py` draws on its own copy.
3. **SESSION_ENDED is last.** Stop joins the worker before releasing the session, and the sink guard rejects writes after release.
4. **Cooperative stop.** `threading.Event`; the worker waits on it (`wait(0.03)`) instead of `sleep`, so stop is immediate even while the camera is down.
5. **Bounded join.** `join(timeout)` returns `None` in both cases, so `is_alive()` decides whether it timed out.

## Known limits
- Stopping while `Camera.maybe_reopen()` is mid-attempt waits for that attempt (about 1 s on a missing device in the sandbox).
- A truly hung driver call cannot be interrupted from Python; the thread is a daemon and is abandoned (flagged in `StopResult`).
- Detector cannot be switched mid-session (the old `m` key in `preview.py` was removed); stop and start a new session instead.
- Real webcam and the preview window were not run (no device or display in the sandbox).

## Sources
Retrieved and read while building this:
- OpenCV 4.9 documentation: `VideoCapture` is synchronous and not thread-safe, read in a separate thread: https://docs.opencv.org/4.9.0/d4/d65/tutorial_orbbec_astra.html
- OpenCV issue #13145: `read()` waits for the next frame; the usual fix is a reader thread keeping the most recent frame: https://github.com/opencv/opencv/issues/13145
- Python `Thread.join(timeout)` / `is_alive()` semantics (CPython docs commit, mirror): https://git.joshthomas.dev/mirrors/cpython/commit/2c9f1042c62ca0fcd1b962da3b532c9cfe543ccd?files=Doc
- `Event.wait()` returns whether the flag was set (secondary tutorial): https://mail.bogotobogo.com/python/Multithread/python_multithreading_Event_Objects_between_Threads.php
- Main thread owns GUI windows while a thread reads frames (gist by crackwitz, `FreshestFrame`): https://gist.github.com/crackwitz/15c3910f243a42dcd9d4a40fcdb24e40
- Prior art for background threads in proctoring: https://github.com/vardanagarwal/Proctoring-AI (MIT; its audio part calls the speech API on a separate thread so recording is not disturbed). Idea only, no code reused.
- Prior art for "events only, no stored video, never auto-disqualify": https://github.com/renilthereal/exam-system (no license; design reference only).

Referenced but NOT opened this session (verify before citing): https://docs.python.org/3/library/threading.html

The implementation is original to this project. Nothing was copied from the repositories above.
