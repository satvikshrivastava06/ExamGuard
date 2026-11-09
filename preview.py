"""Live preview for one exam session. A thin client of MonitoringService:
the worker thread owns the camera and detection; this (main) thread owns the window.

Keys: q = stop monitoring and quit.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import cv2

from exam_monitor.config import Config
from exam_monitor.events import JsonlSink
from exam_monitor.monitoring_service import CameraCheckFailed, MonitoringService, MonitoringStartError
from exam_monitor.overlay import blank_frame, draw_detections, draw_hud
from exam_monitor.session_binding import MonitoringRegistry, SessionBindingError


def parse_source(s: str):
    return int(s) if s.isdigit() else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-id", required=True, help="exam session ID issued by the backend")
    ap.add_argument("--source", default="0", help="webcam index, video file, or image file")
    ap.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    ap.add_argument("--every", type=int, default=Config.process_every_n, help="analyse 1 of every N frames")
    ap.add_argument("--sample-fps", type=float, default=0.0,
                    help="analyse at most this many frames per second (time-based). 0 = use --every")
    ap.add_argument("--absence-threshold", type=float, default=Config.prolonged_absence_s,
                    help="seconds a face must be absent before the absence is RECORDED")
    ap.add_argument("--process-width", type=int, default=Config.process_width)
    ap.add_argument("--require-camera", action="store_true",
                    help="refuse to start if the camera is unusable (default: start and log the problem)")
    ap.add_argument("--no-gui", action="store_true", help="no window (CI / headless)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (0 = until q)")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()

    cfg = Config(process_every_n=max(1, args.every), process_width=args.process_width,
                 sample_target_fps=max(0.0, args.sample_fps), prolonged_absence_s=max(0.0, args.absence_threshold))

    # `kill` / service managers send SIGTERM. Turn it into a normal exit so the finally-block below
    # still stops the session, writes SESSION_ENDED and closes the log. (Ctrl+C already raises KeyboardInterrupt.)
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(0)))
    registry = MonitoringRegistry(cfg, lambda sid: JsonlSink(Path(args.log_dir) / f"events_{sid}.jsonl"))
    service = MonitoringService(cfg, registry)
    sid = args.session_id

    try:
        service.start_monitoring(sid, parse_source(args.source), args.detector,
                                 publish_frames=not args.no_gui, require_camera=args.require_camera)
    except CameraCheckFailed as exc:
        raise SystemExit(f"Camera not usable [{exc.check.problem.value}]: {exc.check.detail}\n  -> {exc.check.hint}")
    except (SessionBindingError, MonitoringStartError) as exc:
        raise SystemExit(f"Cannot start monitoring: {exc}")
    print(f"monitoring started: session={sid} source={args.source} detector={args.detector}")

    started, last_id = time.monotonic(), -2
    try:
        while service.is_running(sid):
            if args.max_seconds and time.monotonic() - started > args.max_seconds:
                break
            snap = service.snapshot(sid)
            if args.no_gui:
                time.sleep(0.05)
                continue
            if snap.frame_id == last_id:                 # nothing new yet; do not redraw
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue
            last_id = snap.frame_id
            status = service.get_status(sid)
            img = snap.frame.copy() if snap.frame is not None else blank_frame(
                cfg.capture_width, cfg.capture_height, "WAITING FOR CAMERA")
            if status.state in ("CAMERA_UNAVAILABLE", "CAMERA_PERMISSION_DENIED"):
                img = blank_frame(cfg.capture_width, cfg.capture_height,
                                  "CAMERA PERMISSION DENIED" if status.state == "CAMERA_PERMISSION_DENIED"
                                  else "CAMERA UNAVAILABLE")
            draw_detections(img, list(snap.detections), multi_face_alert=status.state == "MULTIPLE_FACES")
            draw_hud(img, [
                f"session={sid} state={status.state or 'INIT'} faces={len(snap.detections)}",
                f"FPS {snap.fps:5.1f} | analyze p50 {snap.analyze_p50_ms:.1f} / p95 {snap.analyze_p95_ms:.1f} ms",
                f"cpu={snap.proc_cpu_pct:.0f}% rss={snap.rss_mb:.0f}MB events={status.event_count} "
                f"analysed {snap.frames_analysed}/{snap.frames_read}",
                f"absence now {status.absence_ongoing_s:.0f}s | recorded prolonged absences "
                f"{status.prolonged_absence_count} (>= {cfg.prolonged_absence_s:.0f}s)",
            ])
            cv2.imshow("Exam monitor - preview", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Runs on every exit path: q, max-seconds, Ctrl+C, SIGTERM, or an error in the loop above.
        last = service.snapshot(sid) if service.is_running(sid) else None
        result = service.stop_monitoring(sid)
        if not args.no_gui:
            cv2.destroyAllWindows()
        if last:
            print(f"last fps={last.fps:.1f} analyse p50={last.analyze_p50_ms:.1f}ms p95={last.analyze_p95_ms:.1f}ms "
                  f"analysed {last.frames_analysed}/{last.frames_read} frames")
        print("stopped cleanly:", result.worker_stopped_cleanly, "| worker error:", result.worker_error)
        print("final status:", json.dumps(result.status.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
