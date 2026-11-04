"""Exam monitor that captures from a live video-meeting URL.

Usage
-----
    python meet_monitor.py \\
        --session-id EXAM-2026-S0001 \\
        --meet-url "https://meet.google.com/abc-defg-hij"

The script:
  1. Opens a real Chromium browser and navigates to the meeting URL.
  2. Waits for the meeting UI to load, then auto-clicks the "Join" button.
  3. Injects a JavaScript snippet that reads the student's local camera
     preview (the <video> element) and streams JPEG frames to Python.
  4. Runs the same face-detection pipeline as preview.py.
  5. Logs events to logs/events_<session_id>.jsonl (same format as before).
  6. On exit, prints a full accuracy report.

Keys (in the OpenCV preview window that mirrors what the browser sees):
  q  –  quit and print the accuracy report
  m  –  switch detector (haar <-> yunet)

Prerequisites
-------------
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from exam_monitor.config import Config
from exam_monitor.detector import build_detector
from exam_monitor.events import EventType, JsonlSink
from exam_monitor.meet_capture import MeetCapture
from exam_monitor.session_binding import MonitoringRegistry, SessionBindingError
from exam_monitor.metrics import FpsMeter, LatencyTracker, ResourceMonitor
from exam_monitor.overlay import blank_frame, draw_detections, draw_hud
from exam_monitor.pipeline import analyze_frame
from exam_monitor.accuracy_report import compute_accuracy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-id", required=True, help="Exam session ID (e.g. EXAM-2026-S0001)")
    ap.add_argument("--meet-url", required=True, help="Full meeting URL (Google Meet, Zoom, Teams …)")
    ap.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    ap.add_argument("--every", type=int, default=Config.process_every_n,
                    help="Analyse 1 of every N frames (default: %(default)s)")
    ap.add_argument("--process-width", type=int, default=Config.process_width)
    ap.add_argument("--join-wait", type=float, default=8.0,
                    help="Seconds to wait for the meeting UI to load before injecting capture (default: 8)")
    ap.add_argument("--capture-fps", type=int, default=15,
                    help="Frames per second to request from the browser JS injector (default: 15)")
    ap.add_argument("--no-gui", action="store_true", help="No OpenCV preview window")
    ap.add_argument("--max-seconds", type=float, default=0, help="Stop after N seconds (0 = run until q)")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()

    cfg = Config(process_every_n=max(1, args.every), process_width=args.process_width)
    log_path = Path(args.log_dir) / f"events_{args.session_id}.jsonl"

    # Bind session first (same guard as preview.py)
    registry = MonitoringRegistry(cfg, lambda sid: JsonlSink(Path(args.log_dir) / f"events_{sid}.jsonl"))
    try:
        monitor = registry.bind(args.session_id)
    except SessionBindingError as exc:
        raise SystemExit(f"Cannot associate monitoring with session: {exc}")

    detector = build_detector(args.detector, cfg)
    cam = MeetCapture(args.meet_url, target_fps=args.capture_fps, join_wait_s=args.join_wait)
    lat, fps, res = LatencyTracker(), FpsMeter(), ResourceMonitor()

    start = time.monotonic()
    print(f"session={args.session_id}  meet={args.meet_url}  detector={detector.name}")
    print("Waiting for first frame from browser …")

    frame_id, dets, resources, last_res = 0, [], {}, 0.0
    try:
        while True:
            t_loop = time.perf_counter()
            ok, frame = cam.read()
            t_cap = time.perf_counter()
            now = time.monotonic()

            if args.max_seconds and now - start > args.max_seconds:
                break

            if not ok:
                monitor.observe(EventType.CAMERA_UNAVAILABLE, now, frame_id)
                dets = []
                shown = blank_frame(cfg.capture_width, cfg.capture_height, "WAITING FOR MEET FRAME …")
                time.sleep(0.05)
            else:
                if frame_id % cfg.process_every_n == 0:
                    t0 = time.perf_counter()
                    obs, dets, details = analyze_frame(frame, detector, cfg)
                    lat.add("analyze_ms", (time.perf_counter() - t0) * 1e3)
                    for k, v in detector.last_timing.items():
                        lat.add(k, v)
                    monitor.observe(obs, now, frame_id, details)
                shown = frame
                draw_detections(shown, dets,
                                multi_face_alert=monitor.status().state == "MULTIPLE_FACES")
                lat.add("capture_wait_ms", (t_cap - t_loop) * 1e3)
                lat.add("process_ms", (time.perf_counter() - t_cap) * 1e3)

            frame_id += 1
            fps.tick()

            if now - last_res >= 1.0:
                resources, last_res = res.sample(), now

            if not args.no_gui:
                a, p = lat.stats("analyze_ms"), lat.stats("process_ms")
                draw_hud(shown, [
                    f"FPS {fps.fps:5.1f} | detector={detector.name} | every {cfg.process_every_n} frames",
                    f"analyze p50 {a['p50']:.1f} / p95 {a['p95']:.1f} ms | loop p95 {p['p95']:.1f} ms",
                    f"session={args.session_id}  state={monitor.status().state or 'INIT'}  faces={len(dets)} "
                    f"cpu={resources.get('proc_cpu_pct', 0):.0f}%  rss={resources.get('rss_mb', 0):.0f}MB",
                    f"meet={args.meet_url[:60]}{'…' if len(args.meet_url) > 60 else ''}",
                ])
                cv2.imshow("Exam monitor – Meet", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    try:
                        detector = build_detector("yunet" if detector.name == "haar" else "haar", cfg)
                    except FileNotFoundError as exc:
                        print(exc)

    finally:
        final_status = registry.release(args.session_id)
        cam.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    # --- Latency stats (same as preview.py) ---
    for stage in lat.stages():
        s = lat.stats(stage)
        print(f"{stage:16s} p50={s['p50']:.2f} p95={s['p95']:.2f} max={s['max']:.2f} n={s['n']}")
    print(f"avg fps={fps.fps:.1f}")
    print("final status:", json.dumps(final_status.to_dict()))

    # --- Accuracy report ---
    if log_path.exists():
        try:
            report = compute_accuracy(log_path)
            report.print()
        except Exception as exc:
            print(f"[accuracy] Could not generate report: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
