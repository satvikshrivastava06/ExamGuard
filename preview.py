"""Live webcam preview with face detection, event state machine, FPS/latency HUD.

Keys: q = quit, m = switch detector (haar <-> yunet).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from exam_monitor.camera import Camera
from exam_monitor.config import Config
from exam_monitor.detector import build_detector
from exam_monitor.events import EventType, JsonlSink
from exam_monitor.session_binding import MonitoringRegistry, SessionBindingError
from exam_monitor.metrics import FpsMeter, LatencyTracker, ResourceMonitor
from exam_monitor.overlay import blank_frame, draw_detections, draw_hud
from exam_monitor.pipeline import analyze_frame


def parse_source(s: str):
    return int(s) if s.isdigit() else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session-id", required=True, help="exam session ID issued by the backend")
    ap.add_argument("--source", default="0", help="webcam index, video file, or image file")
    ap.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    ap.add_argument("--every", type=int, default=Config.process_every_n, help="analyse 1 of every N frames")
    ap.add_argument("--process-width", type=int, default=Config.process_width)
    ap.add_argument("--no-gui", action="store_true", help="no window (CI / headless)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop after N seconds (0 = run until q)")
    ap.add_argument("--log-dir", default="logs")
    args = ap.parse_args()

    cfg = Config(process_every_n=max(1, args.every), process_width=args.process_width)
    # Bind FIRST: a bad or duplicate session ID must fail before the camera is touched.
    registry = MonitoringRegistry(cfg, lambda sid: JsonlSink(Path(args.log_dir) / f"events_{sid}.jsonl"))
    try:
        monitor = registry.bind(args.session_id)
    except SessionBindingError as exc:
        raise SystemExit(f"Cannot associate monitoring with session: {exc}")
    session = args.session_id

    detector = build_detector(args.detector, cfg)
    cam = Camera(parse_source(args.source), cfg, loop=True)
    lat, fps, res = LatencyTracker(), FpsMeter(), ResourceMonitor()

    start = time.monotonic()
    print(f"session={session} source={args.source} detector={detector.name}")

    frame_id, dets, resources, last_res = 0, [], {}, 0.0
    min_period = 0.0 if cam.is_webcam else 1 / cfg.capture_fps  # throttle files/stills to real time
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
                cam.maybe_reopen(now)
                dets = []
                shown = blank_frame(cfg.capture_width, cfg.capture_height, "CAMERA UNAVAILABLE")
                time.sleep(0.03)
            else:
                if frame_id % cfg.process_every_n == 0:
                    t0 = time.perf_counter()
                    obs, dets, details = analyze_frame(frame, detector, cfg)
                    lat.add("analyze_ms", (time.perf_counter() - t0) * 1e3)
                    for k, v in detector.last_timing.items():
                        lat.add(k, v)
                    monitor.observe(obs, now, frame_id, details)
                shown = frame
                draw_detections(shown, dets, multi_face_alert=monitor.status().state == "MULTIPLE_FACES")
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
                    f"analyze p50 {a['p50']:.1f} / p95 {a['p95']:.1f} ms | loop work p95 {p['p95']:.1f} ms",
                    f"session={session} state={monitor.status().state or 'INIT'} faces={len(dets)} "
                    f"cpu={resources.get('proc_cpu_pct', 0):.0f}% rss={resources.get('rss_mb', 0):.0f}MB",
                ])
                cv2.imshow("Exam monitor - preview", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    try:
                        detector = build_detector("yunet" if detector.name == "haar" else "haar", cfg)
                    except FileNotFoundError as exc:
                        print(exc)
            if min_period:
                time.sleep(max(0.0, min_period - (time.perf_counter() - t_loop)))
    finally:
        final_status = registry.release(session)
        cam.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
    for stage in lat.stages():
        s = lat.stats(stage)
        print(f"{stage:16s} p50={s['p50']:.2f} p95={s['p95']:.2f} max={s['max']:.2f} n={s['n']}")
    print(f"avg fps={fps.fps:.1f}")
    print("final status:", json.dumps(final_status.to_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
