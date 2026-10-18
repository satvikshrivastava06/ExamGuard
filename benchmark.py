"""Per-stage latency, achievable FPS and resource use for a given source.

`capture_wait` is time blocked in cap.read(); on a 30 fps webcam it is ~33 ms by construction
and is NOT processing cost. Budget against `analyze`, not against total loop time.
"""
from __future__ import annotations

import argparse
import json
import time

from exam_monitor.camera import Camera
from exam_monitor.config import Config
from exam_monitor.detector import build_detector
from exam_monitor.metrics import FpsMeter, LatencyTracker, ResourceMonitor
from exam_monitor.pipeline import analyze_frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="0")
    ap.add_argument("--detector", choices=["haar", "yunet"], default="haar")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--every", type=int, default=1, help="1 = analyse every frame (worst case)")
    ap.add_argument("--process-width", type=int, default=Config.process_width)
    ap.add_argument("--json", help="write summary to this path")
    args = ap.parse_args()

    cfg = Config(process_every_n=max(1, args.every), process_width=args.process_width)
    detector = build_detector(args.detector, cfg)
    cam = Camera(int(args.source) if args.source.isdigit() else args.source, cfg, loop=True)
    if not cam.is_open:
        raise SystemExit(f"Cannot open source {args.source!r}")

    lat, fps, res = LatencyTracker(window=args.frames), FpsMeter(window=args.frames), ResourceMonitor()
    faces_seen = analysed = bad_reads = 0
    samples: list[dict] = []

    for i in range(args.warmup + args.frames):
        t0 = time.perf_counter()
        ok, frame = cam.read()
        t1 = time.perf_counter()
        if not ok:
            bad_reads += 1
            continue
        measuring = i >= args.warmup
        if measuring:
            lat.add("capture_wait_ms", (t1 - t0) * 1e3)
            fps.tick()
        if i % cfg.process_every_n == 0:
            obs, dets, _ = analyze_frame(frame, detector, cfg)
            t2 = time.perf_counter()
            if measuring:
                lat.add("analyze_ms", (t2 - t1) * 1e3)
                for k, v in detector.last_timing.items():
                    lat.add(k, v)
                analysed += 1
                faces_seen += bool(dets)
        if measuring:
            lat.add("loop_total_ms", (time.perf_counter() - t0) * 1e3)
        if i % 30 == 0:
            samples.append(res.sample())
    cam.release()

    summary = {
        "source": args.source, "detector": detector.name, "frames": args.frames,
        "process_width": cfg.process_width, "analyse_every": cfg.process_every_n,
        "loop_fps": round(fps.fps, 1), "bad_reads": bad_reads,
        "face_detected_ratio": round(faces_seen / analysed, 3) if analysed else None,
        "latency_ms": {s: {k: round(v, 2) for k, v in lat.stats(s).items()} for s in lat.stages()},
        "proc_cpu_pct_mean": round(sum(s["proc_cpu_pct"] for s in samples) / max(1, len(samples)), 1),
        "rss_mb_max": round(max((s["rss_mb"] for s in samples), default=0), 1),
    }
    a = summary["latency_ms"].get("analyze_ms")
    if a and a["p50"] > 0:
        summary["max_sustainable_analyze_fps_p95"] = round(1000 / a["p95"], 1) if a["p95"] else None
    print(json.dumps(summary, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
