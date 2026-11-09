"""Replay a video through the monitor and compare its conclusions with a ground-truth file.

    python evaluate_video.py --video sample_videos/timeline_presence.mp4
    python evaluate_video.py --all                                   # every *.truth.json in sample_videos/

Ground truth sits next to the video as <name>.truth.json:
{
  "fps": 10,
  "segments": [ {"start": 0, "end": 6, "expect": "FACE_PRESENT"},      # scored
                {"start": 6, "end": 15, "expect": "ANY"} ],            # not scored, only reported
  "expected_prolonged": [ {"min_s": 15, "max_s": 18.5} ],              # optional: recorded absences
  "present_throughout": true,                                          # optional: person never left, so NO absence may be recorded
  "param": {"name": "roll_deg", "values": [ ...one number per video frame... ]}   # optional
}
expect is one of FACE_PRESENT, FACE_ABSENT, MULTIPLE_FACES, DETECTION_UNAVAILABLE, ANY.

Use your own recordings: film a short clip, write its .truth.json by hand from a stopwatch.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from exam_monitor.config import Config
from exam_monitor.events import EventType
from exam_monitor.replay import FrameRecord, ReplayResult, confirmed_state_at, replay_video

PASS_FRACTION = 0.90        # a scored segment passes if the confirmed state matches this often
EXTRA_LAG_S = 1.0           # slack after the hold time before we start judging a segment


@dataclass(frozen=True)
class SegmentScore:
    start: float
    end: float
    expect: str
    status: str             # PASS | FAIL | SKIP (segment too short to judge) | INFO (ANY)
    match_fraction: Optional[float]
    observed: str           # what the monitor actually said


def load_truth(video: Path) -> dict:
    truth_path = video.with_suffix(".truth.json")
    if not truth_path.exists():
        raise FileNotFoundError(f"missing ground truth: {truth_path}")
    return json.loads(truth_path.read_text())


def _hold_for(expect: str, cfg: Config) -> float:
    return {"FACE_PRESENT": cfg.hold_face_present_s, "FACE_ABSENT": cfg.hold_face_absent_s,
            "MULTIPLE_FACES": cfg.hold_multiple_faces_s,
            "DETECTION_UNAVAILABLE": cfg.hold_detection_failure_s}.get(expect, 1.0)


def score_segments(result: ReplayResult, truth: dict, cfg: Config) -> list[SegmentScore]:
    out = []
    for seg in truth["segments"]:
        a, b, expect = float(seg["start"]), float(seg["end"]), seg["expect"]
        if expect == "ANY":
            c = Counter(r.observation for r in result.records if a <= r.t < b)
            n = sum(c.values()) or 1
            out.append(SegmentScore(a, b, expect, "INFO", None,
                                    ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in c.most_common())))
            continue
        lag = _hold_for(expect, cfg) + EXTRA_LAG_S
        times = [a + lag + k * 0.1 for k in range(int(max(0.0, b - a - lag) / 0.1))]
        if not times:
            out.append(SegmentScore(a, b, expect, "SKIP", None, "segment shorter than hold time + slack"))
            continue
        states = [confirmed_state_at(result.events, t) for t in times]
        frac = sum(s == expect for s in states) / len(states)
        seen = Counter(s or "none" for s in states).most_common(2)
        out.append(SegmentScore(a, b, expect, "PASS" if frac >= PASS_FRACTION else "FAIL", frac,
                                ", ".join(f"{k} {100 * v / len(states):.0f}%" for k, v in seen)))
    return out


def score_prolonged(result: ReplayResult, truth: dict) -> Optional[tuple[bool, list[float], list[dict]]]:
    """Recorded absences vs expectation. If the person never left ("present_throughout"), ANY record is false."""
    if "expected_prolonged" not in truth and not truth.get("present_throughout"):
        return None
    found = [e.details["duration_s"] for e in result.events if e.type == EventType.PROLONGED_ABSENCE]
    expected = truth.get("expected_prolonged", [])
    ok = len(found) == len(expected) and all(x["min_s"] <= d <= x["max_s"] for d, x in zip(found, expected))
    return ok, found, expected


# ------------------------------------------------------------------ detection rate by condition
_PARAM_EDGES = {
    "roll_deg": [0, 10, 20, 30, 40, 46],
    "yaw_squeeze": [0, 0.1, 0.2, 0.3, 0.4, 0.51],
}
_LUMA_EDGES = [0, 20, 30, 60, 100, 256]


def _bucket_label(edges: list[float], v: float, unit: str = "") -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"{lo:g}-{hi:g}{unit}"
    return f">={edges[-1]:g}{unit}"


def condition_table(result: ReplayResult, truth: dict) -> tuple[str, list[tuple[str, int, int, int, int]]]:
    """For frames inside ANY segments: how often did the monitor say present/absent/unavailable per condition?
    Conditions are the truth file's param (head angle) if given, otherwise measured brightness."""
    ranges = [(float(s["start"]), float(s["end"])) for s in truth["segments"] if s["expect"] == "ANY"]
    param = truth.get("param")
    rows: dict[str, Counter] = {}
    for r in result.records:
        if not any(a <= r.t < b for a, b in ranges):
            continue
        if param and param["name"] in _PARAM_EDGES:
            v = abs(param["values"][min(r.frame_index, len(param["values"]) - 1)])
            label = _bucket_label(_PARAM_EDGES[param["name"]], v)
            title = param["name"] + " (absolute value)"
        else:
            label = _bucket_label(_LUMA_EDGES, r.luma)
            title = "mean brightness (0-255)"
        rows.setdefault(label, Counter())[r.observation] += 1
    table = []
    for label, c in sorted(rows.items(), key=lambda kv: float(kv[0].split("-")[0].lstrip(">="))):
        n = sum(c.values())
        table.append((label, n, c["FACE_PRESENT"] + c["MULTIPLE_FACES"], c["FACE_ABSENT"], c["DETECTION_UNAVAILABLE"]))
    return title if rows else "", table


def evaluate(video: Path, cfg: Optional[Config] = None, quiet: bool = False) -> tuple[bool, ReplayResult]:
    truth = load_truth(video)
    cfg = cfg or Config(sample_target_fps=4.0, prolonged_absence_s=float(
        (truth.get("expected_prolonged") or [{}])[0].get("threshold_s", Config.prolonged_absence_s)))
    result = replay_video(video, cfg)
    scores = score_segments(result, truth, cfg)
    prolonged = score_prolonged(result, truth)
    ok = all(s.status in ("PASS", "SKIP", "INFO") for s in scores) and (prolonged is None or prolonged[0])

    if not quiet:
        print(f"\n=== {video.name}: {truth.get('description', '')}")
        print(f"    {result.frames_total} frames @ {result.fps:g} fps = {result.duration_s:.0f}s of video, "
              f"{result.frames_analysed} analysed ({100 * result.frames_analysed / max(1, result.frames_total):.0f}%), "
              f"replayed in {result.wall_s:.1f}s")
        print(f"    {'segment':>13s}  {'expected':22s} {'result':6s} monitor said")
        for s in scores:
            print(f"    {s.start:5.1f}-{s.end:5.1f}s  {s.expect:22s} {s.status:6s} {s.observed}")
        if prolonged:
            label = "FALSE prolonged absences (person was present throughout)" if truth.get("present_throughout") \
                else "prolonged absences recorded"
            exp = "none allowed" if truth.get("present_throughout") else f"expected {[(x['min_s'], x['max_s']) for x in prolonged[2]]}"
            print(f"    {label}: {[round(d, 1) for d in prolonged[1]]} s   {exp}   -> {'PASS' if prolonged[0] else 'FAIL'}")
        else:
            n = sum(e.type == EventType.PROLONGED_ABSENCE for e in result.events)
            print(f"    prolonged absences recorded: {n}")
        title, table = condition_table(result, truth)
        if table:
            print(f"    detection by {title}:")
            print(f"      {'bucket':>10s} {'frames':>6s} {'present':>8s} {'absent':>7s} {'unavailable':>11s}")
            for label, n, present, absent, unavail in table:
                print(f"      {label:>10s} {n:>6d} {100 * present / n:7.0f}% {100 * absent / n:6.0f}% {100 * unavail / n:10.0f}%")
        print(f"    events: {[e.type.value for e in result.events if e.type != EventType.SESSION_STARTED]}")
        print(f"    OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="one video (needs <name>.truth.json beside it)")
    ap.add_argument("--all", action="store_true", help="every video in --dir")
    ap.add_argument("--dir", default="sample_videos")
    ap.add_argument("--sample-fps", type=float, default=4.0, help="analysis rate on the video clock")
    ap.add_argument("--absence-threshold", type=float, default=None, help="override prolonged_absence_s")
    ap.add_argument("--report-only", action="store_true", help="always exit 0")
    a = ap.parse_args()

    videos = ([Path(a.video)] if a.video else
              sorted(p for p in Path(a.dir).iterdir() if p.suffix in (".mp4", ".avi") and p.with_suffix(".truth.json").exists()))
    if not videos:
        raise SystemExit("no videos found. Run: python make_test_assets.py")

    failures = 0
    for v in videos:
        cfg = None
        if a.absence_threshold is not None or a.sample_fps != 4.0:
            cfg = Config(sample_target_fps=a.sample_fps,
                         prolonged_absence_s=a.absence_threshold if a.absence_threshold is not None
                         else Config.prolonged_absence_s)
        ok, _ = evaluate(v, cfg)
        failures += not ok
    print(f"\n{len(videos) - failures}/{len(videos)} videos passed")
    return 0 if (a.report_only or failures == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
