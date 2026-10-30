"""Preflight: is the camera usable, and if not, why? Run this before an exam session.

Exit code 0 = usable, 1 = problem (details and a fix on stdout).
"""
from __future__ import annotations

import argparse
import json

from exam_monitor.camera_probe import check_camera
from exam_monitor.config import Config


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="0", help="webcam index, video file, or image file")
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for the camera to open")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = check_camera(int(a.source) if a.source.isdigit() else a.source, Config(), open_timeout_s=a.timeout)
    if a.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"camera check: {'OK' if res.ok else 'PROBLEM'}  [{res.problem.value}]")
        print(f"  platform  : {res.platform}")
        print(f"  permission: {res.permission.value}")
        print(f"  detail    : {res.detail}")
        if res.hint:
            print(f"  what to do: {res.hint}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
