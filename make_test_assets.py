"""Generate a labelled robustness test set from one base face image.

Images  -> sample_images/robustness/<category>/<name>_<N>face.png      (N = faces that really are there)
Videos  -> sample_videos/<name>.mp4  + <name>.truth.json               (what the monitor SHOULD conclude, when)

IMPORTANT: everything here is SYNTHETIC. Darkening, rotation, perspective squeeze and cut-and-paste
approximate real conditions; they do not replace recordings of real people in real rooms.
Add real, consented recordings next to these and describe them in a .truth.json (see evaluate_video.py).

Usage:  python make_test_assets.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from exam_monitor.config import Config
from exam_monitor.detector import HaarFaceDetector

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "sample_images" / "astronaut_1face.png"
BACKGROUND = ROOT / "sample_images" / "coffee_0face.png"      # a face-free, busy "room"


# =========================================================================== image operations
def brightness(img, k):
    return np.clip(img.astype(np.float32) * k, 0, 255).astype(np.uint8)


def add_offset(img, v):
    return np.clip(img.astype(np.int16) + v, 0, 255).astype(np.uint8)


def gamma(img, g):
    lut = (np.power(np.arange(256) / 255.0, g) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def contrast(img, alpha, beta=0):
    return np.clip((img.astype(np.float32) - 128) * alpha + 128 + beta, 0, 255).astype(np.uint8)


def noise(img, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


def side_light(img):
    """One side bright, the other in shadow (a window beside the desk)."""
    ramp = np.linspace(1.3, 0.25, img.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def backlit(img, box):
    """Bright light behind the person: the face is in shadow while everything around it is blown out."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - (x + bw / 2)) / (bw * 1.3)) ** 2 + ((yy - (y + bh / 2)) / (bh * 1.3)) ** 2)
    near_face = np.clip(1.2 - d, 0, 1)[..., None]               # 1 on the face, fading to 0 away from it
    out = img.astype(np.float32) * (1 - 0.65 * near_face) + 140 * (1 - near_face)
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate(img, deg):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def perspective_squeeze(img, amount, side="right", axis="yaw"):
    """Fake head turn/tilt: one edge is made shorter (yaw) or narrower (pitch) by `amount` (0..0.5)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    a = amount
    if axis == "yaw":
        dst = (np.float32([[0, 0], [w * (1 - a * 0.4), h * a / 2], [w * (1 - a * 0.4), h * (1 - a / 2)], [0, h]])
               if side == "right" else
               np.float32([[w * a * 0.4, h * a / 2], [w, 0], [w, h], [w * a * 0.4, h * (1 - a / 2)]]))
    else:  # pitch: top edge narrower, like looking down at the desk
        dst = np.float32([[w * a / 2, 0], [w * (1 - a / 2), 0], [w, h], [0, h]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def shift_out_of_frame(img, box, frac):
    """Push the image right so `frac` of the face's width is cut off by the frame edge."""
    h, w = img.shape[:2]
    x, y, bw, bh = box
    move = (w + int(bw * frac)) - (x + bw)          # moves the face's right edge `frac * bw` past the frame edge
    m = np.float32([[1, 0, move], [0, 1, 0]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def far_away(img, scale, bg):
    """Person sitting far from the camera: the face image shrunk and pasted into a face-free room."""
    h, w = bg.shape[:2]
    canvas = cv2.resize(bg, (img.shape[1], img.shape[0]))
    small = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    y0, x0 = (canvas.shape[0] - small.shape[0]) // 2, (canvas.shape[1] - small.shape[1]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def occlude(img, box, region):
    x, y, w, h = box
    out = img.copy()
    rects = {
        "mouth": (x, y + int(h * 0.62), w, int(h * 0.38)),
        "eyes": (x, y + int(h * 0.20), w, int(h * 0.28)),
        "half": (x + w // 2, y, w - w // 2, h),
    }
    rx, ry, rw, rh = rects[region]
    cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (45, 45, 45), -1)    # a hand, a mask, a notebook
    return out


def downscale(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# =========================================================================== image set
def build_image_variants(base: np.ndarray, background: np.ndarray) -> Iterator[tuple[str, str, np.ndarray, int]]:
    """Yield (category, name, image, expected_face_count)."""
    box = HaarFaceDetector(Config()).detect(base)[0].box          # where the face really is in the base image
    rng = np.random.default_rng(1)

    yield "control", "normal", base, 1

    for name, img in [("dim_x0.5", brightness(base, 0.5)), ("dim_x0.3", brightness(base, 0.3)),
                      ("dark_x0.15", brightness(base, 0.15)), ("very_dark_x0.05", brightness(base, 0.05)),
                      ("gamma_2.0", gamma(base, 2.0)), ("bright_x1.5", brightness(base, 1.5)),
                      ("overexposed_+90", add_offset(base, 90)), ("washed_out", contrast(base, 0.35, 40)),
                      ("low_contrast", contrast(base, 0.4)), ("noise_s25", noise(base, 25)),
                      ("noise_s50", noise(base, 50)), ("side_light", side_light(base)), ("backlit", backlit(base, box)),
                      ("defocus_k9", blur(base, 9)), ("defocus_k21", blur(base, 21))]:
        yield "lighting", name, img, 1

    for deg in (15, 30, 45):
        yield "angle_roll", f"roll_+{deg}", rotate(base, deg), 1
        yield "angle_roll", f"roll_-{deg}", rotate(base, -deg), 1
    for amt in (0.2, 0.4):
        yield "angle_yaw", f"yaw_squeeze_{amt}", perspective_squeeze(base, amt, "right", "yaw"), 1
        yield "angle_pitch", f"pitch_squeeze_{amt}", perspective_squeeze(base, amt, axis="pitch"), 1
    yield "angle_other", "mirrored", cv2.flip(base, 1), 1
    yield "angle_other", "upside_down", cv2.rotate(base, cv2.ROTATE_180), 1

    for frac in (0.3, 0.5):
        yield "framing", f"cut_off_{int(frac * 100)}pct", shift_out_of_frame(base, box, frac), 1
    for sc in (0.5, 0.3, 0.2):
        yield "distance", f"far_scale_{sc}", far_away(base, sc, background), 1

    for region in ("mouth", "eyes", "half"):
        yield "occlusion", f"occluded_{region}", occlude(base, box, region), 1

    for size in (320, 160, 96):
        yield "resolution", f"res_{size}px", downscale(base, size), 1

    # Negative controls: nothing here is a face, so any detection is a false positive.
    yield "negative", "room_no_person", cv2.resize(background, (512, 512)), 0
    yield "negative", "uniform_wall", np.full((512, 512, 3), 140, np.uint8), 0
    yield "negative", "random_noise", rng.integers(0, 256, (512, 512, 3), dtype=np.uint8), 0
    stripes = (np.indices((512, 512)).sum(axis=0) // 32 % 2 * 255).astype(np.uint8)
    yield "negative", "diagonal_stripes", cv2.cvtColor(stripes, cv2.COLOR_GRAY2BGR), 0
    yield "negative", "dark_empty_room", brightness(cv2.resize(background, (512, 512)), 0.04), 0
    yield "negative", "blown_out_blank", np.full((512, 512, 3), 252, np.uint8), 0


def write_images(out_dir: Path) -> int:
    base, bg = cv2.imread(str(BASE)), cv2.imread(str(BACKGROUND))
    if base is None or bg is None:
        raise SystemExit(f"need {BASE} and {BACKGROUND}")
    n = 0
    for cat, name, img, expected in build_image_variants(base, bg):
        d = out_dir / cat
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{name}_{expected}face.png"), img)
        n += 1
    return n


# =========================================================================== videos
W, H, FPS = 480, 360, 10


def _fit(img):
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def _person(base):                      # the face image centred on a grey wall, full height
    canvas = np.full((H, W, 3), 110, np.uint8)
    s = cv2.resize(base, (H, H), interpolation=cv2.INTER_AREA)
    x0 = (W - H) // 2
    canvas[:, x0:x0 + H] = s
    return canvas


def _timeline(segments: list[tuple[float, float, str]], render: Callable[[str, float], np.ndarray]):
    def gen():
        for i in range(int(segments[-1][1] * FPS)):
            t = i / FPS
            kind = next(k for a, b, k in segments if a <= t < b)
            yield render(kind, t)
    return gen


def _save(path: Path, make_frames, truth: dict) -> Path:
    for fourcc, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        target = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*fourcc), FPS, (W, H))
        if not writer.isOpened():
            continue
        n = 0
        for f in make_frames():
            writer.write(f)
            n += 1
        writer.release()
        if n and target.exists() and target.stat().st_size > 0:
            truth = {**truth, "video": target.name, "fps": FPS}
            target.with_suffix(".truth.json").write_text(json.dumps(truth, indent=1))
            return target
    raise RuntimeError(f"no usable video encoder in this OpenCV build for {path.name}")


def write_videos(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base, bg = cv2.imread(str(BASE)), cv2.imread(str(BACKGROUND))
    person, room = _person(base), _fit(bg)
    two = _fit(np.hstack([base, base]))
    made = []

    # 1. presence / absence / dark, with the ground truth for every second
    segs = [(0, 6, "present"), (6, 12, "absent"), (12, 18, "present"), (18, 36, "absent"),
            (36, 42, "present"), (42, 48, "dark"), (48, 54, "present")]
    render = lambda k, t: {"present": person, "absent": room, "dark": brightness(person, 0.04)}[k]
    made.append(_save(out_dir / "timeline_presence", _timeline(segs, render), {
        "description": "present 6s | absent 6s (short: state yes, record no) | present 6s | absent 18s (recorded) | present 6s | too dark 6s | present 6s",
        "segments": [{"start": 0, "end": 6, "expect": "FACE_PRESENT"}, {"start": 6, "end": 12, "expect": "FACE_ABSENT"},
                     {"start": 12, "end": 18, "expect": "FACE_PRESENT"}, {"start": 18, "end": 36, "expect": "FACE_ABSENT"},
                     {"start": 36, "end": 42, "expect": "FACE_PRESENT"}, {"start": 42, "end": 48, "expect": "DETECTION_UNAVAILABLE"},
                     {"start": 48, "end": 54, "expect": "FACE_PRESENT"}],
        "expected_prolonged": [{"min_s": 17.0, "max_s": 18.6, "threshold_s": 10.0}]}))

    # 2. one false-positive flash inside a long absence must NOT split it into two records
    segs = [(0, 4, "present"), (4, 15, "absent"), (15, 15.3, "present"), (15.3, 30, "absent"), (30, 36, "present")]
    made.append(_save(out_dir / "blip_in_absence", _timeline(segs, render), {
        "description": "absent 4-30s with a 0.3s flash of a face at 15s (a poster, a reflection)",
        "segments": [{"start": 0, "end": 4, "expect": "FACE_PRESENT"}, {"start": 4, "end": 30, "expect": "FACE_ABSENT"},
                     {"start": 30, "end": 36, "expect": "FACE_PRESENT"}],
        "expected_prolonged": [{"min_s": 24.0, "max_s": 26.5, "threshold_s": 10.0}]}))

    # 3. two people
    segs = [(0, 5, "one"), (5, 12, "two"), (12, 18, "one")]
    render2 = lambda k, t: {"one": person, "two": two}[k]
    made.append(_save(out_dir / "two_people", _timeline(segs, render2), {
        "description": "one person | two people side by side | one person",
        "segments": [{"start": 0, "end": 5, "expect": "FACE_PRESENT"}, {"start": 5, "end": 12, "expect": "MULTIPLE_FACES"},
                     {"start": 12, "end": 18, "expect": "FACE_PRESENT"}]}))

    # 4. lighting ramp: person present the whole time, light fades to almost nothing and returns
    def factor(t):
        if t < 5 or t >= 30:
            return 1.0
        if t < 15:
            return math.exp(math.log(0.03) * (t - 5) / 10)
        if t < 20:
            return 0.03
        return math.exp(math.log(0.03) * (30 - t) / 10)

    def ramp():
        for i in range(35 * FPS):
            yield brightness(person, factor(i / FPS))
    made.append(_save(out_dir / "lighting_ramp", ramp, {
        "description": "person present throughout; light fades x1.0 -> x0.03 over 10s, holds 5s, returns over 10s",
        "present_throughout": True,
        "segments": [{"start": 0, "end": 5, "expect": "FACE_PRESENT"}, {"start": 5, "end": 15, "expect": "ANY"},
                     {"start": 15, "end": 20, "expect": "DETECTION_UNAVAILABLE"}, {"start": 20, "end": 30, "expect": "ANY"},
                     {"start": 30, "end": 35, "expect": "FACE_PRESENT"}],
        "param": {"name": "light_factor", "values": [round(factor(i / FPS), 4) for i in range(35 * FPS)]}}))

    # 5/6. head-angle sweeps (synthetic: in-plane roll, and a perspective squeeze that fakes a head turn)
    def sweep_value(t, peak):
        phase = t % 40
        return peak * (phase / 10 if phase < 10 else (20 - phase) / 10 if phase < 20
                       else -(phase - 20) / 10 if phase < 30 else -(40 - phase) / 10)

    def roll_frames():
        for i in range(40 * FPS):
            yield rotate(person, sweep_value(i / FPS, 45))

    def yaw_frames():
        for i in range(40 * FPS):
            yield perspective_squeeze(person, abs(sweep_value(i / FPS, 0.5)), "right" if sweep_value(i / FPS, 1) >= 0 else "left", "yaw")

    made.append(_save(out_dir / "angle_sweep_roll", roll_frames, {
        "description": "person present throughout; head rolled 0 -> +45 -> 0 -> -45 -> 0 degrees (in-plane)",
        "present_throughout": True,
        "segments": [{"start": 0, "end": 40, "expect": "ANY"}],
        "param": {"name": "roll_deg", "values": [round(sweep_value(i / FPS, 45), 2) for i in range(40 * FPS)]}}))
    made.append(_save(out_dir / "angle_sweep_yaw", yaw_frames, {
        "description": "person present throughout; fake head-turn (perspective squeeze 0 -> 0.5 and back, both sides)",
        "present_throughout": True,
        "segments": [{"start": 0, "end": 40, "expect": "ANY"}],
        "param": {"name": "yaw_squeeze", "values": [round(abs(sweep_value(i / FPS, 0.5)), 3) for i in range(40 * FPS)]}}))
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=str(ROOT / "sample_images" / "robustness"))
    ap.add_argument("--videos", default=str(ROOT / "sample_videos"))
    ap.add_argument("--skip-videos", action="store_true")
    a = ap.parse_args()
    print(f"wrote {write_images(Path(a.images))} images to {a.images}")
    if not a.skip_videos:
        for v in write_videos(Path(a.videos)):
            print("wrote", v)


if __name__ == "__main__":
    main()
