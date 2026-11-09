"""Lighting, camera angle, occlusion and sample-video tests.

Three kinds of expectation:
  MUST          regression guard: this has to keep working
  SAFE FAILURE  when the monitor cannot judge it must SAY so (DETECTION_UNAVAILABLE), never claim "absent"
  KNOWN LIMIT   documented weakness of a frontal Haar detector. Marked xfail(strict=False): it does not break
                the build, and if an improvement fixes one it shows up as XPASS so the docs can be updated.

All inputs are SYNTHETIC (see make_test_assets.py). They show how the pipeline reacts to the conditions,
not how real students on real cameras will behave.
"""
from pathlib import Path

import pytest

from evaluate_images import evaluate_dir
from evaluate_video import evaluate
from exam_monitor.config import Config
from exam_monitor.detector import HaarFaceDetector
from exam_monitor.events import EventType as E
from make_test_assets import write_images, write_videos

# ------------------------------------------------------------------ expectations (from measured results)
MUST_DETECT = [
    "normal", "mirrored", "roll_+15", "roll_-15", "pitch_squeeze_0.2", "pitch_squeeze_0.4",
    "yaw_squeeze_0.2", "yaw_squeeze_0.4", "dim_x0.5", "dim_x0.3", "gamma_2.0", "bright_x1.5",
    "overexposed_+90", "washed_out", "low_contrast", "noise_s25", "noise_s50", "side_light",
    "defocus_k9", "defocus_k21", "res_320px", "res_160px", "res_96px", "far_scale_0.5",
]
MUST_BE_FLAGGED = ["very_dark_x0.05", "dark_x0.15", "dark_empty_room", "blown_out_blank"]
KNOWN_LIMITS = [
    "roll_+30", "roll_-30", "roll_+45", "roll_-45", "upside_down", "far_scale_0.3", "far_scale_0.2",
    "cut_off_30pct", "cut_off_50pct", "occluded_eyes", "occluded_mouth", "occluded_half", "backlit",
]


@pytest.fixture(scope="module")
def image_results(tmp_path_factory):
    root = tmp_path_factory.mktemp("robustness")
    write_images(root)
    cfg = Config()
    return {r.name: r for r in evaluate_dir(root, HaarFaceDetector(cfg), cfg)}


# ------------------------------------------------------------------ images
@pytest.mark.parametrize("name", MUST_DETECT)
def test_detects_under_ordinary_variation(image_results, name):
    assert image_results[name].outcome == "detected", image_results[name]


@pytest.mark.parametrize("name", MUST_BE_FLAGGED)
def test_unusable_frames_are_flagged_not_called_absent(image_results, name):
    r = image_results[name]
    assert r.observation == E.DETECTION_UNAVAILABLE.value and r.outcome == "flagged", r


def test_no_false_positives_on_negative_controls(image_results):
    negatives = [r for r in image_results.values() if r.category == "negative"]
    assert len(negatives) == 6
    assert all(r.faces == 0 for r in negatives), [r.name for r in negatives if r.faces]


@pytest.mark.parametrize("name", KNOWN_LIMITS)
@pytest.mark.xfail(strict=False, reason="known limit of a frontal Haar cascade (see docs/sprint3)")
def test_known_limit_is_still_a_limit(image_results, name):
    assert image_results[name].outcome == "detected"


def test_every_silent_miss_is_a_documented_one(image_results):
    """A NEW way to miss a face silently is a regression; the known ones are listed above."""
    silent = {r.name for r in image_results.values() if r.outcome == "silent_miss"}
    assert silent <= set(KNOWN_LIMITS), f"undocumented silent misses: {sorted(silent - set(KNOWN_LIMITS))}"


def test_darkness_boundary_matches_config(image_results):
    cfg = Config()
    for r in image_results.values():
        if r.luma < cfg.dark_mean_luma or r.luma > cfg.bright_mean_luma:
            assert r.observation == E.DETECTION_UNAVAILABLE.value, r
        else:
            assert r.observation != E.DETECTION_UNAVAILABLE.value, r


# ------------------------------------------------------------------ videos
@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    out = tmp_path_factory.mktemp("videos")
    return {p.stem: p for p in write_videos(out)}


@pytest.mark.slow
def test_video_presence_absence_dark_timeline(videos):
    ok, res = evaluate(videos["timeline_presence"], quiet=True)
    assert ok
    types = [e.type for e in res.events]
    assert types.count(E.FACE_ABSENT) == 2                    # the 6 s and the 18 s absence both became states...
    assert types.count(E.PROLONGED_ABSENCE) == 1              # ...but only the 18 s one was recorded
    assert E.DETECTION_UNAVAILABLE in types and types[-1] == E.SESSION_ENDED


@pytest.mark.slow
def test_video_flash_inside_long_absence_stays_one_record(videos):
    ok, res = evaluate(videos["blip_in_absence"], quiet=True)
    assert ok and [e.type for e in res.events].count(E.PROLONGED_ABSENCE) == 1


@pytest.mark.slow
def test_video_two_people_is_multiple_faces(videos):
    ok, res = evaluate(videos["two_people"], quiet=True)
    assert ok and E.MULTIPLE_FACES in [e.type for e in res.events]


@pytest.mark.slow
def test_video_fading_light_never_reports_absence(videos):
    ok, res = evaluate(videos["lighting_ramp"], quiet=True)
    assert ok
    assert not [r for r in res.records if r.luma < 20 and r.observation != E.DETECTION_UNAVAILABLE.value]
    assert E.FACE_ABSENT not in [e.type for e in res.events]


@pytest.mark.slow
def test_video_head_turn_does_not_create_a_false_absence_record(videos):
    ok, res = evaluate(videos["angle_sweep_yaw"], quiet=True)
    assert ok, [e.details for e in res.events if e.type == E.PROLONGED_ABSENCE]


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="known limit: a frontal Haar cascade cannot see a head rolled >= ~30 degrees, "
                                        "so a 10 s+ tilt is recorded as an absence")
def test_video_head_roll_does_not_create_a_false_absence_record(videos):
    ok, res = evaluate(videos["angle_sweep_roll"], quiet=True)
    assert ok


@pytest.mark.slow
def test_replay_is_deterministic(videos):
    a = evaluate(videos["two_people"], quiet=True)[1]
    b = evaluate(videos["two_people"], quiet=True)[1]
    key = lambda r: [(e.type.value, round(e.ts_mono, 3)) for e in r.events]
    assert key(a) == key(b)
