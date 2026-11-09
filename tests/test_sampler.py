import pytest

from exam_monitor.config import Config
from exam_monitor.sampler import FrameSampler


def run(sampler, camera_fps, seconds, jitter=None):
    """Simulate a camera: return how many frames were selected."""
    n = int(camera_fps * seconds)
    chosen = []
    for i in range(n):
        t = i / camera_fps + (jitter(i) if jitter else 0.0)
        if sampler.should_process(i, t):
            chosen.append(i)
    return chosen


def test_every_n_picks_frame_zero_and_multiples():
    assert run(FrameSampler(every_n=3), 30, 1) == list(range(0, 30, 3))


def test_every_one_selects_everything():
    assert len(run(FrameSampler(every_n=1), 30, 1)) == 30


@pytest.mark.parametrize("camera_fps", [15, 30, 60])
def test_time_based_rate_is_independent_of_camera_fps(camera_fps):
    chosen = run(FrameSampler(target_fps=5.0), camera_fps, 20)
    assert 95 <= len(chosen) <= 105                     # ~5 per second whatever the camera does


def test_time_based_first_frame_is_taken_immediately():
    assert run(FrameSampler(target_fps=2.0), 30, 1)[0] == 0


def test_time_based_does_not_burst_after_a_stall():
    s = FrameSampler(target_fps=5.0)
    assert s.should_process(0, 0.0)
    # process froze for 30 s, then frames arrive again every 33 ms
    got = [s.should_process(i, 30.0 + i * 0.033) for i in range(1, 31)]
    assert sum(got) <= 6                                # ~1 s of frames -> ~5 picks, NOT 150 catch-up picks


def test_time_based_with_jittery_timestamps_stays_near_target():
    chosen = run(FrameSampler(target_fps=5.0), 30, 20, jitter=lambda i: 0.004 * ((i * 7) % 5 - 2))
    assert 90 <= len(chosen) <= 110


def test_counters():
    s = FrameSampler(every_n=2)
    run(s, 30, 1)
    assert s.seen == 30 and s.selected == 15


def test_validation():
    with pytest.raises(ValueError):
        FrameSampler(every_n=0)
    with pytest.raises(ValueError):
        FrameSampler(target_fps=-1)


def test_from_config_prefers_time_based_when_set():
    assert FrameSampler.from_config(Config(sample_target_fps=4.0)).target_fps == 4.0
    assert FrameSampler.from_config(Config(process_every_n=3)).every_n == 3
