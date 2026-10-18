import pytest

from exam_monitor.config import Config
from exam_monitor.events import EventType as E, IntegrityStateMachine, ListSink

CFG = Config()


def make():
    sink = ListSink()
    return IntegrityStateMachine(CFG, "t", sink), sink


def feed(sm, obs, t0, t1, step=0.1):
    t = t0
    while t <= t1 + 1e-9:
        sm.observe(obs, t)
        t += step


def types(sink):
    return [e.type for e in sink.events]


def test_face_present_after_hold():
    sm, sink = make()
    feed(sm, E.FACE_PRESENT, 0, 0.4)
    assert types(sink) == []
    feed(sm, E.FACE_PRESENT, 0.5, 1.0)
    assert types(sink) == [E.FACE_PRESENT]


def test_single_missed_detection_does_not_fire_absent():
    sm, sink = make()
    feed(sm, E.FACE_PRESENT, 0, 1)
    feed(sm, E.FACE_ABSENT, 1.1, 1.5)       # 0.4 s blip
    feed(sm, E.FACE_PRESENT, 1.6, 3)
    assert E.FACE_ABSENT not in types(sink)


def test_absent_fires_once_after_hold_and_carries_previous_duration():
    sm, sink = make()
    feed(sm, E.FACE_PRESENT, 0, 5)
    feed(sm, E.FACE_ABSENT, 5.1, 12)
    absent = [e for e in sink.events if e.type == E.FACE_ABSENT]
    assert len(absent) == 1                  # entry event only, not one per frame
    assert absent[0].details["previous_state"] == "FACE_PRESENT"
    assert absent[0].details["previous_duration_s"] > 4


def test_recovery_emits_face_present():
    sm, sink = make()
    feed(sm, E.FACE_PRESENT, 0, 1)
    feed(sm, E.FACE_ABSENT, 1.1, 5)
    feed(sm, E.FACE_PRESENT, 5.1, 6)
    assert types(sink)[-2:] == [E.FACE_ABSENT, E.FACE_PRESENT]


def test_stall_emits_interruption_and_resets_debounce():
    sm, sink = make()
    feed(sm, E.FACE_PRESENT, 0, 1)
    sm.observe(E.FACE_ABSENT, 1.1)           # candidate starts
    sm.observe(E.FACE_ABSENT, 30.0)          # 29 s gap: must NOT confirm absence from the gap
    assert E.SESSION_INTERRUPTED in types(sink)
    assert E.FACE_ABSENT not in types(sink)


def test_camera_unavailable_and_multiple_faces():
    sm, sink = make()
    feed(sm, E.MULTIPLE_FACES, 0, 2)
    feed(sm, E.CAMERA_UNAVAILABLE, 2.1, 5)
    assert types(sink) == [E.MULTIPLE_FACES, E.CAMERA_UNAVAILABLE]


def test_non_state_observation_rejected():
    sm, _ = make()
    with pytest.raises(ValueError):
        sm.observe(E.SESSION_STARTED, 0)
