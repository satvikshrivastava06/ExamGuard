import pytest

from exam_monitor.absence import AbsenceTracker
from exam_monitor.config import Config
from exam_monitor.events import EventType as E, ListSink
from exam_monitor.session_binding import MonitoringRegistry

SID = "exam-sess-0001"


# ===================================================================== tracker on its own
def test_leave_without_enter_is_nothing():
    assert AbsenceTracker(5).leave(10.0, "face_returned") is None


def test_below_threshold_not_recorded_at_or_above_recorded():
    tr = AbsenceTracker(10.0)
    tr.enter(0.0, 3.0)
    assert tr.leave(9.99, "face_returned") is None
    tr.enter(0.0, 3.0)
    ep = tr.leave(10.0, "face_returned")                     # boundary is inclusive
    assert ep is not None and ep.duration_s == 10.0


def test_enter_twice_keeps_the_first_start():
    tr = AbsenceTracker(1.0)
    tr.enter(2.0, 5.0)
    tr.enter(4.0, 6.0)                                       # ignored
    assert tr.leave(12.0, "x").duration_s == 10.0


def test_start_time_is_when_absence_began_not_when_it_was_confirmed():
    tr = AbsenceTracker(1.0)
    tr.enter(started=10.0, now=13.0)                         # confirmed 3 s after it began
    assert tr.ongoing_s(15.0) == 5.0                         # measured from 10, not 13


def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        AbsenceTracker(-1)


# ===================================================================== through the session monitor
def make(threshold=10.0, **kw):
    cfg = Config(prolonged_absence_s=threshold, hold_face_present_s=0.5, hold_face_absent_s=3.0,
                 hold_detection_failure_s=1.0, stall_s=2.0, **kw)
    sink = ListSink()
    reg = MonitoringRegistry(cfg, lambda sid: sink, clock=lambda: 0.0)
    return reg, reg.bind(SID), sink


def feed(mon, obs, t0, t1, step=0.1):
    t = t0
    while t <= t1 + 1e-9:
        mon.observe(obs, round(t, 6))
        t += step


def records(sink):
    return [e for e in sink.events if e.type == E.PROLONGED_ABSENCE]


def test_long_absence_recorded_once_with_session_id_and_status():
    reg, mon, sink = make()
    feed(mon, E.FACE_PRESENT, 0, 2)
    feed(mon, E.FACE_ABSENT, 2.1, 22)
    assert mon.status().absence_ongoing_s > 18                 # live view while it is happening
    feed(mon, E.FACE_PRESENT, 22.1, 24)
    (ev,) = records(sink)
    assert ev.session_id == SID
    d = ev.details
    assert 19.5 < d["duration_s"] < 20.5 and d["threshold_s"] == 10.0 and d["end_reason"] == "face_returned"
    st = mon.status()
    assert st.prolonged_absence_count == 1 and st.absence_ongoing_s == 0.0
    assert 19.5 < st.prolonged_absence_total_s < 20.5


def test_short_absence_gives_absent_state_but_no_record():
    reg, mon, sink = make()
    feed(mon, E.FACE_PRESENT, 0, 2)
    feed(mon, E.FACE_ABSENT, 2.1, 8)                           # ~6 s: confirmed (>3 s), not recorded (<10 s)
    feed(mon, E.FACE_PRESENT, 8.1, 10)
    types = [e.type for e in sink.events]
    assert E.FACE_ABSENT in types and E.PROLONGED_ABSENCE not in types


def test_threshold_is_configurable():
    for threshold, expected in [(4.0, 1), (60.0, 0)]:
        reg, mon, sink = make(threshold)
        feed(mon, E.FACE_PRESENT, 0, 1)
        feed(mon, E.FACE_ABSENT, 1.1, 13)                      # ~12 s
        feed(mon, E.FACE_PRESENT, 13.1, 15)
        assert len(records(sink)) == expected


def test_flickering_detector_never_becomes_a_fake_absence():
    """REGRESSION. Found by the angle-sweep videos: detection alternating present/absent at the edge of
    the detector's range was stitched into one long 'absence' although the person never left."""
    reg, mon, sink = make()
    t = 0.0
    while t < 60:                                              # 0.25 s absent, 0.25 s present, for a minute
        mon.observe(E.FACE_ABSENT, round(t, 3))
        mon.observe(E.FACE_PRESENT, round(t + 0.25, 3))
        t += 0.5
    types = [e.type for e in sink.events]
    assert E.PROLONGED_ABSENCE not in types and E.FACE_ABSENT not in types


def test_brief_false_positive_inside_a_long_absence_does_not_split_it():
    reg, mon, sink = make()
    feed(mon, E.FACE_PRESENT, 0, 2)
    feed(mon, E.FACE_ABSENT, 2.1, 14.9)
    feed(mon, E.FACE_PRESENT, 15.0, 15.3)                      # 0.3 s flash (a poster, a reflection) < 0.5 s hold
    feed(mon, E.FACE_ABSENT, 15.4, 28)
    feed(mon, E.FACE_PRESENT, 28.1, 30)
    (ev,) = records(sink)
    assert ev.details["duration_s"] > 25                       # one record, not 13 s + 12 s


def test_a_real_return_ends_the_absence_and_a_second_one_is_a_new_record():
    reg, mon, sink = make(threshold=5.0)
    feed(mon, E.FACE_PRESENT, 0, 2)
    feed(mon, E.FACE_ABSENT, 2.1, 12)
    feed(mon, E.FACE_PRESENT, 12.1, 16)                        # 4 s back: a real return
    feed(mon, E.FACE_ABSENT, 16.1, 30)
    feed(mon, E.FACE_PRESENT, 30.1, 32)
    assert len(records(sink)) == 2


@pytest.mark.parametrize("blind,reason", [(E.DETECTION_UNAVAILABLE, "detection_unavailable"),
                                          (E.CAMERA_UNAVAILABLE, "camera_unavailable"),
                                          (E.CAMERA_PERMISSION_DENIED, "camera_unavailable")])
def test_blind_time_is_never_counted_as_absence(blind, reason):
    reg, mon, sink = make(threshold=5.0)
    feed(mon, E.FACE_PRESENT, 0, 1)
    feed(mon, E.FACE_ABSENT, 1.1, 12)
    feed(mon, blind, 12.1, 90)                                 # 78 s outage right after the absence
    (ev,) = records(sink)
    assert ev.details["end_reason"] == reason
    assert 10.5 < ev.details["duration_s"] < 12.0              # the outage is NOT added


def test_absence_does_not_resume_across_a_blind_gap():
    reg, mon, sink = make(threshold=10.0)
    feed(mon, E.FACE_PRESENT, 0, 1)
    feed(mon, E.FACE_ABSENT, 1.1, 8)                           # 7 s absent
    feed(mon, E.DETECTION_UNAVAILABLE, 8.1, 11)                # blind
    feed(mon, E.FACE_ABSENT, 11.1, 18)                         # 7 s absent again: 14 s total must NOT be reported
    feed(mon, E.FACE_PRESENT, 18.1, 20)
    assert records(sink) == []


def test_stall_ends_the_absence_without_counting_the_gap_then_tracking_resumes():
    reg, mon, sink = make(threshold=5.0)
    feed(mon, E.FACE_PRESENT, 0, 1)
    feed(mon, E.FACE_ABSENT, 1.1, 12)
    feed(mon, E.FACE_ABSENT, 600.0, 612)                       # process frozen ~10 minutes, still no face
    feed(mon, E.FACE_PRESENT, 612.1, 614)
    first, second = records(sink)
    assert first.details["end_reason"] == "interrupted" and first.details["duration_s"] < 12     # gap not counted
    assert 11 < second.details["duration_s"] < 13                                               # new episode after the stall


def test_student_never_appears_and_session_ends_during_absence():
    reg, mon, sink = make()
    feed(mon, E.FACE_ABSENT, 0, 15)                            # absent from the very first frame
    final = reg.release(SID)
    assert final.absence_ongoing_s == 0.0                      # a closed session has no absence in progress
    ev = [e for e in sink.events if e.type == E.PROLONGED_ABSENCE]
    assert len(ev) == 1 and ev[0].details["end_reason"] == "session_ended"
    assert 14.5 < ev[0].details["duration_s"] <= 15.0
    assert [e.type for e in sink.events][-2:] == [E.PROLONGED_ABSENCE, E.SESSION_ENDED]    # ENDED still last


def test_release_during_a_short_absence_records_nothing():
    reg, mon, sink = make()
    feed(mon, E.FACE_PRESENT, 0, 1)
    feed(mon, E.FACE_ABSENT, 1.1, 6)
    reg.release(SID)
    assert E.PROLONGED_ABSENCE not in [e.type for e in sink.events]
