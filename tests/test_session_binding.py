import threading

import pytest

from exam_monitor.config import Config
from exam_monitor.events import Event, EventType as E, ListSink
from exam_monitor.session_binding import (
    InvalidSessionId, MonitoringRegistry, SessionAlreadyMonitored, SessionMismatch,
    SessionNotMonitored, UnknownSession,
)

SID_A, SID_B = "exam-sess-0001", "exam-sess-0002"


def make(validator=None):
    sinks: dict[str, ListSink] = {}

    def factory(sid):
        sinks[sid] = ListSink()
        return sinks[sid]

    clock = iter(range(10_000)).__next__          # deterministic monotonic clock
    return MonitoringRegistry(Config(), factory, validator, clock=lambda: float(clock())), sinks


def feed(mon, obs, t0, t1, step=0.1):
    t = t0
    while t <= t1 + 1e-9:
        mon.observe(obs, t)
        t += step


@pytest.mark.parametrize("bad", ["", "short", "../../etc/passwd", "has space 12345", "abc12345\n",
                                 "a" * 65, "semi;colon-123", None, 12345678, b"bytes-id-12345"])
def test_invalid_ids_rejected(bad):
    reg, sinks = make()
    with pytest.raises(InvalidSessionId):
        reg.bind(bad)
    assert sinks == {}                              # nothing was created for a rejected ID


def test_bind_creates_status_for_that_session_only():
    reg, _ = make()
    mon = reg.bind(SID_A)
    st = reg.get_status(SID_A)
    assert st.session_id == SID_A and st.active and st.state is None
    assert mon.status().event_count == 1            # SESSION_STARTED
    with pytest.raises(SessionNotMonitored):
        reg.get_status(SID_B)


def test_duplicate_bind_rejected():
    reg, _ = make()
    reg.bind(SID_A)
    with pytest.raises(SessionAlreadyMonitored):
        reg.bind(SID_A)


def test_validator_rejection_creates_nothing():
    reg, sinks = make(validator=lambda sid: sid == SID_A)
    reg.bind(SID_A)
    with pytest.raises(UnknownSession):
        reg.bind(SID_B)
    assert SID_B not in sinks


def test_two_sessions_do_not_cross_contaminate():
    reg, sinks = make()
    a, b = reg.bind(SID_A), reg.bind(SID_B)
    feed(a, E.FACE_PRESENT, 0, 1)
    feed(b, E.FACE_ABSENT, 0, 4)
    assert reg.get_status(SID_A).state == "FACE_PRESENT"
    assert reg.get_status(SID_B).state == "FACE_ABSENT"
    assert {e.session_id for e in sinks[SID_A].events} == {SID_A}
    assert {e.session_id for e in sinks[SID_B].events} == {SID_B}


def test_sink_guard_rejects_foreign_event():
    reg, sinks = make()
    mon = reg.bind(SID_A)
    foreign = Event(E.FACE_ABSENT, SID_B, "2026-01-01T00:00:00.000+00:00", 0.0)
    with pytest.raises(SessionMismatch):
        mon._sink.write(foreign)
    assert all(e.session_id == SID_A for e in sinks[SID_A].events)
    assert mon.status().event_count == 1            # rejected event not counted


def test_release_returns_final_status_and_frees_id():
    reg, sinks = make()
    mon = reg.bind(SID_A)
    feed(mon, E.FACE_PRESENT, 0, 1)
    final = reg.release(SID_A)
    assert final.active is False and final.state == "FACE_PRESENT"
    assert sinks[SID_A].events[-1].type == E.SESSION_ENDED
    with pytest.raises(SessionNotMonitored):
        reg.get_status(SID_A)
    with pytest.raises(SessionNotMonitored):
        mon.observe(E.FACE_PRESENT, 99.0)           # stale handle cannot write after release
    reg.bind(SID_A)                                  # same ID can be bound again (resume)


def test_release_unknown_session():
    reg, _ = make()
    with pytest.raises(SessionNotMonitored):
        reg.release(SID_A)


def test_status_snapshot_is_immutable():
    reg, _ = make()
    st = reg.bind(SID_A).status()
    with pytest.raises(Exception):
        st.state = "FACE_ABSENT"


def test_concurrent_sessions_stay_isolated():
    reg, sinks = make()
    a, b = reg.bind(SID_A), reg.bind(SID_B)
    errors = []

    def worker(mon, obs):
        try:
            for i in range(2000):
                mon.observe(obs, i * 0.01)
                reg.get_status(mon.session_id)
        except Exception as exc:                    # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=worker, args=(a, E.FACE_PRESENT)),
          threading.Thread(target=worker, args=(b, E.MULTIPLE_FACES))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors
    assert reg.get_status(SID_A).state == "FACE_PRESENT"
    assert reg.get_status(SID_B).state == "MULTIPLE_FACES"
    assert {e.session_id for e in sinks[SID_A].events} == {SID_A}
    assert {e.session_id for e in sinks[SID_B].events} == {SID_B}
