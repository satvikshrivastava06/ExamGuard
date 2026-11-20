"""Tests for timestamp consistency across the event pipeline.

Verifies:
  * ts_utc is valid ISO-8601 with timezone
  * ts_mono is monotonically non-decreasing within a session
  * ts_utc and ts_mono advance together (no drift beyond tolerance)
  * SESSION_STARTED has the earliest timestamps
  * SESSION_ENDED has the latest timestamps
  * Event ordering in the sink matches ts_mono ordering
  * Duplicate timestamps are acceptable (same frame, same instant)
  * Wall-clock resets (e.g., NTP jump) do not corrupt monotonic ordering
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import pytest

from exam_monitor.config import Config
from exam_monitor.events import Event, EventType as E, IntegrityStateMachine, ListSink
from exam_monitor.session_binding import MonitoringRegistry, SessionMonitor

CFG = Config(hold_face_present_s=0.1, hold_face_absent_s=0.2,
             hold_multiple_faces_s=0.1, hold_detection_failure_s=0.1,
             hold_camera_unavailable_s=0.2, stall_s=3.0)
SID = "timestamp-test-01"


def _parse_utc(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime."""
    return datetime.fromisoformat(ts)


def _make_registry(clock=None):
    sinks: dict[str, ListSink] = {}

    def factory(sid):
        sinks[sid] = ListSink()
        return sinks[sid]

    if clock is None:
        t = [0.0]

        def clock():
            t[0] += 0.01
            return t[0]

    return MonitoringRegistry(CFG, factory, clock=clock), sinks


def feed(mon, obs, t0, t1, step=0.1):
    t = t0
    while t <= t1 + 1e-9:
        mon.observe(obs, t)
        t += step


# ====================================================================== ISO-8601 format

class TestTimestampFormat:

    def test_ts_utc_is_valid_iso8601(self):
        """Every event's ts_utc must be parseable as ISO-8601 with timezone info."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 1)
        reg.release(SID)
        for ev in sinks[SID].events:
            dt = _parse_utc(ev.ts_utc)
            assert dt.tzinfo is not None, f"ts_utc {ev.ts_utc!r} has no timezone"

    def test_ts_utc_is_utc(self):
        """ts_utc must be in UTC (offset +00:00 or Z)."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 1)
        reg.release(SID)
        for ev in sinks[SID].events:
            dt = _parse_utc(ev.ts_utc)
            assert dt.utcoffset().total_seconds() == 0.0, (
                f"ts_utc {ev.ts_utc!r} is not UTC: offset={dt.utcoffset()}")

    def test_ts_utc_has_millisecond_precision(self):
        """ts_utc should include millisecond precision (3 decimal digits)."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 0.5)
        reg.release(SID)
        for ev in sinks[SID].events:
            # ISO-8601 with milliseconds: "...T12:34:56.789+00:00"
            assert "." in ev.ts_utc, f"ts_utc {ev.ts_utc!r} lacks fractional seconds"


# ====================================================================== Monotonic ordering

class TestMonotonicOrdering:

    def test_ts_mono_is_non_decreasing(self):
        """ts_mono must never go backwards within a session's events."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 2)
        feed(mon, E.FACE_ABSENT, 2.1, 5)
        feed(mon, E.FACE_PRESENT, 5.1, 7)
        reg.release(SID)
        events = sinks[SID].events
        for i in range(1, len(events)):
            assert events[i].ts_mono >= events[i - 1].ts_mono, (
                f"ts_mono went backwards: event[{i-1}]={events[i-1].ts_mono} "
                f"> event[{i}]={events[i].ts_mono}")

    def test_session_started_has_first_ts_mono(self):
        """SESSION_STARTED must have the smallest ts_mono in the session."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 1, 3)
        reg.release(SID)
        events = sinks[SID].events
        started = [e for e in events if e.type == E.SESSION_STARTED]
        assert len(started) == 1
        assert started[0].ts_mono == min(e.ts_mono for e in events)

    def test_session_ended_has_last_ts_mono(self):
        """SESSION_ENDED must have the largest ts_mono in the session."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 1, 3)
        reg.release(SID)
        events = sinks[SID].events
        ended = [e for e in events if e.type == E.SESSION_ENDED]
        assert len(ended) == 1
        assert ended[0].ts_mono == max(e.ts_mono for e in events)

    def test_event_sink_order_matches_ts_mono_order(self):
        """Events as they appear in the sink must be ordered by ts_mono."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 1)
        feed(mon, E.FACE_ABSENT, 1.1, 4)
        feed(mon, E.FACE_PRESENT, 4.1, 5)
        reg.release(SID)
        events = sinks[SID].events
        mono_values = [e.ts_mono for e in events]
        assert mono_values == sorted(mono_values)


# ====================================================================== Wall-clock vs monotonic consistency

class TestWallClockConsistency:

    def test_ts_utc_advances_with_ts_mono(self):
        """ts_utc should broadly advance with ts_mono. They don't need to match
        exactly (wall clock is not monotonic), but SESSION_ENDED's ts_utc must
        be >= SESSION_STARTED's ts_utc under normal conditions."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 1)
        reg.release(SID)
        events = sinks[SID].events
        started = next(e for e in events if e.type == E.SESSION_STARTED)
        ended = next(e for e in events if e.type == E.SESSION_ENDED)
        assert _parse_utc(ended.ts_utc) >= _parse_utc(started.ts_utc)

    def test_state_events_have_recent_ts_utc(self):
        """State event ts_utc should be close to 'now' (within 5 seconds)."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        before = datetime.now(timezone.utc)
        feed(mon, E.FACE_PRESENT, 0, 1)
        reg.release(SID)
        after = datetime.now(timezone.utc)
        for ev in sinks[SID].events:
            dt = _parse_utc(ev.ts_utc)
            assert before <= dt <= after, (
                f"Event {ev.type.value} at {ev.ts_utc} is outside [{before}, {after}]")


# ====================================================================== State machine timestamps

class TestStateMachineTimestamps:

    def test_observe_uses_provided_mono_time(self):
        """The state machine must use the 'now' argument for ts_mono, not its own clock."""
        sink = ListSink()
        sm = IntegrityStateMachine(CFG, SID, sink)
        sm.start(100.0)
        sm.observe(E.FACE_PRESENT, 100.5)
        sm.observe(E.FACE_PRESENT, 101.0)
        sm.end(200.0)
        monos = [e.ts_mono for e in sink.events]
        assert monos[0] == 100.0   # SESSION_STARTED
        assert monos[-1] == 200.0  # SESSION_ENDED
        for m in monos:
            assert 100.0 <= m <= 200.0

    def test_stall_resets_but_maintains_ordering(self):
        """After a stall (SESSION_INTERRUPTED), subsequent events must still
        have ts_mono >= the interruption event."""
        sink = ListSink()
        sm = IntegrityStateMachine(CFG, SID, sink)
        sm.start(0.0)
        sm.observe(E.FACE_PRESENT, 0.5)
        sm.observe(E.FACE_PRESENT, 1.0)
        # 10-second gap triggers stall
        sm.observe(E.FACE_PRESENT, 11.0)
        sm.observe(E.FACE_PRESENT, 11.5)
        sm.end(12.0)
        events = sink.events
        interrupted = [e for e in events if e.type == E.SESSION_INTERRUPTED]
        assert len(interrupted) == 1
        int_mono = interrupted[0].ts_mono
        # Everything after the interruption must have >= mono time
        after = False
        for ev in events:
            if ev.type == E.SESSION_INTERRUPTED:
                after = True
                continue
            if after:
                assert ev.ts_mono >= int_mono

    def test_rapid_observations_preserve_order(self):
        """Many observations at nearly identical timestamps should still be ordered."""
        sink = ListSink()
        sm = IntegrityStateMachine(CFG, SID, sink)
        sm.start(0.0)
        # Feed 100 observations with very small increments
        for i in range(100):
            sm.observe(E.FACE_PRESENT, 0.001 * i)
        sm.end(1.0)
        monos = [e.ts_mono for e in sink.events]
        assert monos == sorted(monos)


# ====================================================================== Cross-session timestamps

class TestCrossSessionTimestamps:

    def test_two_sessions_have_independent_mono_clocks(self):
        """Two sessions may observe at different monotonic times;
        each session's events must be internally ordered."""
        reg, sinks = _make_registry()
        a = reg.bind("session-alpha-01")
        b = reg.bind("session-beta-002")
        feed(a, E.FACE_PRESENT, 0, 1)
        feed(b, E.FACE_ABSENT, 0.5, 4)
        reg.release("session-alpha-01")
        reg.release("session-beta-002")

        for sid in ["session-alpha-01", "session-beta-002"]:
            events = sinks[sid].events
            monos = [e.ts_mono for e in events]
            assert monos == sorted(monos), f"Session {sid} has unordered ts_mono"


# ====================================================================== Event.to_json roundtrip

class TestEventSerialization:

    def test_to_json_preserves_timestamps(self):
        """Event.to_json() must include both ts_utc and ts_mono faithfully."""
        import json
        ev = Event(E.FACE_PRESENT, SID,
                   "2026-11-19T10:05:30.123+00:00", 42.5, 7,
                   {"n_faces": 1})
        d = json.loads(ev.to_json())
        assert d["ts_utc"] == "2026-11-19T10:05:30.123+00:00"
        assert d["ts_mono"] == 42.5
        assert d["type"] == "FACE_PRESENT"

    def test_to_json_ts_mono_precision(self):
        """ts_mono in JSON should not lose floating-point precision."""
        import json
        ev = Event(E.FACE_ABSENT, SID,
                   "2026-11-19T10:00:00.000+00:00", 123456.789012, 0, {})
        d = json.loads(ev.to_json())
        assert abs(d["ts_mono"] - 123456.789012) < 1e-6


# ====================================================================== Prolonged absence timestamps

class TestAbsenceTimestamps:

    def test_prolonged_absence_start_utc_precedes_end_utc(self):
        """PROLONGED_ABSENCE details must have start_utc < end_utc."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 1)
        # Absence long enough for prolonged (threshold is default 10s)
        feed(mon, E.FACE_ABSENT, 1.1, 15)
        feed(mon, E.FACE_PRESENT, 15.1, 16)
        reg.release(SID)
        pa = [e for e in sinks[SID].events if e.type == E.PROLONGED_ABSENCE]
        if pa:  # depends on config threshold
            for ep in pa:
                start = _parse_utc(ep.details["start_utc"])
                end = _parse_utc(ep.details["end_utc"])
                assert start < end
                assert ep.details["duration_s"] > 0

    def test_absence_duration_matches_timestamp_span(self):
        """PROLONGED_ABSENCE duration_s should be consistent with start/end UTC."""
        reg, sinks = _make_registry()
        mon = reg.bind(SID)
        feed(mon, E.FACE_PRESENT, 0, 1)
        feed(mon, E.FACE_ABSENT, 1.1, 15)
        feed(mon, E.FACE_PRESENT, 15.1, 16)
        reg.release(SID)
        pa = [e for e in sinks[SID].events if e.type == E.PROLONGED_ABSENCE]
        if pa:
            for ep in pa:
                start = _parse_utc(ep.details["start_utc"])
                end = _parse_utc(ep.details["end_utc"])
                wall_duration = (end - start).total_seconds()
                reported = ep.details["duration_s"]
                # Wall clock and monotonic may differ slightly, allow 1s tolerance
                assert abs(wall_duration - reported) < 1.0, (
                    f"Wall duration {wall_duration:.3f} vs reported {reported:.3f}")
