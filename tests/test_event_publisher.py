"""Tests for the event publishing interface.

Covers:
  * Fan-out to multiple subscribers via CompositePublisher
  * Buffered async delivery with retry under temporary failures
  * Graceful degradation when a subscriber raises
  * EventBus integration with the existing EventSink protocol
  * Callback-style subscribers
  * Thread-safety of the buffered publisher
"""
from __future__ import annotations

import threading
import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from exam_monitor.config import Config
from exam_monitor.events import Event, EventType as E, ListSink
from exam_monitor.event_publisher import (
    BufferedPublisher,
    CallbackPublisher,
    CompositePublisher,
    EventBus,
)

CFG = Config()
SID = "test-session-0001"


def _make_event(etype: E = E.FACE_PRESENT, ts_mono: float = 1.0,
                frame_id: Optional[int] = 0, **details) -> Event:
    return Event(etype, SID, "2026-11-19T10:00:00.000+00:00", ts_mono, frame_id, details)


# ====================================================================== CompositePublisher

class TestCompositePublisher:

    def test_fan_out_to_multiple_sinks(self):
        sink_a, sink_b = ListSink(), ListSink()
        pub = CompositePublisher([sink_a, sink_b])
        ev = _make_event()
        pub.publish(ev)
        assert sink_a.events == [ev]
        assert sink_b.events == [ev]

    def test_single_subscriber(self):
        sink = ListSink()
        pub = CompositePublisher([sink])
        ev = _make_event()
        pub.publish(ev)
        assert sink.events == [ev]

    def test_no_subscribers_is_fine(self):
        pub = CompositePublisher([])
        pub.publish(_make_event())  # should not raise

    def test_broken_subscriber_does_not_affect_others(self):
        good = ListSink()

        class Bad:
            def write(self, event):
                raise RuntimeError("boom")

        pub = CompositePublisher([Bad(), good])
        ev = _make_event()
        pub.publish(ev)  # must not raise
        assert good.events == [ev]

    def test_close_closes_all_sinks(self):
        a, b = ListSink(), ListSink()
        # Patch close onto ListSink for this test
        a.close = MagicMock()  # type: ignore[attr-defined]
        b.close = MagicMock()  # type: ignore[attr-defined]
        pub = CompositePublisher([a, b])
        pub.close()
        a.close.assert_called_once()
        b.close.assert_called_once()

    def test_add_subscriber_at_runtime(self):
        sink = ListSink()
        pub = CompositePublisher([])
        pub.add(sink)
        ev = _make_event()
        pub.publish(ev)
        assert sink.events == [ev]

    def test_publishes_all_event_types(self):
        sink = ListSink()
        pub = CompositePublisher([sink])
        for etype in [E.FACE_PRESENT, E.FACE_ABSENT, E.MULTIPLE_FACES,
                      E.SESSION_STARTED, E.SESSION_ENDED, E.PROLONGED_ABSENCE]:
            pub.publish(_make_event(etype))
        assert len(sink.events) == 6
        assert {e.type for e in sink.events} == {E.FACE_PRESENT, E.FACE_ABSENT,
                                                  E.MULTIPLE_FACES, E.SESSION_STARTED,
                                                  E.SESSION_ENDED, E.PROLONGED_ABSENCE}


# ====================================================================== CallbackPublisher

class TestCallbackPublisher:

    def test_callback_receives_event(self):
        received = []
        pub = CallbackPublisher(received.append)
        ev = _make_event()
        pub.publish(ev)
        assert received == [ev]

    def test_callback_exception_does_not_propagate(self):
        def boom(ev):
            raise ValueError("exploded")

        pub = CallbackPublisher(boom)
        pub.publish(_make_event())  # must not raise

    def test_filters_face_events_only(self):
        received = []
        pub = CallbackPublisher(
            received.append,
            filter_types={E.FACE_PRESENT, E.FACE_ABSENT},
        )
        pub.publish(_make_event(E.FACE_PRESENT))
        pub.publish(_make_event(E.SESSION_STARTED))
        pub.publish(_make_event(E.FACE_ABSENT))
        assert len(received) == 2
        assert received[0].type == E.FACE_PRESENT
        assert received[1].type == E.FACE_ABSENT


# ====================================================================== BufferedPublisher

class TestBufferedPublisher:

    def test_async_delivery(self):
        sink = ListSink()
        pub = BufferedPublisher(sink, max_retries=0)
        ev = _make_event()
        pub.publish(ev)
        pub.close()
        assert sink.events == [ev]

    def test_temporary_failure_retries(self):
        """Sink fails twice then succeeds on the third attempt."""
        attempts = {"count": 0}
        written = []

        class FlakyNet:
            def write(self, event):
                attempts["count"] += 1
                if attempts["count"] <= 2:
                    raise ConnectionError("network down")
                written.append(event)

        pub = BufferedPublisher(FlakyNet(), max_retries=3, retry_delay_s=0.01)
        ev = _make_event()
        pub.publish(ev)
        pub.close()
        assert written == [ev]
        assert attempts["count"] == 3

    def test_permanent_failure_drops_after_retries(self):
        class AlwaysFails:
            def write(self, event):
                raise ConnectionError("permanently down")

        pub = BufferedPublisher(AlwaysFails(), max_retries=2, retry_delay_s=0.01)
        pub.publish(_make_event())
        pub.close()
        assert pub.dropped_count == 1

    def test_multiple_events_delivered_in_order(self):
        sink = ListSink()
        pub = BufferedPublisher(sink, max_retries=0)
        events = [_make_event(ts_mono=float(i)) for i in range(10)]
        for ev in events:
            pub.publish(ev)
        pub.close()
        assert sink.events == events

    def test_thread_safety(self):
        """Multiple producer threads publishing concurrently."""
        sink = ListSink()
        pub = BufferedPublisher(sink, max_retries=0)
        errors = []
        n_per_thread = 50

        def producer(tid):
            try:
                for i in range(n_per_thread):
                    pub.publish(_make_event(ts_mono=float(tid * 1000 + i)))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=producer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        pub.close()
        assert not errors
        assert len(sink.events) == 4 * n_per_thread

    def test_close_is_idempotent(self):
        sink = ListSink()
        pub = BufferedPublisher(sink, max_retries=0)
        pub.close()
        pub.close()  # second close should not raise

    def test_publish_after_close_no_crash(self):
        sink = ListSink()
        pub = BufferedPublisher(sink, max_retries=0)
        pub.close()
        pub.publish(_make_event())  # should not crash; event may be dropped


# ====================================================================== Network failure simulation

class TestNetworkFailureDelivery:
    """Simulate temporary network failures and verify events are eventually delivered."""

    def test_network_recovers_mid_stream(self):
        """Events published during an outage are retried and delivered once the network returns."""
        delivered = []
        network_up = threading.Event()
        network_up.set()  # start healthy

        class NetworkSink:
            def write(self, event):
                if not network_up.is_set():
                    raise ConnectionError("connection refused")
                delivered.append(event)

        pub = BufferedPublisher(NetworkSink(), max_retries=5, retry_delay_s=0.01)

        # Publish first event successfully
        ev1 = _make_event(E.FACE_PRESENT, ts_mono=1.0)
        pub.publish(ev1)
        time.sleep(0.05)  # let delivery thread run

        # Take down the network
        network_up.clear()
        ev2 = _make_event(E.FACE_ABSENT, ts_mono=2.0)
        pub.publish(ev2)
        time.sleep(0.05)

        # Bring network back up
        network_up.set()
        time.sleep(0.2)  # allow retries to complete

        pub.close()
        assert ev1 in delivered
        assert ev2 in delivered

    def test_intermittent_failures_do_not_lose_good_events(self):
        """Even with some events failing permanently, others still get through."""
        delivered = []
        call_count = {"n": 0}

        class IntermittentSink:
            def write(self, event):
                call_count["n"] += 1
                # Fail on the very first call for each event but succeed on retry
                if event.ts_mono == 2.0 and call_count["n"] <= 2:
                    raise IOError("transient")
                delivered.append(event)

        pub = BufferedPublisher(IntermittentSink(), max_retries=3, retry_delay_s=0.01)
        for i in range(5):
            pub.publish(_make_event(ts_mono=float(i)))
        pub.close()
        ts_values = {ev.ts_mono for ev in delivered}
        # All 5 events should eventually arrive
        assert ts_values == {0.0, 1.0, 2.0, 3.0, 4.0}


# ====================================================================== EventBus

class TestEventBus:

    def test_bus_writes_to_primary_and_publisher(self):
        primary = ListSink()
        secondary = ListSink()
        pub = CompositePublisher([secondary])
        bus = EventBus(primary, pub)
        ev = _make_event()
        bus.write(ev)
        assert primary.events == [ev]
        assert secondary.events == [ev]

    def test_bus_works_as_event_sink(self):
        """EventBus is a valid EventSink (it has write())."""
        primary = ListSink()
        bus = EventBus(primary, CompositePublisher([]))
        ev = _make_event()
        bus.write(ev)
        assert primary.events == [ev]

    def test_publisher_failure_does_not_block_primary(self):
        primary = ListSink()

        class BrokenPub:
            def publish(self, ev):
                raise RuntimeError("pub crashed")

            def close(self):
                pass

        bus = EventBus(primary, BrokenPub())  # type: ignore[arg-type]
        ev = _make_event()
        bus.write(ev)
        assert primary.events == [ev]  # primary still received it

    def test_bus_close_cascades(self):
        primary = ListSink()
        pub = CompositePublisher([])
        bus = EventBus(primary, pub)
        bus.close()  # should not raise
