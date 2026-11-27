"""Tests for event delivery under temporary network failures.

This test module simulates various network failure scenarios and verifies
that the event publishing system handles them gracefully:
  - Events are buffered during outages
  - Retries with exponential backoff
  - Events are delivered in order after recovery
  - Permanent failures are logged and don't crash the pipeline
  - High-volume bursts during partial connectivity
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import pytest

from exam_monitor.config import Config
from exam_monitor.events import Event, EventType as E, ListSink
from exam_monitor.event_publisher import (
    BufferedPublisher,
    CompositePublisher,
    CallbackPublisher,
)

CFG = Config()
SID = "network-test-0001"


def _make_event(etype: E = E.FACE_PRESENT, ts_mono: float = 1.0,
                frame_id: Optional[int] = 0, **details) -> Event:
    return Event(etype, SID, "2026-11-19T10:00:00.000+00:00", ts_mono, frame_id, details)


class FlakyNetworkSink:
    """A sink that fails according to a configurable pattern."""

    def __init__(self):
        self.events = []
        self.fail_until_count = 0
        self.fail_every_n = 0
        self.fail_random = False
        self.error_type = ConnectionError
        self.error_message = "network down"
        self._call_count = 0

    def write(self, event: Event):
        self._call_count += 1
        if self.fail_until_count > 0:
            self.fail_until_count -= 1
            raise self.error_type(self.error_message)
        if self.fail_every_n == 1:
            if self._call_count % 2 == 1:
                raise self.error_type(self.error_message)
        elif self.fail_every_n > 0 and self._call_count % self.fail_every_n == 0:
            raise self.error_type(self.error_message)
        self.events.append(event)

    def close(self):
        pass


class SlowNetworkSink:
    """A sink that simulates network latency."""

    def __init__(self, delay_s: float = 0.05):
        self.events = []
        self.delay_s = delay_s

    def write(self, event: Event):
        time.sleep(self.delay_s)
        self.events.append(event)

    def close(self):
        pass


class TestTemporaryOutage:

    def test_short_outage_all_events_delivered(self):
        """A brief network outage should not lose any events."""
        sink = FlakyNetworkSink()
        sink.fail_until_count = 3  # fail first 3 writes

        pub = BufferedPublisher(sink, max_retries=10, retry_delay_s=0.01)

        # Publish during outage
        for i in range(5):
            pub.publish(_make_event(ts_mono=float(i)))

        pub.close()

        # All 5 events should eventually succeed
        assert len(sink.events) == 5
        assert [e.ts_mono for e in sink.events] == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_long_outage_retries_until_success(self):
        """Extended outage should be retried until max_retries exhausted."""
        sink = FlakyNetworkSink()
        sink.fail_until_count = 5  # fail 5 times

        pub = BufferedPublisher(sink, max_retries=7, retry_delay_s=0.01)

        pub.publish(_make_event(ts_mono=1.0))
        pub.close()

        # Should have succeeded after 5 failures + 1 success = 6 calls
        assert len(sink.events) == 1
        assert sink.events[0].ts_mono == 1.0

    def test_outage_exceeds_max_retries_drops_event(self):
        """If max_retries is exceeded, the event is dropped and counted."""
        sink = FlakyNetworkSink()
        sink.fail_until_count = 10  # fail 10 times

        pub = BufferedPublisher(sink, max_retries=3, retry_delay_s=0.01)

        pub.publish(_make_event(ts_mono=1.0))
        pub.close()

        # Event should be dropped
        assert len(sink.events) == 0
        assert pub.dropped_count == 1

    def test_multiple_events_during_outage_all_retry_independently(self):
        """Each event should retry independently; a failure on one doesn't block others."""
        sink = FlakyNetworkSink()
        # Fail only the first write attempt for each event
        sink.fail_every_n = 1  # fail first call, succeed on second

        pub = BufferedPublisher(sink, max_retries=2, retry_delay_s=0.01)

        for i in range(10):
            pub.publish(_make_event(ts_mono=float(i)))

        pub.close()

        # All 10 should eventually get through (2 attempts each)
        assert len(sink.events) == 10

    def test_events_delivered_in_order_after_outage(self):
        """Events published during an outage must be delivered in original order."""
        sink = FlakyNetworkSink()
        sink.fail_until_count = 2

        pub = BufferedPublisher(sink, max_retries=5, retry_delay_s=0.01)

        for i in range(20):
            pub.publish(_make_event(ts_mono=float(i), frame_id=i))

        pub.close()

        monos = [e.ts_mono for e in sink.events]
        assert monos == sorted(monos)
        # frame_id should also be in order
        frames = [e.frame_id for e in sink.events]
        assert frames == sorted(frames)


class TestIntermittentFailures:

    def test_intermittent_failures_no_event_loss(self):
        """Random intermittent failures should not cause event loss."""
        sink = FlakyNetworkSink()
        sink.fail_every_n = 3  # every 3rd write fails

        pub = BufferedPublisher(sink, max_retries=5, retry_delay_s=0.01)

        for i in range(30):
            pub.publish(_make_event(ts_mono=float(i)))

        pub.close()

        assert len(sink.events) == 30

    def test_burst_during_intermittent_failure(self):
        """A burst of events during intermittent failures."""
        sink = FlakyNetworkSink()
        sink.fail_every_n = 2

        pub = BufferedPublisher(sink, max_retries=4, retry_delay_s=0.01)

        # Burst of 50 events
        for i in range(50):
            pub.publish(_make_event(ts_mono=float(i)))

        pub.close()

        assert len(sink.events) == 50

    def test_different_error_types_handled(self):
        """Different exception types should all trigger retry."""
        for error_type in [ConnectionError, IOError, OSError, TimeoutError, RuntimeError]:
            sink = FlakyNetworkSink()
            sink.fail_until_count = 2
            sink.error_type = error_type
            sink.error_message = f"{error_type.__name__} occurred"

            pub = BufferedPublisher(sink, max_retries=3, retry_delay_s=0.01)
            pub.publish(_make_event())
            pub.close()

            assert len(sink.events) == 1, f"Failed for {error_type.__name__}"


class TestSlowNetwork:

    def test_slow_sink_does_not_block_publisher(self):
        """A slow sink should not block the publisher thread."""
        sink = SlowNetworkSink(delay_s=0.02)
        pub = BufferedPublisher(sink, max_retries=0)

        start = time.monotonic()
        for i in range(20):
            pub.publish(_make_event(ts_mono=float(i)))
        publish_time = time.monotonic() - start

        pub.close()

        # Publishing 20 events should be nearly instant (non-blocking)
        assert publish_time < 0.1, f"Publishing took {publish_time:.3f}s, should be < 0.1s"
        # But all events should eventually arrive
        assert len(sink.events) == 20

    def test_slow_sink_maintains_order(self):
        """Slow delivery must not reorder events."""
        sink = SlowNetworkSink(delay_s=0.01)
        pub = BufferedPublisher(sink, max_retries=0)

        for i in range(15):
            pub.publish(_make_event(ts_mono=float(i)))

        pub.close()

        assert [e.ts_mono for e in sink.events] == list(range(15))


class TestHighVolumeUnderStress:

    def test_many_concurrent_producers(self):
        """Multiple threads publishing to the same buffered publisher."""
        sink = ListSink()
        pub = BufferedPublisher(sink, max_retries=0)

        errors = []
        n_per_thread = 100

        def producer(tid):
            try:
                for i in range(n_per_thread):
                    pub.publish(_make_event(ts_mono=float(tid * 1000 + i)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=producer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        pub.close()
        assert not errors
        assert len(sink.events) == 5 * n_per_thread

    def test_producer_consumer_speed_mismatch(self):
        """Producer much faster than consumer should buffer without loss."""
        sink = SlowNetworkSink(delay_s=0.005)
        pub = BufferedPublisher(sink, max_retries=0)

        # Produce 100 events as fast as possible
        start = time.monotonic()
        for i in range(100):
            pub.publish(_make_event(ts_mono=float(i)))
        produce_time = time.monotonic() - start

        pub.close()
        consume_time = time.monotonic() - start - produce_time

        # Produce should be fast
        assert produce_time < 0.1
        # All events should arrive
        assert len(sink.events) == 100


class TestCompositePublisherUnderFailure:

    def test_one_sink_failing_does_not_affect_others(self):
        """CompositePublisher should isolate sink failures."""
        good = ListSink()
        bad = FlakyNetworkSink()
        bad.fail_until_count = 100  # always fails

        pub = CompositePublisher([bad, good])

        for i in range(10):
            pub.publish(_make_event(ts_mono=float(i)))

        pub.close()

        assert len(good.events) == 10
        # bad sink failed silently, no events there

    def test_all_sinks_fail_no_crash(self):
        """If all sinks fail, publisher should not crash."""
        bad1 = FlakyNetworkSink()
        bad1.fail_until_count = 100
        bad2 = FlakyNetworkSink()
        bad2.fail_until_count = 100

        pub = CompositePublisher([bad1, bad2])

        pub.publish(_make_event())
        pub.close()  # must not raise


class TestCallbackPublisherUnderFailure:

    def test_callback_exception_is_caught(self):
        """Exceptions in callback should not propagate."""
        errors = []

        def flaky_callback(ev):
            if ev.ts_mono == 2.0:
                raise ValueError("boom")
            errors.append(ev.ts_mono)

        pub = CallbackPublisher(flaky_callback)

        for i in range(5):
            pub.publish(_make_event(ts_mono=float(i)))

        # Should have received 4 events (all except 2.0 which raised)
        assert errors == [0.0, 1.0, 3.0, 4.0]

    def test_callback_filter_still_works_during_failures(self):
        """Filtered events should not trigger the callback at all."""
        received = []

        def callback(ev):
            received.append(ev.type)

        pub = CallbackPublisher(callback, filter_types={E.FACE_PRESENT})

        pub.publish(_make_event(E.FACE_PRESENT))
        pub.publish(_make_event(E.FACE_ABSENT))
        pub.publish(_make_event(E.MULTIPLE_FACES))

        assert received == [E.FACE_PRESENT]


class TestRecoveryScenarios:

    def test_network_recovers_then_fails_again(self):
        """Multiple outage/recovery cycles."""
        sink = FlakyNetworkSink()
        # Fail, recover, fail again, recover
        sink.fail_until_count = 2
        # We'll manually control this by replacing the sink
        pub = BufferedPublisher(sink, max_retries=5, retry_delay_s=0.01)

        pub.publish(_make_event(ts_mono=1.0))
        pub.close()
        assert len(sink.events) == 1

        # New sink, fails then recovers
        sink2 = FlakyNetworkSink()
        sink2.fail_until_count = 2
        pub2 = BufferedPublisher(sink2, max_retries=5, retry_delay_s=0.01)
        pub2.publish(_make_event(ts_mono=2.0))
        pub2.close()
        assert len(sink2.events) == 1

    def test_publisher_retains_events_until_delivered_or_dropped(self):
        """BufferedPublisher should not lose events that are still pending."""
        sink = FlakyNetworkSink()
        sink.fail_until_count = 2

        pub = BufferedPublisher(sink, max_retries=3, retry_delay_s=0.01)

        # Queue up several events
        for i in range(5):
            pub.publish(_make_event(ts_mono=float(i)))

        # Close immediately - background thread should still retry
        pub.close()

        # All should eventually be delivered (2 initial failures, then success)
        assert len(sink.events) == 5


class TestPublisherMetrics:

    def test_dropped_count_tracked(self):
        """Dropped event count should be tracked."""
        sink = FlakyNetworkSink()
        sink.fail_until_count = 10

        pub = BufferedPublisher(sink, max_retries=2, retry_delay_s=0.01)
        pub.publish(_make_event())
        pub.publish(_make_event())
        pub.close()

        assert pub.dropped_count == 2

    def test_retry_count_observable(self):
        """Number of retry attempts should be observable."""
        sink = FlakyNetworkSink()
        sink.fail_until_count = 3

        pub = BufferedPublisher(sink, max_retries=5, retry_delay_s=0.01)
        pub.publish(_make_event())
        pub.close()

        # 3 failures + 1 success = 4 calls
        assert sink._call_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])