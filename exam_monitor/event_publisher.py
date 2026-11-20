"""Common event publishing interface for face-presence events.

Decouples event production (the monitoring pipeline) from event consumption
(JSONL logs, HTTP endpoints, WebSocket feeds, message queues, ...) through a
thin publish/subscribe layer that sits on top of the existing EventSink protocol.

Architecture
------------
    IntegrityStateMachine / SessionMonitor
        |  Event
        v
    EventBus (implements EventSink)
        |  write(event) ----> primary sink (e.g. JsonlSink)
        |                +--> publisher.publish(event) ----> subscriber 1
        |                                             ----> subscriber 2
        |                                             ----> ...

Subscribers can be any callable or EventSink. The publisher layer adds:
  * Fan-out to multiple consumers without the producer knowing about them.
  * Async buffered delivery with retry, so a slow or temporarily down consumer
    (network endpoint, remote queue) does not block the pipeline.
  * Graceful degradation: a broken subscriber is logged and skipped, never
    crashes the monitoring worker.

Thread safety
-------------
Publishers are used from the monitoring worker thread. Subscribers may live on
other threads (API handlers, UI). The BufferedPublisher's internal queue
bridges the two safely.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional, Protocol, Union, runtime_checkable

from .events import Event, EventSink, EventType

log = logging.getLogger(__name__)

# A subscriber is anything that accepts an Event: a callable, or an EventSink.
Subscriber = Union[Callable[[Event], None], EventSink]


@runtime_checkable
class EventPublisher(Protocol):
    """Minimal interface for publishing events to external consumers."""

    def publish(self, event: Event) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# CallbackPublisher
# ---------------------------------------------------------------------------

class CallbackPublisher:
    """Wraps a simple callback as a publisher. Supports optional event-type filtering.
    Failures are logged and swallowed."""

    def __init__(
        self,
        callback: Callable[[Event], None],
        name: str = "callback",
        filter_types: Optional[set[EventType]] = None,
    ):
        self._cb = callback
        self._name = name
        self._filter = filter_types

    def publish(self, event: Event) -> None:
        if self._filter is not None and event.type not in self._filter:
            return
        try:
            self._cb(event)
        except Exception:
            log.exception("subscriber %r failed on %s", self._name, event.type.value)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# SinkPublisher
# ---------------------------------------------------------------------------

class SinkPublisher:
    """Wraps an EventSink as a publisher. Failures are logged and swallowed."""

    def __init__(self, sink: EventSink, name: str = "sink"):
        self._sink = sink
        self._name = name

    def publish(self, event: Event) -> None:
        try:
            self._sink.write(event)
        except Exception:
            log.exception("subscriber sink %r failed on %s", self._name, event.type.value)

    def close(self) -> None:
        close = getattr(self._sink, "close", None)
        if callable(close):
            close()


# ---------------------------------------------------------------------------
# CompositePublisher
# ---------------------------------------------------------------------------

class CompositePublisher:
    """Fans out events to multiple publishers/subscribers.
    A broken subscriber never blocks the others.

    Accepts raw EventSink objects, callables, or EventPublisher instances.
    """

    def __init__(self, publishers: Optional[list[Union[EventPublisher, Subscriber]]] = None):
        self._lock = threading.Lock()
        self._publishers: list[EventPublisher] = []
        if publishers:
            for p in publishers:
                self.add(p)

    def _wrap(self, p: Union[EventPublisher, Subscriber]) -> EventPublisher:
        if isinstance(p, EventPublisher):
            return p
        if callable(getattr(p, "write", None)):
            return SinkPublisher(p, name=repr(p))  # type: ignore[arg-type]
        return CallbackPublisher(p, name=repr(p))  # type: ignore[arg-type]

    def add(self, publisher: Union[EventPublisher, Subscriber]) -> None:
        with self._lock:
            self._publishers.append(self._wrap(publisher))

    def remove(self, publisher: EventPublisher) -> None:
        with self._lock:
            try:
                self._publishers.remove(publisher)
            except ValueError:
                pass

    def subscribe(self, subscriber: Subscriber, name: str = "") -> EventPublisher:
        """Convenience: wrap a raw callable or EventSink and register it. Returns the
        created publisher so the caller can later remove() it."""
        if callable(getattr(subscriber, "write", None)):
            pub = SinkPublisher(subscriber, name or repr(subscriber))  # type: ignore[arg-type]
        else:
            pub = CallbackPublisher(subscriber, name or repr(subscriber))  # type: ignore[arg-type]
        self.add(pub)
        return pub

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._publishers)

    def publish(self, event: Event) -> None:
        with self._lock:
            targets = list(self._publishers)
        for pub in targets:
            try:
                pub.publish(event)
            except Exception:
                log.exception("publisher %r raised on %s", pub, event.type.value)

    def close(self) -> None:
        with self._lock:
            targets = list(self._publishers)
            self._publishers.clear()
        for pub in targets:
            try:
                pub.close()
            except Exception:
                log.exception("close failed for publisher %r", pub)


# ---------------------------------------------------------------------------
# BufferedPublisher
# ---------------------------------------------------------------------------

class BufferedPublisher:
    """Delivers events asynchronously through an internal queue, with retry.

    This bridges a fast producer (the monitoring worker) and a slow or unreliable
    consumer (a remote HTTP endpoint, a message queue) without blocking the
    pipeline. Events are buffered in a bounded queue and drained by a background
    thread that retries failed deliveries.
    """

    def __init__(
        self,
        target: Union[EventPublisher, Callable[[Event], None], EventSink],
        max_retries: int = 3,
        retry_delay_s: float = 1.0,
        max_queue_size: int = 1000,
        name: str = "buffered",
    ):
        if callable(getattr(target, "publish", None)):
            self._target: EventPublisher = target  # type: ignore[assignment]
        elif callable(getattr(target, "write", None)):
            self._target = SinkPublisher(target, name)  # type: ignore[arg-type]
        else:
            self._target = CallbackPublisher(target, name)  # type: ignore[arg-type]
        self._max_retries = max_retries
        self._retry_delay = retry_delay_s
        self._name = name
        self._q: queue.Queue[Optional[Event]] = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name=f"event-publisher-{name}")
        self._dropped = 0
        self._delivered = 0
        self._thread.start()

    @property
    def dropped_count(self) -> int:
        return self._dropped

    @property
    def delivered_count(self) -> int:
        return self._delivered

    @property
    def pending(self) -> int:
        return self._q.qsize()

    def publish(self, event: Event) -> None:
        if self._stop.is_set():
            return
        try:
            self._q.put_nowait(event)
        except queue.Full:
            try:
                self._q.get_nowait()
                self._dropped += 1
                log.warning("publisher %r queue full: dropped oldest event", self._name)
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(event)
            except queue.Full:
                self._dropped += 1

    def close(self, timeout: float = 5.0) -> None:
        """Signal the drain thread to finish remaining items and stop."""
        self._stop.set()
        self._q.put(None)  # sentinel to wake the drain thread
        self._thread.join(timeout)
        # FINAL DRAIN: Process all remaining items in the queue synchronously
        # to avoid race conditions in tests where close() is called immediately
        # after publish().
        while not self._q.empty():
            try:
                ev = self._q.get_nowait()
                if ev is not None:
                    self._try_deliver(ev)
            except queue.Empty:
                break

    def _drain(self) -> None:
        while True:
            try:
                # Use a short timeout to check for stop signal
                event = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if event is None:
                # Drain any remaining items after the stop sentinel.
                while True:
                    try:
                        ev = self._q.get_nowait()
                    except queue.Empty:
                        break
                    if ev is not None:
                        self._try_deliver(ev)
                break
            self._try_deliver(event)

    def _try_deliver(self, event: Event) -> None:
        for attempt in range(1, self._max_retries + 1):
            try:
                self._target.publish(event)
                self._delivered += 1
                return
            except Exception:
                if attempt == self._max_retries:
                    log.exception("publisher %r: dropping %s after %d retries",
                                  self._name, event.type.value, self._max_retries)
                    self._dropped += 1
                else:
                    log.warning("publisher %r: retry %d/%d for %s",
                                self._name, attempt, self._max_retries, event.type.value)
                    # Only sleep if we haven't been told to stop
                    if not self._stop.is_set():
                        time.sleep(self._retry_delay)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """Drop-in EventSink that writes to a primary sink and publishes to subscribers."""

    def __init__(self, primary: EventSink,
                 publisher: Optional[CompositePublisher] = None):
        self._primary = primary
        self._publisher = publisher or CompositePublisher()

    @property
    def publisher(self) -> CompositePublisher:
        return self._publisher

    def subscribe(self, subscriber: Subscriber, name: str = "") -> EventPublisher:
        return self._publisher.subscribe(subscriber, name)

    @property
    def subscriber_count(self) -> int:
        return self._publisher.subscriber_count

    def write(self, event: Event) -> None:
        self._primary.write(event)
        try:
            self._publisher.publish(event)
        except Exception:
            log.exception("EventBus publisher failed for %s", event.type.value)

    def close(self) -> None:
        self._publisher.close()
        close = getattr(self._primary, "close", None)
        if callable(close):
            close()
