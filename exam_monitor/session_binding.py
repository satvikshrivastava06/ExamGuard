"""Associate CV monitoring status with exactly one exam session.

Guarantees:
  1. A monitor is created for one session ID and can never be re-pointed at another.
  2. Every event it emits carries that ID (enforced on write, not trusted).
  3. At most one live monitor exists per session ID.
  4. Status is readable from another thread (e.g. an API handler) without tearing.

Not covered here: whether the session exists / is ACTIVE / belongs to this candidate. That is the
backend's decision (Team 1). Plug it in through the `validator` argument of MonitoringRegistry.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Callable, Optional

from .config import Config
from .events import Event, EventSink, EventType, IntegrityStateMachine, STATE_TYPES

# The ID ends up in log file names, so accept only a known-good alphabet and length.
# Use fullmatch(): `$` would also accept a trailing "\n".
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")


class SessionBindingError(Exception):
    """Base class so callers can catch every association failure in one place."""


class InvalidSessionId(SessionBindingError): ...
class UnknownSession(SessionBindingError): ...
class SessionAlreadyMonitored(SessionBindingError): ...
class SessionNotMonitored(SessionBindingError): ...
class SessionMismatch(SessionBindingError): ...


SessionValidator = Callable[[str], bool]
SinkFactory = Callable[[str], EventSink]


def validate_session_id(raw: object) -> str:
    if not isinstance(raw, str) or not _SESSION_ID_RE.fullmatch(raw):
        raise InvalidSessionId("session id must be 8-64 chars of [A-Za-z0-9_-]")
    return raw


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class MonitoringStatus:
    """Immutable snapshot. Safe to hand to another thread or serialise into an API response."""
    session_id: str
    bound_at_utc: str
    active: bool = True
    state: Optional[str] = None              # last CONFIRMED state (after debounce), None until first
    state_since_utc: Optional[str] = None
    last_event_type: Optional[str] = None
    last_observation_utc: Optional[str] = None   # liveness: advances on every analysed frame
    event_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class _BoundSink:
    """Wraps the real sink. Refuses any event whose session_id differs from the bound one."""

    def __init__(self, session_id: str, inner: EventSink, on_event: Callable[[Event], None]):
        self._sid, self._inner, self._on_event = session_id, inner, on_event

    def write(self, event: Event) -> None:
        if event.session_id != self._sid:
            raise SessionMismatch(f"event for {event.session_id!r} written to monitor of {self._sid!r}")
        self._inner.write(event)
        self._on_event(event)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


class SessionMonitor:
    """One monitor = one session. Created only through MonitoringRegistry.bind()."""

    def __init__(self, session_id: str, cfg: Config, sink: EventSink, clock: Callable[[], float]):
        self.session_id = session_id            # read-only by convention; the sink guard enforces it
        self._clock = clock
        self._lock = threading.Lock()           # guards _status and the state machine
        self._status = MonitoringStatus(session_id=session_id, bound_at_utc=_utc_now())
        self._sink = _BoundSink(session_id, sink, self._apply_event)
        self._machine = IntegrityStateMachine(cfg, session_id, self._sink)

    # Called only from inside a locked section (machine -> sink -> here), so it takes no lock itself.
    def _apply_event(self, ev: Event) -> None:
        changes: dict = {"last_event_type": ev.type.value, "event_count": self._status.event_count + 1}
        if ev.type in STATE_TYPES:
            changes.update(state=ev.type.value, state_since_utc=ev.ts_utc)
        self._status = replace(self._status, **changes)

    def observe(self, obs: EventType, now: float, frame_id: Optional[int] = None,
                details: Optional[dict] = None) -> list[Event]:
        with self._lock:
            if not self._status.active:
                raise SessionNotMonitored(f"monitor for {self.session_id!r} is closed")
            events = self._machine.observe(obs, now, frame_id, details)
            self._status = replace(self._status, last_observation_utc=_utc_now())
            return events

    def status(self) -> MonitoringStatus:
        with self._lock:
            return self._status

    def _start(self) -> None:
        with self._lock:
            self._machine.start(self._clock())

    def _close(self) -> MonitoringStatus:
        with self._lock:
            if self._status.active:
                self._machine.end(self._clock())
                self._status = replace(self._status, active=False)
            self._sink.close()
            return self._status


class MonitoringRegistry:
    """Owns the session_id -> monitor mapping.

    Lock order is registry lock -> monitor lock. Monitors never take the registry lock,
    so the two cannot deadlock.
    """

    def __init__(self, cfg: Config, sink_factory: SinkFactory,
                 validator: Optional[SessionValidator] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._cfg, self._sink_factory, self._validator, self._clock = cfg, sink_factory, validator, clock
        self._lock = threading.Lock()
        self._active: dict[str, SessionMonitor] = {}

    def bind(self, session_id: str) -> SessionMonitor:
        sid = validate_session_id(session_id)
        # Backend lookup may be slow, so it runs outside the registry lock. The duplicate check
        # below is what actually prevents two monitors, even if two callers pass validation together.
        if self._validator is not None and not self._validator(sid):
            raise UnknownSession(f"session {sid!r} was rejected by the validator")
        with self._lock:
            if sid in self._active:
                raise SessionAlreadyMonitored(f"session {sid!r} already has a live monitor")
            sink = self._sink_factory(sid)
            monitor = SessionMonitor(sid, self._cfg, sink, self._clock)
            try:
                monitor._start()
            except Exception:
                getattr(sink, "close", lambda: None)()
                raise
            self._active[sid] = monitor
            return monitor

    def get_status(self, session_id: str) -> MonitoringStatus:
        sid = validate_session_id(session_id)
        with self._lock:
            monitor = self._active.get(sid)
        if monitor is None:
            raise SessionNotMonitored(f"no live monitor for session {sid!r}")
        return monitor.status()

    def list_active(self) -> list[MonitoringStatus]:
        with self._lock:
            monitors = list(self._active.values())
        return [m.status() for m in monitors]

    def release(self, session_id: str) -> MonitoringStatus:
        """Emit SESSION_ENDED, close the sink, free the ID. Returns the final status."""
        sid = validate_session_id(session_id)
        with self._lock:
            monitor = self._active.pop(sid, None)
            if monitor is None:
                raise SessionNotMonitored(f"no live monitor for session {sid!r}")
            return monitor._close()
