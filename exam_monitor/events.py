from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol

from .config import Config


class EventType(str, Enum):
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"
    SESSION_INTERRUPTED = "SESSION_INTERRUPTED"
    FACE_PRESENT = "FACE_PRESENT"
    FACE_ABSENT = "FACE_ABSENT"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    DETECTION_FAILURE = "DETECTION_FAILURE"
    CAMERA_UNAVAILABLE = "CAMERA_UNAVAILABLE"
    CAMERA_PERMISSION_DENIED = "CAMERA_PERMISSION_DENIED"


# Mutually exclusive monitoring states. Everything else is a point-in-time event.
STATE_TYPES = (
    EventType.FACE_PRESENT,
    EventType.FACE_ABSENT,
    EventType.MULTIPLE_FACES,
    EventType.DETECTION_FAILURE,
    EventType.CAMERA_UNAVAILABLE,
    EventType.CAMERA_PERMISSION_DENIED,
)


@dataclass(frozen=True)
class Event:
    type: EventType
    session_id: str
    ts_utc: str
    ts_mono: float
    frame_id: Optional[int] = None
    details: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["type"] = self.type.value
        return json.dumps(d, separators=(",", ":"))


class EventSink(Protocol):
    def write(self, event: Event) -> None: ...


class JsonlSink:
    """Append-only JSON Lines. Stores event metadata only: never frames, crops or embeddings."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
        self.path = path

    def write(self, event: Event) -> None:
        self._f.write(event.to_json() + "\n")

    def close(self) -> None:
        self._f.close()


class ListSink:
    def __init__(self):
        self.events: list[Event] = []

    def write(self, event: Event) -> None:
        self.events.append(event)


class IntegrityStateMachine:
    """Debounced state machine over per-frame observations.

    A raw observation must persist for hold_* seconds before the state changes. This stops a single
    missed detection from producing FACE_ABSENT, and a single spurious box from producing
    MULTIPLE_FACES. One event is emitted per state ENTRY, not per frame.
    """

    def __init__(self, cfg: Config, session_id: str, sink: EventSink):
        self.cfg = cfg
        self.session_id = session_id
        self.sink = sink
        self.state: Optional[EventType] = None
        self._state_since = 0.0
        self._cand: Optional[EventType] = None
        self._cand_since = 0.0
        self._last_obs: Optional[float] = None
        self._hold = {
            EventType.FACE_PRESENT: cfg.hold_face_present_s,
            EventType.FACE_ABSENT: cfg.hold_face_absent_s,
            EventType.MULTIPLE_FACES: cfg.hold_multiple_faces_s,
            EventType.DETECTION_FAILURE: cfg.hold_detection_failure_s,
            EventType.CAMERA_UNAVAILABLE: cfg.hold_camera_unavailable_s,
            EventType.CAMERA_PERMISSION_DENIED: cfg.hold_camera_permission_denied_s,
        }

    def _emit(self, etype: EventType, now: float, frame_id: Optional[int], details: dict) -> Event:
        ev = Event(etype, self.session_id, datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                   now, frame_id, details)
        self.sink.write(ev)
        return ev

    def start(self, now: float) -> Event:
        return self._emit(EventType.SESSION_STARTED, now, None, {})

    def end(self, now: float) -> Event:
        return self._emit(EventType.SESSION_ENDED, now, None, {"final_state": self.state.value if self.state else None})

    def observe(self, obs: EventType, now: float, frame_id: Optional[int] = None,
                details: Optional[dict] = None) -> list[Event]:
        if obs not in STATE_TYPES:
            raise ValueError(f"{obs} is not an observable state")
        out: list[Event] = []

        # Our own loop stalled (laptop sleep, UI hang, blocked I/O). Do not let the gap count as
        # "face absent" time or as debounce time for a pending candidate.
        if self._last_obs is not None and now - self._last_obs > self.cfg.stall_s:
            out.append(self._emit(EventType.SESSION_INTERRUPTED, now, frame_id,
                                  {"gap_s": round(now - self._last_obs, 3)}))
            self._cand = None
        self._last_obs = now

        if obs == self.state:
            self._cand = None
            return out
        if obs != self._cand:
            self._cand, self._cand_since = obs, now
        if now - self._cand_since >= self._hold[obs]:
            prev = self.state
            d = dict(details or {})
            d["previous_state"] = prev.value if prev else None
            d["previous_duration_s"] = round(now - self._state_since, 3) if prev else None
            self.state, self._state_since, self._cand = obs, now, None
            out.append(self._emit(obs, now, frame_id, d))
        return out
