"""Prolonged-absence episodes.

ONE definition of absence for the whole system:
    an absence is a confirmed FACE_ABSENT state (the state machine's debounced decision).
    It is RECORDED only if it lasted at least `threshold_s` (config: prolonged_absence_s).

Two thresholds therefore exist, with different jobs:
    hold_face_absent_s   (3 s)   "is the face absent right now?"          -> FACE_ABSENT state event
    prolonged_absence_s  (10 s)  "was the absence long enough to record?" -> PROLONGED_ABSENCE event

This tracker does not look at raw frames. It follows the state machine's confirmed transitions, so a
flickering detector (present, absent, present, ...) cannot be stitched into one long fake absence:
if the machine never confirms FACE_ABSENT, nothing is recorded.

Timing rules:
  * start = when the absent observations that were later confirmed began (not when confirmation came)
  * end   = when the observations that ended the absence began (face back, dark frame, camera down)
  * time we could not see (detection unavailable, camera down, our own stall) is never counted as absence
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from .events import EventType

_REASON = {
    EventType.FACE_PRESENT: "face_returned",
    EventType.MULTIPLE_FACES: "face_returned",
    EventType.DETECTION_UNAVAILABLE: "detection_unavailable",
    EventType.CAMERA_UNAVAILABLE: "camera_unavailable",
    EventType.CAMERA_PERMISSION_DENIED: "camera_unavailable",
}


def reason_for(successor: EventType) -> str:
    return _REASON.get(successor, "face_returned")


@dataclass(frozen=True)
class AbsenceEpisode:
    start_utc: str
    end_utc: str
    duration_s: float
    end_reason: str        # face_returned | detection_unavailable | camera_unavailable | interrupted | session_ended

    def details(self, threshold_s: float) -> dict:
        return {"start_utc": self.start_utc, "end_utc": self.end_utc,
                "duration_s": round(self.duration_s, 3), "threshold_s": threshold_s,
                "end_reason": self.end_reason}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AbsenceTracker:
    def __init__(self, threshold_s: float, wall_clock: Callable[[], datetime] = _utc_now):
        if threshold_s < 0:
            raise ValueError("threshold_s must be >= 0")
        self.threshold_s = threshold_s
        self._wall = wall_clock
        self._start: Optional[float] = None
        self._start_wall: Optional[datetime] = None

    @property
    def active(self) -> bool:
        return self._start is not None

    def ongoing_s(self, now: float) -> float:
        """Length of the absence in progress (0 if none). For live status only."""
        return 0.0 if self._start is None else max(0.0, now - self._start)

    def enter(self, started: float, now: float) -> None:
        """A FACE_ABSENT state was confirmed at `now`; the absent observations began at `started`."""
        if self._start is not None:
            return
        self._start = started
        self._start_wall = self._wall() - timedelta(seconds=max(0.0, now - started))

    def leave(self, ended: float, reason: str) -> Optional[AbsenceEpisode]:
        """The absence ended at `ended`. Returns an episode only if it reached the threshold."""
        if self._start is None:
            return None
        duration = max(0.0, ended - self._start)
        start_wall = self._start_wall
        self._start, self._start_wall = None, None
        if duration < self.threshold_s:
            return None                                   # too short: not recorded
        return AbsenceEpisode(
            start_utc=start_wall.isoformat(timespec="milliseconds"),
            end_utc=(start_wall + timedelta(seconds=duration)).isoformat(timespec="milliseconds"),
            duration_s=duration, end_reason=reason)
