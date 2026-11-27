"""Health and monitoring diagnostics for the exam monitoring pipeline.

Tracks the operational health of every component in the pipeline and exposes
a single ``diagnose()`` snapshot that an API endpoint or ops dashboard can
serve. Components push status updates into the ``HealthMonitor``; consumers
pull the latest ``DiagnosticReport`` at any time.

Health levels
-------------
  HEALTHY     – the component is operating within normal parameters.
  DEGRADED    – functional, but outside ideal bounds (high latency, low FPS,
                elevated CPU, retries happening). Alerts/warnings are appropriate.
  UNHEALTHY   – the component has failed or is not producing useful output
                (camera down, detector crashing, sink write errors).

Thread safety
-------------
The worker thread calls ``update_*`` methods; API/UI threads call ``diagnose()``
and ``summary()``. A single lock guards the mutable state, held only for short
copies so the worker is never blocked by a slow reader.
"""
from __future__ import annotations

import time
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class HealthCheck:
    """Status of one pipeline component at one point in time."""
    component: str
    status: HealthStatus
    detail: str
    last_checked_utc: str
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value if hasattr(self.status, "value") else self.status
        return d


@dataclass(frozen=True)
class DiagnosticReport:
    """Aggregated snapshot of all component health plus system-level stats."""
    overall_status: HealthStatus
    uptime_s: float
    checks: tuple[HealthCheck, ...]
    sessions_active: int = 0
    total_events_published: int = 0
    events_per_minute: float = 0.0
    errors_total: int = 0
    generated_utc: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["overall_status"] = self.overall_status.value
        d["checks"] = [asdict(c) for c in self.checks]
        for c in d["checks"]:
            c["status"] = c["status"].value if hasattr(c["status"], "value") else c["status"]
        return d


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class HealthMonitor:
    """Central health tracker for the exam monitoring pipeline.

    The monitoring worker calls the ``update_*`` methods as it runs; external
    readers (API handlers, dashboards) call ``diagnose()`` at any time.

    Parameters
    ----------
    fps_degraded_threshold:
        FPS below this is DEGRADED.
    fps_unhealthy_threshold:
        FPS below this is UNHEALTHY.
    latency_degraded_ms:
        Analysis p95 above this is DEGRADED.
    latency_unhealthy_ms:
        Analysis p95 above this is UNHEALTHY.
    cpu_degraded_pct:
        Process CPU% above this is DEGRADED.
    error_rate_degraded:
        Errors per minute above this is DEGRADED.
    """

    def __init__(self, *,
                 fps_degraded_threshold: float = 5.0,
                 fps_unhealthy_threshold: float = 1.0,
                 latency_degraded_ms: float = 200.0,
                 latency_unhealthy_ms: float = 500.0,
                 cpu_degraded_pct: float = 80.0,
                 error_rate_degraded: float = 5.0):
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._start_wall = _utc_now()

        # Thresholds
        self._fps_degraded = fps_degraded_threshold
        self._fps_unhealthy = fps_unhealthy_threshold
        self._lat_degraded = latency_degraded_ms
        self._lat_unhealthy = latency_unhealthy_ms
        self._cpu_degraded = cpu_degraded_pct
        self._error_rate_degraded = error_rate_degraded

        # Component state (updated by the worker)
        self._camera: Optional[HealthCheck] = None
        self._detector: Optional[HealthCheck] = None
        self._event_sink: Optional[HealthCheck] = None
        self._resources: Optional[HealthCheck] = None

        # Counters
        self._sessions_active = 0
        self._total_events = 0
        self._errors_total = 0
        self._event_times: list[float] = []  # monotonic timestamps for rate calc

    # ------------------------------------------------------------------ updates (worker thread)

    def update_camera(self, status: HealthStatus, detail: str, **metrics) -> None:
        with self._lock:
            self._camera = HealthCheck("camera", status, detail, _utc_now(), dict(metrics))

    def update_detector(self, status: HealthStatus, detail: str, **metrics) -> None:
        with self._lock:
            self._detector = HealthCheck("detector", status, detail, _utc_now(), dict(metrics))

    def update_event_sink(self, status: HealthStatus, detail: str, **metrics) -> None:
        with self._lock:
            self._event_sink = HealthCheck("event_sink", status, detail, _utc_now(), dict(metrics))

    def update_resources(self, status: HealthStatus, detail: str, **metrics) -> None:
        with self._lock:
            self._resources = HealthCheck("resources", status, detail, _utc_now(), dict(metrics))

    def record_event(self) -> None:
        """Called each time an event is published."""
        now = time.monotonic()
        with self._lock:
            self._total_events += 1
            self._event_times.append(now)
            # Keep only last 5 minutes of timestamps for rate calculation
            cutoff = now - 300
            self._event_times = [t for t in self._event_times if t > cutoff]

    def record_error(self) -> None:
        with self._lock:
            self._errors_total += 1

    def set_active_sessions(self, count: int) -> None:
        with self._lock:
            self._sessions_active = count

    # ------------------------------------------------------------------ auto-check helpers

    def check_fps(self, fps: float) -> HealthCheck:
        """Evaluate FPS and return a health check for it."""
        if fps <= self._fps_unhealthy:
            status = HealthStatus.UNHEALTHY
            detail = f"FPS critically low: {fps:.1f}"
        elif fps <= self._fps_degraded:
            status = HealthStatus.DEGRADED
            detail = f"FPS below target: {fps:.1f}"
        else:
            status = HealthStatus.HEALTHY
            detail = f"FPS normal: {fps:.1f}"
        return HealthCheck("fps", status, detail, _utc_now(), {"fps": round(fps, 1)})

    def check_latency(self, p95_ms: float) -> HealthCheck:
        """Evaluate analysis latency and return a health check."""
        if p95_ms >= self._lat_unhealthy:
            status = HealthStatus.UNHEALTHY
            detail = f"Analysis p95 critically high: {p95_ms:.0f}ms"
        elif p95_ms >= self._lat_degraded:
            status = HealthStatus.DEGRADED
            detail = f"Analysis p95 elevated: {p95_ms:.0f}ms"
        else:
            status = HealthStatus.HEALTHY
            detail = f"Analysis p95 normal: {p95_ms:.0f}ms"
        return HealthCheck("latency", status, detail, _utc_now(), {"p95_ms": round(p95_ms, 1)})

    def check_resources(self, cpu_pct: float, rss_mb: float) -> HealthCheck:
        """Evaluate resource usage and return a health check."""
        if cpu_pct >= self._cpu_degraded:
            status = HealthStatus.DEGRADED
            detail = f"High CPU usage: {cpu_pct:.0f}%"
        else:
            status = HealthStatus.HEALTHY
            detail = f"Resources normal: CPU {cpu_pct:.0f}%, RSS {rss_mb:.0f}MB"
        check = HealthCheck("resources", status, detail, _utc_now(),
                            {"cpu_pct": round(cpu_pct, 1), "rss_mb": round(rss_mb, 1)})
        with self._lock:
            self._resources = check
        return check

    # ------------------------------------------------------------------ read side

    def diagnose(self) -> DiagnosticReport:
        """Return a full diagnostic snapshot. Thread-safe."""
        now = time.monotonic()
        with self._lock:
            checks = tuple(c for c in [self._camera, self._detector,
                                        self._event_sink, self._resources] if c is not None)
            events_per_min = self._events_per_minute(now)
            return DiagnosticReport(
                overall_status=self._worst(checks),
                uptime_s=round(now - self._start_time, 1),
                checks=checks,
                sessions_active=self._sessions_active,
                total_events_published=self._total_events,
                events_per_minute=round(events_per_min, 1),
                errors_total=self._errors_total,
                generated_utc=_utc_now(),
            )

    def summary(self) -> HealthStatus:
        """Overall health: the worst status across all tracked components."""
        with self._lock:
            checks = tuple(c for c in [self._camera, self._detector,
                                        self._event_sink, self._resources] if c is not None)
        return self._worst(checks)

    # ------------------------------------------------------------------ internals

    def _events_per_minute(self, now: float) -> float:
        """Events per minute based on timestamps in the last 60 seconds."""
        cutoff = now - 60
        recent = [t for t in self._event_times if t > cutoff]
        return len(recent)  # count in last 60s = events per minute

    @staticmethod
    def _worst(checks: tuple[HealthCheck, ...]) -> HealthStatus:
        if not checks:
            return HealthStatus.HEALTHY
        priority = {HealthStatus.UNHEALTHY: 2, HealthStatus.DEGRADED: 1, HealthStatus.HEALTHY: 0}
        worst = max(checks, key=lambda c: priority.get(c.status, 0))
        return worst.status
