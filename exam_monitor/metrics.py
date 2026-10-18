from __future__ import annotations

from collections import defaultdict, deque
from time import perf_counter

import numpy as np
import psutil


class LatencyTracker:
    """Rolling per-stage latency in milliseconds."""

    def __init__(self, window: int = 300):
        self._d: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def add(self, stage: str, ms: float) -> None:
        self._d[stage].append(ms)

    def stats(self, stage: str) -> dict:
        v = self._d.get(stage)
        if not v:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
        a = np.fromiter(v, dtype=np.float64)
        return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    def stages(self) -> list[str]:
        return list(self._d)


class FpsMeter:
    def __init__(self, window: int = 60):
        self._t: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._t.append(perf_counter())

    @property
    def fps(self) -> float:
        if len(self._t) < 2 or self._t[-1] == self._t[0]:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


class ResourceMonitor:
    """Process CPU% is per-core (can exceed 100 on multi-core); system CPU% is 0-100."""

    def __init__(self):
        self._proc = psutil.Process()
        self._proc.cpu_percent(None)      # prime the counters; first call always returns 0
        psutil.cpu_percent(None)

    def sample(self) -> dict:
        return {
            "proc_cpu_pct": self._proc.cpu_percent(None),
            "sys_cpu_pct": psutil.cpu_percent(None),
            "rss_mb": self._proc.memory_info().rss / 1e6,
            "threads": self._proc.num_threads(),
        }
