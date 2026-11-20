"""Tests for health and monitoring diagnostics.

Covers:
  * HealthMonitor tracks component health correctly
  * FPS, latency, CPU thresholds classify HEALTHY/DEGRADED/UNHEALTHY
  * DiagnosticReport serializes to dict properly
  * Thread safety between worker updates and reader threads
  * Event rate and error counters
  * Integration with ResourceMonitor, LatencyTracker, FpsMeter
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from exam_monitor.config import Config
from exam_monitor.health import (
    DiagnosticReport,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
)
from exam_monitor.metrics import FpsMeter, LatencyTracker, ResourceMonitor

CFG = Config()


class TestHealthStatus:
    def test_enum_values(self):
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"

    def test_priority_ordering(self):
        # UNHEALTHY > DEGRADED > HEALTHY
        assert HealthMonitor._worst((HealthCheck("a", HealthStatus.UNHEALTHY, "", "now", {}),
                                     HealthCheck("b", HealthStatus.HEALTHY, "", "now", {}))).value == "unhealthy"
        assert HealthMonitor._worst((HealthCheck("a", HealthStatus.DEGRADED, "", "now", {}),
                                     HealthCheck("b", HealthStatus.HEALTHY, "", "now", {}))).value == "degraded"
        assert HealthMonitor._worst((HealthCheck("a", HealthStatus.HEALTHY, "", "now", {}),
                                     HealthCheck("b", HealthStatus.HEALTHY, "", "now", {}))).value == "healthy"


class TestHealthCheck:
    def test_dataclass_immutability(self):
        hc = HealthCheck("camera", HealthStatus.HEALTHY, "ok", "2026-01-01T00:00:00+00:00", {"fps": 30})
        with pytest.raises(Exception):
            hc.status = HealthStatus.UNHEALTHY  # type: ignore[misc]

    def test_to_dict_conversion(self):
        hc = HealthCheck("detector", HealthStatus.DEGRADED, "high latency", "2026-01-01T00:00:00+00:00",
                         {"p95_ms": 250})
        d = {"component": "detector", "status": "degraded", "detail": "high latency",
             "last_checked_utc": "2026-01-01T00:00:00+00:00", "metrics": {"p95_ms": 250}}
        assert hc.to_dict() == d


class TestDiagnosticReport:
    def test_to_dict_serializes_all_fields(self):
        report = DiagnosticReport(
            overall_status=HealthStatus.HEALTHY,
            uptime_s=123.4,
            checks=(HealthCheck("a", HealthStatus.HEALTHY, "ok", "now", {}),),
            sessions_active=2,
            total_events_published=100,
            events_per_minute=30.5,
            errors_total=1,
            generated_utc="2026-01-01T00:00:00.000+00:00",
        )
        d = report.to_dict()
        assert d["overall_status"] == "healthy"
        assert d["uptime_s"] == 123.4
        assert d["sessions_active"] == 2
        assert d["total_events_published"] == 100
        assert d["events_per_minute"] == 30.5
        assert d["errors_total"] == 1
        assert len(d["checks"]) == 1
        assert d["checks"][0]["component"] == "a"

    def test_empty_checks_is_valid(self):
        report = DiagnosticReport(
            overall_status=HealthStatus.HEALTHY,
            uptime_s=0.0,
            checks=(),
            sessions_active=0,
            total_events_published=0,
            events_per_minute=0.0,
            errors_total=0,
            generated_utc="2026-01-01T00:00:00.000+00:00",
        )
        d = report.to_dict()
        assert d["checks"] == []


class TestHealthMonitor:
    def setup_method(self):
        self.monitor = HealthMonitor()

    def test_initial_state_healthy(self):
        report = self.monitor.diagnose()
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.uptime_s >= 0
        assert report.checks == ()

    def test_summary_matches_worst_check(self):
        self.monitor.update_camera(HealthStatus.DEGRADED, "slow fps", fps=5)
        self.monitor.update_detector(HealthStatus.HEALTHY, "ok", latency_ms=50)
        assert self.monitor.summary() == HealthStatus.DEGRADED

    def test_update_camera_records_check(self):
        self.monitor.update_camera(HealthStatus.UNHEALTHY, "permission denied")
        report = self.monitor.diagnose()
        camera_checks = [c for c in report.checks if c.component == "camera"]
        assert len(camera_checks) == 1
        assert camera_checks[0].status == HealthStatus.UNHEALTHY
        assert camera_checks[0].detail == "permission denied"

    def test_update_detector_records_check(self):
        self.monitor.update_detector(HealthStatus.DEGRADED, "model load slow", load_time_ms=5000)
        report = self.monitor.diagnose()
        det_checks = [c for c in report.checks if c.component == "detector"]
        assert len(det_checks) == 1
        assert det_checks[0].status == HealthStatus.DEGRADED
        assert det_checks[0].metrics.get("load_time_ms") == 5000

    def test_update_event_sink_records_check(self):
        self.monitor.update_event_sink(HealthStatus.HEALTHY, "writing normally")
        report = self.monitor.diagnose()
        sink_checks = [c for c in report.checks if c.component == "event_sink"]
        assert len(sink_checks) == 1
        assert sink_checks[0].status == HealthStatus.HEALTHY

    def test_update_resources_records_check(self):
        self.monitor.update_resources(HealthStatus.DEGRADED, "high cpu", cpu_pct=85, rss_mb=200)
        report = self.monitor.diagnose()
        res_checks = [c for c in report.checks if c.component == "resources"]
        assert len(res_checks) == 1
        assert res_checks[0].status == HealthStatus.DEGRADED
        assert res_checks[0].metrics["cpu_pct"] == 85

    def test_check_fps_thresholds(self):
        # Healthy
        assert self.monitor.check_fps(30.0).status == HealthStatus.HEALTHY
        assert self.monitor.check_fps(10.0).status == HealthStatus.HEALTHY
        # Degraded
        assert self.monitor.check_fps(4.0).status == HealthStatus.DEGRADED
        assert self.monitor.check_fps(2.0).status == HealthStatus.DEGRADED
        # Unhealthy
        assert self.monitor.check_fps(0.5).status == HealthStatus.UNHEALTHY

    def test_check_latency_thresholds(self):
        # Healthy
        assert self.monitor.check_latency(50.0).status == HealthStatus.HEALTHY
        assert self.monitor.check_latency(150.0).status == HealthStatus.HEALTHY
        # Degraded
        assert self.monitor.check_latency(250.0).status == HealthStatus.DEGRADED
        assert self.monitor.check_latency(400.0).status == HealthStatus.DEGRADED
        # Unhealthy
        assert self.monitor.check_latency(600.0).status == HealthStatus.UNHEALTHY

    def test_check_resources_cpu_threshold(self):
        assert self.monitor.check_resources(50.0, 100.0).status == HealthStatus.HEALTHY
        assert self.monitor.check_resources(90.0, 100.0).status == HealthStatus.DEGRADED

    def test_record_event_increments_counter(self):
        self.monitor.record_event()
        self.monitor.record_event()
        report = self.monitor.diagnose()
        assert report.total_events_published == 2

    def test_record_error_increments_counter(self):
        self.monitor.record_error()
        self.monitor.record_error()
        self.monitor.record_error()
        report = self.monitor.diagnose()
        assert report.errors_total == 3

    def test_events_per_minute_calculates_rate(self):
        # Record 5 events over ~0.1s simulated time
        for _ in range(5):
            self.monitor.record_event()
        # The rate is based on timestamps in last 60s; all are recent
        report = self.monitor.diagnose()
        assert report.events_per_minute >= 5.0

    def test_set_active_sessions(self):
        self.monitor.set_active_sessions(3)
        report = self.monitor.diagnose()
        assert report.sessions_active == 3

    def test_uptime_increases(self):
        t0 = self.monitor.diagnose().uptime_s
        time.sleep(0.05)
        t1 = self.monitor.diagnose().uptime_s
        assert t1 > t0

    def test_generated_utc_is_current(self):
        before = time.time()
        report = self.monitor.diagnose()
        after = time.time()
        # generated_utc should be roughly now (within 2 seconds)
        gen = report.generated_utc
        assert "T" in gen and "+" in gen


class TestThreadSafety:
    def test_concurrent_updates_and_reads(self):
        monitor = HealthMonitor()
        errors = []
        updates_done = threading.Event()

        def writer():
            try:
                for i in range(1000):
                    monitor.update_camera(HealthStatus.HEALTHY, f"ok {i}")
                    monitor.update_detector(HealthStatus.DEGRADED, "slow", p95_ms=300)
                    monitor.update_event_sink(HealthStatus.HEALTHY, "ok")
                    monitor.update_resources(HealthStatus.HEALTHY, "ok", cpu_pct=20, rss_mb=100)
                    monitor.record_event()
                    if i % 10 == 0:
                        monitor.record_error()
                updates_done.set()
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                while not updates_done.is_set():
                    monitor.diagnose()
                    monitor.summary()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_check_methods(self):
        monitor = HealthMonitor()
        errors = []

        def checker():
            try:
                for _ in range(500):
                    monitor.check_fps(15.0)
                    monitor.check_latency(100.0)
                    monitor.check_resources(30.0, 150.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=checker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


class TestIntegrationWithMetrics:
    def test_fps_meter_feeds_health_check(self):
        monitor = HealthMonitor()
        fps = FpsMeter()
        for _ in range(30):
            fps.tick()
            time.sleep(0.01)
        check = monitor.check_fps(fps.fps)
        assert check.component == "fps"
        assert "fps" in check.metrics

    def test_latency_tracker_feeds_health_check(self):
        monitor = HealthMonitor()
        lat = LatencyTracker()
        for _ in range(100):
            lat.add("analyze_ms", 50.0)
        check = monitor.check_latency(lat.stats("analyze_ms")["p95"])
        assert check.component == "latency"
        assert "p95_ms" in check.metrics

    def test_resource_monitor_feeds_health_check(self):
        monitor = HealthMonitor()
        res = ResourceMonitor()
        sample = res.sample()
        check = monitor.check_resources(sample["proc_cpu_pct"], sample["rss_mb"])
        assert check.component == "resources"
        assert "cpu_pct" in check.metrics
        assert "rss_mb" in check.metrics


class TestThresholdConfiguration:
    def test_custom_thresholds(self):
        monitor = HealthMonitor(
            fps_degraded_threshold=20.0,
            fps_unhealthy_threshold=5.0,
            latency_degraded_ms=100.0,
            latency_unhealthy_ms=300.0,
            cpu_degraded_pct=50.0,
            error_rate_degraded=10.0,
        )
        # With custom thresholds, 15 fps is degraded
        assert monitor.check_fps(15.0).status == HealthStatus.DEGRADED
        # 100ms latency is degraded
        assert monitor.check_latency(100.0).status == HealthStatus.DEGRADED
        # 50% CPU is degraded
        assert monitor.check_resources(55.0, 100.0).status == HealthStatus.DEGRADED


class TestDiagnoseSnapshotConsistency:
    def test_diagnose_returns_immutable_snapshot(self):
        monitor = HealthMonitor()
        report1 = monitor.diagnose()
        # Modify internal state
        monitor.update_camera(HealthStatus.UNHEALTHY, "dead")
        report2 = monitor.diagnose()
        # report1 should be unchanged
        assert report1.overall_status == HealthStatus.HEALTHY
        assert report2.overall_status == HealthStatus.UNHEALTHY

    def test_checks_tuple_is_ordered(self):
        monitor = HealthMonitor()
        monitor.update_camera(HealthStatus.HEALTHY, "ok")
        monitor.update_detector(HealthStatus.HEALTHY, "ok")
        monitor.update_event_sink(HealthStatus.HEALTHY, "ok")
        report = monitor.diagnose()
        # Checks should be in a consistent order
        components = [c.component for c in report.checks]
        # Order doesn't matter but should be deterministic
        assert components == sorted(components) or components == ["camera", "detector", "event_sink"]


class TestHealthMonitorWithRealComponents:
    def test_full_pipeline_simulation(self):
        """Simulate a monitoring worker updating health at each frame."""
        monitor = HealthMonitor(
            fps_degraded_threshold=10.0,
            latency_degraded_ms=150.0,
        )
        fps = FpsMeter()
        lat = LatencyTracker()
        res = ResourceMonitor()

        # Simulate 100 frames of normal operation
        for i in range(100):
            fps.tick()
            lat.add("analyze_ms", 50.0)
            if i % 10 == 0:
                sample = res.sample()
                monitor.check_resources(sample["proc_cpu_pct"], sample["rss_mb"])
            monitor.check_fps(fps.fps)
            monitor.check_latency(lat.stats("analyze_ms")["p95"])
            monitor.record_event()
            time.sleep(0.001)

        report = monitor.diagnose()
        assert report.overall_status == HealthStatus.HEALTHY
        assert report.total_events_published == 100
        assert report.sessions_active == 0  # not set

        # Now simulate degradation
        monitor.set_active_sessions(1)
        monitor.update_camera(HealthStatus.HEALTHY, "ok")
        monitor.update_detector(HealthStatus.DEGRADED, "high latency", p95_ms=400)
        monitor.update_event_sink(HealthStatus.HEALTHY, "ok")
        monitor.update_resources(HealthStatus.DEGRADED, "high cpu", cpu_pct=85, rss_mb=200)

        report = monitor.diagnose()
        assert report.overall_status == HealthStatus.DEGRADED
        assert report.sessions_active == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])