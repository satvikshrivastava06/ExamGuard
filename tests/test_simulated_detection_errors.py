"""Tests for score inputs under simulated detection errors."""

from __future__ import annotations

import numpy as np
import pytest

from exam_monitor import (
    ErrorInjectionConfig,
    SimulatedErrorDetector,
    SimulatedErrorType,
    StaticTruthDetector,
    UncertaintyLevel,
    simulate_score_inputs,
)
from exam_monitor.config import Config
from exam_monitor.events import EventType
from exam_monitor.pipeline import analyze_frame


class TestSimulatedDetectionErrors:
    def test_clean_baseline_scores_100(self):
        res = simulate_score_inputs(
            [1] * 80,
            ErrorInjectionConfig(),
            step_s=0.5,
        )
        assert res.metrics.quality_score == pytest.approx(100.0)
        assert res.metrics.silent_miss == 0
        assert res.metrics.false_positive == 0
        assert res.uncertainty.uncertainty_level == UncertaintyLevel.LOW

    def test_false_negatives_penalize_quality_score(self):
        clean = simulate_score_inputs([1] * 80, ErrorInjectionConfig(), step_s=0.5)
        noisy = simulate_score_inputs(
            [1] * 80,
            ErrorInjectionConfig(false_negative_rate=0.25, seed=123),
            step_s=0.5,
        )
        assert noisy.metrics.silent_miss > 0
        assert noisy.metrics.silent_miss_rate > 0.0
        assert noisy.metrics.quality_score < clean.metrics.quality_score

    def test_false_positives_on_empty_frames_reduce_score(self):
        res = simulate_score_inputs(
            [0] * 40,
            ErrorInjectionConfig(false_positive_rate=0.30, seed=7),
            step_s=0.5,
        )
        assert res.metrics.false_positive > 0
        assert res.metrics.false_positive_rate > 0.0
        assert res.metrics.quality_score < 100.0

    def test_detector_exceptions_are_flagged_safely_not_silent_misses(self):
        cfg = Config()
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        sim = SimulatedErrorDetector(
            StaticTruthDetector(expected_faces=1),
            ErrorInjectionConfig(force_errors_at={0: SimulatedErrorType.DETECTOR_EXCEPTION}),
        )
        obs, dets, details = analyze_frame(frame, sim, cfg)
        assert obs == EventType.DETECTION_UNAVAILABLE
        assert dets == []
        assert details["reason"] == "detector_exception"

    def test_blind_frames_and_exceptions_raise_uncertainty_indicators(self):
        res = simulate_score_inputs(
            [1] * 80,
            ErrorInjectionConfig(blind_frame_rate=0.30, exception_rate=0.10, seed=99),
            step_s=0.5,
        )
        assert res.metrics.flagged > 0
        assert res.metrics.silent_miss == 0
        assert res.report.detection_unavailable_s > 0.0
        codes = {i.code for i in res.uncertainty.indicators}
        assert "DETECTOR_BLIND_TIME" in codes
        assert res.uncertainty.uncertainty_level in (
            UncertaintyLevel.HIGH,
            UncertaintyLevel.CRITICAL,
            UncertaintyLevel.MODERATE,
        )

    def test_confidence_noise_degrades_confidence_stats(self):
        res = simulate_score_inputs(
            [1] * 60,
            ErrorInjectionConfig(confidence_noise_std=0.25, seed=5),
            step_s=0.5,
        )
        assert res.metrics.total == 60
        assert len(res.metrics.confidence_samples) > 0
        assert any(c != pytest.approx(0.92) for c in res.metrics.confidence_samples)
