"""Unit tests for detection quality comparison across sample conditions."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluate_images import ImageResult
from exam_monitor.condition_comparison import (
    ConditionComparisonReport,
    ConditionQualityMetrics,
    MultiDetectorComparisonReport,
    compare_image_results,
)


class TestConditionQualityMetrics:

    def test_perfect_condition_metrics(self):
        m = ConditionQualityMetrics(
            name="control",
            category="control",
            total=10,
            expected_faces=10,
            detected=10,
            silent_miss=0,
            flagged=0,
            false_positive=0,
            correct_negative=0,
            confidence_samples=[0.9, 0.95, 0.88],
            luma_samples=[115.0, 120.0],
        )
        assert m.accuracy == 1.0
        assert m.detection_rate == 1.0
        assert m.silent_miss_rate == 0.0
        assert m.flagged_rate == 0.0
        assert m.quality_score == 100.0
        assert m.mean_confidence == pytest.approx(0.91, abs=0.01)

    def test_vulnerable_condition_silent_misses(self):
        m = ConditionQualityMetrics(
            name="occlusion",
            category="occlusion",
            total=5,
            expected_faces=5,
            detected=1,
            silent_miss=4,
            flagged=0,
            false_positive=0,
            correct_negative=0,
        )
        assert m.accuracy == 0.2
        assert m.detection_rate == 0.2
        assert m.silent_miss_rate == 0.8
        # Quality score heavily penalizes silent misses
        assert m.quality_score < 10.0

    def test_flagged_safe_failure_metrics(self):
        m = ConditionQualityMetrics(
            name="dark",
            category="lighting",
            total=4,
            expected_faces=4,
            detected=0,
            silent_miss=0,
            flagged=4,  # DETECTION_UNAVAILABLE (safe failure)
            false_positive=0,
            correct_negative=0,
        )
        assert m.silent_miss_rate == 0.0
        assert m.flagged_rate == 1.0
        # Not penalized as heavily as silent miss
        assert m.quality_score >= 0.0

    def test_negative_controls_metrics(self):
        m = ConditionQualityMetrics(
            name="negative",
            category="negative",
            total=6,
            expected_faces=0,
            detected=0,
            silent_miss=0,
            flagged=1,
            false_positive=0,
            correct_negative=5,
        )
        assert m.accuracy == pytest.approx(5.0 / 6.0)
        assert m.false_positive_rate == 0.0
        assert m.quality_score > 70.0


class TestConditionComparisonReport:

    def _make_report(self) -> ConditionComparisonReport:
        rep = ConditionComparisonReport(
            detector_name="haar",
            baseline_condition="control",
        )
        rep.add_metric(ConditionQualityMetrics(
            name="control", category="control", total=10, expected_faces=10,
            detected=10, silent_miss=0, flagged=0, false_positive=0, correct_negative=0
        ))
        rep.add_metric(ConditionQualityMetrics(
            name="lighting", category="lighting", total=10, expected_faces=10,
            detected=8, silent_miss=0, flagged=2, false_positive=0, correct_negative=0
        ))
        rep.add_metric(ConditionQualityMetrics(
            name="angle_roll", category="angle", total=10, expected_faces=10,
            detected=4, silent_miss=6, flagged=0, false_positive=0, correct_negative=0
        ))
        return rep

    def test_ranking_by_quality(self):
        rep = self._make_report()
        ranking = rep.rank_conditions_by_quality()
        # Control is best, lighting is second, angle_roll is worst (silent misses)
        assert ranking[0][0] == "control"
        assert ranking[1][0] == "lighting"
        assert ranking[2][0] == "angle_roll"

    def test_vulnerability_detection(self):
        rep = self._make_report()
        vulns = rep.get_vulnerabilities(silent_miss_threshold=0.20)
        assert "angle_roll" in vulns
        assert "lighting" not in vulns
        assert "control" not in vulns

    def test_relative_baseline_deltas(self):
        rep = self._make_report()
        deltas = rep.relative_comparison_to_baseline()
        assert "angle_roll" in deltas
        assert deltas["angle_roll"]["delta_accuracy"] < 0
        assert deltas["angle_roll"]["delta_quality_score"] < 0
        assert deltas["angle_roll"]["relative_drop_pct"] > 50.0

    def test_json_and_markdown_exports(self, tmp_path):
        rep = self._make_report()
        json_file = tmp_path / "comp.json"
        rep.export_json(json_file)
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert data["detector_name"] == "haar"
        assert len(data["ranking"]) == 3

        md_file = tmp_path / "comp.md"
        rep.export_markdown(md_file)
        assert md_file.exists()
        md_text = md_file.read_text(encoding="utf-8")
        assert "# Detection Quality Condition Comparison (haar)" in md_text
        assert "angle_roll" in md_text

        csv_file = tmp_path / "comp.csv"
        rep.export_csv(csv_file)
        assert csv_file.exists()
        lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4  # header + 3 rows


class TestCompareImageResultsIntegration:

    def test_aggregate_image_results_to_comparison_report(self):
        mock_results = [
            ImageResult("control", "normal", 1, 1, "FACE_PRESENT", 0.95, 115.0, "detected"),
            ImageResult("lighting", "bright", 1, 1, "FACE_PRESENT", 0.91, 150.0, "detected"),
            ImageResult("lighting", "dark", 1, 0, "DETECTION_UNAVAILABLE", 0.0, 15.0, "flagged"),
            ImageResult("occlusion", "mask", 1, 0, "FACE_ABSENT", 0.0, 110.0, "silent_miss"),
            ImageResult("negative", "empty", 0, 0, "FACE_ABSENT", 0.0, 100.0, "correct_negative"),
        ]

        report = compare_image_results(mock_results, detector_name="haar", baseline_condition="control")
        assert "control" in report.conditions
        assert "lighting" in report.conditions
        assert "occlusion" in report.conditions
        assert "negative" in report.conditions

        assert report.conditions["control"].accuracy == 1.0
        assert report.conditions["lighting"].flagged == 1
        assert report.conditions["occlusion"].silent_miss == 1
        assert report.conditions["negative"].correct_negative == 1

        vulns = report.get_vulnerabilities(silent_miss_threshold=0.5)
        assert "occlusion" in vulns


class TestMultiDetectorComparison:

    def test_multi_detector_side_by_side(self, tmp_path):
        haar_rep = ConditionComparisonReport(detector_name="haar")
        haar_rep.add_metric(ConditionQualityMetrics("lighting", "lighting", 10, 10, 8, 0, 2, 0, 0))
        haar_rep.add_metric(ConditionQualityMetrics("angle_roll", "angle", 10, 10, 3, 7, 0, 0, 0))

        yunet_rep = ConditionComparisonReport(detector_name="yunet")
        yunet_rep.add_metric(ConditionQualityMetrics("lighting", "lighting", 10, 10, 9, 0, 1, 0, 0))
        yunet_rep.add_metric(ConditionQualityMetrics("angle_roll", "angle", 10, 10, 8, 2, 0, 0, 0))

        multi = MultiDetectorComparisonReport(
            detectors=["haar", "yunet"],
            reports={"haar": haar_rep, "yunet": yunet_rep},
        )

        md = multi.to_markdown()
        assert "Multi-Detector Robustness Comparison" in md
        assert "yunet" in md
        assert "haar" in md

        json_file = tmp_path / "multi.json"
        multi.export_json(json_file)
        assert json_file.exists()
        data = json.loads(json_file.read_text(encoding="utf-8"))
        assert "side_by_side" in data
        assert "angle_roll" in data["side_by_side"]
