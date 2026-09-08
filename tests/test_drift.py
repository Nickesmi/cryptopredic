"""Tests for distribution-drift monitoring (Phase 15 of the Phase-2 validation brief)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.drift import (
    detect_drift,
    monitor_prediction_pipeline_drift,
    population_stability_index,
    summarise_drift,
)


def test_identical_distributions_have_near_zero_psi() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0, 1, 2000)
    psi = population_stability_index(ref, cur, n_bins=10)
    assert psi < 0.1


def test_shifted_distribution_has_significant_psi() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(3, 1, 2000)  # large mean shift
    psi = population_stability_index(ref, cur, n_bins=10)
    assert psi > 0.25


def test_detect_drift_classifies_severity() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 1000)
    cur_same = rng.normal(0, 1, 1000)
    cur_shifted = rng.normal(4, 1, 1000)

    report_same = detect_drift(ref, cur_same, "test_metric")
    report_shifted = detect_drift(ref, cur_shifted, "test_metric")

    assert report_same.severity == "none"
    assert report_shifted.severity == "significant"
    assert report_shifted.reference_n == 1000
    assert report_shifted.current_n == 1000


def test_raises_on_insufficient_reference_data() -> None:
    with pytest.raises(ValueError):
        population_stability_index([1.0, 2.0], [1.0, 2.0, 3.0], n_bins=10)


def test_monitor_prediction_pipeline_drift_covers_all_available_metrics() -> None:
    rng = np.random.default_rng(2)
    reference_rows = [
        {
            "anchor_price": 100.0,
            "predicted_price": 100.0 + rng.normal(0, 2),
            "actual_price": 100.0 + rng.normal(0, 2),
            "absolute_error": abs(rng.normal(1, 0.5)),
            "confidence": float(np.clip(rng.normal(0.6, 0.1), 0, 1)),
        }
        for _ in range(200)
    ]
    # A drifted "current" population: predictions much more bullish, errors
    # much larger, confidence artificially inflated.
    current_rows = [
        {
            "anchor_price": 100.0,
            "predicted_price": 100.0 + rng.normal(10, 2),
            "actual_price": 100.0 + rng.normal(0, 2),
            "absolute_error": abs(rng.normal(8, 0.5)),
            "confidence": float(np.clip(rng.normal(0.95, 0.02), 0, 1)),
        }
        for _ in range(200)
    ]

    reports = monitor_prediction_pipeline_drift(reference_rows, current_rows)
    assert "prediction_distribution" in reports
    assert "error_distribution" in reports
    assert "confidence_distribution" in reports
    assert reports["prediction_distribution"].severity == "significant"
    assert reports["error_distribution"].severity == "significant"

    summary = summarise_drift(reports)
    assert summary["any_significant_drift"] is True
    assert "prediction_distribution" in summary["significant_drift"]


def test_feature_drift_detected_per_column() -> None:
    rng = np.random.default_rng(3)
    reference_rows = [{"rsi_14": float(x)} for x in rng.normal(50, 10, 200)]
    current_rows = [{"rsi_14": float(x)} for x in rng.normal(85, 5, 200)]  # drifted overbought

    reports = monitor_prediction_pipeline_drift(
        reference_rows, current_rows, feature_columns=["rsi_14"]
    )
    assert "feature:rsi_14" in reports
    assert reports["feature:rsi_14"].severity == "significant"


def test_missing_fields_are_skipped_not_fabricated() -> None:
    reference_rows = [{"confidence": 0.5} for _ in range(50)]
    current_rows = [{"confidence": 0.5} for _ in range(10)]
    reports = monitor_prediction_pipeline_drift(reference_rows, current_rows)
    # No anchor_price/predicted_price/actual_price/absolute_error present.
    assert "prediction_distribution" not in reports
    assert "target_distribution" not in reports
