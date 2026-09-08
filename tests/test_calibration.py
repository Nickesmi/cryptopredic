"""Tests for bucketed confidence calibration (Phase 16 / Phase-2 #8 of the audit)."""

from __future__ import annotations

import pytest

from src.evaluation.calibration import calibration_from_rows, compute_calibration
from src.evaluation.prediction_store import PredictionStore


def test_perfectly_calibrated_bucket_has_zero_gap() -> None:
    # 10 predictions at 0.9 confidence, exactly 9 succeed -> 90% actual.
    confidences = [0.9] * 10
    successes = [True] * 9 + [False]
    report = compute_calibration(confidences, successes)
    bucket = next(b for b in report.buckets if b.label == "90%-100%")
    assert bucket.n == 10
    assert bucket.predicted_confidence_mean == pytest.approx(0.9)
    assert bucket.actual_success_rate == pytest.approx(0.9)
    assert bucket.to_dict()["gap"] == pytest.approx(0.0, abs=1e-9)


def test_overconfident_bucket_shows_positive_gap() -> None:
    # System says 90% confident, but only succeeds 30% of the time.
    confidences = [0.95] * 20
    successes = [True] * 6 + [False] * 14
    report = compute_calibration(confidences, successes)
    bucket = next(b for b in report.buckets if b.n > 0)
    gap = bucket.to_dict()["gap"]
    assert gap > 0.5  # predicted much higher than actual
    assert not report.is_well_calibrated(max_gap=0.10)


def test_brier_score_zero_for_perfect_predictions() -> None:
    confidences = [1.0, 0.0, 1.0, 0.0]
    successes = [True, False, True, False]
    report = compute_calibration(confidences, successes, bucket_edges=(0.0, 0.5, 1.0001))
    assert report.brier_score == pytest.approx(0.0)


def test_brier_score_worst_case_is_one() -> None:
    confidences = [1.0, 0.0]
    successes = [False, True]  # maximally wrong in both directions
    report = compute_calibration(confidences, successes, bucket_edges=(0.0, 0.5, 1.0001))
    assert report.brier_score == pytest.approx(1.0)


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError):
        compute_calibration([0.5, 0.6], [True])


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        compute_calibration([], [])


def test_calibration_from_rows_reads_named_fields() -> None:
    rows = [
        {"confidence": 0.8, "direction_correct": True},
        {"confidence": 0.8, "direction_correct": False},
        {"confidence": 0.55, "direction_correct": True},
    ]
    report = calibration_from_rows(rows)
    assert report.n_total == 3


class TestPredictionStoreCalibration:
    def test_returns_none_when_store_is_empty(self, tmp_path) -> None:
        store = PredictionStore(tmp_path / "empty.sqlite3")
        assert store.calibration_report() is None
        assert store.dashboard()["calibration"] is None

    def test_returns_report_once_predictions_are_evaluated(self, tmp_path) -> None:
        store = PredictionStore(tmp_path / "predictions.sqlite3")
        record = store.create_prediction(
            expires_at=1_700_000_300,
            model_name="xgboost",
            model_version="1.0.0",
            symbol="BTCUSDT",
            timeframe="1H",
            prediction_horizon=5,
            anchor_time=1_700_000_000,
            anchor_price=100.0,
            predicted_price=110.0,
            confidence=0.8,
            bullish_probability=0.8,
            bearish_probability=0.2,
            expected_volatility=0.02,
            prediction_values={"expected_price": 110.0},
        )
        store.evaluate_prediction(record.prediction_id, actual_time=1_700_000_300, actual_price=112.0)

        report = store.calibration_report()
        assert report is not None
        assert report["n_total"] == 1
