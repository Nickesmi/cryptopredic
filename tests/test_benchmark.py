"""Tests for baseline forecasters (Phase 13 of the timing audit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.benchmark import (
    buy_and_hold_return,
    compare_to_baselines,
    naive_drift,
    naive_persistence,
)


def _trending_df(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(np.linspace(100, 200, n), index=dates)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=dates,
    )


def test_naive_persistence_always_predicts_no_change() -> None:
    df = _trending_df()
    metrics = naive_persistence(df, horizon_candles=5, lookback=5)
    assert metrics["predictions"] > 0
    # Persistence predicts flat (predicted_return == 0), which the
    # direction_correct definition (`> 0` / `< 0`) never counts as a match
    # against a nonzero actual return — a real edge case in evaluate_prediction
    # worth knowing about, not a bug in the persistence baseline itself.
    assert metrics["directional_accuracy"] == 0.0
    assert metrics["mae"] > 0


def test_naive_drift_tracks_steady_trend_well() -> None:
    df = _trending_df()
    metrics = naive_drift(df, horizon_candles=5, lookback=10)
    assert metrics["predictions"] > 0
    # A perfectly linear trend is exactly what linear drift extrapolation
    # is built to capture.
    assert metrics["mae"] < 1.0
    assert metrics["directional_accuracy"] == 1.0


def test_buy_and_hold_matches_total_return() -> None:
    df = _trending_df()
    result = buy_and_hold_return(df)
    assert result["total_return_pct"] == (200.0 - 100.0) / 100.0 * 100.0


def test_compare_to_baselines_reports_all_three() -> None:
    df = _trending_df()
    drift_metrics = naive_drift(df, horizon_candles=5, lookback=14)
    comparison = compare_to_baselines(
        df, horizon_candles=5, model_metrics=drift_metrics, lookback=14
    )
    assert set(comparison) == {
        "model",
        "naive_persistence",
        "naive_drift",
        "buy_and_hold",
        "model_beats_persistence",
        "model_beats_drift",
    }
    # The "model" here literally is the drift baseline, so it must tie itself.
    assert comparison["model_beats_drift"] is True
