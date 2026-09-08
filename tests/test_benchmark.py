"""Tests for baseline forecasters (Phase 13 of the timing audit)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.benchmark import (
    buy_and_hold_return,
    compare_to_baselines,
    moving_average,
    naive_drift,
    naive_persistence,
    random_direction,
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


def test_moving_average_bets_on_mean_reversion() -> None:
    df = _trending_df()
    metrics = moving_average(df, horizon_candles=5, lookback=20)
    assert metrics["predictions"] > 0
    # On a strictly increasing series the trailing mean is always below the
    # current price, so MA's implied direction (predicted < anchor) is
    # always wrong -- the opposite failure mode from naive_drift.
    assert metrics["directional_accuracy"] == 0.0


def test_random_direction_is_reproducible_and_roughly_a_coin_flip() -> None:
    df = _trending_df(n=500)
    metrics_a = random_direction(df, horizon_candles=5, lookback=5, seed=1)
    metrics_b = random_direction(df, horizon_candles=5, lookback=5, seed=1)
    assert metrics_a == metrics_b  # same seed -> reproducible
    assert metrics_a["predictions"] > 0
    # Not asserting an exact 50% -- just that it isn't trivially 0% or 100%,
    # i.e. it actually behaves like a coin flip rather than a fixed call.
    assert 0.0 < metrics_a["directional_accuracy"] < 1.0


def test_compare_to_baselines_reports_all_four() -> None:
    df = _trending_df()
    drift_metrics = naive_drift(df, horizon_candles=5, lookback=14)
    comparison = compare_to_baselines(
        df, horizon_candles=5, model_metrics=drift_metrics, lookback=14
    )
    assert set(comparison) == {
        "model",
        "naive_persistence",
        "naive_drift",
        "moving_average",
        "random_direction",
        "buy_and_hold",
        "model_beats_persistence",
        "model_beats_drift",
        "model_beats_moving_average",
        "model_beats_random_direction",
    }
    # The "model" here literally is the drift baseline, so it must tie itself.
    assert comparison["model_beats_drift"] is True
