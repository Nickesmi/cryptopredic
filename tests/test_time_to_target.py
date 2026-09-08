"""Tests for the empirical time-to-target estimator (Phase 10 of the timing audit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.time_to_target import (
    first_passage_candles,
    most_likely_horizon,
    time_to_target_report,
)


def test_first_passage_candles_exact_values() -> None:
    # Hand-worked example — see the derivation in the audit notes.
    closes = [100, 101, 102, 103, 110, 111, 90, 89]
    result = first_passage_candles(closes, target_return=0.05, max_candles=3)

    assert len(result) == len(closes) - 3
    assert np.isnan(result[0])       # window [101,102,103]: max +3%, never hits +5%
    assert result[1] == 3            # window [102,103,110]: hits at position 3 (110)
    assert result[2] == 2            # window [103,110,111]: hits at position 2 (110)
    assert result[3] == 1            # window [110,111,90]: hits immediately (111)
    assert np.isnan(result[4])       # window [111,90,89]: never reaches +5% from 110


def test_negative_target_return_uses_downside_threshold() -> None:
    closes = [100, 99, 95, 94, 93]
    result = first_passage_candles(closes, target_return=-0.05, max_candles=3)
    # From 100: window [99,95,94] -> -5% first hit at index1 (95, -5%) -> candle 2
    assert result[0] == 2


def test_probability_within_is_nondecreasing_with_horizon() -> None:
    n = 1000
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + rng.normal(0.0001, 0.01, n))
    df = pd.DataFrame({"close": close}, index=dates)

    report = time_to_target_report(df, "1H", target_return=0.02, max_candles=200)
    probs = [p for _, p in sorted(report.probability_within.items())]
    assert all(a <= b + 1e-9 for a, b in zip(probs, probs[1:]))
    assert 0.0 <= report.hit_rate <= 1.0


def test_most_likely_horizon_none_when_target_rarely_reached() -> None:
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(100.0, index=dates)  # perfectly flat — target never reached
    df = pd.DataFrame({"close": close})

    report = time_to_target_report(df, "1H", target_return=0.10, max_candles=100)
    assert report.hit_rate == 0.0
    assert most_likely_horizon(report) is None


def test_raises_on_insufficient_history() -> None:
    df = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(ValueError):
        time_to_target_report(df, "1H", target_return=0.02, max_candles=50)
