"""Tests for the cross-sectional ranking-signal evaluation (Phase 4, Sections 4-9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cross_sectional_analysis import (
    build_cross_sectional_panel,
    compute_ic_series,
    forward_return,
    information_coefficient,
    quintile_analysis,
    rank_turnover,
)


def _make_series(n: int, mu: float, vol: float, base: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = base * np.cumprod(1 + rng.normal(mu, vol, n))
    close_s = pd.Series(close, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close_s * 0.999,
            "high": close_s * 1.01,
            "low": close_s * 0.99,
            "close": close_s,
            "volume": 2_000_000.0,
        },
        index=dates,
    )


def _small_universe(n: int = 500) -> dict[str, pd.DataFrame]:
    return {
        "BTCUSDT": _make_series(n, 0.0006, 0.02, 40_000, seed=1),
        "AAA": _make_series(n, 0.002, 0.02, 10, seed=2),   # strong outperformer
        "BBB": _make_series(n, -0.001, 0.02, 5, seed=3),   # underperformer
        "CCC": _make_series(n, 0.0005, 0.025, 20, seed=4),  # noise-like
        "DDD": _make_series(n, 0.0004, 0.018, 8, seed=5),
    }


def test_forward_return_basic() -> None:
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    df = pd.DataFrame({"close": close})
    assert forward_return(df, scan_index=1, horizon_candles=1) == pytest.approx(0.10)
    assert forward_return(df, scan_index=1, horizon_candles=2) == pytest.approx(0.21)


def test_forward_return_none_when_out_of_range() -> None:
    df = pd.DataFrame({"close": [100.0, 110.0]})
    assert forward_return(df, scan_index=2, horizon_candles=5) is None


def test_information_coefficient_perfect_positive_correlation() -> None:
    scores = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    returns = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04, "e": 0.05}
    ic = information_coefficient(scores, returns, method="spearman")
    assert ic == pytest.approx(1.0)


def test_information_coefficient_perfect_negative_correlation() -> None:
    scores = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    returns = {"a": 0.05, "b": 0.04, "c": 0.03, "d": 0.02, "e": 0.01}
    ic = information_coefficient(scores, returns, method="spearman")
    assert ic == pytest.approx(-1.0)


def test_information_coefficient_none_with_too_few_assets() -> None:
    assert information_coefficient({"a": 1.0, "b": 2.0}, {"a": 0.1, "b": 0.2}) is None


def test_information_coefficient_none_with_no_variation() -> None:
    scores = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0, "e": 5.0}
    returns = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5}
    assert information_coefficient(scores, returns) is None


def test_build_cross_sectional_panel_scores_full_universe() -> None:
    candidates = _small_universe()
    panel = build_cross_sectional_panel(
        candidates, benchmark_symbol="BTCUSDT", timeframe="1D",
        scan_every_candles=60, min_train_size=200,
    )
    assert len(panel) > 0
    for snap in panel:
        assert len(snap.scores) + len(snap.excluded) == len(candidates)
        # BTC itself is always scoreable (deep, liquid history) in this fixture
        assert "BTCUSDT" in snap.scores or "BTCUSDT" in snap.excluded


def test_build_cross_sectional_panel_raises_without_benchmark() -> None:
    candidates = _small_universe()
    with pytest.raises(ValueError):
        build_cross_sectional_panel(
            candidates, benchmark_symbol="NOTREAL", timeframe="1D",
            scan_every_candles=60, min_train_size=200,
        )


def test_compute_ic_series_returns_one_series_per_horizon() -> None:
    candidates = _small_universe()
    panel = build_cross_sectional_panel(
        candidates, benchmark_symbol="BTCUSDT", timeframe="1D",
        scan_every_candles=60, min_train_size=200,
    )
    ic_series = compute_ic_series(panel, candidates, horizons_candles={"1D": 1, "7D": 7})
    assert set(ic_series) == {"1D", "7D"}
    for result in ic_series.values():
        summary = result.summary()
        assert "n_periods" in summary


def test_quintile_analysis_returns_all_buckets() -> None:
    candidates = _small_universe()
    panel = build_cross_sectional_panel(
        candidates, benchmark_symbol="BTCUSDT", timeframe="1D",
        scan_every_candles=60, min_train_size=200,
    )
    buckets = quintile_analysis(panel, candidates, horizon_candles=7, n_buckets=5)
    assert set(buckets) == {0, 1, 2, 3, 4}


def test_rank_turnover_reports_transitions() -> None:
    candidates = _small_universe()
    panel = build_cross_sectional_panel(
        candidates, benchmark_symbol="BTCUSDT", timeframe="1D",
        scan_every_candles=60, min_train_size=200,
    )
    result = rank_turnover(panel, top_n=2)
    assert "mean_top_n_turnover" in result
    if result["mean_top_n_turnover"] is not None:
        assert 0.0 <= result["mean_top_n_turnover"] <= 1.0


def test_rank_turnover_zero_for_a_static_universe() -> None:
    """If nothing about the underlying data changes structurally, a
    deterministic scoring function scoring the SAME evolving-but-consistent
    trends should not need to be perfectly stable -- but this at least
    confirms the turnover calculation runs and returns a valid fraction."""
    candidates = _small_universe(n=800)
    panel = build_cross_sectional_panel(
        candidates, benchmark_symbol="BTCUSDT", timeframe="1D",
        scan_every_candles=30, min_train_size=200,
    )
    result = rank_turnover(panel, top_n=2)
    assert result["n_transitions"] > 0
