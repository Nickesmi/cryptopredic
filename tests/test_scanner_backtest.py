"""Tests for the point-in-time opportunity-scanner backtest (Phase 11 of
the Phase-2 validation brief)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.scanner_backtest import backtest_scanner


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


def _universe(n: int = 900):
    btc = _make_series(n, mu=0.0008, vol=0.02, base=40_000, seed=1)
    strong = _make_series(n, mu=0.002, vol=0.02, base=10, seed=2)
    weak = _make_series(n, mu=-0.001, vol=0.02, base=5, seed=3)
    return {"BTCUSDT": btc, "AAA": strong, "BBB": weak}


def test_produces_trades_and_sane_summary() -> None:
    candidates = _universe()
    report = backtest_scanner(
        candidates,
        benchmark_symbol="BTCUSDT",
        timeframe="1D",
        scan_every_candles=30,
        forward_window_candles=14,
        min_train_size=200,
        min_opportunity_score=55,
        top_n=2,
    )
    summary = report.summary()
    assert summary["n_scans"] > 0
    assert summary["n_trades"] > 0
    assert -1.0 <= summary["win_rate"] <= 1.0
    assert 0.0 <= summary["target_hit_rate"] <= 1.0
    assert 0.0 <= summary["invalidation_hit_rate"] <= 1.0
    assert summary["caveats"], "survivorship-bias caveat must always be surfaced"


def test_no_lookahead_changing_the_future_does_not_change_past_picks() -> None:
    """The defining no-survivorship-bias check: scores at scan point t must
    be identical whether or not the data after t is corrupted."""
    candidates = _universe()
    report_a = backtest_scanner(
        candidates,
        benchmark_symbol="BTCUSDT",
        timeframe="1D",
        scan_every_candles=200,
        forward_window_candles=14,
        min_train_size=200,
        min_opportunity_score=1.0,  # force recommendations through
        top_n=3,
    )

    corrupted = {k: v.copy() for k, v in candidates.items()}
    # Blow up every candidate's future (everything after the first scan
    # point) with an extreme, obviously-detectable price shock.
    first_scan_idx = 200
    for symbol, df in corrupted.items():
        df.iloc[first_scan_idx:, df.columns.get_loc("close")] *= 100.0

    report_b = backtest_scanner(
        corrupted,
        benchmark_symbol="BTCUSDT",
        timeframe="1D",
        scan_every_candles=200,
        forward_window_candles=14,
        min_train_size=200,
        min_opportunity_score=1.0,
        top_n=3,
    )

    scores_a = {(t.scan_index, t.symbol): t.opportunity_score for t in report_a.trades if t.scan_index == first_scan_idx}
    scores_b = {(t.scan_index, t.symbol): t.opportunity_score for t in report_b.trades if t.scan_index == first_scan_idx}
    assert scores_a  # sanity: the first scan actually produced trades
    assert scores_a == scores_b


def test_raises_when_benchmark_not_in_candidates() -> None:
    candidates = _universe()
    with pytest.raises(ValueError):
        backtest_scanner(
            candidates,
            benchmark_symbol="NOTREAL",
            timeframe="1D",
            scan_every_candles=30,
            forward_window_candles=14,
        )


def test_no_recommendation_scans_are_tracked_separately() -> None:
    candidates = _universe()
    report = backtest_scanner(
        candidates,
        benchmark_symbol="BTCUSDT",
        timeframe="1D",
        scan_every_candles=60,
        forward_window_candles=14,
        min_train_size=200,
        min_opportunity_score=99.9,  # essentially impossible to clear
        top_n=2,
    )
    assert report.n_scans_with_no_recommendation == report.n_scans
    assert report.trades == []
    assert "qualifying recommendation" in report.summary()["message"].lower()
