"""Tests for the transaction-cost-aware portfolio backtest (Phase 4, Sections 10-13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cross_sectional_analysis import build_cross_sectional_panel
from src.evaluation.portfolio_backtest import (
    backtest_top_n_portfolio,
    buy_and_hold_baseline,
    compute_risk_adjusted_metrics,
    equal_weight_universe_baseline,
    momentum_ranking_baseline,
    random_selection_baseline,
)


def _make_series(n: int, mu: float, vol: float, base: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = base * np.cumprod(1 + rng.normal(mu, vol, n))
    close_s = pd.Series(close, index=dates, dtype=float)
    return pd.DataFrame(
        {"open": close_s * 0.999, "high": close_s * 1.01, "low": close_s * 0.99, "close": close_s, "volume": 2_000_000.0},
        index=dates,
    )


def _universe(n: int = 700) -> dict[str, pd.DataFrame]:
    return {
        "BTCUSDT": _make_series(n, 0.0006, 0.02, 40_000, seed=1),
        "AAA": _make_series(n, 0.002, 0.02, 10, seed=2),
        "BBB": _make_series(n, -0.001, 0.02, 5, seed=3),
        "CCC": _make_series(n, 0.0005, 0.025, 20, seed=4),
        "DDD": _make_series(n, 0.0004, 0.018, 8, seed=5),
    }


class TestComputeRiskAdjustedMetrics:
    def test_empty_returns(self) -> None:
        assert compute_risk_adjusted_metrics([], periods_per_year=365) == {"n_periods": 0}

    def test_all_positive_returns_have_full_win_rate_and_no_drawdown(self) -> None:
        metrics = compute_risk_adjusted_metrics([0.01, 0.02, 0.015], periods_per_year=365)
        assert metrics["win_rate"] == 1.0
        assert metrics["max_drawdown"] == pytest.approx(0.0)
        assert metrics["profit_factor"] == float("inf")

    def test_mixed_returns_report_finite_profit_factor(self) -> None:
        metrics = compute_risk_adjusted_metrics([0.05, -0.02, 0.03, -0.01], periods_per_year=365)
        assert 0.0 < metrics["profit_factor"] < float("inf")
        assert metrics["max_drawdown"] <= 0.0

    def test_all_losses_have_zero_win_rate(self) -> None:
        metrics = compute_risk_adjusted_metrics([-0.01, -0.02, -0.03], periods_per_year=365)
        assert metrics["win_rate"] == 0.0
        assert metrics["profit_factor"] == 0.0


class TestBacktestTopNPortfolio:
    def test_produces_one_period_return_per_snapshot(self) -> None:
        candidates = _universe()
        panel = build_cross_sectional_panel(
            candidates, benchmark_symbol="BTCUSDT", timeframe="1D", scan_every_candles=60, min_train_size=300
        )
        result = backtest_top_n_portfolio(panel, candidates, top_n=2, horizon_candles=7, cost_pct=0.001)
        assert len(result.period_returns) == len(panel)
        assert len(result.holdings_per_period) == len(panel)
        assert "sharpe_like" in result.metrics

    def test_higher_cost_never_improves_return(self) -> None:
        candidates = _universe()
        panel = build_cross_sectional_panel(
            candidates, benchmark_symbol="BTCUSDT", timeframe="1D", scan_every_candles=60, min_train_size=300
        )
        low_cost = backtest_top_n_portfolio(panel, candidates, top_n=2, horizon_candles=7, cost_pct=0.0)
        high_cost = backtest_top_n_portfolio(panel, candidates, top_n=2, horizon_candles=7, cost_pct=0.05)
        assert high_cost.metrics["total_return"] <= low_cost.metrics["total_return"]

    def test_min_score_can_exclude_every_period_and_holds_cash(self) -> None:
        candidates = _universe()
        panel = build_cross_sectional_panel(
            candidates, benchmark_symbol="BTCUSDT", timeframe="1D", scan_every_candles=60, min_train_size=300
        )
        result = backtest_top_n_portfolio(
            panel, candidates, top_n=2, horizon_candles=7, min_score=99.99
        )
        assert all(r == 0.0 for r in result.period_returns)
        assert all(h == [] for h in result.holdings_per_period)


class TestBaselines:
    def _panel(self, candidates):
        return build_cross_sectional_panel(
            candidates, benchmark_symbol="BTCUSDT", timeframe="1D", scan_every_candles=60, min_train_size=300
        )

    def test_buy_and_hold_baseline_runs(self) -> None:
        candidates = _universe()
        panel = self._panel(candidates)
        metrics = buy_and_hold_baseline(candidates, "BTCUSDT", panel, horizon_candles=7)
        assert "total_return" in metrics

    def test_equal_weight_universe_baseline_runs(self) -> None:
        candidates = _universe()
        panel = self._panel(candidates)
        metrics = equal_weight_universe_baseline(candidates, panel, horizon_candles=7)
        assert "total_return" in metrics

    def test_momentum_ranking_baseline_runs(self) -> None:
        candidates = _universe()
        panel = self._panel(candidates)
        metrics = momentum_ranking_baseline(candidates, panel, top_n=2, horizon_candles=7)
        assert "total_return" in metrics

    def test_random_selection_baseline_is_reproducible(self) -> None:
        candidates = _universe()
        panel = self._panel(candidates)
        a = random_selection_baseline(candidates, panel, top_n=2, horizon_candles=7, seed=5)
        b = random_selection_baseline(candidates, panel, top_n=2, horizon_candles=7, seed=5)
        assert a == b
