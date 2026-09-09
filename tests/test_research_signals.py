"""Anti-leakage regression tests for src/research/signals.py (Phase 5, Section 3).

For every registered signal: corrupt every candle strictly after a fixed
evaluation point (including extreme 50x and 0.01x price shocks, and a
volume shock), and assert the signal's value AT the evaluation point is
completely unchanged. This is the empirical proof the audit requires --
"causal by construction" (rolling/shift/pct_change) is the engineering
argument; this test is the evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.signals import (
    HORIZON_CANDLES,
    SIGNAL_REGISTRY,
    cross_sectional_relative_strength,
)


def _make_df(n: int = 900, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n)), index=dates)
    volume = pd.Series(rng.lognormal(10, 0.4, n), index=dates)
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": volume},
        index=dates,
    )


def _make_benchmark(n: int = 900, seed: int = 6) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.Series(40_000 * np.cumprod(1 + rng.normal(0.0001, 0.012, n)), index=dates)


EVAL_POINT = 700  # well past every signal's warmup at lookback<=720... use a lookback-safe point per-signal below


@pytest.mark.parametrize("spec", SIGNAL_REGISTRY, ids=lambda s: f"{s.family}:{s.name}")
@pytest.mark.parametrize("shock_multiplier", [50.0, 0.01])
def test_signal_unchanged_by_future_price_shock(spec, shock_multiplier) -> None:
    df = _make_df()
    benchmark = _make_benchmark()
    lookback = HORIZON_CANDLES["1D"]  # 24 candles: short enough that EVAL_POINT has full warmup
    eval_point = 400  # far from both ends; leaves >450 future candles to corrupt

    baseline_series = spec.series(df, benchmark, lookback)
    baseline_value = baseline_series.iloc[eval_point]

    corrupted = df.copy()
    price_cols = ["open", "high", "low", "close"]
    corrupted.loc[corrupted.index[eval_point + 1 :], price_cols] *= shock_multiplier
    corrupted.loc[corrupted.index[eval_point + 1 :], "volume"] *= 50.0

    corrupted_benchmark = benchmark.copy()
    corrupted_benchmark.iloc[eval_point + 1 :] *= shock_multiplier

    corrupted_series = spec.series(corrupted, corrupted_benchmark, lookback)
    corrupted_value = corrupted_series.iloc[eval_point]

    if pd.isna(baseline_value):
        assert pd.isna(corrupted_value)
    else:
        assert corrupted_value == pytest.approx(baseline_value, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("spec", SIGNAL_REGISTRY, ids=lambda s: f"{s.family}:{s.name}")
def test_signal_at_every_earlier_point_is_unaffected_by_one_future_shock(spec) -> None:
    """A single shock far in the future must not change ANY earlier value, not just the boundary one."""
    df = _make_df()
    benchmark = _make_benchmark()
    lookback = HORIZON_CANDLES["4H"]
    shock_at = 850

    baseline_series = spec.series(df, benchmark, lookback)

    corrupted = df.copy()
    corrupted.loc[corrupted.index[shock_at:], ["open", "high", "low", "close"]] *= 50.0
    corrupted_benchmark = benchmark.copy()
    corrupted_benchmark.iloc[shock_at:] *= 50.0

    corrupted_series = spec.series(corrupted, corrupted_benchmark, lookback)

    before = baseline_series.iloc[: shock_at - 1]
    after = corrupted_series.iloc[: shock_at - 1]
    pd.testing.assert_series_equal(before, after, check_exact=False, rtol=1e-9, atol=1e-12)


def test_cross_sectional_relative_strength_ignores_universe_composition_beyond_inputs() -> None:
    asset_return = 0.05
    universe_returns = {"A": 0.01, "B": 0.02, "C": 0.03}
    result = cross_sectional_relative_strength(asset_return, universe_returns)
    assert result == pytest.approx(0.05 - (0.01 + 0.02 + 0.03) / 3)


def test_cross_sectional_relative_strength_empty_universe_returns_none() -> None:
    assert cross_sectional_relative_strength(0.05, {}) is None


def test_all_signals_produce_finite_or_nan_values_no_inf() -> None:
    df = _make_df()
    benchmark = _make_benchmark()
    for spec in SIGNAL_REGISTRY:
        series = spec.series(df, benchmark, HORIZON_CANDLES["1D"])
        finite_or_nan = series.dropna()
        assert np.isfinite(finite_or_nan.to_numpy()).all(), f"{spec.name} produced non-finite values"
