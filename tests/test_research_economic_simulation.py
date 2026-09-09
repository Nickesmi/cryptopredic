"""Tests for src/research/economic_simulation.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cross_sectional_analysis import CrossSectionalSnapshot
from src.research.economic_simulation import (
    compute_mfe_mae,
    cost_sensitivity_sweep,
    mfe_mae_report,
)


def _trending_df(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = pd.Series(np.linspace(100, 150, n), index=dates)
    return pd.DataFrame(
        {"open": close, "high": close * 1.02, "low": close * 0.98, "close": close, "volume": 1.0},
        index=dates,
    )


def test_compute_mfe_mae_on_a_clean_uptrend() -> None:
    df = _trending_df()
    result = compute_mfe_mae(df, scan_index=51, horizon_candles=10)
    assert result is not None
    mfe, mae = result
    assert mfe > 0  # uptrend: favorable excursion should be positive
    assert mae <= 0 or mae < mfe  # high/low bounds around an uptrend


def test_compute_mfe_mae_out_of_range_returns_none() -> None:
    df = _trending_df(n=50)
    assert compute_mfe_mae(df, scan_index=45, horizon_candles=20) is None


def test_mfe_mae_report_aggregates_across_panel() -> None:
    candidates = {"A": _trending_df(), "B": _trending_df()}
    panel = [
        CrossSectionalSnapshot(scan_index=60, scan_time=0, scores={"A": 1.0, "B": 0.5}, excluded=[]),
        CrossSectionalSnapshot(scan_index=80, scan_time=0, scores={"A": 0.8, "B": 0.9}, excluded=[]),
    ]
    report = mfe_mae_report(panel, candidates, top_n=1, horizon_candles=10)
    assert report.n_positions == 2
    assert report.mean_mfe is not None


def test_mfe_mae_report_empty_panel() -> None:
    report = mfe_mae_report([], {}, top_n=1, horizon_candles=10)
    assert report.n_positions == 0
    assert report.mean_mfe is None


def test_cost_sensitivity_sweep_increasing_cost_never_increases_return() -> None:
    candidates = {"A": _trending_df(), "B": _trending_df()}
    panel = [
        CrossSectionalSnapshot(scan_index=60 + i * 10, scan_time=0, scores={"A": 1.0, "B": 0.5}, excluded=[])
        for i in range(5)
    ]
    result = cost_sensitivity_sweep(
        panel, candidates, top_n=1, horizon_candles=5, cost_grid=[0.0, 0.01, 0.05]
    )
    returns = [r for r in result.total_return if r is not None]
    assert returns == sorted(returns, reverse=True)  # higher cost -> lower or equal return
