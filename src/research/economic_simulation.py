"""Economic simulation extensions for the Alpha Research Engine (Phase 5, Sections 9, 10, 13).

Reuses ``src/evaluation/portfolio_backtest.py`` for the transaction-cost-
aware top-N backtest and its baselines (buy-and-hold, equal-weight,
momentum, random) rather than re-implementing them — those functions are
signal-agnostic (they only consume a ``CrossSectionalSnapshot`` panel and
a candidates dict) and were already built generically enough for this
package to use directly. What's new here is what Phase 4 didn't need:
Maximum Favorable/Adverse Excursion (Section 10) and a transaction-cost
sensitivity sweep (Section 9's "run sensitivity analysis with
conservative transaction-cost assumptions").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.cross_sectional_analysis import CrossSectionalSnapshot
from src.evaluation.portfolio_backtest import backtest_top_n_portfolio, compute_risk_adjusted_metrics


def compute_mfe_mae(full_df: pd.DataFrame, scan_index: int, horizon_candles: int) -> tuple[float, float] | None:
    """Maximum Favorable/Adverse Excursion for a long position entered at the scan anchor.

    Anchor convention matches ``forward_return``: entry price is
    ``close[scan_index - 1]``, and the holding window is the
    ``horizon_candles`` candles strictly after it. Returns
    ``(mfe, mae)`` as fractional returns (mfe >= 0 unless the position
    never trades above entry; mae <= 0 unless it never trades below
    entry), or ``None`` if the window falls outside the available data.
    """
    anchor_idx = scan_index - 1
    start = anchor_idx + 1
    end = anchor_idx + horizon_candles  # inclusive
    if anchor_idx < 0 or end >= len(full_df):
        return None
    entry_price = float(full_df["close"].iloc[anchor_idx])
    if entry_price <= 0:
        return None
    window = full_df.iloc[start : end + 1]
    if window.empty:
        return None
    mfe = float(window["high"].max() / entry_price - 1.0)
    mae = float(window["low"].min() / entry_price - 1.0)
    return mfe, mae


@dataclass
class MfeMaeReport:
    n_positions: int
    mean_mfe: float | None
    mean_mae: float | None
    median_mfe: float | None
    median_mae: float | None
    mfe_mae_ratio: float | None  # |mean_mfe / mean_mae|; >1 means favorable excursion typically exceeds adverse

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_positions": self.n_positions,
            "mean_mfe": self.mean_mfe,
            "mean_mae": self.mean_mae,
            "median_mfe": self.median_mfe,
            "median_mae": self.median_mae,
            "mfe_mae_ratio": self.mfe_mae_ratio,
        }


def mfe_mae_report(
    panel: list[CrossSectionalSnapshot],
    candidates: dict[str, pd.DataFrame],
    top_n: int,
    horizon_candles: int,
    min_score: float | None = None,
) -> MfeMaeReport:
    """MFE/MAE pooled across every top-N position taken over the whole panel (Section 10)."""
    mfes: list[float] = []
    maes: list[float] = []

    for snap in panel:
        eligible = snap.scores
        if min_score is not None:
            eligible = {s: v for s, v in eligible.items() if v >= min_score}
        if not eligible:
            continue
        top = sorted(eligible, key=lambda s: -eligible[s])[:top_n]
        for symbol in top:
            result = compute_mfe_mae(candidates[symbol], snap.scan_index, horizon_candles)
            if result is not None:
                mfes.append(result[0])
                maes.append(result[1])

    if not mfes:
        return MfeMaeReport(0, None, None, None, None, None)

    mean_mfe = float(np.mean(mfes))
    mean_mae = float(np.mean(maes))
    ratio = abs(mean_mfe / mean_mae) if mean_mae != 0 else None
    return MfeMaeReport(
        n_positions=len(mfes),
        mean_mfe=mean_mfe,
        mean_mae=mean_mae,
        median_mfe=float(np.median(mfes)),
        median_mae=float(np.median(maes)),
        mfe_mae_ratio=ratio,
    )


@dataclass
class SensitivitySweepResult:
    parameter: str
    values: list[float]
    sharpe_like: list[float | None]
    total_return: list[float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "values": self.values,
            "sharpe_like": self.sharpe_like,
            "total_return": self.total_return,
        }

    def collapses(self, relative_tolerance: float = 0.5) -> bool:
        """True if Sharpe swings sign or drops by more than *relative_tolerance* across the sweep.

        Used to flag fragility (Section 13/19: "if performance disappears
        when a parameter changes ... treat it as overfitting").
        """
        valid = [s for s in self.sharpe_like if s is not None]
        if len(valid) < 2:
            return True
        if max(valid) <= 0:
            return True
        if min(valid) < 0 and max(valid) > 0:
            return True
        return (max(valid) - min(valid)) / abs(max(valid)) > relative_tolerance


def cost_sensitivity_sweep(
    panel: list[CrossSectionalSnapshot],
    candidates: dict[str, pd.DataFrame],
    top_n: int,
    horizon_candles: int,
    cost_grid: list[float] = (0.0005, 0.001, 0.0025, 0.005, 0.01),
    min_score: float | None = None,
    periods_per_year: float = 365.0,
) -> SensitivitySweepResult:
    """Re-run the top-N backtest across a grid of round-trip transaction costs."""
    sharpe: list[float | None] = []
    total_return: list[float | None] = []
    for cost in cost_grid:
        result = backtest_top_n_portfolio(
            panel, candidates, top_n=top_n, horizon_candles=horizon_candles,
            cost_pct=cost, min_score=min_score, periods_per_year=periods_per_year,
        )
        sharpe.append(result.metrics.get("sharpe_like"))
        total_return.append(result.metrics.get("total_return"))
    return SensitivitySweepResult(parameter="cost_pct", values=list(cost_grid), sharpe_like=sharpe, total_return=total_return)


def parameter_sensitivity_sweep(
    build_panel_fn,
    candidates: dict[str, pd.DataFrame],
    top_n: int,
    horizon_candles: int,
    parameter_values: list[int],
    parameter_name: str,
    cost_pct: float = 0.001,
    periods_per_year: float = 365.0,
) -> SensitivitySweepResult:
    """Generic sweep over a signal-construction parameter (e.g. lookback window, top_n, rebalance stride).

    ``build_panel_fn(value) -> list[CrossSectionalSnapshot]`` must build a
    fresh panel for one value of the swept parameter (e.g. re-run
    ``build_signal_panel`` with a different lookback or
    ``scan_every_candles``); this function then backtests each resulting
    panel identically so only the swept parameter differs.
    """
    sharpe: list[float | None] = []
    total_return: list[float | None] = []
    for value in parameter_values:
        panel = build_panel_fn(value)
        result = backtest_top_n_portfolio(
            panel, candidates, top_n=top_n, horizon_candles=horizon_candles,
            cost_pct=cost_pct, periods_per_year=periods_per_year,
        )
        sharpe.append(result.metrics.get("sharpe_like"))
        total_return.append(result.metrics.get("total_return"))
    return SensitivitySweepResult(parameter=parameter_name, values=list(parameter_values), sharpe_like=sharpe, total_return=total_return)
