"""Transaction-cost-aware top-N portfolio backtest for the opportunity scanner.

Phase 4, Sections 10-13 of the audit: a positive Information Coefficient
(cross_sectional_analysis.py) proves a *ranking* signal exists — it does
not by itself prove a *tradable* strategy exists once realistic costs and
concentration (top-N, not the whole cross-section) are introduced. This
module simulates the simplest such strategy: at each scan point, take the
scanner's top-N eligible assets, equal-weight them, hold for one horizon,
pay a round-trip cost per position, then rebalance — and reports the
standard risk-adjusted performance suite (Section 13) rather than
cumulative return alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.cross_sectional_analysis import CrossSectionalSnapshot, forward_return


@dataclass
class PortfolioBacktestResult:
    period_returns: list[float] = field(default_factory=list)  # net of costs, one per rebalance
    holdings_per_period: list[list[str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_periods": len(self.period_returns),
            "holdings_per_period": self.holdings_per_period,
            "metrics": self.metrics,
        }


def compute_risk_adjusted_metrics(returns: list[float], periods_per_year: float) -> dict[str, Any]:
    """Standard risk-adjusted performance suite (Section 13) from a period-return series."""
    arr = np.array(returns, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n_periods": 0}

    cumulative = np.cumprod(1 + arr)
    total_return = float(cumulative[-1] - 1)
    mean_r = float(arr.mean())
    std_r = float(arr.std(ddof=1)) if n > 1 else 0.0
    sharpe = (mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0

    downside = arr[arr < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean_r / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else 0.0

    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())
    # CAGR-like annualisation of total_return over the sample, for Calmar.
    years = n / periods_per_year if periods_per_year > 0 else 1.0
    cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 and (1 + total_return) > 0 else None
    calmar = (cagr / abs(max_dd)) if (cagr is not None and max_dd < 0) else None

    wins = arr[arr > 0]
    losses = arr[arr < 0]
    win_rate = float((arr > 0).mean())
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "n_periods": n,
        "total_return": total_return,
        "cagr": cagr,
        "volatility_per_period": std_r,
        "max_drawdown": max_dd,
        "sharpe_like": sharpe,
        "sortino_like": sortino,
        "calmar_like": calmar,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
    }


def backtest_top_n_portfolio(
    panel: list[CrossSectionalSnapshot],
    candidates: dict[str, pd.DataFrame],
    top_n: int,
    horizon_candles: int,
    cost_pct: float = 0.001,
    min_score: float | None = None,
    periods_per_year: float = 365.0,
) -> PortfolioBacktestResult:
    """Equal-weight top-N rebalancing strategy, net of a round-trip cost per position.

    Args:
        panel:            Cross-sectional snapshots from
                          ``cross_sectional_analysis.build_cross_sectional_panel``.
        candidates:       ``{symbol: full OHLCV history}`` (same dict passed
                          to build the panel).
        top_n:            Number of assets to hold each period.
        horizon_candles:  Holding period, in candles, before the next rebalance.
        cost_pct:         Round-trip cost (fees + slippage) charged per
                          position held each period — charged every period
                          regardless of whether the holding changed, since a
                          position "rebalanced" into the same asset again
                          still typically means closing and reopening a
                          sized position in a real venue; this is the more
                          conservative (not zero-cost-on-no-change) assumption.
        min_score:        Optional minimum opportunity score to be eligible
                          at all (mirrors ``DEFAULT_MIN_OPPORTUNITY_SCORE`` —
                          a period where nothing clears the bar holds cash,
                          i.e. a zero return for that period, consistent
                          with the scanner's "no forced recommendation" rule).
        periods_per_year: Used to annualise Sharpe/Sortino/Calmar; set to
                          ``365 / (holding-period-in-days)`` for daily data,
                          or the equivalent for other timeframes.
    """
    period_returns: list[float] = []
    holdings_per_period: list[list[str]] = []

    for snap in panel:
        eligible = snap.scores
        if min_score is not None:
            eligible = {s: v for s, v in eligible.items() if v >= min_score}

        if not eligible:
            period_returns.append(0.0)
            holdings_per_period.append([])
            continue

        top = sorted(eligible, key=lambda s: -eligible[s])[:top_n]
        position_returns = []
        for symbol in top:
            r = forward_return(candidates[symbol], snap.scan_index, horizon_candles)
            if r is not None:
                position_returns.append(r - cost_pct)

        period_return = float(np.mean(position_returns)) if position_returns else 0.0
        period_returns.append(period_return)
        holdings_per_period.append(top)

    metrics = compute_risk_adjusted_metrics(period_returns, periods_per_year=periods_per_year)

    return PortfolioBacktestResult(
        period_returns=period_returns,
        holdings_per_period=holdings_per_period,
        metrics=metrics,
    )


def buy_and_hold_baseline(
    candidates: dict[str, pd.DataFrame],
    symbol: str,
    panel: list[CrossSectionalSnapshot],
    horizon_candles: int,
    periods_per_year: float = 365.0,
) -> dict[str, Any]:
    """Baseline: hold a single asset (e.g. BTC) over the same rebalance schedule as the portfolio backtest."""
    returns = [
        r for snap in panel
        if (r := forward_return(candidates[symbol], snap.scan_index, horizon_candles)) is not None
    ]
    return compute_risk_adjusted_metrics(returns, periods_per_year=periods_per_year)


def equal_weight_universe_baseline(
    candidates: dict[str, pd.DataFrame],
    panel: list[CrossSectionalSnapshot],
    horizon_candles: int,
    exclude: set[str] | None = None,
    periods_per_year: float = 365.0,
) -> dict[str, Any]:
    """Baseline: equal-weight the entire candidate universe (no ranking at all) each period."""
    exclude = exclude or set()
    period_returns = []
    for snap in panel:
        symbols = [s for s in candidates if s not in exclude]
        rets = [
            r for s in symbols
            if (r := forward_return(candidates[s], snap.scan_index, horizon_candles)) is not None
        ]
        period_returns.append(float(np.mean(rets)) if rets else 0.0)
    return compute_risk_adjusted_metrics(period_returns, periods_per_year=periods_per_year)


def momentum_ranking_baseline(
    candidates: dict[str, pd.DataFrame],
    panel: list[CrossSectionalSnapshot],
    top_n: int,
    horizon_candles: int,
    lookback_candles: int = 14,
    cost_pct: float = 0.001,
    periods_per_year: float = 365.0,
) -> dict[str, Any]:
    """Baseline: rank purely by trailing return over *lookback_candles* (no scanner at all).

    The audit brief is explicit that the scanner must demonstrate value
    beyond trivial momentum -- this is that trivial-momentum ranking,
    evaluated on the exact same scan points and holding period.
    """
    period_returns = []
    for snap in panel:
        trailing = {}
        for symbol in candidates:
            df = candidates[symbol]
            anchor_idx = snap.scan_index - 1
            past_idx = anchor_idx - lookback_candles
            if past_idx < 0 or anchor_idx >= len(df):
                continue
            past_price = float(df["close"].iloc[past_idx])
            now_price = float(df["close"].iloc[anchor_idx])
            if past_price > 0:
                trailing[symbol] = now_price / past_price - 1.0

        top = sorted(trailing, key=lambda s: -trailing[s])[:top_n]
        rets = [
            r - cost_pct for s in top
            if (r := forward_return(candidates[s], snap.scan_index, horizon_candles)) is not None
        ]
        period_returns.append(float(np.mean(rets)) if rets else 0.0)

    return compute_risk_adjusted_metrics(period_returns, periods_per_year=periods_per_year)


def random_selection_baseline(
    candidates: dict[str, pd.DataFrame],
    panel: list[CrossSectionalSnapshot],
    top_n: int,
    horizon_candles: int,
    cost_pct: float = 0.001,
    periods_per_year: float = 365.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Baseline: pick top_n assets uniformly at random each period (reproducible via seed)."""
    rng = np.random.default_rng(seed)
    period_returns = []
    symbols = list(candidates)
    for snap in panel:
        chosen = rng.choice(symbols, size=min(top_n, len(symbols)), replace=False)
        rets = [
            r - cost_pct for s in chosen
            if (r := forward_return(candidates[s], snap.scan_index, horizon_candles)) is not None
        ]
        period_returns.append(float(np.mean(rets)) if rets else 0.0)
    return compute_risk_adjusted_metrics(period_returns, periods_per_year=periods_per_year)
