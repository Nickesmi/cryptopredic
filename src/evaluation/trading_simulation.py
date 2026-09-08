"""Simple economic-value trading simulation (Phase 3, Section 14).

Prediction accuracy alone is not the point of a trading system — Section
14 of the Phase 3 brief asks whether acting on a formulation's
predictions would actually make money, net of realistic costs. This
module simulates the simplest possible policy on top of an existing
:class:`~src.evaluation.formulation_experiments.FormulationResult`'s
per-prediction records:

    enter LONG  if implied_return > +entry_threshold
    enter SHORT if implied_return < -entry_threshold
    else stay FLAT

with a fixed round-trip cost (fees + slippage) charged on every trade,
and reports total/mean return, win rate, max drawdown, and a Sharpe-like
ratio, compared against buy-and-hold and a naive momentum baseline
computed over the same window.

The entry threshold must be chosen using design-region data BEFORE this
is ever run against a frozen test region — this module does not choose
one for you, precisely so a caller can't accidentally fit it to the data
it's about to be scored on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TradingSimResult:
    n_predictions: int
    n_trades: int
    entry_threshold: float
    cost_pct: float
    total_return: float
    mean_trade_return: float
    win_rate: float
    max_drawdown: float
    sharpe_like: float
    buy_and_hold_return: float
    momentum_baseline_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_predictions": self.n_predictions,
            "n_trades": self.n_trades,
            "entry_threshold": self.entry_threshold,
            "cost_pct": self.cost_pct,
            "total_return": self.total_return,
            "mean_trade_return": self.mean_trade_return,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "sharpe_like": self.sharpe_like,
            "buy_and_hold_return": self.buy_and_hold_return,
            "momentum_baseline_return": self.momentum_baseline_return,
        }


def simulate_trading(
    records: list[dict[str, Any]],
    entry_threshold: float,
    cost_pct: float = 0.001,
) -> TradingSimResult:
    """Simulate the enter-long/short/flat policy over a list of prediction records.

    Args:
        records: Per-prediction records from
            ``formulation_experiments.run_formulation_walk_forward`` (each
            needs ``implied_return`` and ``actual_return``). Records
            should be in chronological order and, for a clean simulation,
            should not overlap in time more than the horizon allows —
            this function does not itself de-overlap them (the caller's
            ``stride`` should already be `>= horizon_candles` for a
            realistic non-overlapping-positions simulation; overlapping
            positions are a modelling choice the caller must make
            deliberately, not an accident of this function).
        entry_threshold: Minimum |implied_return| to take a position.
            Must be chosen from data NOT included in *records* if
            *records* is a frozen test set (Section 12/14 discipline).
        cost_pct: Round-trip transaction cost (fees + slippage), as a
            fraction of notional, charged on every trade taken.
    """
    if not records:
        raise ValueError("No records to simulate.")

    trade_returns = []
    actual_returns_all = [r["actual_return"] for r in records]

    for r in records:
        implied = r["implied_return"]
        actual = r["actual_return"]
        if implied > entry_threshold:
            trade_returns.append(actual - cost_pct)
        elif implied < -entry_threshold:
            trade_returns.append(-actual - cost_pct)
        # else: flat, no trade, no entry in trade_returns

    n_trades = len(trade_returns)
    if n_trades == 0:
        total_return = 0.0
        mean_trade_return = 0.0
        win_rate = 0.0
        max_drawdown = 0.0
        sharpe_like = 0.0
    else:
        arr = np.array(trade_returns)
        total_return = float(np.sum(arr))
        mean_trade_return = float(np.mean(arr))
        win_rate = float(np.mean(arr > 0))
        cumulative = np.cumsum(arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_drawdown = float(drawdowns.min())
        std = float(np.std(arr, ddof=1)) if n_trades > 1 else 0.0
        sharpe_like = mean_trade_return / std * np.sqrt(n_trades) if std > 0 else 0.0

    buy_and_hold_return = float(np.sum(actual_returns_all))

    # Naive momentum baseline: always take the position implied by the
    # SIGN of the actual return two predictions ago (a trivial, free
    # "momentum continues" rule) -- included so the simulated policy is
    # judged against something simpler than itself, not just against
    # doing nothing.
    momentum_positions = [0.0] + list(np.sign(actual_returns_all[:-1]))
    momentum_returns = [
        pos * ret - (cost_pct if pos != 0 else 0.0)
        for pos, ret in zip(momentum_positions, actual_returns_all)
    ]
    momentum_baseline_return = float(np.sum(momentum_returns))

    return TradingSimResult(
        n_predictions=len(records),
        n_trades=n_trades,
        entry_threshold=entry_threshold,
        cost_pct=cost_pct,
        total_return=total_return,
        mean_trade_return=mean_trade_return,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        sharpe_like=sharpe_like,
        buy_and_hold_return=buy_and_hold_return,
        momentum_baseline_return=momentum_baseline_return,
    )
