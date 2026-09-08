"""Risk scoring for candidate assets.

Produces a 0-100 risk score (higher = riskier) from volatility, drawdown,
and liquidity — kept deliberately separate from the opportunity score so
that "this could move a lot" and "this is dangerous to hold" are never
collapsed into one number. ``score_coins.py`` treats risk as a penalty
subtracted from — not blended into — the raw opportunity score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskAssessment:
    volatility: float           # stdev of log returns, per-candle
    max_drawdown: float         # most negative peak-to-trough return over the window, as a fraction
    liquidity_score: float      # 0-100, higher = more liquid
    risk_score: float           # 0-100, higher = riskier
    risk_level: str             # "low" | "medium" | "high"


def _max_drawdown(closes: pd.Series) -> float:
    running_max = closes.cummax()
    drawdown = (closes - running_max) / running_max
    return float(drawdown.min()) if len(drawdown) else 0.0


def _liquidity_score(avg_quote_volume: float) -> float:
    """Log-scaled liquidity score: $100k -> ~0, $1M -> ~50, $100M+ -> ~100."""
    if avg_quote_volume <= 0:
        return 0.0
    score = (np.log10(avg_quote_volume) - 5.0) / 3.0 * 100.0
    return float(np.clip(score, 0.0, 100.0))


def assess_risk(df: pd.DataFrame, avg_quote_volume: float, lookback: int = 60) -> RiskAssessment:
    """Assess risk from recent volatility, drawdown, and liquidity.

    Args:
        df:               OHLCV history, ascending.
        avg_quote_volume: Recent average quote (USD) volume, as computed
                          by ``liquidity_filter.check_liquidity``.
        lookback:         Number of trailing candles to assess.
    """
    recent = df.tail(lookback)
    log_returns = np.log(recent["close"] / recent["close"].shift(1)).dropna()
    volatility = float(log_returns.std()) if len(log_returns) > 1 else 0.0
    drawdown = _max_drawdown(recent["close"])
    liquidity_score = _liquidity_score(avg_quote_volume)

    # Volatility and drawdown push risk up; liquidity pulls it down.
    vol_component = float(np.clip(volatility / 0.10, 0.0, 1.0)) * 100.0  # 10%/candle vol -> max
    drawdown_component = float(np.clip(abs(drawdown) / 0.50, 0.0, 1.0)) * 100.0  # 50% drawdown -> max
    illiquidity_component = 100.0 - liquidity_score

    risk_score = float(
        np.clip(0.45 * vol_component + 0.30 * drawdown_component + 0.25 * illiquidity_component, 0.0, 100.0)
    )

    if risk_score >= 66:
        risk_level = "high"
    elif risk_score >= 33:
        risk_level = "medium"
    else:
        risk_level = "low"

    return RiskAssessment(
        volatility=volatility,
        max_drawdown=drawdown,
        liquidity_score=liquidity_score,
        risk_score=risk_score,
        risk_level=risk_level,
    )
