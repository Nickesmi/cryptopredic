"""Liquidity / data-quality gate for the opportunity scanner.

Phase 19 of the timing audit is explicit that "high potential" must not
be confused with "thin, easily-manipulated market". This module is the
hard gate applied *before* any asset is scored: assets that fail it are
excluded outright, never scored-then-ranked-low, so a single bad candle
in an illiquid book can't accidentally surface as a "recommendation" if
enough other sub-scores happen to look good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidityFilterConfig:
    min_history_candles: int = 90
    min_avg_quote_volume: float = 1_000_000.0  # 24h-equivalent quote volume, in USD
    max_spread_proxy: float = 0.08  # (high-low)/close, averaged — a proxy for bid/ask spread
    max_single_candle_return: float = 0.60  # >60% single-candle move smells like bad data/manipulation


@dataclass(frozen=True)
class LiquidityVerdict:
    passed: bool
    reasons: list[str]
    avg_quote_volume: float
    spread_proxy: float
    history_candles: int


def check_liquidity(
    df: pd.DataFrame, config: LiquidityFilterConfig = LiquidityFilterConfig()
) -> LiquidityVerdict:
    """Evaluate *df* (OHLCV, ascending) against minimum liquidity/quality bars.

    Returns a :class:`LiquidityVerdict` explaining exactly which checks
    failed, so callers/reports can say *why* an asset was excluded rather
    than silently dropping it.
    """
    reasons: list[str] = []
    n = len(df)

    if n < config.min_history_candles:
        reasons.append(
            f"insufficient_history: {n} candles < {config.min_history_candles} required"
        )

    quote_volume = (df["close"] * df["volume"]).tail(min(n, 30))
    avg_quote_volume = float(quote_volume.mean()) if len(quote_volume) else 0.0
    if avg_quote_volume < config.min_avg_quote_volume:
        reasons.append(
            f"low_liquidity: avg quote volume {avg_quote_volume:,.0f} < "
            f"{config.min_avg_quote_volume:,.0f} required"
        )

    spread_proxy_series = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    spread_proxy = float(spread_proxy_series.tail(min(n, 30)).mean() or 0.0)
    if spread_proxy > config.max_spread_proxy:
        reasons.append(
            f"wide_spread_proxy: {spread_proxy:.2%} > {config.max_spread_proxy:.2%} threshold"
        )

    if n >= 2:
        candle_returns = df["close"].pct_change().abs()
        max_move = float(candle_returns.tail(min(n, 60)).max() or 0.0)
        if max_move > config.max_single_candle_return:
            reasons.append(
                f"abnormal_price_action: single-candle move {max_move:.0%} exceeds "
                f"{config.max_single_candle_return:.0%} — possible bad data or manipulation"
            )

    return LiquidityVerdict(
        passed=not reasons,
        reasons=reasons,
        avg_quote_volume=avg_quote_volume,
        spread_proxy=spread_proxy,
        history_candles=n,
    )
