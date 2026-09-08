"""Point-in-time backtest of the High-Potential Asset Discovery Engine.

Phase 11 of the Phase-2 validation brief is explicit that testing "the
ranking code runs" is not a backtest. This module historically
reconstructs *what the scanner would actually have picked* at each of a
series of past timestamps, using only data available as of that
timestamp, then measures what happened next.

No-lookahead guarantee: at scan point ``t`` (a candle index into each
candidate's full history), :func:`backtest_scanner` calls
``scan_opportunities()`` with ``df.iloc[:t]`` for every candidate — the
scanner never sees a single row at or after ``t``. Only *after* scoring is
complete does the backtest look at ``df.iloc[t : t + forward_window]`` to
measure the outcome. This mirrors exactly the no-leakage discipline
``src/evaluation/backtest.py`` and ``walk_forward.py`` already enforce for
the price forecaster.

What this does **not** solve: classic survivorship bias, where an asset
that later got delisted or died is missing from today's candidate universe
entirely, so a backtest run today can never "pick" a token that no longer
exists to look back at. That requires a historical universe snapshot (the
actual set of tradeable symbols as of each past timestamp, delisted ones
included) which this repository does not have a data source for. This is
called out explicitly in the results this module produces rather than
silently assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.ranking.liquidity_filter import LiquidityFilterConfig
from src.ranking.recommend import scan_opportunities
from src.utils.timeframes import timeframe_to_seconds


@dataclass
class ScannerTrade:
    scan_index: int
    scan_time: int
    symbol: str
    opportunity_score: float
    direction: str
    entry_price: float
    target_price: float
    invalidation_price: float
    forward_window_candles: int
    exit_price: float
    subsequent_return: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    max_drawdown: float
    target_hit: bool
    target_hit_candles: int | None
    invalidation_hit: bool
    invalidation_hit_candles: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_index": self.scan_index,
            "scan_time": self.scan_time,
            "symbol": self.symbol,
            "opportunity_score": self.opportunity_score,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "invalidation_price": self.invalidation_price,
            "forward_window_candles": self.forward_window_candles,
            "exit_price": self.exit_price,
            "subsequent_return": self.subsequent_return,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "max_drawdown": self.max_drawdown,
            "target_hit": self.target_hit,
            "target_hit_candles": self.target_hit_candles,
            "invalidation_hit": self.invalidation_hit,
            "invalidation_hit_candles": self.invalidation_hit_candles,
        }


@dataclass
class ScannerBacktestReport:
    trades: list[ScannerTrade] = field(default_factory=list)
    n_scans: int = 0
    n_scans_with_no_recommendation: int = 0
    caveats: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        if not self.trades:
            return {
                "n_scans": self.n_scans,
                "n_scans_with_no_recommendation": self.n_scans_with_no_recommendation,
                "n_trades": 0,
                "message": "Scanner never produced a qualifying recommendation in this window.",
                "caveats": self.caveats,
            }

        returns = np.array([t.subsequent_return for t in self.trades])
        mfe = np.array([t.max_favorable_excursion for t in self.trades])
        mae = np.array([t.max_adverse_excursion for t in self.trades])
        drawdowns = np.array([t.max_drawdown for t in self.trades])
        target_hits = np.array([t.target_hit for t in self.trades])
        invalidation_hits = np.array([t.invalidation_hit for t in self.trades])

        mean_ret = float(returns.mean())
        std_ret = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
        risk_adjusted_return = mean_ret / std_ret if std_ret > 0 else 0.0

        hit_times = [t.target_hit_candles for t in self.trades if t.target_hit_candles is not None]

        return {
            "n_scans": self.n_scans,
            "n_scans_with_no_recommendation": self.n_scans_with_no_recommendation,
            "n_trades": len(self.trades),
            "mean_subsequent_return": mean_ret,
            "median_subsequent_return": float(np.median(returns)),
            "std_subsequent_return": std_ret,
            "risk_adjusted_return": risk_adjusted_return,
            "win_rate": float((returns > 0).mean()),
            "mean_max_favorable_excursion": float(mfe.mean()),
            "mean_max_adverse_excursion": float(mae.mean()),
            "mean_max_drawdown": float(drawdowns.mean()),
            "target_hit_rate": float(target_hits.mean()),
            "invalidation_hit_rate": float(invalidation_hits.mean()),
            "median_time_to_target_candles": (
                float(np.median(hit_times)) if hit_times else None
            ),
            "caveats": self.caveats,
        }


def _evaluate_trade(
    scan_index: int,
    scan_time: int,
    symbol: str,
    opportunity_score: float,
    direction: str,
    entry_price: float,
    target_price: float,
    invalidation_price: float,
    forward: pd.DataFrame,
) -> ScannerTrade:
    forward_window_candles = len(forward)
    exit_price = float(forward["close"].iloc[-1])
    subsequent_return = (exit_price - entry_price) / entry_price

    if direction == "bullish":
        mfe = float(((forward["high"] - entry_price) / entry_price).max())
        mae = float(((forward["low"] - entry_price) / entry_price).min())
        running_peak = forward["close"].cummax()
        drawdown = float(((forward["close"] - running_peak) / running_peak).min())
        target_hits = np.where(forward["high"].to_numpy() >= target_price)[0]
        invalidation_hits = np.where(forward["low"].to_numpy() <= invalidation_price)[0]
    else:
        mfe = float(((entry_price - forward["low"]) / entry_price).max())
        mae = float(((entry_price - forward["high"]) / entry_price).min())
        running_trough = forward["close"].cummin()
        drawdown = float(((running_trough - forward["close"]) / forward["close"]).min())
        target_hits = np.where(forward["low"].to_numpy() <= target_price)[0]
        invalidation_hits = np.where(forward["high"].to_numpy() >= invalidation_price)[0]

    target_hit = target_hits.size > 0
    invalidation_hit = invalidation_hits.size > 0
    target_hit_candles = int(target_hits[0]) + 1 if target_hit else None
    invalidation_hit_candles = int(invalidation_hits[0]) + 1 if invalidation_hit else None

    return ScannerTrade(
        scan_index=scan_index,
        scan_time=scan_time,
        symbol=symbol,
        opportunity_score=opportunity_score,
        direction=direction,
        entry_price=entry_price,
        target_price=target_price,
        invalidation_price=invalidation_price,
        forward_window_candles=forward_window_candles,
        exit_price=exit_price,
        subsequent_return=subsequent_return,
        max_favorable_excursion=mfe,
        max_adverse_excursion=mae,
        max_drawdown=drawdown,
        target_hit=target_hit,
        target_hit_candles=target_hit_candles,
        invalidation_hit=invalidation_hit,
        invalidation_hit_candles=invalidation_hit_candles,
    )


def backtest_scanner(
    candidates: dict[str, pd.DataFrame],
    benchmark_symbol: str,
    timeframe: str,
    scan_every_candles: int,
    forward_window_candles: int,
    min_train_size: int = 200,
    min_opportunity_score: float = 65.0,
    liquidity_config: LiquidityFilterConfig = LiquidityFilterConfig(),
    top_n: int = 3,
) -> ScannerBacktestReport:
    """Reconstruct historical scanner picks and measure their outcomes.

    Args:
        candidates:           ``{symbol: full ascending OHLCV history}`` —
                              the *entire* history each symbol has, so the
                              function itself can enforce the point-in-time
                              cut rather than trusting the caller to have
                              pre-sliced it (which is exactly the mistake
                              that would silently reintroduce lookahead).
        benchmark_symbol:     Symbol used as the relative-strength reference
                              (typically BTC); must be a key of *candidates*.
        timeframe:            Candle period string shared by all inputs.
        scan_every_candles:   Stride between historical scan points.
        forward_window_candles: How many candles forward to evaluate each pick.
        min_train_size:       Minimum candles of history required before the
                              first scan point (so early scans aren't scored
                              off a handful of candles).
        min_opportunity_score: Same meaning as in ``scan_opportunities``.
        top_n:                Max recommendations taken per scan point.

    Returns:
        A :class:`ScannerBacktestReport`.
    """
    if benchmark_symbol not in candidates:
        raise ValueError(f"benchmark_symbol '{benchmark_symbol}' not in candidates")

    lengths = {symbol: len(df) for symbol, df in candidates.items()}
    max_start = min(lengths.values()) - forward_window_candles
    if max_start <= min_train_size:
        raise ValueError(
            "Not enough shared history across candidates for the requested "
            "min_train_size + forward_window_candles."
        )

    trades: list[ScannerTrade] = []
    n_scans = 0
    n_no_recommendation = 0

    for t in range(min_train_size, max_start, scan_every_candles):
        n_scans += 1
        # Point-in-time cut: every candidate is sliced to [:t] -- the
        # scanner cannot see t or anything after it. This is the single
        # line that prevents lookahead in this backtest.
        point_in_time = {symbol: df.iloc[:t].copy() for symbol, df in candidates.items()}
        benchmark_df = point_in_time[benchmark_symbol]

        result = scan_opportunities(
            candidates=point_in_time,
            benchmark_df=benchmark_df,
            timeframe=timeframe,
            min_opportunity_score=min_opportunity_score,
            liquidity_config=liquidity_config,
            top_n=top_n,
        )

        if not result.recommendations:
            n_no_recommendation += 1
            continue

        scan_time = int(point_in_time[benchmark_symbol].index[-1].timestamp())
        for rec in result.recommendations:
            full_df = candidates[rec.symbol]
            forward = full_df.iloc[t : t + forward_window_candles]
            if len(forward) < forward_window_candles:
                continue
            trades.append(
                _evaluate_trade(
                    scan_index=t,
                    scan_time=scan_time,
                    symbol=rec.symbol,
                    opportunity_score=rec.opportunity_score,
                    direction=rec.direction,
                    entry_price=rec.current_price,
                    target_price=rec.potential_target,
                    invalidation_price=rec.invalidation_price,
                    forward=forward,
                )
            )

    caveats = [
        "Uses a fixed, present-day candidate universe for the entire backtest "
        "window -- it cannot 'pick' an asset that existed historically but is "
        "not in the candidates dict passed in, and it excludes nothing for "
        "having later been delisted. Classic survivorship bias (a real "
        "historical universe snapshot including since-delisted tokens) is "
        "NOT eliminated by this module alone; it is only as good as the "
        "point-in-time universe the caller supplies.",
    ]

    return ScannerBacktestReport(
        trades=trades,
        n_scans=n_scans,
        n_scans_with_no_recommendation=n_no_recommendation,
        caveats=caveats,
    )
