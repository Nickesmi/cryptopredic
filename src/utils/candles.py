"""Shared conversion between exchange-adapter candle bars and OHLCV frames."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.data.exchange import CandleBar


def bars_to_frame(bars: Sequence[CandleBar], drop_unclosed: bool = True) -> pd.DataFrame:
    """Convert a list of :class:`~src.data.exchange.CandleBar` to an OHLCV frame.

    Args:
        bars:          Candle bars, oldest-first.
        drop_unclosed: If the last bar is still forming (``is_closed=False``
                       — Binance's REST kline endpoint returns the
                       in-progress candle as the last element), drop it.
                       This is the single, shared enforcement point for
                       "never let a not-yet-closed candle enter feature
                       engineering or a prediction anchor" (timing-audit
                       Phase 5) — every caller that builds model input from
                       live candles goes through this function, so fixing
                       it here fixes it everywhere at once. Only pass
                       ``False`` for callers that explicitly want the live,
                       still-updating tick (e.g. a UI ticker), never for
                       anything that feeds a model.
    """
    if not bars:
        raise ValueError("No candle data returned by the exchange adapter.")
    if drop_unclosed and not bars[-1].is_closed:
        bars = bars[:-1]
    if not bars:
        raise ValueError(
            "No closed candle data available (only an in-progress candle was returned)."
        )
    df = pd.DataFrame(
        {
            "time": [pd.Timestamp(b.time, unit="s", tz="UTC") for b in bars],
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [b.volume for b in bars],
        }
    ).set_index("time")
    return df[["open", "high", "low", "close", "volume"]]
