"""Shared conversion between exchange-adapter candle bars and OHLCV frames."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from src.data.exchange import CandleBar


def bars_to_frame(bars: Sequence[CandleBar]) -> pd.DataFrame:
    """Convert a list of :class:`~src.data.exchange.CandleBar` to an OHLCV frame."""
    if not bars:
        raise ValueError("No candle data returned by the exchange adapter.")
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
