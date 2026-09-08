"""Shared market-regime classification.

Previously duplicated as a private ``_market_regime`` inside
``src/models/model_manager.py`` with no equivalent in the opportunity
scanner — meaning the scanner could recommend a bullish pick without ever
surfacing whether the *overall market* (or the asset itself) was in a
bear regime at the time. Extracted here so both the forecaster and the
scanner classify regimes identically.
"""

from __future__ import annotations

import pandas as pd


def classify_regime(closes: pd.Series, lookback: int = 30, threshold: float = 0.05) -> str:
    """Classify recent price action as ``"bull"``, ``"bear"``, or ``"sideways"``.

    Args:
        closes:    Close-price series, ascending.
        lookback:  Number of trailing candles to assess.
        threshold: Minimum fractional change over *lookback* to call it a
                   trend rather than sideways action.
    """
    recent = closes.tail(lookback)
    if len(recent) < 2:
        return "sideways"
    change = (float(recent.iloc[-1]) - float(recent.iloc[0])) / float(recent.iloc[0])
    if change >= threshold:
        return "bull"
    if change <= -threshold:
        return "bear"
    return "sideways"
