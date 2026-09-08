"""Canonical timeframe <-> seconds mapping.

Single source of truth for candle-interval durations. Before this module
existed, ``src/models/model_manager.py`` and ``src/api/routes_forecast.py``
each hard-coded their own copy of this mapping, which is exactly the kind
of duplication that let the two fall out of sync. Every module that needs
to convert a timeframe string to a duration (or vice versa) must import
from here rather than defining a local copy.
"""

from __future__ import annotations

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1H": 3_600,
    "4H": 14_400,
    "1D": 86_400,
    "1W": 604_800,
}


def timeframe_to_seconds(tf: str) -> int:
    """Return the duration of one *tf* candle, in seconds.

    Raises:
        ValueError: If *tf* is not a recognised timeframe string.
    """
    try:
        return TIMEFRAME_SECONDS[tf]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported timeframe '{tf}'. Supported: {list(TIMEFRAME_SECONDS)}"
        ) from exc


def supported_timeframes() -> list[str]:
    return list(TIMEFRAME_SECONDS)
