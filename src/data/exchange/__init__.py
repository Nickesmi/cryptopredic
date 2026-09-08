"""Exchange data layer — shared types.

``CandleBar`` is the canonical data transfer object for a single
OHLCV candle, shared across all exchange adapters, the API layer,
and the WebSocket relay.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandleBar:
    """Immutable OHLCV candle representation.

    Attributes:
        time:      Unix timestamp in **seconds** (UTC).
        open:      Opening price (USD).
        high:      High price (USD).
        low:       Low price (USD).
        close:     Closing price (USD).
        volume:    Trade volume (base asset units).
        is_closed: True when the candle period is fully closed.
    """

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True
