"""Regression tests for the timeframe/horizon confusion bug.

Root cause (see docs/TIMING_AUDIT_REPORT.md): ``ModelManager`` used to map
the *chart* timeframe to a completely different, hard-coded forecast
horizon in *calendar days* (``_TF_TO_DAYS``), train the model on that
mismatched horizon, then label/expire the resulting prediction as if it
were an ``n_candles``-ahead forecast at the requested timeframe. E.g. a
"4H, n=20" request trained a 30-calendar-day-ahead model but displayed and
expired it as an 80-hour (~3.3 day) forecast.

These tests pin down the invariant that must hold forever: the model's
target timestamp always equals ``anchor_time + n_candles * timeframe``,
for every timeframe, and that value is what the system uses for display,
expiry and evaluation. If someone reintroduces a timeframe -> horizon
lookup table, these tests fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.model_manager import ModelManager
from src.utils.timeframes import timeframe_to_seconds


def _synthetic_ohlcv(n: int, freq_seconds: int, period_candles: int = 30) -> pd.DataFrame:
    """A smooth, bounded, oscillating OHLCV series (no leakage / no headline noise)."""
    dates = pd.date_range("2024-01-01", periods=n, freq=pd.Timedelta(seconds=freq_seconds))
    t = np.arange(n)
    close = pd.Series(
        200 + 30 * np.sin(2 * np.pi * t / period_candles), index=dates, dtype=float
    )
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000.0,
        },
        index=dates,
    )


@pytest.mark.parametrize(
    "timeframe,n_candles",
    [
        ("1H", 4),
        ("4H", 6),
        ("1D", 7),
        ("1m", 30),
    ],
)
def test_target_timestamp_matches_requested_horizon(timeframe: str, n_candles: int) -> None:
    interval = timeframe_to_seconds(timeframe)
    df = _synthetic_ohlcv(n=400, freq_seconds=interval)

    manager = ModelManager()
    forecast = manager.predict_from_frame(
        symbol="BTCUSDT", timeframe=timeframe, raw_df=df, n_candles=n_candles
    )

    expected_horizon_seconds = n_candles * interval
    assert forecast.horizon_seconds == expected_horizon_seconds
    assert forecast.target_timestamp == forecast.anchor_time + expected_horizon_seconds

    # The last timestamp on the projected price path must be the target
    # timestamp — this is what expires_at is built from in routes_forecast.
    assert forecast.timestamps[-1] == forecast.target_timestamp


def test_same_n_candles_different_timeframes_yield_different_horizons() -> None:
    """A "4 candle" request on 1H data must NOT be trained/labelled the same
    as a "4 candle" request on 1D data — they are different horizons
    (4 hours vs. 4 days). Before the fix, both could be silently remapped
    to the same bucketed horizon-in-days.
    """
    manager = ModelManager()

    df_1h = _synthetic_ohlcv(n=400, freq_seconds=timeframe_to_seconds("1H"))
    df_1d = _synthetic_ohlcv(n=400, freq_seconds=timeframe_to_seconds("1D"))

    forecast_1h = manager.predict_from_frame("BTCUSDT", "1H", df_1h, n_candles=4)
    forecast_1d = manager.predict_from_frame("BTCUSDT", "1D", df_1d, n_candles=4)

    assert forecast_1h.horizon_seconds == 4 * 3600
    assert forecast_1d.horizon_seconds == 4 * 86400
    assert forecast_1h.horizon_seconds != forecast_1d.horizon_seconds


def test_different_timeframes_never_share_a_cached_model() -> None:
    """Cache key must include timeframe, not just (symbol, horizon-in-days)."""
    manager = ModelManager()

    df_1h = _synthetic_ohlcv(n=400, freq_seconds=timeframe_to_seconds("1H"))
    df_4h = _synthetic_ohlcv(n=400, freq_seconds=timeframe_to_seconds("4H"))

    manager.predict_from_frame("BTCUSDT", "1H", df_1h, n_candles=7)
    manager.predict_from_frame("BTCUSDT", "4H", df_4h, n_candles=7)

    assert ("BTCUSDT", "1H", 7) in manager._forecasters
    assert ("BTCUSDT", "4H", 7) in manager._forecasters
    assert manager._forecasters[("BTCUSDT", "1H", 7)] is not manager._forecasters[("BTCUSDT", "4H", 7)]
