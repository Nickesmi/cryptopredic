"""Tests for the unclosed-candle data-quality gate (Phase 6, Section 4).

src/utils/candles.py::bars_to_frame already drops a trailing unclosed
candle at the point a CandleBar list first becomes a DataFrame -- this
gate is the second, independent check for any OHLCV frame that carries
its own is_closed metadata through a different path (Section 4 lists
"unclosed candles" as its own explicit rejection condition, not merely
an assumption inherited from one call site).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.quality import validate_ohlcv


def _clean_frame(n: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = pd.Series(100 + np.arange(n, dtype=float), index=dates)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=dates,
    )


def test_all_closed_candles_pass() -> None:
    df = _clean_frame()
    is_closed = pd.Series(True, index=df.index)
    report = validate_ohlcv(df, "1H", is_closed=is_closed, now=df.index[-1] + pd.Timedelta(minutes=30))
    assert report.is_safe_to_use
    assert not any(i.code == "unclosed_candle" for i in report.issues)


def test_trailing_unclosed_candle_is_critical() -> None:
    df = _clean_frame()
    is_closed = pd.Series(True, index=df.index)
    is_closed.iloc[-1] = False
    report = validate_ohlcv(df, "1H", is_closed=is_closed, now=df.index[-1] + pd.Timedelta(minutes=30))
    assert not report.is_safe_to_use
    codes = [i.code for i in report.critical_issues]
    assert "unclosed_candle" in codes


def test_unclosed_candle_anywhere_not_just_trailing_is_critical() -> None:
    df = _clean_frame()
    is_closed = pd.Series(True, index=df.index)
    is_closed.iloc[10] = False  # mid-series, not the trailing candle
    report = validate_ohlcv(df, "1H", is_closed=is_closed, now=df.index[-1] + pd.Timedelta(minutes=30))
    assert not report.is_safe_to_use
    assert any(i.code == "unclosed_candle" for i in report.critical_issues)


def test_no_is_closed_series_means_the_check_is_skipped_not_failed() -> None:
    df = _clean_frame()
    report = validate_ohlcv(df, "1H", now=df.index[-1] + pd.Timedelta(minutes=30))
    assert not any(i.code == "unclosed_candle" for i in report.issues)


def test_misaligned_is_closed_series_is_flagged_critical() -> None:
    df = _clean_frame()
    is_closed = pd.Series(True, index=df.index[:-5])  # doesn't cover the whole frame
    report = validate_ohlcv(df, "1H", is_closed=is_closed, now=df.index[-1] + pd.Timedelta(minutes=30))
    assert not report.is_safe_to_use
    assert any(i.code == "is_closed_alignment_mismatch" for i in report.critical_issues)
