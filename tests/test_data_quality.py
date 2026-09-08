"""Tests for the OHLCV data-quality gate (Phase 14 of the Phase-2 validation brief)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.quality import DataQualityError, enforce_quality_gate, validate_ohlcv


def _clean_df(n: int = 50, freq="h", start="2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = pd.Series(np.linspace(100, 110, n), index=dates)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=dates,
    )


def test_clean_data_passes() -> None:
    df = _clean_df()
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = validate_ohlcv(df, "1H", now=now)
    assert report.is_safe_to_use
    assert report.critical_issues == []


def test_empty_frame_is_critical() -> None:
    report = validate_ohlcv(pd.DataFrame(), "1H")
    assert not report.is_safe_to_use
    assert any(i.code == "empty_frame" for i in report.issues)


def test_duplicate_timestamps_are_critical() -> None:
    df = _clean_df()
    df = pd.concat([df, df.iloc[[-1]]])
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = validate_ohlcv(df, "1H", now=now)
    assert not report.is_safe_to_use
    assert any(i.code == "duplicate_candles" for i in report.issues)


def test_impossible_ohlc_is_critical() -> None:
    df = _clean_df()
    df.iloc[10, df.columns.get_loc("high")] = df["low"].iloc[10] - 5.0  # high < low
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = validate_ohlcv(df, "1H", now=now)
    assert not report.is_safe_to_use
    assert any(i.code == "impossible_ohlc" for i in report.issues)


def test_non_positive_price_is_critical() -> None:
    df = _clean_df()
    df.iloc[5, df.columns.get_loc("close")] = -1.0
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = validate_ohlcv(df, "1H", now=now)
    assert not report.is_safe_to_use
    assert any(i.code == "non_positive_price" for i in report.issues)


def test_nan_value_is_critical() -> None:
    df = _clean_df()
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = validate_ohlcv(df, "1H", now=now)
    assert not report.is_safe_to_use
    assert any(i.code == "non_finite_values" for i in report.issues)


def test_stale_data_is_critical() -> None:
    df = _clean_df()
    now = df.index[-1].to_pydatetime() + timedelta(hours=10)  # way past max_stale_candles
    report = validate_ohlcv(df, "1H", now=now)
    assert not report.is_safe_to_use
    assert any(i.code == "stale_data" for i in report.issues)


def test_small_gap_is_warning_not_critical() -> None:
    df = _clean_df(n=100)
    # Drop a single candle in the middle -- one gap out of 100 is well
    # under the default 5% missing-fraction threshold.
    df = df.drop(df.index[50])
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = validate_ohlcv(df, "1H", now=now)
    gap_issues = [i for i in report.issues if i.code == "missing_candles"]
    assert gap_issues
    assert gap_issues[0].severity == "warning"
    assert report.is_safe_to_use


def test_large_missing_fraction_is_critical() -> None:
    df = _clean_df(n=100)
    # Keep only every 3rd candle but preserve the original time span --
    # most of the series is missing.
    sparse = df.iloc[::3]
    now = sparse.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = validate_ohlcv(sparse, "1H", now=now)
    assert not report.is_safe_to_use
    assert any(i.code == "missing_candles" and i.severity == "critical" for i in report.issues)


def test_enforce_quality_gate_raises_on_critical_issue() -> None:
    df = _clean_df()
    df.iloc[5, df.columns.get_loc("close")] = -1.0
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    with pytest.raises(DataQualityError):
        enforce_quality_gate(df, "1H", now=now)


def test_enforce_quality_gate_passes_clean_data() -> None:
    df = _clean_df()
    now = df.index[-1].to_pydatetime() + timedelta(minutes=1)
    report = enforce_quality_gate(df, "1H", now=now)
    assert report.is_safe_to_use
