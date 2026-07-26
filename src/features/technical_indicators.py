"""Technical indicator functions for time-series feature engineering.

Each function accepts a DataFrame and returns an augmented copy with
additional columns.  They are **pure** — no mutation of the input
DataFrame and no I/O side-effects.

Indicators implemented
----------------------
* Simple Moving Average (SMA)
* Exponential Moving Average (EMA)
* Relative Strength Index (RSI)
* MACD (Moving Average Convergence/Divergence)
* Bollinger Bands
* Lag features
* Rolling volatility (std of log returns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------


def add_sma(
    df: pd.DataFrame,
    price_col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Append Simple Moving Average columns for each window.

    Args:
        df:        Input OHLCV DataFrame.
        price_col: Price column to compute SMA on.
        windows:   List of rolling windows in days.

    Returns:
        DataFrame with added ``sma_{w}`` columns.
    """
    if windows is None:
        windows = [7, 14, 30]

    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df[price_col].rolling(window=w, min_periods=1).mean()
    return df


def add_ema(
    df: pd.DataFrame,
    price_col: str = "close",
    spans: list[int] | None = None,
) -> pd.DataFrame:
    """Append Exponential Moving Average columns for each span.

    Args:
        df:        Input OHLCV DataFrame.
        price_col: Price column to compute EMA on.
        spans:     List of EMA span values in days.

    Returns:
        DataFrame with added ``ema_{span}`` columns.
    """
    if spans is None:
        spans = [12, 26]

    df = df.copy()
    for span in spans:
        df[f"ema_{span}"] = df[price_col].ewm(span=span, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def add_rsi(
    df: pd.DataFrame,
    price_col: str = "close",
    period: int = 14,
) -> pd.DataFrame:
    """Append the Relative Strength Index (RSI) column.

    Uses Wilder's smoothing method (equivalent to EMA with alpha=1/period).

    Args:
        df:        Input OHLCV DataFrame.
        price_col: Price column to compute RSI on.
        period:    Look-back period in days.

    Returns:
        DataFrame with added ``rsi_{period}`` column (values in [0, 100]).
    """
    df = df.copy()
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # When avg_loss is 0 (pure uptrend), RSI is 100 by definition.
    # When avg_gain is 0 (pure downtrend), RSI is 0 by definition.
    # Using np.where avoids division-by-zero producing NaN.
    rsi = np.where(
        avg_loss == 0,
        100.0,
        np.where(
            avg_gain == 0,
            0.0,
            100.0 - (100.0 / (1.0 + avg_gain / avg_loss)),
        ),
    )
    df[f"rsi_{period}"] = pd.array(rsi, dtype="float64")
    return df


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def add_macd(
    df: pd.DataFrame,
    price_col: str = "close",
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> pd.DataFrame:
    """Append MACD line, signal line, and histogram columns.

    Args:
        df:          Input OHLCV DataFrame.
        price_col:   Price column to compute MACD on.
        fast_span:   Fast EMA span.
        slow_span:   Slow EMA span.
        signal_span: Signal line EMA span.

    Returns:
        DataFrame with added ``macd``, ``macd_signal``, and
        ``macd_hist`` columns.
    """
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast_span, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow_span, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal_span, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


def add_bollinger_bands(
    df: pd.DataFrame,
    price_col: str = "close",
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Append Bollinger Band columns (upper, middle/SMA, lower, bandwidth).

    Args:
        df:        Input OHLCV DataFrame.
        price_col: Price column to compute bands on.
        window:    Rolling window size.
        num_std:   Number of standard deviations for band width.

    Returns:
        DataFrame with added ``bb_upper``, ``bb_mid``, ``bb_lower``,
        and ``bb_width`` columns.
    """
    df = df.copy()
    rolling = df[price_col].rolling(window=window, min_periods=1)
    mid = rolling.mean()
    std = rolling.std()

    df["bb_mid"] = mid
    df["bb_upper"] = mid + num_std * std
    df["bb_lower"] = mid - num_std * std
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def add_lag_features(
    df: pd.DataFrame,
    price_col: str = "close",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Append lagged price columns (``close_lag_{n}``).

    These give the model explicit access to recent price history without
    requiring a sequence-based architecture.

    Args:
        df:        Input OHLCV DataFrame (sorted by date ascending).
        price_col: Price column to lag.
        lags:      List of lag offsets in days.

    Returns:
        DataFrame with added ``{price_col}_lag_{n}`` columns.
    """
    if lags is None:
        lags = [1, 2, 3, 7, 14]

    df = df.copy()
    for lag in lags:
        df[f"{price_col}_lag_{lag}"] = df[price_col].shift(lag)
    return df


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------


def add_rolling_volatility(
    df: pd.DataFrame,
    log_return_col: str = "log_return",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Append rolling standard deviation of log returns (volatility proxy).

    Args:
        df:             Input DataFrame containing *log_return_col*.
        log_return_col: Name of the log-return column.
        windows:        List of rolling windows in days.

    Returns:
        DataFrame with added ``volatility_{w}`` columns.
    """
    if windows is None:
        windows = [7, 14]

    df = df.copy()
    if log_return_col not in df.columns:
        # Silently skip — log returns may not have been added yet.
        return df

    for w in windows:
        df[f"volatility_{w}"] = (
            df[log_return_col].rolling(window=w, min_periods=1).std()
        )
    return df
