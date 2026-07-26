"""OHLCV data preprocessing — cleaning and return computation.

All functions are pure transformations of DataFrames with no I/O side
effects, making them trivially testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitise a raw OHLCV DataFrame.

    Steps performed:
      1. Sort ascending by date index.
      2. Forward-fill then backward-fill any missing OHLCV values.
      3. Drop any remaining rows that are still fully NaN.
      4. Remove duplicate index entries (keep last occurrence).
      5. Ensure all numeric columns are ``float64``.

    Args:
        df: Raw OHLCV DataFrame with columns ``[open, high, low, close, volume]``.

    Returns:
        Cleaned DataFrame with the same schema.

    Raises:
        ValueError: If the DataFrame is empty after cleaning.
    """
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    ohlcv_cols = [
        c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
    ]
    df[ohlcv_cols] = df[ohlcv_cols].ffill().bfill()

    df = df.dropna(subset=ohlcv_cols)
    df[ohlcv_cols] = df[ohlcv_cols].astype(np.float64)

    if df.empty:
        raise ValueError("DataFrame is empty after cleaning — cannot proceed.")

    return df


def add_log_returns(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Append a ``log_return`` column computed from *price_col*.

    ``log_return[t] = ln(close[t] / close[t-1])``

    Args:
        df:         DataFrame containing *price_col*.
        price_col:  Name of the price column to compute returns from.

    Returns:
        DataFrame with an additional ``log_return`` column.
        The first row will contain ``NaN`` (no prior price).

    Raises:
        KeyError: If *price_col* is not in *df*.
    """
    if price_col not in df.columns:
        raise KeyError(f"Column '{price_col}' not found in DataFrame.")

    df = df.copy()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df
