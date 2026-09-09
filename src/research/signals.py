"""Independent alpha-signal family definitions (Phase 5, Section 2).

Every signal here is a **pure, vectorised, causal transform** of an OHLCV
series: it is built entirely from ``pandas`` rolling/shift/``pct_change``
operations, each of which by construction only reads rows at or before
the current one. There is no ``scan_candidate`` call, no
``FeaturePipeline`` fit-on-whole-series step, and no coupling of any kind
to ``src/ranking/`` — this is the "clean research module that can
evaluate independent signal families without coupling them to the
production scanner" the audit brief requires.

Design choice — whole-series vectorisation instead of point-in-time
slicing: Phase 4's ``cross_sectional_analysis.py`` had to re-slice the
dataframe to ``.iloc[:t]`` and rebuild the full scanner at every scan
point, because the production scanner's ``FeaturePipeline`` step is not
guaranteed causal by inspection alone (it had to be *proven* causal by a
"corrupt the future" test). Every function below is causal by
construction (a rolling/shift/pct_change operation at row ``i`` cannot
read row ``i+1``), so the signal series for an entire asset can be
computed once and then indexed at any scan point — this is what makes
testing dozens of signals across eight horizons on multiple universes
tractable within a bounded experiment budget (Section 5 / Section 15).
The "corrupt the future" regression test in
``tests/test_research_signals.py`` still exists and is still mandatory:
"causal by construction" is an engineering argument, not a substitute for
the empirical proof the audit requires.

Every function takes and returns ``pd.Series``/``pd.DataFrame`` aligned
to the input's index. A value at position ``i`` uses only data at
positions ``<= i``. Insufficient history produces ``NaN``, never a
zero-filled or interpolated value (which would be a subtle look-ahead —
see ``docs/PHASE5_ALPHA_RESEARCH_REPORT.md`` Section 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Candle-count lookback per named horizon, assuming 1-hour candles (the
# granularity Phase 5's synthetic universes use throughout — see
# src/research/universes.py). A different candle granularity would need a
# different mapping; this one is not reused from src/utils/timeframes.py
# because that module maps *timeframe strings* to seconds, not *research
# horizons* to lookback-candle counts for a fixed granularity.
HORIZON_CANDLES: dict[str, int] = {
    "1H": 1,
    "4H": 4,
    "12H": 12,
    "1D": 24,
    "3D": 72,
    "7D": 168,
    "14D": 336,
    "30D": 720,
}


# ---------------------------------------------------------------------------
# A. Momentum
# ---------------------------------------------------------------------------

def momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Trailing simple return over *lookback* candles: close[t]/close[t-lookback] - 1."""
    return close.pct_change(lookback)


def multi_horizon_momentum(close: pd.Series, lookbacks: list[int]) -> pd.Series:
    """Equal-weight average of momentum computed at several lookbacks.

    Tests whether *combining* horizons (vs. any single one) carries more
    signal — Section 2H requires measuring this, not assuming it.
    """
    parts = [momentum(close, lb) for lb in lookbacks]
    return pd.concat(parts, axis=1).mean(axis=1)


def multi_timeframe_confirmation(close: pd.Series, lookbacks: list[int]) -> pd.Series:
    """Reward agreement across timeframes; zero out disagreement.

    ``mean(momentums)`` when every lookback's momentum has the same sign,
    else 0 — a direct, measurable test of "do short/medium/long agreeing
    make the signal more reliable" rather than a plain average that
    can't distinguish agreement from cancellation.
    """
    parts = pd.concat([momentum(close, lb) for lb in lookbacks], axis=1)
    same_sign = (parts.gt(0).all(axis=1)) | (parts.lt(0).all(axis=1))
    combined = parts.mean(axis=1)
    return combined.where(same_sign, 0.0)


# ---------------------------------------------------------------------------
# B. Mean reversion
# ---------------------------------------------------------------------------

def distance_from_ma(close: pd.Series, window: int) -> pd.Series:
    """(close - trailing SMA) / trailing SMA. Positive = above its own average."""
    ma = close.rolling(window).mean()
    return (close - ma) / ma


def return_zscore(close: pd.Series, window: int) -> pd.Series:
    """Z-score of the latest 1-candle return against its own trailing distribution."""
    r = close.pct_change()
    mean = r.rolling(window).mean()
    std = r.rolling(window).std()
    return (r - mean) / std.replace(0.0, np.nan)


def bollinger_deviation(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Position within Bollinger bands, in units of num_std: (close-mid)/(num_std*std)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mid) / (num_std * std.replace(0.0, np.nan))


def short_term_reversal(close: pd.Series, lookback: int = 1) -> pd.Series:
    """Negative of very-short-lookback momentum: a large recent move is scored as 'due to revert'."""
    return -momentum(close, lookback)


def medium_term_reversal(close: pd.Series, lookback: int = 14) -> pd.Series:
    """Same construction as short_term_reversal, at a medium lookback."""
    return -momentum(close, lookback)


# ---------------------------------------------------------------------------
# C. Volume / flow
# ---------------------------------------------------------------------------

def volume_acceleration(volume: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """Ratio of trailing short-window avg volume to trailing long-window avg volume, minus 1."""
    short_avg = volume.rolling(short).mean()
    long_avg = volume.rolling(long).mean()
    return short_avg / long_avg.replace(0.0, np.nan) - 1.0


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Latest volume relative to its own trailing rolling mean."""
    baseline = volume.rolling(window).mean()
    return volume / baseline.replace(0.0, np.nan)


def price_volume_confirmation(close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """Rolling correlation between returns and volume changes.

    Positive = up-moves tend to come with rising volume (a textbook
    "confirmed" trend); negative = moves happening on fading volume.
    """
    r = close.pct_change()
    dv = volume.pct_change()
    return r.rolling(window).corr(dv)


def abnormal_volume(volume: pd.Series, window: int = 30) -> pd.Series:
    """Z-score of the latest volume against its own trailing distribution."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0.0, np.nan)


def volume_trend(volume: pd.Series, short: int = 7, long: int = 30) -> pd.Series:
    """EMA-ratio proxy for the slope of volume: short EMA / long EMA - 1."""
    short_ema = volume.ewm(span=short, adjust=False).mean()
    long_ema = volume.ewm(span=long, adjust=False).mean()
    return short_ema / long_ema.replace(0.0, np.nan) - 1.0


# ---------------------------------------------------------------------------
# D. Volatility
# ---------------------------------------------------------------------------

def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of log returns over *window* candles."""
    log_r = np.log(close / close.shift(1))
    return log_r.rolling(window).std()


def volatility_ratio(close: pd.Series, short: int = 7, long: int = 30) -> pd.Series:
    """Short-window realized vol / long-window realized vol.

    ``volatility_expansion = ratio - 1`` (positive when vol is expanding);
    ``volatility_contraction = 1 - ratio`` (positive when vol is
    contracting) are the two signed views of this same ratio requested
    separately in Section 2D — both are exact mirror images of each
    other, so both are derived from this one function rather than
    duplicating the rolling-std computation twice.
    """
    short_vol = realized_volatility(close, short)
    long_vol = realized_volatility(close, long)
    return short_vol / long_vol.replace(0.0, np.nan)


def volatility_expansion(close: pd.Series, short: int = 7, long: int = 30) -> pd.Series:
    return volatility_ratio(close, short, long) - 1.0


def volatility_contraction(close: pd.Series, short: int = 7, long: int = 30) -> pd.Series:
    return 1.0 - volatility_ratio(close, short, long)


def volatility_adjusted_momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Momentum scaled by its own trailing volatility -- a Sharpe-like signal."""
    mom = momentum(close, lookback)
    vol = realized_volatility(close, lookback).replace(0.0, np.nan)
    return mom / vol


def volatility_regime_percentile(close: pd.Series, window: int = 14, history: int = 90) -> pd.Series:
    """Percentile rank (0-1) of the current realized-vol reading against its own trailing history.

    High values = "currently in a high-volatility regime relative to
    itself"; used both as a standalone signal and as a regime-conditioning
    variable in the stability breakdown.
    """
    vol = realized_volatility(close, window)
    return vol.rolling(history).rank(pct=True)


# ---------------------------------------------------------------------------
# E. Relative strength
# ---------------------------------------------------------------------------

def relative_strength(close: pd.Series, benchmark_close: pd.Series, lookback: int) -> pd.Series:
    """Asset momentum minus benchmark momentum over the same lookback.

    *benchmark_close* must already be aligned to *close*'s index (same
    timestamps) — callers (the walk-forward harness) are responsible for
    that alignment; this function does not reindex, precisely so a silent
    misalignment cannot be papered over here.
    """
    return momentum(close, lookback) - momentum(benchmark_close, lookback)


def cross_sectional_relative_strength(
    asset_return: float, universe_returns: dict[str, float]
) -> float | None:
    """One asset's trailing return minus the equal-weight universe's trailing return, at one point in time.

    Unlike the other functions in this module, this is evaluated by the
    walk-forward harness at each scan point (it needs the whole universe's
    contemporaneous returns, which a single-asset series cannot see) —
    see ``src/research/walk_forward.py::cross_sectional_relative_strength_panel``.
    """
    if not universe_returns:
        return None
    mean_return = float(np.mean(list(universe_returns.values())))
    return asset_return - mean_return


# ---------------------------------------------------------------------------
# Registry: uniform (df, benchmark_close, lookback) -> pd.Series wrappers
# ---------------------------------------------------------------------------

# Every wrapper takes the SAME signature so the experiment runner can loop
# over (signal, horizon) pairs with no per-signal special-casing. *df* must
# have "close" and "volume" columns. A signal marked
# ``horizon_parameterized=False`` ignores *lookback* (it uses its own fixed
# multi-window construction internally) and is run once, not swept.


def _wrap_momentum(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return momentum(df["close"], lookback)


def _wrap_multi_horizon_momentum(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return multi_horizon_momentum(df["close"], [4, 24, 168])


def _wrap_multi_timeframe_confirmation(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return multi_timeframe_confirmation(df["close"], [4, 24, 168])


def _wrap_distance_from_ma(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return distance_from_ma(df["close"], window=max(lookback, 2))


def _wrap_return_zscore(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return return_zscore(df["close"], window=max(lookback, 5))


def _wrap_bollinger_deviation(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return bollinger_deviation(df["close"], window=max(lookback, 5))


def _wrap_short_term_reversal(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return short_term_reversal(df["close"], lookback=lookback)


def _wrap_medium_term_reversal(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return medium_term_reversal(df["close"], lookback=lookback)


def _wrap_volume_acceleration(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return volume_acceleration(df["volume"], short=max(lookback // 4, 2), long=max(lookback, 8))


def _wrap_relative_volume(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return relative_volume(df["volume"], window=max(lookback, 5))


def _wrap_price_volume_confirmation(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return price_volume_confirmation(df["close"], df["volume"], window=max(lookback, 3))


def _wrap_abnormal_volume(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return abnormal_volume(df["volume"], window=max(lookback, 5))


def _wrap_volume_trend(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return volume_trend(df["volume"], short=max(lookback // 4, 2), long=max(lookback, 8))


def _wrap_realized_volatility(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return realized_volatility(df["close"], window=max(lookback, 2))


def _wrap_volatility_expansion(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return volatility_expansion(df["close"], short=max(lookback // 4, 2), long=max(lookback, 8))


def _wrap_volatility_contraction(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return volatility_contraction(df["close"], short=max(lookback // 4, 2), long=max(lookback, 8))


def _wrap_volatility_adjusted_momentum(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    return volatility_adjusted_momentum(df["close"], lookback=lookback)


def _wrap_volatility_regime_percentile(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    # `history` is deliberately a fixed constant, not scaled by `lookback`:
    # scaling it (e.g. lookback * 6) would push the 30D-horizon variant's
    # warmup past 4000+ candles, starving it of usable history within a
    # bounded experiment window for no analytical benefit -- "how far back
    # to compare the current vol regime against" is a separate design
    # choice from "how long a window realized vol itself is measured over".
    return volatility_regime_percentile(df["close"], window=max(lookback, 2), history=500)


def _wrap_relative_strength(df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
    if benchmark_close is None:
        return pd.Series(np.nan, index=df.index)
    return relative_strength(df["close"], benchmark_close, lookback)


@dataclass(frozen=True)
class SignalSpec:
    family: str
    name: str
    fn: Callable[[pd.DataFrame, pd.Series | None, int], pd.Series]
    horizon_parameterized: bool

    def series(self, df: pd.DataFrame, benchmark_close: pd.Series | None, lookback: int) -> pd.Series:
        return self.fn(df, benchmark_close, lookback)


SIGNAL_REGISTRY: list[SignalSpec] = [
    SignalSpec("momentum", "momentum", _wrap_momentum, True),
    SignalSpec("momentum", "multi_horizon_momentum", _wrap_multi_horizon_momentum, False),
    SignalSpec("momentum", "multi_timeframe_confirmation", _wrap_multi_timeframe_confirmation, False),
    SignalSpec("mean_reversion", "distance_from_ma", _wrap_distance_from_ma, True),
    SignalSpec("mean_reversion", "return_zscore", _wrap_return_zscore, True),
    SignalSpec("mean_reversion", "bollinger_deviation", _wrap_bollinger_deviation, True),
    SignalSpec("mean_reversion", "short_term_reversal", _wrap_short_term_reversal, True),
    SignalSpec("mean_reversion", "medium_term_reversal", _wrap_medium_term_reversal, True),
    SignalSpec("volume", "volume_acceleration", _wrap_volume_acceleration, True),
    SignalSpec("volume", "relative_volume", _wrap_relative_volume, True),
    SignalSpec("volume", "price_volume_confirmation", _wrap_price_volume_confirmation, True),
    SignalSpec("volume", "abnormal_volume", _wrap_abnormal_volume, True),
    SignalSpec("volume", "volume_trend", _wrap_volume_trend, True),
    SignalSpec("volatility", "realized_volatility", _wrap_realized_volatility, True),
    SignalSpec("volatility", "volatility_expansion", _wrap_volatility_expansion, True),
    SignalSpec("volatility", "volatility_contraction", _wrap_volatility_contraction, True),
    SignalSpec("volatility", "volatility_adjusted_momentum", _wrap_volatility_adjusted_momentum, True),
    SignalSpec("volatility", "volatility_regime_percentile", _wrap_volatility_regime_percentile, True),
    SignalSpec("relative_strength", "relative_strength", _wrap_relative_strength, True),
]
