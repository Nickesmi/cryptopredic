"""Empirical time-to-target estimation.

This is the module the timing audit calls for in Phase 10: instead of a
single deterministic claim ("$100,000 in 4 hours"), estimate — from
actual historical price action — how long a given percentage move has
typically taken to arrive, and report it as a probability-of-arrival
curve over several horizons rather than one falsely-precise number.

Definitions
-----------
For a starting index ``i`` and a target return ``r`` (e.g. ``+0.02`` for
+2%), the **first-passage time** is the smallest ``k >= 1`` such that::

    close[i + k] / close[i] - 1 >= r     (r > 0, "reaches at least +r")
    close[i + k] / close[i] - 1 <= r     (r < 0, "reaches at most r")

within a maximum look-ahead window. If the move never happens within
that window the observation is **censored** (right-censored) — we know
only that it took longer than the window, not exactly how long. Censored
observations are excluded from the median/mean time-to-target (which
would otherwise be biased downward) but are counted in the denominator
of the hit-rate / P(reach within H) figures.

This module deliberately works off simple close-price arrays — it has
no dependency on the forecasting model — so it can be used to sanity
check *any* horizon claim ("BTC will reach $X within 4 hours") against
what similarly-sized historical moves actually took.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.timeframes import timeframe_to_seconds

# Canonical horizon buckets used across the timing-audit deliverables
# (Phase 8/10/11): short/medium/long term, expressed in seconds so they
# are timeframe-independent.
DEFAULT_HORIZON_BUCKETS_SECONDS: tuple[int, ...] = (
    3600,     # 1h
    4 * 3600,  # 4h
    12 * 3600,  # 12h
    86400,    # 1d
    3 * 86400,  # 3d
    7 * 86400,  # 7d
)


@dataclass
class TimeToTargetReport:
    target_return: float
    max_candles: int
    interval_seconds: int
    n_observations: int
    n_hit: int
    hit_rate: float  # fraction of observations that ever reached the target within max_candles
    median_candles_to_hit: float | None
    mean_candles_to_hit: float | None
    probability_within: dict[int, float] = field(default_factory=dict)  # horizon_seconds -> P(reach by then)

    def to_dict(self) -> dict:
        return {
            "target_return": self.target_return,
            "max_candles": self.max_candles,
            "interval_seconds": self.interval_seconds,
            "n_observations": self.n_observations,
            "n_hit": self.n_hit,
            "hit_rate": self.hit_rate,
            "median_time_to_target_seconds": (
                None
                if self.median_candles_to_hit is None
                else self.median_candles_to_hit * self.interval_seconds
            ),
            "mean_time_to_target_seconds": (
                None
                if self.mean_candles_to_hit is None
                else self.mean_candles_to_hit * self.interval_seconds
            ),
            "probability_within": {
                f"{seconds}s": prob for seconds, prob in self.probability_within.items()
            },
        }


def first_passage_candles(
    closes: Sequence[float], target_return: float, max_candles: int
) -> np.ndarray:
    """Candles-to-first-touch of *target_return* for every starting index.

    Returns an array of the same length as *closes* (minus the trailing
    ``max_candles`` positions, which don't have a full look-ahead window)
    where each entry is the number of candles until the target return was
    first reached, or ``np.nan`` if it was never reached within
    ``max_candles`` (censored).
    """
    arr = np.asarray(closes, dtype=float)
    n = len(arr)
    results = np.full(max(0, n - max_candles), np.nan)

    for i in range(len(results)):
        base = arr[i]
        if base == 0 or not np.isfinite(base):
            continue
        window = arr[i + 1 : i + 1 + max_candles]
        returns = window / base - 1.0
        if target_return >= 0:
            hits = np.where(returns >= target_return)[0]
        else:
            hits = np.where(returns <= target_return)[0]
        if hits.size:
            results[i] = hits[0] + 1  # +1 because window starts at i+1

    return results


def time_to_target_report(
    df: pd.DataFrame,
    timeframe: str,
    target_return: float,
    max_candles: int,
    horizon_buckets_seconds: Sequence[int] = DEFAULT_HORIZON_BUCKETS_SECONDS,
    price_col: str = "close",
) -> TimeToTargetReport:
    """Build an empirical time-to-target report from historical *df*.

    Args:
        df:                Chronologically ordered OHLCV history.
        timeframe:         Candle period string of *df* (used only to
                           convert candle counts to real time).
        target_return:     Target return, e.g. ``0.02`` for +2%, ``-0.05``
                           for -5%.
        max_candles:       Maximum look-ahead window, in candles, before an
                           observation is considered censored.
        horizon_buckets_seconds: Horizons (in seconds) to report
                           P(reach within horizon) for.

    Returns:
        A :class:`TimeToTargetReport`.
    """
    interval_seconds = timeframe_to_seconds(timeframe)
    passage = first_passage_candles(df[price_col].tolist(), target_return, max_candles)
    n_observations = len(passage)
    if n_observations == 0:
        raise ValueError("Not enough history to compute first-passage times.")

    hit_mask = ~np.isnan(passage)
    n_hit = int(hit_mask.sum())
    hit_rate = n_hit / n_observations

    hit_candles = passage[hit_mask]
    median_candles = float(np.median(hit_candles)) if n_hit else None
    mean_candles = float(np.mean(hit_candles)) if n_hit else None

    probability_within: dict[int, float] = {}
    for bucket_seconds in horizon_buckets_seconds:
        bucket_candles = bucket_seconds / interval_seconds
        # P(reach within this horizon) = fraction of ALL observations
        # (including those that eventually got censored at max_candles)
        # that hit within bucket_candles — censoring beyond the bucket
        # doesn't affect this estimate as long as bucket <= max_candles.
        within = hit_mask & (passage <= bucket_candles)
        probability_within[bucket_seconds] = float(within.sum()) / n_observations

    return TimeToTargetReport(
        target_return=target_return,
        max_candles=max_candles,
        interval_seconds=interval_seconds,
        n_observations=n_observations,
        n_hit=n_hit,
        hit_rate=hit_rate,
        median_candles_to_hit=median_candles,
        mean_candles_to_hit=mean_candles,
        probability_within=probability_within,
    )


def most_likely_horizon(report: TimeToTargetReport) -> int | None:
    """Return the smallest horizon (seconds) at which P(reach within) >= 0.5.

    ``None`` if even the largest configured horizon has < 50% probability
    of reaching the target — i.e. the honest answer is "more likely than
    not, this target will not be reached within any horizon we checked".
    """
    for seconds, prob in sorted(report.probability_within.items()):
        if prob >= 0.5:
            return seconds
    return None
