"""Cross-sectional evaluation of the opportunity scanner's ranking signal.

Phase 2's `scanner_backtest.py` only asked "did the scanner's TOP picks
make money" — it never checked the more fundamental question the Phase 4
audit brief demands: **did higher-ranked assets, as a group, outperform
lower-ranked assets, as a group?** A scanner whose top pick occasionally
does well by chance is not the same as a scanner whose *score* carries
real cross-sectional information. This module builds that check:

  1. Score the ENTIRE candidate universe (not just the top N) at each
     historical scan point, using the exact same point-in-time-sliced
     `scan_candidate()` the production scanner uses — no separate scoring
     logic to accidentally diverge from what's actually shipped.
  2. Compute forward returns at several horizons for every scored asset.
  3. Bucket assets into quintiles by score and compare mean forward
     return per bucket (Section 7 of the audit).
  4. Compute the Information Coefficient — the cross-sectional rank
     correlation between score and forward return — per snapshot, then
     summarise its distribution across snapshots (Section 8), since a
     single pooled correlation across all (snapshot, asset) pairs would
     hide whether the signal is real cross-sectionally at each point in
     time or is being driven entirely by market-wide (systemic) moves
     shared by every asset in a given snapshot.
  5. Measure rank turnover / stability (Section 9).

No-lookahead guarantee: identical to `scanner_backtest.py` — every
snapshot slices every candidate (including the benchmark) to `.iloc[:t]`
before scoring; forward returns are read only after scoring is complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.ranking.liquidity_filter import LiquidityFilterConfig
from src.ranking.recommend import scan_candidate


@dataclass
class CrossSectionalSnapshot:
    scan_index: int
    scan_time: int
    scores: dict[str, float]  # symbol -> final opportunity score; only assets that passed liquidity
    excluded: list[str]  # symbols excluded by the liquidity/data-quality gate at this snapshot


def build_cross_sectional_panel(
    candidates: dict[str, pd.DataFrame],
    benchmark_symbol: str,
    timeframe: str,
    scan_every_candles: int,
    min_train_size: int,
    liquidity_config: LiquidityFilterConfig = LiquidityFilterConfig(),
    max_scans: int | None = None,
) -> list[CrossSectionalSnapshot]:
    """Score every candidate at each of a series of historical scan points.

    Args:
        candidates: ``{symbol: full ascending OHLCV history}``.
        benchmark_symbol: Reference asset for relative-strength scoring;
            must be a key of *candidates*.
        scan_every_candles: Stride between scan points.
        min_train_size: Minimum candles of history before the first scan.
        max_scans: Optional cap on the number of scan points (for
            runtime control on large panels).
    """
    if benchmark_symbol not in candidates:
        raise ValueError(f"benchmark_symbol '{benchmark_symbol}' not in candidates")

    lengths = {s: len(df) for s, df in candidates.items()}
    max_start = min(lengths.values())
    if max_start <= min_train_size:
        raise ValueError("Not enough shared history across candidates for min_train_size.")

    snapshots: list[CrossSectionalSnapshot] = []
    scan_points = list(range(min_train_size, max_start, scan_every_candles))
    if max_scans is not None:
        scan_points = scan_points[:max_scans]

    for t in scan_points:
        point_in_time = {s: df.iloc[:t].copy() for s, df in candidates.items()}
        benchmark_df = point_in_time[benchmark_symbol]

        scores: dict[str, float] = {}
        excluded: list[str] = []
        for symbol, df in point_in_time.items():
            rec, reasons = scan_candidate(symbol, df, benchmark_df, timeframe, liquidity_config)
            if rec is None:
                excluded.append(symbol)
            else:
                scores[symbol] = rec.opportunity_score

        scan_time = int(benchmark_df.index[-1].timestamp())
        snapshots.append(
            CrossSectionalSnapshot(scan_index=t, scan_time=scan_time, scores=scores, excluded=excluded)
        )

    return snapshots


def forward_return(full_df: pd.DataFrame, scan_index: int, horizon_candles: int) -> float | None:
    """Return (close[scan_index-1+horizon] / close[scan_index-1]) - 1, or None if out of range.

    ``scan_index - 1`` is the anchor (the last candle visible to the scan
    at ``scan_index``, matching ``scan_candidate``'s own point-in-time
    slice ``df.iloc[:scan_index]``).
    """
    anchor_idx = scan_index - 1
    future_idx = anchor_idx + horizon_candles
    if anchor_idx < 0 or future_idx >= len(full_df):
        return None
    anchor = float(full_df["close"].iloc[anchor_idx])
    future = float(full_df["close"].iloc[future_idx])
    if anchor == 0:
        return None
    return (future - anchor) / anchor


def information_coefficient(
    scores: dict[str, float], returns: dict[str, float], method: str = "spearman"
) -> float | None:
    """Cross-sectional correlation between score and forward return at one snapshot.

    Returns ``None`` if fewer than 5 assets have both a score and a
    computable forward return (too few for a meaningful correlation).
    """
    common = sorted(set(scores) & set(returns))
    if len(common) < 5:
        return None
    x = [scores[s] for s in common]
    y = [returns[s] for s in common]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None  # degenerate (no variation) -- correlation undefined
    if method == "spearman":
        corr, _ = stats.spearmanr(x, y)
    elif method == "pearson":
        corr, _ = stats.pearsonr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    return float(corr) if np.isfinite(corr) else None


@dataclass
class ICSeriesResult:
    horizon_label: str
    method: str
    values: list[float] = field(default_factory=list)  # one IC per snapshot where computable

    def summary(self) -> dict[str, Any]:
        if not self.values:
            return {
                "horizon": self.horizon_label,
                "method": self.method,
                "n_periods": 0,
                "message": "No periods had enough assets with both a score and a forward return.",
            }
        arr = np.array(self.values)
        return {
            "horizon": self.horizon_label,
            "method": self.method,
            "n_periods": len(arr),
            "mean_ic": float(arr.mean()),
            "median_ic": float(np.median(arr)),
            "std_ic": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "pct_periods_positive": float((arr > 0).mean()),
        }


def compute_ic_series(
    panel: list[CrossSectionalSnapshot],
    candidates: dict[str, pd.DataFrame],
    horizons_candles: dict[str, int],
    method: str = "spearman",
) -> dict[str, ICSeriesResult]:
    """Compute the per-snapshot IC time series for each horizon (Section 8)."""
    results: dict[str, list[float]] = {label: [] for label in horizons_candles}

    for snap in panel:
        for label, h in horizons_candles.items():
            returns = {
                symbol: r
                for symbol in snap.scores
                if (r := forward_return(candidates[symbol], snap.scan_index, h)) is not None
            }
            ic = information_coefficient(snap.scores, returns, method=method)
            if ic is not None:
                results[label].append(ic)

    return {label: ICSeriesResult(horizon_label=label, method=method, values=vals) for label, vals in results.items()}


def quintile_analysis(
    panel: list[CrossSectionalSnapshot],
    candidates: dict[str, pd.DataFrame],
    horizon_candles: int,
    n_buckets: int = 5,
) -> dict[int, list[float]]:
    """Pool forward returns by score-quintile across every snapshot (Section 7).

    Bucket 0 = lowest-scored quintile, ``n_buckets - 1`` = highest-scored.
    Returns ``{bucket_index: [forward_returns...]}``; call ``np.mean`` on
    each list for the per-bucket mean forward return, and look for
    monotonicity across buckets as evidence of real cross-sectional signal.
    """
    buckets: dict[int, list[float]] = {i: [] for i in range(n_buckets)}

    for snap in panel:
        if len(snap.scores) < n_buckets:
            continue
        ranked = sorted(snap.scores.items(), key=lambda kv: kv[1])  # ascending
        n = len(ranked)
        for i, (symbol, _score) in enumerate(ranked):
            bucket = min(int(i * n_buckets / n), n_buckets - 1)
            r = forward_return(candidates[symbol], snap.scan_index, horizon_candles)
            if r is not None:
                buckets[bucket].append(r)

    return buckets


def rank_turnover(panel: list[CrossSectionalSnapshot], top_n: int = 5) -> dict[str, Any]:
    """Measure how much the top-N set changes between consecutive snapshots (Section 9)."""
    turnovers: list[float] = []
    score_changes: list[float] = []
    prev_top: set[str] | None = None
    prev_scores: dict[str, float] | None = None

    for snap in panel:
        if len(snap.scores) < top_n:
            continue
        top = set(sorted(snap.scores, key=lambda s: -snap.scores[s])[:top_n])
        if prev_top is not None:
            changed = len(top - prev_top)
            turnovers.append(changed / top_n)
        if prev_scores is not None:
            common = set(snap.scores) & set(prev_scores)
            if common:
                score_changes.extend(abs(snap.scores[s] - prev_scores[s]) for s in common)
        prev_top = top
        prev_scores = snap.scores

    return {
        "n_transitions": len(turnovers),
        "mean_top_n_turnover": float(np.mean(turnovers)) if turnovers else None,
        "mean_absolute_score_change": float(np.mean(score_changes)) if score_changes else None,
    }
