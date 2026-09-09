"""Statistical testing suite for alpha signals (Phase 5, Section 7).

Builds on ``src/evaluation/significance.py`` (paired/independent
bootstrap) rather than duplicating it, and adds the pieces that module
didn't need for forecasting-error comparisons: an IC-specific summary
(information ratio, hit rate, CI), a cross-sectional permutation/null
test, and Benjamini-Hochberg multiple-testing correction — mandatory here
because Phase 5 runs many (signal x horizon x universe) experiments and a
single "p < 0.05" without correction would be exactly the data-mining
trap Section 15 warns against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from src.evaluation.significance import BootstrapResult, bootstrap_statistic


@dataclass
class ICStats:
    """Summary statistics for one IC time series (one signal x horizon x universe cell)."""

    n_periods: int
    mean_ic: float | None
    median_ic: float | None
    std_ic: float | None
    ic_information_ratio: float | None  # mean / std, the IC analogue of a Sharpe ratio
    ci_low: float | None
    ci_high: float | None
    hit_rate: float | None  # fraction of periods with IC matching the sign of the mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_periods": self.n_periods,
            "mean_ic": self.mean_ic,
            "median_ic": self.median_ic,
            "std_ic": self.std_ic,
            "ic_information_ratio": self.ic_information_ratio,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "hit_rate": self.hit_rate,
        }


def compute_ic_stats(ic_values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> ICStats:
    """Mean/median/std/IR/CI/hit-rate for a series of per-snapshot IC values."""
    arr = np.asarray([v for v in ic_values if np.isfinite(v)], dtype=float)
    n = len(arr)
    if n < 5:
        return ICStats(n, None, None, None, None, None, None, None)

    mean_ic = float(arr.mean())
    median_ic = float(np.median(arr))
    std_ic = float(arr.std(ddof=1)) if n > 1 else 0.0
    ir = (mean_ic / std_ic) if std_ic > 0 else None
    sign = np.sign(mean_ic) if mean_ic != 0 else 1.0
    hit_rate = float((np.sign(arr) == sign).mean())

    boot = bootstrap_statistic(arr, np.mean, n_boot=n_boot, seed=seed)
    return ICStats(n, mean_ic, median_ic, std_ic, ir, boot.ci_low, boot.ci_high, hit_rate)


@dataclass
class QuantileResult:
    n_buckets: int
    bucket_means: list[float | None]
    top_minus_bottom: float | None
    monotonic: bool | None  # strictly non-decreasing (or non-increasing) from bottom to top bucket
    monotonicity_spearman: float | None  # Spearman corr between bucket index and bucket mean return

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_buckets": self.n_buckets,
            "bucket_means": self.bucket_means,
            "top_minus_bottom": self.top_minus_bottom,
            "monotonic": self.monotonic,
            "monotonicity_spearman": self.monotonicity_spearman,
        }


def analyze_quantiles(buckets: dict[int, list[float]]) -> QuantileResult:
    """Section 7: top-minus-bottom spread and monotonicity from a bucket -> [returns] map."""
    n_buckets = len(buckets)
    means: list[float | None] = []
    for i in range(n_buckets):
        vals = buckets.get(i, [])
        means.append(float(np.mean(vals)) if vals else None)

    valid = [(i, m) for i, m in enumerate(means) if m is not None]
    if len(valid) < 2:
        return QuantileResult(n_buckets, means, None, None, None)

    top_minus_bottom = valid[-1][1] - valid[0][1]

    idxs = [i for i, _ in valid]
    vals = [m for _, m in valid]
    diffs = np.diff(vals)
    monotonic = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    from scipy import stats as _stats
    if len(set(vals)) > 1:
        rho, _ = _stats.spearmanr(idxs, vals)
        monotonicity_spearman = float(rho) if np.isfinite(rho) else None
    else:
        monotonicity_spearman = None

    return QuantileResult(n_buckets, means, float(top_minus_bottom), monotonic, monotonicity_spearman)


def permutation_test_ic(
    scores_by_snapshot: list[dict[str, float]],
    returns_by_snapshot: list[dict[str, float]],
    observed_mean_ic: float,
    method: str = "spearman",
    n_perm: int = 500,
    seed: int = 0,
) -> float:
    """Null-distribution significance test for a mean IC (Section 7 / Section 14).

    At each permutation, forward returns are reshuffled *within each
    snapshot* (breaking the score-return pairing while preserving each
    snapshot's own cross-sectional return distribution and each signal's
    own cross-sectional score distribution), the mean IC across snapshots
    is recomputed, and a two-sided p-value is the fraction of permuted
    |mean IC| at least as large as the observed |mean IC|.

    Shuffling within each snapshot (rather than pooling all snapshots
    together) is deliberate: it destroys exactly the relationship being
    tested (does this signal's score predict this asset's forward return
    at this point in time) while leaving every other structural property
    of the data (autocorrelation across time, cross-sectional dispersion
    at each snapshot) untouched, so the null distribution isn't
    artificially narrowed or widened by a mismatched shuffle unit.
    """
    from src.evaluation.cross_sectional_analysis import information_coefficient

    rng = np.random.default_rng(seed)
    permuted_means: list[float] = []

    for _ in range(n_perm):
        perm_ics = []
        for scores, returns in zip(scores_by_snapshot, returns_by_snapshot):
            common = sorted(set(scores) & set(returns))
            if len(common) < 5:
                continue
            shuffled_symbols = list(common)
            rng.shuffle(shuffled_symbols)
            shuffled_returns = {sym: returns[orig] for sym, orig in zip(common, shuffled_symbols)}
            ic = information_coefficient(
                {s: scores[s] for s in common}, shuffled_returns, method=method
            )
            if ic is not None:
                perm_ics.append(ic)
        if perm_ics:
            permuted_means.append(float(np.mean(perm_ics)))

    if not permuted_means:
        return 1.0

    permuted_arr = np.abs(np.array(permuted_means))
    p_value = float((permuted_arr >= abs(observed_mean_ic)).mean())
    return p_value


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns a same-length list of significance flags.

    Standard step-up procedure: sort p-values ascending, find the largest
    rank ``k`` where ``p_(k) <= (k/m) * alpha``, and reject (flag
    significant) every hypothesis with a p-value at or below
    ``p_(k)``. With zero tested hypotheses, or if no p-value satisfies
    the criterion, nothing is flagged significant.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    thresholds = (np.arange(1, m + 1) / m) * alpha
    below = sorted_p <= thresholds
    if not below.any():
        return [False] * m
    max_k = np.max(np.where(below)[0])  # 0-indexed largest k satisfying the criterion
    cutoff_p = sorted_p[max_k]
    return [p <= cutoff_p for p in p_values]
