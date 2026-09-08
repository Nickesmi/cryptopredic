"""Statistical significance helpers for comparing forecasting experiments.

Phase 3, Section 13 of the audit: "Do not interpret a 1-2% improvement as
meaningful automatically." This module provides the minimum machinery
needed to say whether an observed difference between two experiments (or
between one experiment and chance) is distinguishable from noise, given
how few walk-forward predictions any single experiment typically produces
(tens to a few hundred, per Phase 2/3's runs) — nowhere near enough to
trust a raw point estimate at face value.

Deliberately simple (paired bootstrap, not a full econometric test suite):
these experiments have autocorrelated, non-iid errors (overlapping
horizons, shared regimes), so a naive t-test's independence assumption is
already shaky; a bootstrap over the *same* prediction records at least
avoids compounding that with a false precision from a parametric formula
it also doesn't satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n: int
    n_boot: int

    def excludes(self, value: float) -> bool:
        """True if *value* falls outside the confidence interval."""
        return not (self.ci_low <= value <= self.ci_high)

    def to_dict(self) -> dict:
        return {
            "point_estimate": self.point_estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "n_boot": self.n_boot,
        }


def bootstrap_statistic(
    values: Sequence[float],
    statistic_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap a confidence interval for ``statistic_fn(values)``."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 5:
        raise ValueError("Need at least 5 observations for a meaningful bootstrap.")

    rng = np.random.default_rng(seed)
    point = float(statistic_fn(arr))
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boot_stats[b] = statistic_fn(sample)

    alpha = (1 - ci) / 2
    low, high = np.quantile(boot_stats, [alpha, 1 - alpha])
    return BootstrapResult(point_estimate=point, ci_low=float(low), ci_high=float(high), n=n, n_boot=n_boot)


def paired_bootstrap_diff(
    values_a: Sequence[float],
    values_b: Sequence[float],
    statistic_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap a CI for ``statistic_fn(a) - statistic_fn(b)`` over matched pairs.

    *values_a* and *values_b* must be the same length and index-aligned
    (e.g. per-prediction errors from two experiments evaluated at the
    exact same walk-forward test points) — this is what makes it a
    *paired* comparison, which is far more powerful than comparing two
    independent bootstraps when both experiments share the same
    underlying market moves.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("values_a and values_b must be the same length (paired).")
    n = len(a)
    if n < 5:
        raise ValueError("Need at least 5 paired observations for a meaningful bootstrap.")

    rng = np.random.default_rng(seed)
    point = float(statistic_fn(a) - statistic_fn(b))
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = statistic_fn(a[idx]) - statistic_fn(b[idx])

    alpha = (1 - ci) / 2
    low, high = np.quantile(boot_diffs, [alpha, 1 - alpha])
    return BootstrapResult(point_estimate=point, ci_low=float(low), ci_high=float(high), n=n, n_boot=n_boot)


def auc_confidence_interval(
    labels: Sequence[float], probabilities: Sequence[float], n_boot: int = 2000, seed: int = 0
) -> BootstrapResult:
    """Bootstrap CI for ROC-AUC, to check whether it's distinguishable from 0.5 (chance)."""
    from sklearn.metrics import roc_auc_score

    labels_arr = np.asarray(labels, dtype=float)
    probs_arr = np.asarray(probabilities, dtype=float)
    n = len(labels_arr)
    if n < 10 or len(np.unique(labels_arr)) < 2:
        raise ValueError("Need at least 10 observations with both classes present.")

    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(labels_arr, probs_arr))
    boot_aucs = []
    attempts = 0
    while len(boot_aucs) < n_boot and attempts < n_boot * 3:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels_arr[idx])) < 2:
            continue  # degenerate resample -- AUC undefined, skip and retry
        boot_aucs.append(roc_auc_score(labels_arr[idx], probs_arr[idx]))

    boot_aucs = np.array(boot_aucs)
    low, high = np.quantile(boot_aucs, [0.025, 0.975])
    return BootstrapResult(point_estimate=point, ci_low=float(low), ci_high=float(high), n=n, n_boot=len(boot_aucs))
