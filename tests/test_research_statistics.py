"""Tests for src/research/statistics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.statistics import (
    analyze_quantiles,
    benjamini_hochberg,
    compute_ic_stats,
    permutation_test_ic,
)


def test_compute_ic_stats_basic() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.1, 0.05, 200)
    stats = compute_ic_stats(values)
    assert stats.n_periods == 200
    assert stats.mean_ic == pytest.approx(float(values.mean()), abs=1e-9)
    assert stats.ic_information_ratio is not None
    assert stats.ci_low < stats.mean_ic < stats.ci_high
    assert 0.0 <= stats.hit_rate <= 1.0


def test_compute_ic_stats_too_few_observations_returns_nones() -> None:
    stats = compute_ic_stats([0.1, 0.2])
    assert stats.n_periods == 2
    assert stats.mean_ic is None


def test_compute_ic_stats_filters_non_finite() -> None:
    values = [0.1, 0.2, float("nan"), 0.15, 0.05, 0.12]
    stats = compute_ic_stats(values)
    assert stats.n_periods == 5


def test_analyze_quantiles_detects_monotonic_increasing() -> None:
    buckets = {0: [-0.05, -0.04], 1: [-0.01, 0.0], 2: [0.02, 0.03], 3: [0.05, 0.06], 4: [0.09, 0.10]}
    result = analyze_quantiles(buckets)
    assert result.monotonic is True
    assert result.top_minus_bottom == pytest.approx(np.mean(buckets[4]) - np.mean(buckets[0]))
    assert result.monotonicity_spearman == pytest.approx(1.0)


def test_analyze_quantiles_detects_non_monotonic() -> None:
    buckets = {0: [0.05], 1: [-0.03], 2: [0.08], 3: [-0.02], 4: [0.01]}
    result = analyze_quantiles(buckets)
    assert result.monotonic is False


def test_analyze_quantiles_handles_empty_buckets() -> None:
    result = analyze_quantiles({0: [], 1: [], 2: [], 3: [], 4: []})
    assert result.top_minus_bottom is None
    assert result.monotonic is None


def test_benjamini_hochberg_flags_small_pvalues_and_controls_fdr() -> None:
    # 10 hypotheses: first 2 are strongly significant, rest are null-like p-values.
    pvalues = [0.001, 0.004, 0.30, 0.42, 0.55, 0.61, 0.70, 0.80, 0.90, 0.95]
    flags = benjamini_hochberg(pvalues, alpha=0.05)
    assert bool(flags[0]) is True
    assert bool(flags[1]) is True
    assert not any(flags[2:])


def test_benjamini_hochberg_no_significant_when_all_large() -> None:
    pvalues = [0.5, 0.6, 0.7, 0.8]
    flags = benjamini_hochberg(pvalues, alpha=0.05)
    assert not any(flags)


def test_benjamini_hochberg_empty_input() -> None:
    assert benjamini_hochberg([], alpha=0.05) == []


def test_permutation_test_ic_null_data_gives_large_p_value() -> None:
    rng = np.random.default_rng(1)
    scores_by_snapshot = []
    returns_by_snapshot = []
    for _ in range(60):
        symbols = [f"S{i}" for i in range(10)]
        scores_by_snapshot.append({s: rng.normal() for s in symbols})
        returns_by_snapshot.append({s: rng.normal() for s in symbols})

    from src.evaluation.cross_sectional_analysis import information_coefficient
    ics = [
        ic for s, r in zip(scores_by_snapshot, returns_by_snapshot)
        if (ic := information_coefficient(s, r)) is not None
    ]
    observed_mean_ic = float(np.mean(ics))

    p_value = permutation_test_ic(
        scores_by_snapshot, returns_by_snapshot, observed_mean_ic, n_perm=100, seed=2
    )
    assert p_value > 0.05  # true null: should not be spuriously significant


def test_permutation_test_ic_strong_signal_gives_small_p_value() -> None:
    rng = np.random.default_rng(3)
    scores_by_snapshot = []
    returns_by_snapshot = []
    for _ in range(60):
        symbols = [f"S{i}" for i in range(10)]
        scores = {s: rng.normal() for s in symbols}
        # returns strongly follow scores -- a real, planted relationship
        returns = {s: scores[s] * 2.0 + rng.normal(0, 0.1) for s in symbols}
        scores_by_snapshot.append(scores)
        returns_by_snapshot.append(returns)

    from src.evaluation.cross_sectional_analysis import information_coefficient
    ics = [
        ic for s, r in zip(scores_by_snapshot, returns_by_snapshot)
        if (ic := information_coefficient(s, r)) is not None
    ]
    observed_mean_ic = float(np.mean(ics))

    p_value = permutation_test_ic(
        scores_by_snapshot, returns_by_snapshot, observed_mean_ic, n_perm=100, seed=4
    )
    assert p_value < 0.05
