"""Tests for the bootstrap significance helpers (Phase 3, Section 13)."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.significance import (
    auc_confidence_interval,
    bootstrap_statistic,
    paired_bootstrap_diff,
)


def test_bootstrap_ci_contains_true_mean_for_tight_distribution() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.05, 0.01, 500)
    result = bootstrap_statistic(values, np.mean)
    assert result.ci_low < 0.05 < result.ci_high
    assert result.ci_high - result.ci_low < 0.01  # tight distribution -> tight CI


def test_bootstrap_ci_excludes_far_off_value() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.05, 0.002, 500)
    result = bootstrap_statistic(values, np.mean)
    assert result.excludes(0.5)


def test_raises_on_too_few_observations() -> None:
    with pytest.raises(ValueError):
        bootstrap_statistic([1.0, 2.0], np.mean)


def test_paired_bootstrap_detects_a_real_improvement() -> None:
    rng = np.random.default_rng(2)
    errors_a = np.abs(rng.normal(0.03, 0.01, 300))  # smaller errors
    errors_b = np.abs(rng.normal(0.20, 0.05, 300))  # much larger errors
    result = paired_bootstrap_diff(errors_a, errors_b, statistic_fn=np.mean)
    assert result.point_estimate < 0  # a has lower error than b
    assert result.ci_high < 0  # CI entirely below zero -> statistically real difference


def test_paired_bootstrap_finds_no_difference_for_identical_data() -> None:
    rng = np.random.default_rng(3)
    values = rng.normal(0.1, 0.02, 200)
    result = paired_bootstrap_diff(values, values, statistic_fn=np.mean)
    assert result.point_estimate == pytest.approx(0.0)
    assert not result.excludes(0.0)


def test_paired_bootstrap_requires_equal_length() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0])


def test_auc_ci_excludes_half_for_a_genuinely_predictive_signal() -> None:
    rng = np.random.default_rng(4)
    n = 400
    labels = rng.integers(0, 2, n).astype(float)
    # Probabilities strongly correlated with the label -> real signal.
    probs = np.clip(labels * 0.7 + rng.normal(0, 0.1, n) + 0.15, 0.01, 0.99)
    result = auc_confidence_interval(labels, probs)
    assert result.point_estimate > 0.8
    assert result.excludes(0.5)


def test_auc_ci_includes_half_for_pure_noise() -> None:
    rng = np.random.default_rng(5)
    n = 300
    labels = rng.integers(0, 2, n).astype(float)
    probs = rng.uniform(0, 1, n)  # unrelated to labels
    result = auc_confidence_interval(labels, probs)
    assert not result.excludes(0.5)
