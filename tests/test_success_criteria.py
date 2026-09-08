"""Tests for the independent success-criteria definitions (Phase 9 of the
Phase-2 validation brief)."""

from __future__ import annotations

import pytest

from src.evaluation.success_criteria import evaluate_success


def test_price_right_but_wildly_late_is_target_success_not_horizon_success() -> None:
    """The brief's own worked example: a 4H prediction of $100,000 that
    actually arrives 3 days (72h) later must be TARGET success but NOT
    horizon success."""
    anchor_time = 0
    horizon_seconds = 4 * 3600  # 4h promise
    actual_hit_time = 3 * 86400  # arrives 3 days later

    result = evaluate_success(
        anchor_price=90_000.0,
        predicted_price=100_000.0,
        actual_price_at_horizon=90_500.0,  # barely moved by the promised 4h mark
        predicted_horizon_seconds=horizon_seconds,
        actual_target_hit_time_seconds=actual_hit_time,
        anchor_time_seconds=anchor_time,
    )

    assert result.target_success is True
    assert result.horizon_success is False
    assert result.time_error_seconds == actual_hit_time - horizon_seconds
    # At the promised 4h mark the price barely moved, so it's neither
    # directionally impressive nor numerically close to $100,000 yet.
    assert result.price_success is False


def test_target_hit_within_tolerance_window_is_horizon_success() -> None:
    anchor_time = 0
    horizon_seconds = 4 * 3600
    # Hits at 4.5h -- within the default 25% tolerance band (3h-5h).
    actual_hit_time = int(4.5 * 3600)

    result = evaluate_success(
        anchor_price=90_000.0,
        predicted_price=100_000.0,
        actual_price_at_horizon=99_800.0,
        predicted_horizon_seconds=horizon_seconds,
        actual_target_hit_time_seconds=actual_hit_time,
        anchor_time_seconds=anchor_time,
    )
    assert result.target_success is True
    assert result.horizon_success is True


def test_target_never_hit_is_none_not_false() -> None:
    """'Never hit (within the search window)' must be distinguishable from
    'hit at the wrong time' -- they are different failure modes."""
    result = evaluate_success(
        anchor_price=90_000.0,
        predicted_price=100_000.0,
        actual_price_at_horizon=91_000.0,
        predicted_horizon_seconds=4 * 3600,
        actual_target_hit_time_seconds=None,
    )
    assert result.target_success is None
    assert result.horizon_success is None
    assert result.time_error_seconds is None


def test_directional_and_price_success_are_independent() -> None:
    # Right direction, but nowhere near the predicted price -> directional
    # success True, price success False.
    result = evaluate_success(
        anchor_price=100.0,
        predicted_price=110.0,
        actual_price_at_horizon=101.0,
        predicted_horizon_seconds=3600,
        price_tolerance_pct=0.01,
    )
    assert result.directional_success is True
    assert result.price_success is False


def test_trading_success_accounts_for_cost() -> None:
    # A tiny positive move in the predicted direction is directionally
    # correct but can still be a trading failure once cost is subtracted.
    result = evaluate_success(
        anchor_price=100.0,
        predicted_price=100.2,
        actual_price_at_horizon=100.05,  # +0.05% move
        predicted_horizon_seconds=3600,
        trading_cost_pct=0.001,  # 0.1% cost > 0.05% gain
    )
    assert result.directional_success is True
    assert result.trading_success is False


def test_trading_success_none_when_cost_not_provided() -> None:
    result = evaluate_success(
        anchor_price=100.0,
        predicted_price=110.0,
        actual_price_at_horizon=105.0,
        predicted_horizon_seconds=3600,
        trading_cost_pct=None,
    )
    assert result.trading_success is None


def test_requires_anchor_time_when_hit_time_given() -> None:
    with pytest.raises(ValueError):
        evaluate_success(
            anchor_price=100.0,
            predicted_price=110.0,
            actual_price_at_horizon=105.0,
            predicted_horizon_seconds=3600,
            actual_target_hit_time_seconds=7200,
            anchor_time_seconds=None,
        )
