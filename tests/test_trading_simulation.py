"""Tests for the economic-value trading simulation (Phase 3, Section 14)."""

from __future__ import annotations

import pytest

from src.evaluation.trading_simulation import simulate_trading


def _record(implied_return: float, actual_return: float) -> dict:
    return {"implied_return": implied_return, "actual_return": actual_return}


def test_perfect_predictions_are_always_profitable_net_of_small_cost() -> None:
    records = [_record(0.05, 0.05), _record(-0.05, -0.05), _record(0.03, 0.03)]
    result = simulate_trading(records, entry_threshold=0.01, cost_pct=0.001)
    assert result.n_trades == 3
    assert result.win_rate == 1.0
    assert result.total_return > 0


def test_flat_when_implied_return_below_threshold() -> None:
    records = [_record(0.001, 0.05), _record(-0.002, -0.05)]
    result = simulate_trading(records, entry_threshold=0.01)
    assert result.n_trades == 0
    assert result.total_return == 0.0


def test_wrong_direction_predictions_lose_money() -> None:
    records = [_record(0.05, -0.05), _record(-0.05, 0.05)]
    result = simulate_trading(records, entry_threshold=0.01, cost_pct=0.001)
    assert result.win_rate == 0.0
    assert result.total_return < 0


def test_cost_reduces_return_relative_to_zero_cost() -> None:
    records = [_record(0.05, 0.05) for _ in range(10)]
    no_cost = simulate_trading(records, entry_threshold=0.01, cost_pct=0.0)
    with_cost = simulate_trading(records, entry_threshold=0.01, cost_pct=0.01)
    assert with_cost.total_return < no_cost.total_return


def test_buy_and_hold_and_momentum_baselines_are_reported() -> None:
    records = [_record(0.02, 0.01), _record(0.01, -0.01), _record(-0.02, 0.02)]
    result = simulate_trading(records, entry_threshold=0.015)
    assert isinstance(result.buy_and_hold_return, float)
    assert isinstance(result.momentum_baseline_return, float)


def test_raises_on_empty_records() -> None:
    with pytest.raises(ValueError):
        simulate_trading([], entry_threshold=0.01)


def test_max_drawdown_is_non_positive() -> None:
    records = [_record(0.05, 0.05), _record(0.05, -0.10), _record(0.05, 0.02)]
    result = simulate_trading(records, entry_threshold=0.01)
    assert result.max_drawdown <= 0.0
