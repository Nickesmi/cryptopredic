"""Tests for the shadow-mode auto-evaluation loop (Phase 13 of the Phase-2
validation brief)."""

from __future__ import annotations

import pytest

from src.evaluation.prediction_store import PredictionStore
from src.evaluation.shadow import evaluate_due_predictions


def _make_due_prediction(store: PredictionStore, symbol: str = "BTCUSDT", expires_at: int = 1_700_000_300):
    return store.create_prediction(
        expires_at=expires_at,
        model_name="xgboost",
        model_version="1.0.0",
        symbol=symbol,
        timeframe="1H",
        prediction_horizon=1,
        anchor_time=1_700_000_000,
        anchor_price=100.0,
        predicted_price=110.0,
        confidence=0.8,
        bullish_probability=0.8,
        bearish_probability=0.2,
        expected_volatility=0.02,
        prediction_values={"expected_price": 110.0},
    )


@pytest.mark.asyncio
async def test_evaluates_all_due_predictions(tmp_path) -> None:
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    record = _make_due_prediction(store)

    async def stub_lookup(symbol: str, timeframe: str, at_time: int):
        return 112.0, at_time

    result = await evaluate_due_predictions(store, stub_lookup, now_ts=1_700_001_000)

    assert result.n_due == 1
    assert result.n_evaluated == 1
    assert result.n_errors == 0
    assert record.prediction_id in result.evaluated_prediction_ids
    assert store.get_prediction(record.prediction_id) is not None
    assert store.due_predictions(now_ts=1_700_001_000) == []  # no longer due once evaluated


@pytest.mark.asyncio
async def test_not_due_yet_predictions_are_left_alone(tmp_path) -> None:
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    _make_due_prediction(store, expires_at=1_800_000_000)  # far in the future

    async def stub_lookup(symbol: str, timeframe: str, at_time: int):
        return 112.0, at_time

    result = await evaluate_due_predictions(store, stub_lookup, now_ts=1_700_000_000)
    assert result.n_due == 0
    assert result.n_evaluated == 0


@pytest.mark.asyncio
async def test_one_failed_lookup_does_not_block_the_rest(tmp_path) -> None:
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    bad = _make_due_prediction(store, symbol="BADCOIN", expires_at=1_700_000_100)
    good = _make_due_prediction(store, symbol="BTCUSDT", expires_at=1_700_000_200)

    async def flaky_lookup(symbol: str, timeframe: str, at_time: int):
        if symbol == "BADCOIN":
            raise RuntimeError("provider outage")
        return 112.0, at_time

    result = await evaluate_due_predictions(store, flaky_lookup, now_ts=1_700_001_000)

    assert result.n_due == 2
    assert result.n_evaluated == 1
    assert result.n_errors == 1
    assert good.prediction_id in result.evaluated_prediction_ids
    assert result.errors[0]["prediction_id"] == bad.prediction_id


@pytest.mark.asyncio
async def test_re_running_the_cycle_does_not_double_evaluate(tmp_path) -> None:
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    _make_due_prediction(store)

    calls = []

    async def counting_lookup(symbol: str, timeframe: str, at_time: int):
        calls.append(at_time)
        return 112.0, at_time

    await evaluate_due_predictions(store, counting_lookup, now_ts=1_700_001_000)
    result_second_run = await evaluate_due_predictions(store, counting_lookup, now_ts=1_700_002_000)

    assert len(calls) == 1  # only evaluated once across both cycles
    assert result_second_run.n_due == 0
