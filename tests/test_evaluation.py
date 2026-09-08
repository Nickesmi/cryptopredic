from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.backtest import BacktestConfig, run_backtest
from src.evaluation.prediction_store import PredictionStore


def test_prediction_store_persists_and_evaluates(tmp_path):
    store = PredictionStore(tmp_path / "predictions.sqlite3")
    record = store.create_prediction(
        expires_at=1_700_000_300,
        model_name="xgboost",
        model_version="1.0.0",
        symbol="BTCUSDT",
        timeframe="1H",
        prediction_horizon=5,
        anchor_time=1_700_000_000,
        anchor_price=100.0,
        predicted_price=110.0,
        confidence=0.8,
        bullish_probability=0.8,
        bearish_probability=0.2,
        expected_volatility=0.02,
        prediction_values={"expected_price": 110.0},
    )

    stored = store.get_prediction(record.prediction_id)
    assert stored is not None
    assert stored["prediction_id"] == record.prediction_id
    assert stored["prediction_values"]["expected_price"] == pytest.approx(110.0)

    evaluation = store.evaluate_prediction(
        record.prediction_id,
        actual_time=1_700_000_300,
        actual_price=112.0,
    )
    assert evaluation["direction_correct"] is True
    assert evaluation["absolute_error"] == pytest.approx(2.0)
    assert store.leaderboard()[0]["model"] == "xgboost"
    assert store.dashboard()["overall"]["predictions"] == pytest.approx(1.0)


def test_backtest_replays_without_future_data():
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    close = pd.Series(range(100, 280), dtype=float)
    candles = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        },
        index=dates,
    )

    result = run_backtest(
        candles,
        BacktestConfig(
            symbol="BTCUSDT",
            timeframe="1D",
            horizon=3,
            lookback=30,
        ),
    )

    assert result.metrics["predictions"] > 0
    assert result.metrics["directional_accuracy"] == pytest.approx(1.0)
    first = result.predictions[0]
    assert first["anchor_time"] < first["actual_time"]
    assert first["prediction_id"].startswith("bt-BTCUSDT-1D")
