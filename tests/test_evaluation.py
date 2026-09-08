from __future__ import annotations

import numpy as np
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
    # NOTE on fixture design: `close` must share `dates`' index — building
    # the DataFrame with index=dates while close/open/high/low carry a bare
    # RangeIndex causes pandas to reindex-align them against `dates`,
    # silently producing an all-NaN frame. This previously went unnoticed
    # because the backtest crashed earlier (ModelManager.predict was never
    # awaited) before ever touching the price data.
    #
    # A monotonically increasing series (the original fixture) is also the
    # wrong shape for this test: XGBoost (and tree ensembles generally)
    # cannot extrapolate past the feature-value range seen during training,
    # so on an ever-climbing series every walk-forward window is asked to
    # predict beyond its own training range and the model's predictions
    # systematically undershoot — a real limitation worth knowing about
    # (see docs/TIMING_AUDIT_REPORT.md), but not what this test is checking.
    # A bounded, oscillating series keeps future feature values inside the
    # training range so directional accuracy actually reflects backtest
    # correctness rather than an architectural extrapolation limit.
    n = 250
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    close = pd.Series(200 + 40 * np.sin(2 * np.pi * t / 30), index=dates, dtype=float)
    candles = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
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
            lookback=120,
        ),
    )

    assert result.metrics["predictions"] > 0
    # Direction is genuinely ambiguous right at the sine wave's turning
    # points, so we don't expect literally 1.0 — but a competent no-leakage
    # 3-day-ahead model on a smooth, bounded, cyclical series should still
    # get the great majority of calls right.
    assert result.metrics["directional_accuracy"] >= 0.85
    for prediction in result.predictions:
        assert prediction["anchor_time"] < prediction["actual_time"]
        assert prediction["prediction_id"].startswith("bt-BTCUSDT-1D")
        # The evaluation window must match the horizon the model was
        # actually trained on — this is the exact invariant that was
        # violated by the timeframe/horizon-in-days confusion this audit
        # fixed in src/models/model_manager.py.
        assert prediction["actual_time"] - prediction["anchor_time"] == pytest.approx(
            3 * 86400, abs=1
        )
