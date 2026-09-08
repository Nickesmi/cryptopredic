"""Historical replay backtest engine without data leakage.

Walks forward through a chronologically-ordered OHLCV frame, at each
step training/predicting using only data up to that point
(``window = df.iloc[i - lookback : i]``) and then checking the
prediction against the actual price ``horizon`` candles later
(``df.iloc[i + horizon]``). This mirrors exactly what
``ModelManager.predict_from_frame`` does at live-inference time, so
the same target definition — price ``horizon`` candles of
``timeframe`` ahead — is used in both places (Phase 17 of the timing
audit: no train/live skew).

``horizon`` here is always expressed in *candles of the given
timeframe*, never in calendar days independent of the timeframe. If
you need to backtest a wall-clock horizon (e.g. "4 hours"), convert it
to candles first: ``horizon = horizon_seconds // timeframe_to_seconds(tf)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pandas as pd

from src.evaluation.metrics import aggregate_metrics, evaluate_prediction
from src.models.model_manager import ModelManager
from src.utils.timeframes import timeframe_to_seconds


@dataclass
class BacktestConfig:
    symbol: str
    timeframe: str
    horizon: int
    lookback: int


@dataclass
class BacktestResult:
    config: BacktestConfig
    metrics: Dict[str, Any]
    predictions: List[Dict[str, Any]]

    def to_dict(self):
        return {
            "config": asdict(self.config),
            "metrics": self.metrics,
            "predictions_count": len(self.predictions),
        }


def run_backtest(df: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Runs a no-leakage, walk-forward historical replay backtest.

    At each step ``i`` the model only sees ``df.iloc[i - lookback : i]``
    (never anything at or after ``i``), predicts the close price
    ``config.horizon`` candles ahead, and is scored against
    ``df.iloc[i + config.horizon]`` — the actual close at that exact
    future candle, matching the ``target(t, H) = price[t + H]``
    definition used everywhere else in the system.
    """
    model_manager = ModelManager()

    evaluations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    total_len = len(df)
    if total_len <= config.lookback + config.horizon:
        raise ValueError("Dataset too short for the given lookback and horizon.")

    interval_seconds = timeframe_to_seconds(config.timeframe)

    for i in range(config.lookback, total_len - config.horizon):
        # 1. Provide only data up to current index 'i' (future is unknown).
        window = df.iloc[i - config.lookback : i].copy()

        # 2. Generate prediction using the exact same code path as live
        #    inference (ModelManager.predict_from_frame), so backtest and
        #    live predictions can never drift apart.
        try:
            forecast = model_manager.predict_from_frame(
                symbol=config.symbol,
                timeframe=config.timeframe,
                raw_df=window,
                n_candles=config.horizon,
            )
        except ValueError:
            # Not enough history yet to build features at this window size.
            continue

        current_close = forecast.anchor_price
        current_time = forecast.anchor_time

        # 3. Retrieve the actual future data (the target) — the real close
        #    price exactly `horizon` candles after the anchor, never a
        #    proxy for some other horizon.
        future_idx = i + config.horizon - 1
        actual_close = float(df["close"].iloc[future_idx])

        if isinstance(df.index, pd.DatetimeIndex):
            actual_time = int(df.index[future_idx].timestamp())
        else:
            actual_time = current_time + config.horizon * interval_seconds

        predicted_price = forecast.metadata.expected_price
        confidence = forecast.metadata.confidence
        bullish_probability = (
            confidence if forecast.metadata.direction == "bullish" else 1.0 - confidence
        )

        prediction_record = {
            "prediction_id": f"bt-{config.symbol}-{config.timeframe}-{i}",
            "anchor_time": current_time,
            "anchor_price": current_close,
            "actual_time": actual_time,
            "actual_price": actual_close,
            "predicted_price": predicted_price,
            "confidence": confidence,
            "bullish_probability": bullish_probability,
            "expected_price": predicted_price,
            "horizon_seconds": forecast.horizon_seconds,
        }
        predictions.append(prediction_record)

        # 4. Evaluate prediction.
        metrics = evaluate_prediction(
            anchor_price=current_close,
            predicted_price=predicted_price,
            actual_price=actual_close,
            confidence=confidence,
            bullish_probability=bullish_probability,
        )

        eval_dict = metrics.to_dict()
        eval_dict["anchor_price"] = current_close
        eval_dict["actual_price"] = actual_close
        eval_dict["predicted_price"] = predicted_price
        eval_dict["confidence"] = confidence

        evaluations.append(eval_dict)

    if not evaluations:
        raise ValueError("No predictions were generated.")

    final_metrics = aggregate_metrics(evaluations)

    return BacktestResult(
        config=config,
        metrics=final_metrics,
        predictions=predictions,
    )
