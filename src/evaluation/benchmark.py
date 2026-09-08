"""Baseline forecasters for benchmarking the AI model.

Phase 13 of the timing audit: an AI forecaster is only useful if it beats
simple, essentially-free baselines out-of-sample. This file was an empty
stub before the audit — no baseline had ever actually been computed, so
there was no evidence the XGBoost model outperformed doing nothing.

All baselines here use the exact same ``target(t, H) = price[t + H]``
definition and the same walk-forward replay style as
``src/evaluation/backtest.py`` and ``src/evaluation/walk_forward.py``, so
their metrics are directly comparable to the model's.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from src.evaluation.metrics import aggregate_metrics, evaluate_prediction


def _replay_baseline(
    df: pd.DataFrame,
    horizon_candles: int,
    lookback: int,
    predict_fn: Callable[[pd.Series], float],
) -> list[dict[str, Any]]:
    """Shared walk-forward loop for baseline predictors.

    ``predict_fn`` receives the trailing close-price window available at
    each step (``df['close'].iloc[i - lookback : i]``, i.e. never anything
    at or after ``i``) and returns a point forecast for ``i + horizon - 1``.
    """
    total = len(df)
    evaluations: list[dict[str, Any]] = []
    for i in range(lookback, total - horizon_candles + 1):
        history = df["close"].iloc[i - lookback : i]
        anchor_price = float(history.iloc[-1])
        predicted_price = predict_fn(history)
        actual_price = float(df["close"].iloc[i + horizon_candles - 1])

        metrics = evaluate_prediction(
            anchor_price=anchor_price,
            predicted_price=predicted_price,
            actual_price=actual_price,
            confidence=0.5,
            bullish_probability=0.5,
        )
        eval_dict = metrics.to_dict()
        eval_dict["anchor_price"] = anchor_price
        eval_dict["actual_price"] = actual_price
        eval_dict["predicted_price"] = predicted_price
        eval_dict["confidence"] = 0.5
        evaluations.append(eval_dict)
    return evaluations


def naive_persistence(
    df: pd.DataFrame, horizon_candles: int, lookback: int = 5
) -> dict[str, Any]:
    """"Tomorrow will look like today": predicted price = last known close.

    The minimum bar any forecaster must clear. A model that cannot beat
    this on MAE/RMSE and directional accuracy is not adding value.
    """
    evaluations = _replay_baseline(
        df, horizon_candles, lookback, predict_fn=lambda history: float(history.iloc[-1])
    )
    return aggregate_metrics(evaluations)


def naive_drift(
    df: pd.DataFrame, horizon_candles: int, lookback: int = 14
) -> dict[str, Any]:
    """Linear extrapolation of the recent average per-candle change.

    predicted = last_close + horizon_candles * mean(diff(history)).
    Captures simple momentum without any model at all.
    """

    def _predict(history: pd.Series) -> float:
        diffs = history.diff().dropna()
        mean_step = float(diffs.mean()) if len(diffs) else 0.0
        return float(history.iloc[-1]) + horizon_candles * mean_step

    evaluations = _replay_baseline(df, horizon_candles, lookback, predict_fn=_predict)
    return aggregate_metrics(evaluations)


def buy_and_hold_return(df: pd.DataFrame) -> dict[str, float]:
    """Total and annualised return of simply holding the asset over *df*."""
    if len(df) < 2:
        return {"total_return_pct": 0.0}
    start = float(df["close"].iloc[0])
    end = float(df["close"].iloc[-1])
    return {"total_return_pct": (end - start) / start * 100.0}


def compare_to_baselines(
    df: pd.DataFrame,
    horizon_candles: int,
    model_metrics: dict[str, Any],
    lookback: int = 14,
) -> dict[str, Any]:
    """Return model metrics alongside baseline metrics for a side-by-side comparison.

    Args:
        df:               Full chronological OHLCV history the model was
                          evaluated on (same data used to produce
                          *model_metrics*, e.g. from ``run_backtest`` or
                          ``walk_forward_validate``).
        horizon_candles:  Forecast horizon, in candles.
        model_metrics:    Aggregate metrics dict for the AI model over the
                          same period (from ``aggregate_metrics``).
        lookback:         Trailing window used by the baselines.

    Returns:
        ``{"model": ..., "naive_persistence": ..., "naive_drift": ...,
        "buy_and_hold": ..., "model_beats_persistence": bool,
        "model_beats_drift": bool}``
    """
    persistence = naive_persistence(df, horizon_candles, lookback=lookback)
    drift = naive_drift(df, horizon_candles, lookback=lookback)
    hold = buy_and_hold_return(df)

    def _beats(baseline: dict[str, Any]) -> bool:
        if not baseline.get("predictions") or not model_metrics.get("predictions"):
            return False
        return (
            model_metrics["mae"] <= baseline["mae"]
            and model_metrics["directional_accuracy"] >= baseline["directional_accuracy"]
        )

    return {
        "model": model_metrics,
        "naive_persistence": persistence,
        "naive_drift": drift,
        "buy_and_hold": hold,
        "model_beats_persistence": _beats(persistence),
        "model_beats_drift": _beats(drift),
    }
