"""Walk-forward validation.

Chronological, expanding-window validation: repeatedly (1) train on
everything up to a cut-off, (2) evaluate on a contiguous block of
*future* candles the model has never seen, separated from training by
an embargo gap, then (3) roll the cut-off forward and repeat.

    Fold 1: TRAIN [0 .......... train_end_1] | embargo | TEST [.....]
    Fold 2: TRAIN [0 .................. train_end_2]   | embargo | TEST [.....]
    Fold 3: TRAIN [0 .......................... train_end_3]     | embargo | TEST [.....]

This file was an empty stub before this audit — the ``docs/modeling.md``
claim that walk-forward validation exists (``planned for Issue #5``) was
not backed by any implementation, and the only validation actually run
anywhere in the system was a single, un-embargoed 90/10 hold-out inside
``TimeSeriesForecaster.train``.

Each fold trains a **fresh** model (a new ``ModelManager``) strictly on
``df.iloc[:train_end]`` — a fold never reuses another fold's cached
forecaster, so later folds cannot leak information back into earlier
ones and each fold's score reflects only what was knowable at that
point in history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.evaluation.metrics import aggregate_metrics, evaluate_prediction
from src.models.model_manager import ModelManager
from src.utils.timeframes import timeframe_to_seconds


@dataclass
class FoldResult:
    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: dict[str, Any]
    n_predictions: int


@dataclass
class WalkForwardReport:
    symbol: str
    timeframe: str
    horizon_candles: int
    folds: list[FoldResult] = field(default_factory=list)
    overall: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon_candles": self.horizon_candles,
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "train_start": f.train_start,
                    "train_end": f.train_end,
                    "test_start": f.test_start,
                    "test_end": f.test_end,
                    "n_predictions": f.n_predictions,
                    "metrics": f.metrics,
                }
                for f in self.folds
            ],
            "overall": self.overall,
        }


def walk_forward_validate(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    horizon_candles: int,
    n_folds: int = 5,
    min_train_size: int = 200,
    lookback: int | None = None,
    stride: int = 1,
) -> WalkForwardReport:
    """Run expanding-window walk-forward validation over *df*.

    Args:
        df:              Full chronologically-ordered OHLCV history.
        symbol:          Trading pair (used for cache keys / labelling only).
        timeframe:       Candle period string.
        horizon_candles: Forecast horizon, in candles of *timeframe*. Also
                         used as the embargo gap between each fold's train
                         and test blocks (the same purge logic used inside
                         ``TimeSeriesForecaster.train``, applied here at the
                         fold level too).
        n_folds:         Number of expanding-window folds to run.
        min_train_size:  Minimum number of candles in the first fold's
                         training block.
        lookback:        Candles of trailing history fed into the feature
                         pipeline for each test-block prediction. Defaults
                         to the fold's full training block.
        stride:          Evaluate every ``stride``-th candle in each fold's
                         test block instead of every single one. Default 1
                         (every candle, matching prior behaviour) preserves
                         exact backward compatibility; pass e.g. 6-24 for
                         large multi-horizon studies where evaluating every
                         candle is computationally prohibitive without
                         meaningfully changing the aggregate metrics (the
                         test block is still walked forward chronologically
                         with no leakage -- this only thins how densely it's
                         sampled).

    Returns:
        A :class:`WalkForwardReport` with per-fold and pooled metrics.

    Raises:
        ValueError: If *df* is too short to carve out ``n_folds`` folds.
    """
    total = len(df)
    remaining = total - min_train_size - horizon_candles
    if remaining <= 0 or n_folds < 1:
        raise ValueError(
            f"Not enough data for walk-forward validation: {total} rows, "
            f"need > min_train_size + horizon_candles = {min_train_size + horizon_candles}."
        )

    fold_size = max(1, remaining // n_folds)
    interval_seconds = timeframe_to_seconds(timeframe)

    folds: list[FoldResult] = []
    all_evaluations: list[dict[str, Any]] = []

    for fold_index in range(n_folds):
        train_end = min_train_size + fold_index * fold_size
        test_start = train_end + horizon_candles  # embargo gap
        test_end = min(test_start + fold_size, total - horizon_candles)
        if test_end <= test_start or train_end >= total:
            break

        fold_lookback = lookback or train_end
        window_start = max(0, train_end - fold_lookback)
        manager = ModelManager()  # fresh forecaster cache — no cross-fold leakage

        # Force-train the fold's model on exactly [window_start:train_end) —
        # nothing at or after train_end is visible during training. This one
        # call populates ModelManager's cache; every subsequent call below
        # for this fold hits that cache and therefore reuses the SAME frozen
        # model (see ModelManager.predict_from_frame's cache_key), while
        # still recomputing *features* from whatever data is passed in. That
        # split — train once on the strict train block, keep re-deriving
        # "as of now" features as we roll through the test block — is what
        # lets each test point see up-to-date indicators without the model
        # itself ever being fit on test-period data.
        try:
            manager.predict_from_frame(
                symbol=symbol,
                timeframe=timeframe,
                raw_df=df.iloc[window_start:train_end].copy(),
                n_candles=horizon_candles,
            )
        except ValueError:
            continue

        fold_evaluations: list[dict[str, Any]] = []

        for i in range(test_start, test_end, stride):
            context = df.iloc[window_start:i].copy()
            try:
                forecast = manager.predict_from_frame(
                    symbol=symbol,
                    timeframe=timeframe,
                    raw_df=context,
                    n_candles=horizon_candles,
                )
            except ValueError:
                continue

            future_idx = i + horizon_candles - 1
            if future_idx >= total:
                continue
            actual_price = float(df["close"].iloc[future_idx])

            predicted_price = forecast.metadata.expected_price
            confidence = forecast.metadata.confidence
            bullish_probability = (
                confidence if forecast.metadata.direction == "bullish" else 1.0 - confidence
            )
            metrics = evaluate_prediction(
                anchor_price=forecast.anchor_price,
                predicted_price=predicted_price,
                actual_price=actual_price,
                confidence=confidence,
                bullish_probability=bullish_probability,
            )
            eval_dict = metrics.to_dict()
            eval_dict["anchor_price"] = forecast.anchor_price
            eval_dict["actual_price"] = actual_price
            eval_dict["predicted_price"] = predicted_price
            eval_dict["confidence"] = confidence
            fold_evaluations.append(eval_dict)

        if not fold_evaluations:
            continue

        fold_metrics = aggregate_metrics(fold_evaluations)
        folds.append(
            FoldResult(
                fold_index=fold_index,
                train_start=window_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                metrics=fold_metrics,
                n_predictions=len(fold_evaluations),
            )
        )
        all_evaluations.extend(fold_evaluations)

    if not folds:
        raise ValueError("Walk-forward validation produced no folds with predictions.")

    return WalkForwardReport(
        symbol=symbol,
        timeframe=timeframe,
        horizon_candles=horizon_candles,
        folds=folds,
        overall=aggregate_metrics(all_evaluations),
    )
