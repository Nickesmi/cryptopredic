"""Controlled experiments comparing forecasting-target formulations.

Phase 3 of the audit asks a specific question: is *absolute future price*
(the current architecture's target — see the module docstring of
``src/models/xgboost_model.py`` and the mathematical audit in
``docs/PHASE3_FORMULATION_REPORT.md``) the right thing to predict at all,
or does it structurally cause the poor generalisation documented in
Phase 2 (loses to every baseline at every horizon, ~2-4% horizon success)?

This module trains the *same* feature set and the *same* gradient-boosted
tree family on several different targets built from the same underlying
data, under identical walk-forward discipline (embargoed expanding-window
folds, same as ``src/evaluation/walk_forward.py``), so the only thing that
differs between experiments is the target formulation:

    "price"          y = close[t+H]                              (current architecture)
    "return"         y = (close[t+H] - close[t]) / close[t]
    "log_return"     y = ln(close[t+H] / close[t])
    "direction"      y = 1[close[t+H] > close[t]]                 (binary classification)
    "threshold_up"   y = 1[return > +threshold]                   (binary classification)
    "threshold_down" y = 1[return < -threshold]                   (binary classification)

Every result is converted to a common, comparable unit — an **implied
return** — specifically so "predict price" and "predict return" can be
judged on the same MAE/directional-accuracy scale rather than talking past
each other in different units. This is deliberate: without it, a claim
like "return prediction has lower MAE than price prediction" would be
meaningless (different units, different scales).

This module does NOT decide the final answer — it produces the
per-prediction ledgers `docs/PHASE3_FORMULATION_REPORT.md`'s analysis is
built from. It also does not modify `ModelManager` or the production
prediction path; it is deliberately a separate, standalone research tool
so that running these experiments can never accidentally change what the
live system does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.settings import (
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_RANDOM_STATE,
    XGB_SUBSAMPLE,
)
from src.data.preprocessing import add_log_returns, clean_ohlcv
from src.features.feature_pipeline import FeaturePipeline

TargetType = Literal[
    "price", "return", "log_return", "direction", "threshold_up", "threshold_down"
]
CLASSIFICATION_TARGETS: frozenset[str] = frozenset({"direction", "threshold_up", "threshold_down"})

_XGB_COMMON = dict(
    n_estimators=XGB_N_ESTIMATORS,
    max_depth=XGB_MAX_DEPTH,
    learning_rate=XGB_LEARNING_RATE,
    subsample=XGB_SUBSAMPLE,
    colsample_bytree=XGB_COLSAMPLE_BYTREE,
    random_state=XGB_RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
)


def split_design_and_frozen_test(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split *df* into a design region and a frozen final test region.

    Per the audit brief's Section 12: "The final test period must remain
    untouched until the model design is frozen." All formulation/feature/
    regime/cross-asset experiments in this phase run only on the design
    region; the frozen region is touched exactly once, at the end, to
    report the final scorecard for whichever configuration the design
    region selected.
    """
    split = int(len(df) * (1 - test_fraction))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def build_target(
    feature_df: pd.DataFrame, horizon_candles: int, target_type: TargetType, threshold: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Build the (y, actual_return) arrays for one target formulation.

    ``actual_return`` is returned alongside ``y`` regardless of
    *target_type* — it is the ground truth every formulation is ultimately
    judged against (see module docstring), computed once so every
    formulation's "was the direction/magnitude right" question uses
    exactly the same numbers.
    """
    close = feature_df["close"].to_numpy(dtype=float)
    future_close = np.concatenate([close[horizon_candles:], np.full(horizon_candles, np.nan)])
    actual_return = (future_close - close) / close

    if target_type == "price":
        y = future_close
    elif target_type == "return":
        y = actual_return
    elif target_type == "log_return":
        y = np.log(future_close / close)
    elif target_type == "direction":
        y = (actual_return > 0).astype(float)
    elif target_type == "threshold_up":
        y = (actual_return > threshold).astype(float)
    elif target_type == "threshold_down":
        y = (actual_return < -threshold).astype(float)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return y, actual_return


def _make_model(target_type: TargetType):
    if target_type in CLASSIFICATION_TARGETS:
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **_XGB_COMMON)
    return xgb.XGBRegressor(objective="reg:squarederror", **_XGB_COMMON)


def _implied_return(predicted_value: float, anchor_price: float, target_type: TargetType) -> float:
    """Convert any regression target's prediction to an implied return, for comparability."""
    if target_type == "price":
        return (predicted_value - anchor_price) / anchor_price
    if target_type == "log_return":
        return float(np.exp(predicted_value) - 1.0)
    return predicted_value  # "return" is already a return


@dataclass
class FormulationResult:
    name: str
    target_type: str
    horizon_candles: int
    threshold: float
    n_predictions: int
    metrics: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)  # per-prediction, for bootstrap/pairing
    fold_feature_importances: list[dict[str, float]] = field(default_factory=list)  # one dict per fold

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_type": self.target_type,
            "horizon_candles": self.horizon_candles,
            "threshold": self.threshold,
            "n_predictions": self.n_predictions,
            "metrics": self.metrics,
        }


def run_formulation_walk_forward(
    df: pd.DataFrame,
    horizon_candles: int,
    target_type: TargetType,
    threshold: float = 0.0,
    feature_cols: list[str] | None = None,
    n_folds: int = 2,
    min_train_size: int = 700,
    stride: int = 10,
    embargo: int | None = None,
    name: str | None = None,
    extra_features: pd.DataFrame | None = None,
) -> FormulationResult:
    """Walk-forward evaluate one target formulation.

    Same expanding-window-with-embargo design as
    ``src/evaluation/walk_forward.py`` (train once per fold on
    ``[:train_end]``, embargo ``horizon_candles`` rows, evaluate a strided
    sample of the following block), generalised to any of the six target
    types via :func:`build_target` and :func:`_make_model`.

    Args:
        extra_features: Optional extra columns (index-aligned to *df*,
            e.g. volume-derived or cross-asset features computed by the
            caller) to make available for selection via *feature_cols* —
            used by the ablation/cross-asset experiments (Sections 6/9 of
            the Phase 3 report) without needing to modify the shipped
            ``FeaturePipeline``.

    Returns a :class:`FormulationResult` whose ``metrics`` are always
    computed in **implied-return space** (MAE/RMSE/directional accuracy of
    the return every formulation is ultimately trying to get right), plus
    classification-specific metrics (accuracy/precision/recall/F1/
    ROC-AUC/PR-AUC/Brier) when *target_type* is a classification target.
    """
    pipeline = FeaturePipeline()
    clean_df = add_log_returns(clean_ohlcv(df))
    if extra_features is not None:
        clean_df = clean_df.join(extra_features, how="left")
    feature_df = pipeline.build(clean_df)
    cols = feature_cols or pipeline.feature_columns
    embargo = horizon_candles if embargo is None else embargo

    y_full, ret_full = build_target(feature_df, horizon_candles, target_type, threshold)
    X_full = feature_df[cols].to_numpy(dtype=float)
    anchor_prices = feature_df["close"].to_numpy(dtype=float)

    n = len(feature_df)
    remaining = n - min_train_size - horizon_candles
    if remaining <= 50:
        raise ValueError(
            f"Not enough data for formulation walk-forward: n={n}, "
            f"min_train_size={min_train_size}, horizon_candles={horizon_candles}"
        )
    fold_size = max(1, remaining // n_folds)

    records: list[dict[str, Any]] = []
    fold_feature_importances: list[dict[str, float]] = []

    for fold in range(n_folds):
        train_end = min_train_size + fold * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n - horizon_candles)
        if test_end <= test_start or train_end >= n:
            break

        y_train = y_full[:train_end]
        X_train = X_full[:train_end]
        valid_train = np.isfinite(y_train)
        if valid_train.sum() < 20:
            continue

        model = _make_model(target_type)
        model.fit(X_train[valid_train], y_train[valid_train])
        fold_feature_importances.append(dict(zip(cols, model.feature_importances_.tolist())))

        for i in range(test_start, test_end, stride):
            if not np.isfinite(ret_full[i]):
                continue
            x_i = X_full[i : i + 1]
            actual_return = float(ret_full[i])
            anchor_price = float(anchor_prices[i])

            record: dict[str, Any] = {
                "fold": fold,
                "test_index": i,
                # NOTE: test_index is a position into this call's own
                # feature_df, AFTER FeaturePipeline.build()'s dropna() --
                # two calls with different feature sets (e.g. extra_features
                # with a longer rolling-window warmup) can drop a different
                # number of leading rows and therefore disagree on what
                # test_index N actually refers to. Pair records across
                # experiments by "timestamp", never by raw "test_index".
                "timestamp": feature_df.index[i],
                "actual_return": actual_return,
            }

            if target_type in CLASSIFICATION_TARGETS:
                prob_up = float(model.predict_proba(x_i)[0, 1])
                actual_label = float(y_full[i])
                record["predicted_probability"] = prob_up
                record["actual_label"] = actual_label
                # Implied return sign for cross-formulation directional comparison:
                # "predict P(up) > 0.5" -> implied bullish call.
                record["implied_direction"] = 1.0 if prob_up > 0.5 else -1.0
            else:
                predicted_value = float(model.predict(x_i)[0])
                implied_return = _implied_return(predicted_value, anchor_price, target_type)
                record["predicted_value"] = predicted_value
                record["implied_return"] = implied_return

            records.append(record)

    if not records:
        raise ValueError("Formulation walk-forward produced no evaluable predictions.")

    metrics = _aggregate_formulation_metrics(records, target_type)

    return FormulationResult(
        name=name or f"{target_type}@{horizon_candles}c",
        target_type=target_type,
        horizon_candles=horizon_candles,
        threshold=threshold,
        n_predictions=len(records),
        metrics=metrics,
        records=records,
        fold_feature_importances=fold_feature_importances,
    )


def _aggregate_formulation_metrics(records: list[dict[str, Any]], target_type: TargetType) -> dict[str, Any]:
    actual_returns = np.array([r["actual_return"] for r in records])
    metrics: dict[str, Any] = {"n": len(records)}

    if target_type in CLASSIFICATION_TARGETS:
        probs = np.array([r["predicted_probability"] for r in records])
        labels = np.array([r["actual_label"] for r in records])
        preds = (probs > 0.5).astype(int)

        metrics["accuracy"] = float(accuracy_score(labels, preds))
        metrics["brier_score"] = float(brier_score_loss(labels, probs))
        # precision/recall/F1/AUC are undefined with only one class present
        # in this sample -- report None rather than a misleading 0/1.
        if len(np.unique(labels)) > 1:
            metrics["precision"] = float(precision_score(labels, preds, zero_division=0))
            metrics["recall"] = float(recall_score(labels, preds, zero_division=0))
            metrics["f1"] = float(f1_score(labels, preds, zero_division=0))
            metrics["roc_auc"] = float(roc_auc_score(labels, probs))
            metrics["pr_auc"] = float(average_precision_score(labels, probs))
        else:
            metrics["precision"] = metrics["recall"] = metrics["f1"] = None
            metrics["roc_auc"] = metrics["pr_auc"] = None

        implied_directions = np.array([r["implied_direction"] for r in records])
        metrics["directional_accuracy"] = float(
            np.mean((implied_directions > 0) == (actual_returns > 0))
        )
    else:
        implied_returns = np.array([r["implied_return"] for r in records])
        errors = implied_returns - actual_returns
        metrics["return_mae"] = float(np.mean(np.abs(errors)))
        metrics["return_rmse"] = float(np.sqrt(np.mean(errors**2)))
        metrics["return_bias"] = float(np.mean(errors))
        metrics["directional_accuracy"] = float(
            np.mean((implied_returns > 0) == (actual_returns > 0))
        )

    return metrics


@dataclass
class QuantileResult:
    horizon_candles: int
    quantiles: tuple[float, ...]
    n_predictions: int
    coverage: dict[float, float]  # quantile -> fraction of actuals below the predicted quantile
    mean_interval_width: float  # mean(high_quantile - low_quantile), in return units
    records: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_candles": self.horizon_candles,
            "quantiles": list(self.quantiles),
            "n_predictions": self.n_predictions,
            "coverage": self.coverage,
            "mean_interval_width": self.mean_interval_width,
        }


def run_quantile_walk_forward(
    df: pd.DataFrame,
    horizon_candles: int,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    feature_cols: list[str] | None = None,
    n_folds: int = 2,
    min_train_size: int = 700,
    stride: int = 10,
    embargo: int | None = None,
) -> QuantileResult:
    """Walk-forward evaluate return-distribution (quantile) forecasting.

    Phase 3, Section 4 of the audit: instead of one point price, predict
    several quantiles of the return distribution (e.g. 10th/50th/90th
    percentile) and check **empirical coverage** — if the 10th-percentile
    prediction is honest, roughly 10% of actual returns should fall below
    it; if it's badly calibrated (e.g. 40% of actuals fall below the
    "10th percentile" prediction), the interval is not to be trusted even
    if the median forecast looks reasonable.

    Uses ``xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q)``
    — one independently-trained model per quantile, same features and
    same walk-forward discipline as :func:`run_formulation_walk_forward`.
    """
    pipeline = FeaturePipeline()
    clean_df = add_log_returns(clean_ohlcv(df))
    feature_df = pipeline.build(clean_df)
    cols = feature_cols or pipeline.feature_columns
    embargo = horizon_candles if embargo is None else embargo

    y_full, ret_full = build_target(feature_df, horizon_candles, "return")
    X_full = feature_df[cols].to_numpy(dtype=float)

    n = len(feature_df)
    remaining = n - min_train_size - horizon_candles
    if remaining <= 50:
        raise ValueError("Not enough data for quantile walk-forward.")
    fold_size = max(1, remaining // n_folds)

    records: list[dict[str, Any]] = []

    for fold in range(n_folds):
        train_end = min_train_size + fold * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n - horizon_candles)
        if test_end <= test_start or train_end >= n:
            break

        y_train = y_full[:train_end]
        X_train = X_full[:train_end]
        valid_train = np.isfinite(y_train)
        if valid_train.sum() < 20:
            continue

        models = {}
        for q in quantiles:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=q, **_XGB_COMMON
            )
            model.fit(X_train[valid_train], y_train[valid_train])
            models[q] = model

        for i in range(test_start, test_end, stride):
            if not np.isfinite(ret_full[i]):
                continue
            x_i = X_full[i : i + 1]
            predicted = {q: float(models[q].predict(x_i)[0]) for q in quantiles}
            records.append(
                {
                    "fold": fold,
                    "timestamp": feature_df.index[i],
                    "actual_return": float(ret_full[i]),
                    "predicted_quantiles": predicted,
                }
            )

    if not records:
        raise ValueError("Quantile walk-forward produced no evaluable predictions.")

    coverage = {}
    for q in quantiles:
        below = np.mean(
            [r["actual_return"] <= r["predicted_quantiles"][q] for r in records]
        )
        coverage[q] = float(below)

    low_q, high_q = min(quantiles), max(quantiles)
    widths = [r["predicted_quantiles"][high_q] - r["predicted_quantiles"][low_q] for r in records]

    return QuantileResult(
        horizon_candles=horizon_candles,
        quantiles=quantiles,
        n_predictions=len(records),
        coverage=coverage,
        mean_interval_width=float(np.mean(widths)),
        records=records,
    )
