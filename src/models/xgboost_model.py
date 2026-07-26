"""XGBoost-based time-series forecaster using the Direct Strategy.

The **Direct Strategy** trains a separate regressor for each forecast
horizon:

    model_1d.predict(X)   → price 1 calendar day ahead
    model_7d.predict(X)   → price 7 calendar days ahead
    model_30d.predict(X)  → price 30 calendar days ahead

This avoids error accumulation that plagues multi-step recursive
approaches and keeps each model focused on a single objective
(Single Responsibility Principle).

Usage example
-------------
>>> from src.models.xgboost_model import TimeSeriesForecaster
>>> forecaster = TimeSeriesForecaster(horizon=7)
>>> forecaster.train(X_train, y_train)
>>> predictions = forecaster.predict(X_latest)
>>> forecaster.save("models/saved/BTC_7d.joblib")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config.settings import (
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_RANDOM_STATE,
    XGB_SUBSAMPLE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """Value object encapsulating a single-horizon forecast.

    Attributes:
        symbol:           CoinGecko coin ID (e.g. ``"bitcoin"``).
        horizon:          Forecast horizon in calendar days.
        predicted_price:  Point estimate of the future closing price (USD).
        mae:              Mean Absolute Error on the hold-out validation set.
        rmse:             Root Mean Squared Error on the hold-out set.
        mape:             Mean Absolute Percentage Error (%) on the hold-out set.
        feature_names:    Names of the features used during training.
    """

    symbol: str
    horizon: int
    predicted_price: float
    mae: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0
    feature_names: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ForecastResult(symbol={self.symbol!r}, horizon={self.horizon}d, "
            f"predicted_price={self.predicted_price:.2f}, "
            f"MAE={self.mae:.2f}, RMSE={self.rmse:.2f}, MAPE={self.mape:.2f}%)"
        )


class TimeSeriesForecaster:
    """Wraps an XGBRegressor for direct multi-step price forecasting.

    Args:
        horizon:    Forecast horizon in calendar days (1, 7, or 30).
        n_estimators:     Number of boosting rounds.
        max_depth:        Maximum tree depth.
        learning_rate:    Step size shrinkage.
        subsample:        Fraction of samples used per tree.
        colsample_bytree: Fraction of features used per tree.
        random_state:     Reproducibility seed.
        **kwargs:         Additional keyword arguments forwarded to
                          :class:`xgboost.XGBRegressor`.
    """

    def __init__(
        self,
        horizon: int,
        n_estimators: int = XGB_N_ESTIMATORS,
        max_depth: int = XGB_MAX_DEPTH,
        learning_rate: float = XGB_LEARNING_RATE,
        subsample: float = XGB_SUBSAMPLE,
        colsample_bytree: float = XGB_COLSAMPLE_BYTREE,
        random_state: int = XGB_RANDOM_STATE,
        **kwargs: Any,
    ) -> None:
        self.horizon = horizon
        self._feature_names: list[str] = []

        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
            **kwargs,
        )
        logger.info(
            "Initialised TimeSeriesForecaster (horizon=%dd, n_estimators=%d)",
            horizon,
            n_estimators,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        eval_fraction: float = 0.1,
    ) -> dict[str, float]:
        """Fit the XGBoost model on labelled training data.

        An optional hold-out split is evaluated to compute validation
        metrics (MAE, RMSE, MAPE).

        Args:
            X:              Feature matrix of shape ``(n_samples, n_features)``.
            y:              Target vector of shape ``(n_samples,)`` — the
                            closing price *horizon* days ahead.
            feature_names:  Column names corresponding to *X* columns.
            eval_fraction:  Fraction of *X* reserved for hold-out evaluation.
                            Set to ``0`` to skip validation.

        Returns:
            Dictionary with validation metrics ``{mae, rmse, mape}``.
        """
        if feature_names is not None:
            self._feature_names = list(feature_names)

        n = len(X)
        if eval_fraction > 0 and n > 10:
            split = max(1, int(n * (1 - eval_fraction)))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            X_train, X_val = X, None
            y_train, y_val = y, None

        logger.info(
            "Training %dd forecaster on %d samples.", self.horizon, len(X_train)
        )
        self._model.fit(X_train, y_train)

        metrics: dict[str, float] = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
        if X_val is not None and len(X_val) > 0:
            y_pred_val = self._model.predict(X_val)
            metrics["mae"] = float(mean_absolute_error(y_val, y_pred_val))
            metrics["rmse"] = float(
                np.sqrt(mean_squared_error(y_val, y_pred_val))
            )
            # MAPE — guard against zero actuals
            mask = y_val != 0
            if mask.any():
                metrics["mape"] = float(
                    np.mean(np.abs((y_val[mask] - y_pred_val[mask]) / y_val[mask]))
                    * 100
                )
            logger.info(
                "Validation metrics (horizon=%dd): MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
                self.horizon,
                metrics["mae"],
                metrics["rmse"],
                metrics["mape"],
            )
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate price forecasts for the given feature matrix.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Predicted prices of shape ``(n_samples,)``.
        """
        return self._model.predict(X)

    def predict_latest(self, X: np.ndarray) -> float:
        """Return the scalar forecast for the **most recent** observation.

        Args:
            X: Feature matrix where the **last row** represents today's
               feature vector.

        Returns:
            Scalar predicted closing price (USD).
        """
        latest = X[-1:] if X.ndim == 2 else X.reshape(1, -1)
        return float(self._model.predict(latest)[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Serialise the fitted model to disk using joblib.

        Args:
            path: Destination file path (e.g. ``"models/saved/BTC_7d.joblib"``).

        Returns:
            The resolved path that was written.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self._model, "feature_names": self._feature_names}, p)
        logger.info("Saved %dd forecaster to '%s'.", self.horizon, p)
        return p

    @classmethod
    def load(cls, path: str | Path, horizon: int) -> "TimeSeriesForecaster":
        """Deserialise a previously saved forecaster from disk.

        Args:
            path:    Path to a joblib file created by :meth:`save`.
            horizon: Forecast horizon (must match the saved model's horizon).

        Returns:
            A fully initialised :class:`TimeSeriesForecaster` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        data = joblib.load(p)
        instance = cls.__new__(cls)
        instance.horizon = horizon
        instance._model = data["model"]
        instance._feature_names = data.get("feature_names", [])
        logger.info("Loaded %dd forecaster from '%s'.", horizon, p)
        return instance

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def feature_importances(self) -> dict[str, float]:
        """Return feature importances keyed by feature name (if available).

        Returns:
            Dictionary mapping feature name → importance score,
            or an empty dict if the model has not been trained.
        """
        try:
            importances = self._model.feature_importances_
        except AttributeError:
            return {}

        if self._feature_names:
            return dict(zip(self._feature_names, importances.tolist()))
        return {f"f{i}": float(v) for i, v in enumerate(importances)}

    def __repr__(self) -> str:
        return (
            f"TimeSeriesForecaster(horizon={self.horizon}d, "
            f"n_estimators={self._model.n_estimators})"
        )
