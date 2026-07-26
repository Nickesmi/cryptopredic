"""Unit tests for src/models/xgboost_model.py.

Tests cover:
- Model initialisation and repr
- Training with and without validation split
- Prediction (batch and single latest row)
- Save / load round-trip
- ForecastResult value object
- feature_importances introspection
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.xgboost_model import ForecastResult, TimeSeriesForecaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synthetic_dataset(
    n_samples: int = 200,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) with y ~ linear combination of X + noise."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    coeffs = rng.standard_normal(n_features)
    y = X @ coeffs + rng.standard_normal(n_samples) * 0.5 + 100.0  # ~100 USD baseline
    return X, y.astype(np.float64)


FEATURE_NAMES = [f"feature_{i}" for i in range(10)]


# ---------------------------------------------------------------------------
# TimeSeriesForecaster — initialisation
# ---------------------------------------------------------------------------


class TestTimeSeriesForecasterInit:
    def test_horizon_stored(self) -> None:
        f = TimeSeriesForecaster(horizon=7)
        assert f.horizon == 7

    def test_repr(self) -> None:
        f = TimeSeriesForecaster(horizon=30)
        assert "30" in repr(f)
        assert "TimeSeriesForecaster" in repr(f)

    def test_custom_hyperparams(self) -> None:
        f = TimeSeriesForecaster(horizon=1, n_estimators=100, max_depth=3)
        assert f._model.n_estimators == 100
        assert f._model.max_depth == 3


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTimeSeriesForecasterTrain:
    def test_train_returns_metrics_dict(self) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=50)
        metrics = f.train(X, y, feature_names=FEATURE_NAMES)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "mape" in metrics

    def test_metrics_are_nonnegative(self) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=50)
        metrics = f.train(X, y)
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mape"] >= 0

    def test_train_without_eval_fraction(self) -> None:
        X, y = make_synthetic_dataset(n_samples=10)
        f = TimeSeriesForecaster(horizon=1, n_estimators=10)
        metrics = f.train(X, y, eval_fraction=0)
        # No validation set — metrics should be zero defaults
        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_feature_names_stored(self) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=10)
        f.train(X, y, feature_names=FEATURE_NAMES)
        assert f._feature_names == FEATURE_NAMES

    def test_rmse_geq_mae(self) -> None:
        """RMSE ≥ MAE by the QM-AM inequality."""
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=50)
        metrics = f.train(X, y)
        assert metrics["rmse"] >= metrics["mae"]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestTimeSeriesForecasterPredict:
    @pytest.fixture()
    def trained_forecaster(self) -> TimeSeriesForecaster:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=7, n_estimators=50)
        f.train(X, y)
        return f

    def test_predict_shape(self, trained_forecaster: TimeSeriesForecaster) -> None:
        X, _ = make_synthetic_dataset(n_samples=10)
        preds = trained_forecaster.predict(X)
        assert preds.shape == (10,)

    def test_predict_returns_numpy_array(
        self, trained_forecaster: TimeSeriesForecaster
    ) -> None:
        X, _ = make_synthetic_dataset(n_samples=5)
        preds = trained_forecaster.predict(X)
        assert isinstance(preds, np.ndarray)

    def test_predict_latest_returns_scalar(
        self, trained_forecaster: TimeSeriesForecaster
    ) -> None:
        X, _ = make_synthetic_dataset(n_samples=20)
        result = trained_forecaster.predict_latest(X)
        assert isinstance(result, float)

    def test_predict_latest_uses_last_row(
        self, trained_forecaster: TimeSeriesForecaster
    ) -> None:
        X, _ = make_synthetic_dataset(n_samples=20)
        scalar = trained_forecaster.predict_latest(X)
        batch = trained_forecaster.predict(X[-1:])
        assert scalar == pytest.approx(float(batch[0]))

    def test_predictions_finite(self, trained_forecaster: TimeSeriesForecaster) -> None:
        X, _ = make_synthetic_dataset(n_samples=10)
        preds = trained_forecaster.predict(X)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


class TestTimeSeriesForecasterPersistence:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=20)
        f.train(X, y)
        model_path = tmp_path / "model_1d.joblib"
        returned_path = f.save(model_path)
        assert returned_path.exists()

    def test_load_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TimeSeriesForecaster.load(tmp_path / "nonexistent.joblib", horizon=1)

    def test_roundtrip_predictions_identical(self, tmp_path: Path) -> None:
        X, y = make_synthetic_dataset()
        f_original = TimeSeriesForecaster(horizon=7, n_estimators=30)
        f_original.train(X, y, feature_names=FEATURE_NAMES)

        model_path = tmp_path / "model_7d.joblib"
        f_original.save(model_path)

        f_loaded = TimeSeriesForecaster.load(model_path, horizon=7)

        X_test, _ = make_synthetic_dataset(n_samples=10, seed=99)
        preds_original = f_original.predict(X_test)
        preds_loaded = f_loaded.predict(X_test)

        np.testing.assert_array_almost_equal(preds_original, preds_loaded)

    def test_roundtrip_feature_names_preserved(self, tmp_path: Path) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=10)
        f.train(X, y, feature_names=FEATURE_NAMES)

        model_path = tmp_path / "model.joblib"
        f.save(model_path)
        f_loaded = TimeSeriesForecaster.load(model_path, horizon=1)
        assert f_loaded._feature_names == FEATURE_NAMES

    def test_roundtrip_horizon_preserved(self, tmp_path: Path) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=30, n_estimators=10)
        f.train(X, y)
        model_path = tmp_path / "model_30d.joblib"
        f.save(model_path)
        f_loaded = TimeSeriesForecaster.load(model_path, horizon=30)
        assert f_loaded.horizon == 30


# ---------------------------------------------------------------------------
# feature_importances
# ---------------------------------------------------------------------------


class TestFeatureImportances:
    def test_importances_returned_after_training(self) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=50)
        f.train(X, y, feature_names=FEATURE_NAMES)
        importances = f.feature_importances
        assert len(importances) == len(FEATURE_NAMES)
        assert all(v >= 0 for v in importances.values())

    def test_importances_sum_to_one(self) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=50)
        f.train(X, y, feature_names=FEATURE_NAMES)
        total = sum(f.feature_importances.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_importances_keyed_by_feature_names(self) -> None:
        X, y = make_synthetic_dataset()
        f = TimeSeriesForecaster(horizon=1, n_estimators=20)
        f.train(X, y, feature_names=FEATURE_NAMES)
        for name in FEATURE_NAMES:
            assert name in f.feature_importances

    def test_importances_empty_before_training(self) -> None:
        f = TimeSeriesForecaster(horizon=1)
        assert f.feature_importances == {}


# ---------------------------------------------------------------------------
# ForecastResult value object
# ---------------------------------------------------------------------------


class TestForecastResult:
    def test_construction(self) -> None:
        r = ForecastResult(
            symbol="bitcoin",
            horizon=7,
            predicted_price=65_000.0,
            mae=500.0,
            rmse=700.0,
            mape=1.2,
        )
        assert r.symbol == "bitcoin"
        assert r.horizon == 7
        assert r.predicted_price == pytest.approx(65_000.0)

    def test_repr_contains_key_info(self) -> None:
        r = ForecastResult(
            symbol="ethereum", horizon=30, predicted_price=3500.0
        )
        text = repr(r)
        assert "ethereum" in text
        assert "30" in text
        assert "3500" in text

    def test_default_metrics_zero(self) -> None:
        r = ForecastResult(symbol="bitcoin", horizon=1, predicted_price=50_000.0)
        assert r.mae == 0.0
        assert r.rmse == 0.0
        assert r.mape == 0.0

    def test_feature_names_default_empty(self) -> None:
        r = ForecastResult(symbol="bitcoin", horizon=1, predicted_price=50_000.0)
        assert r.feature_names == []
