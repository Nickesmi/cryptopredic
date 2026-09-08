"""Tests for real model/feature version provenance (Phase 17 / Phase-2
production-safety checklist: "model versions are recorded", "feature
versions are recorded")."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_pipeline import FeaturePipeline
from src.models.model_manager import ModelManager


def _synthetic_ohlcv(n: int = 400, freq="h") -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    t = np.arange(n)
    close = pd.Series(200 + 20 * np.sin(2 * np.pi * t / 30), index=dates, dtype=float)
    return pd.DataFrame(
        {"open": close - 0.1, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1000.0},
        index=dates,
    )


def test_feature_version_is_stable_for_same_config() -> None:
    a = FeaturePipeline().feature_version
    b = FeaturePipeline().feature_version
    assert a == b
    assert len(a) == 12


def test_feature_version_changes_when_config_changes() -> None:
    default_version = FeaturePipeline().feature_version
    changed_version = FeaturePipeline(sma_windows=[5, 10, 15]).feature_version
    assert default_version != changed_version


def test_forecast_has_non_placeholder_model_and_feature_version() -> None:
    manager = ModelManager()
    df = _synthetic_ohlcv()
    forecast = manager.predict_from_frame("BTCUSDT", "1H", df, n_candles=4)
    assert forecast.model_version != "1.0.0"
    assert len(forecast.model_version) == 12
    assert forecast.feature_version == manager._pipeline.feature_version


def test_same_training_window_yields_the_same_model_version() -> None:
    df = _synthetic_ohlcv()
    manager_a = ModelManager()
    manager_b = ModelManager()
    forecast_a = manager_a.predict_from_frame("BTCUSDT", "1H", df, n_candles=4)
    forecast_b = manager_b.predict_from_frame("BTCUSDT", "1H", df, n_candles=4)
    assert forecast_a.model_version == forecast_b.model_version


def test_different_training_window_yields_a_different_model_version() -> None:
    df = _synthetic_ohlcv(n=400)
    manager = ModelManager()
    forecast_full = manager.predict_from_frame("BTCUSDT", "1H", df, n_candles=4)

    manager2 = ModelManager()
    forecast_partial = manager2.predict_from_frame(
        "BTCUSDT", "1H", df.iloc[:300].copy(), n_candles=4
    )
    assert forecast_full.model_version != forecast_partial.model_version


def test_different_horizon_yields_a_different_model_version() -> None:
    df = _synthetic_ohlcv()
    manager = ModelManager()
    forecast_h4 = manager.predict_from_frame("BTCUSDT", "1H", df, n_candles=4)
    forecast_h7 = manager.predict_from_frame("BTCUSDT", "1H", df, n_candles=7)
    assert forecast_h4.model_version != forecast_h7.model_version
