"""Unit tests for src/features/technical_indicators.py and FeaturePipeline.

All tests use synthetic DataFrames with deterministic values so they
run offline (no network required) and are fast (<1 s total).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FeaturePipeline
from src.features.technical_indicators import (
    add_bollinger_bands,
    add_ema,
    add_lag_features,
    add_macd,
    add_rolling_volatility,
    add_rsi,
    add_sma,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_ohlcv(n: int = 60, start_price: float = 100.0) -> pd.DataFrame:
    """Return a synthetic OHLCV DataFrame of length *n* with a linear trend."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = np.linspace(start_price, start_price * 1.5, n)
    df = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.default_rng(42)
            .integers(1_000, 10_000, n)
            .astype(float),
        },
        index=idx,
    )
    return df


@pytest.fixture()
def ohlcv_df() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture()
def ohlcv_with_returns(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame with log_return column pre-computed."""
    df = ohlcv_df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


# ---------------------------------------------------------------------------
# add_sma
# ---------------------------------------------------------------------------


class TestAddSma:
    def test_adds_expected_columns(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_sma(ohlcv_df, windows=[7, 14])
        assert "sma_7" in result.columns
        assert "sma_14" in result.columns

    def test_does_not_mutate_input(self, ohlcv_df: pd.DataFrame) -> None:
        before = ohlcv_df.columns.tolist()
        add_sma(ohlcv_df, windows=[7])
        assert ohlcv_df.columns.tolist() == before

    def test_sma_constant_series(self) -> None:
        """SMA of a constant series equals that constant."""
        idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [50.0] * 20}, index=idx)
        result = add_sma(df, windows=[5])
        assert (result["sma_5"] == 50.0).all()

    def test_sma_values_monotone_for_linear_trend(self, ohlcv_df: pd.DataFrame) -> None:
        """SMA should be monotonically increasing when prices are linearly rising."""
        result = add_sma(ohlcv_df, windows=[7])
        sma = result["sma_7"].dropna()
        assert (sma.diff().dropna() >= 0).all()

    def test_no_nan_with_min_periods_1(self, ohlcv_df: pd.DataFrame) -> None:
        """With min_periods=1, SMA should not produce NaN values."""
        result = add_sma(ohlcv_df, windows=[30])
        assert result["sma_30"].isna().sum() == 0


# ---------------------------------------------------------------------------
# add_ema
# ---------------------------------------------------------------------------


class TestAddEma:
    def test_adds_expected_columns(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_ema(ohlcv_df, spans=[12, 26])
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns

    def test_ema_constant_series(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [100.0] * 20}, index=idx)
        result = add_ema(df, spans=[5])
        np.testing.assert_allclose(result["ema_5"].values, 100.0, rtol=1e-6)

    def test_does_not_mutate_input(self, ohlcv_df: pd.DataFrame) -> None:
        before_cols = ohlcv_df.columns.tolist()
        add_ema(ohlcv_df, spans=[12])
        assert ohlcv_df.columns.tolist() == before_cols


# ---------------------------------------------------------------------------
# add_rsi
# ---------------------------------------------------------------------------


class TestAddRsi:
    def test_adds_rsi_column(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_rsi(ohlcv_df, period=14)
        assert "rsi_14" in result.columns

    def test_rsi_range(self, ohlcv_df: pd.DataFrame) -> None:
        """RSI must always fall in [0, 100]."""
        result = add_rsi(ohlcv_df, period=14)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_monotone_uptrend(self) -> None:
        """Steadily rising prices should produce RSI > 50 once warmed up.

        Uses a longer (80-row) series to ensure the EWM warm-up period
        (14 rows for diff + 14 rows for avg_gain/avg_loss) completes well
        before the assertion window.
        """
        idx = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
        prices = np.linspace(100, 200, 80)
        df = pd.DataFrame({"close": prices}, index=idx)
        result = add_rsi(df, period=14)
        # Rows 40+ are comfortably past the ~28-row EWM warm-up.
        late_rsi = result["rsi_14"].iloc[40:].dropna()
        assert len(late_rsi) > 0, "No non-NaN RSI values after warm-up period"
        assert (late_rsi > 50).all()


# ---------------------------------------------------------------------------
# add_macd
# ---------------------------------------------------------------------------


class TestAddMacd:
    def test_adds_macd_columns(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_macd(ohlcv_df)
        for col in ["macd", "macd_signal", "macd_hist"]:
            assert col in result.columns

    def test_macd_hist_is_difference(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_macd(ohlcv_df)
        expected = result["macd"] - result["macd_signal"]
        pd.testing.assert_series_equal(result["macd_hist"], expected, check_names=False)

    def test_does_not_mutate_input(self, ohlcv_df: pd.DataFrame) -> None:
        before_cols = ohlcv_df.columns.tolist()
        add_macd(ohlcv_df)
        assert ohlcv_df.columns.tolist() == before_cols


# ---------------------------------------------------------------------------
# add_bollinger_bands
# ---------------------------------------------------------------------------


class TestAddBollingerBands:
    def test_adds_band_columns(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv_df)
        for col in ["bb_upper", "bb_mid", "bb_lower", "bb_width"]:
            assert col in result.columns

    def test_upper_above_lower(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv_df).dropna()
        assert len(result) > 0
        assert (result["bb_upper"] >= result["bb_lower"]).all()

    def test_mid_between_bands(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv_df).dropna()
        assert len(result) > 0
        assert (result["bb_upper"] >= result["bb_mid"]).all()
        assert (result["bb_mid"] >= result["bb_lower"]).all()

    def test_bb_width_nonnegative(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_bollinger_bands(ohlcv_df).dropna()
        assert len(result) > 0
        assert (result["bb_width"] >= 0).all()


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------


class TestAddLagFeatures:
    def test_adds_lag_columns(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_lag_features(ohlcv_df, lags=[1, 7])
        assert "close_lag_1" in result.columns
        assert "close_lag_7" in result.columns

    def test_lag_1_equals_shifted_close(self, ohlcv_df: pd.DataFrame) -> None:
        result = add_lag_features(ohlcv_df, lags=[1])
        expected = ohlcv_df["close"].shift(1)
        pd.testing.assert_series_equal(
            result["close_lag_1"], expected, check_names=False
        )

    def test_does_not_mutate_input(self, ohlcv_df: pd.DataFrame) -> None:
        before_cols = ohlcv_df.columns.tolist()
        add_lag_features(ohlcv_df, lags=[1])
        assert ohlcv_df.columns.tolist() == before_cols


# ---------------------------------------------------------------------------
# add_rolling_volatility
# ---------------------------------------------------------------------------


class TestAddRollingVolatility:
    def test_adds_volatility_columns(self, ohlcv_with_returns: pd.DataFrame) -> None:
        result = add_rolling_volatility(ohlcv_with_returns, windows=[7])
        assert "volatility_7" in result.columns

    def test_volatility_nonnegative(self, ohlcv_with_returns: pd.DataFrame) -> None:
        result = add_rolling_volatility(ohlcv_with_returns, windows=[7])
        assert (result["volatility_7"].dropna() >= 0).all()

    def test_constant_returns_zero_volatility(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        df = pd.DataFrame({"close": [100.0] * 30}, index=idx)
        df["log_return"] = 0.0
        result = add_rolling_volatility(df, windows=[7])
        # std of a constant series is 0
        assert (result["volatility_7"].dropna() == 0.0).all()

    def test_returns_original_df_if_no_log_return(self, ohlcv_df: pd.DataFrame) -> None:
        """Should silently skip if log_return column is absent."""
        result = add_rolling_volatility(ohlcv_df, windows=[7])
        assert "volatility_7" not in result.columns


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    def test_build_returns_dataframe(self, ohlcv_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline()
        result = pipeline.build(ohlcv_df)
        assert isinstance(result, pd.DataFrame)

    def test_build_no_nan_rows(self, ohlcv_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline()
        result = pipeline.build(ohlcv_df)
        assert result.isna().sum().sum() == 0

    def test_build_adds_expected_feature_columns(self, ohlcv_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline()
        result = pipeline.build(ohlcv_df)
        for col in pipeline.feature_columns:
            assert col in result.columns, f"Missing expected feature: {col}"

    def test_build_reduces_row_count_due_to_lag(self, ohlcv_df: pd.DataFrame) -> None:
        """NaN rows from large lags should be dropped."""
        pipeline = FeaturePipeline(lag_offsets=[1, 14])
        result = pipeline.build(ohlcv_df)
        assert len(result) < len(ohlcv_df)

    def test_feature_columns_property_consistent(self, ohlcv_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline()
        result = pipeline.build(ohlcv_df)
        for col in pipeline.feature_columns:
            assert col in result.columns

    def test_build_is_deterministic(self, ohlcv_df: pd.DataFrame) -> None:
        """Two calls with the same input should produce identical results."""
        pipeline = FeaturePipeline()
        r1 = pipeline.build(ohlcv_df)
        r2 = pipeline.build(ohlcv_df)
        pd.testing.assert_frame_equal(r1, r2)

    def test_custom_windows_reflected_in_columns(self, ohlcv_df: pd.DataFrame) -> None:
        pipeline = FeaturePipeline(sma_windows=[5], ema_spans=[10])
        result = pipeline.build(ohlcv_df)
        assert "sma_5" in result.columns
        assert "ema_10" in result.columns
        # Default windows should not appear
        assert "sma_7" not in result.columns
        assert "ema_12" not in result.columns
