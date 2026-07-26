"""End-to-end data fetch and model training pipeline for Issue #1.

This module wires together all the Clean Architecture layers:

    Data (fetch → preprocess) → Features → Model (train → forecast)

It is designed to be called from scripts, scheduled jobs, or tests and
returns structured :class:`~src.models.xgboost_model.ForecastResult`
objects rather than writing directly to stdout.

Usage
-----
From a Python shell or notebook::

    from src.data.fetch_market_data import run_pipeline
    result = run_pipeline("bitcoin", horizon=7)
    print(result)

From the CLI::

    python -m src.data.fetch_market_data
"""

from __future__ import annotations

import numpy as np

from src.config.settings import (
    DATA_DIR,
    FORECAST_HORIZONS,
    HISTORY_DAYS,
    MODEL_DIR,
    TIER1_SYMBOLS,
)
from src.data.fetch_prices import CoinGeckoPriceRepository, PriceRepository
from src.data.preprocessing import add_log_returns, clean_ohlcv
from src.features.feature_pipeline import FeaturePipeline
from src.models.xgboost_model import ForecastResult, TimeSeriesForecaster
from src.utils.helpers import save_parquet
from src.utils.logger import get_logger
from src.utils.validators import validate_horizon, validate_symbol

logger = get_logger(__name__)


def run_pipeline(
    symbol: str,
    horizon: int,
    history_days: int = HISTORY_DAYS,
    repository: PriceRepository | None = None,
    save_data: bool = True,
) -> ForecastResult:
    """Run the full fetch → preprocess → feature → train → forecast pipeline.

    Args:
        symbol:       CoinGecko coin ID (e.g. ``"bitcoin"``).
        horizon:      Forecast horizon in calendar days (1, 7, or 30).
        history_days: Number of trailing days of price history to fetch.
        repository:   Inject an alternative :class:`PriceRepository` for
                      testing without network access.
        save_data:    If True, persist the raw OHLCV DataFrame as a Parquet
                      file and the trained model to disk.

    Returns:
        A :class:`~src.models.xgboost_model.ForecastResult` with the
        predicted price and validation metrics.
    """
    validate_symbol(symbol)
    validate_horizon(horizon)

    repo = repository or CoinGeckoPriceRepository()

    # ------------------------------------------------------------------
    # 1. Fetch
    # ------------------------------------------------------------------
    logger.info(
        "[%s/%dd] Fetching %d days of price data.",
        symbol,
        horizon,
        history_days,
    )
    raw_df = repo.fetch(symbol, history_days)

    if save_data:
        parquet_path = DATA_DIR / f"{symbol}_ohlcv.parquet"
        save_parquet(raw_df, parquet_path)
        logger.info("Saved raw OHLCV to '%s'.", parquet_path)

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    clean_df = clean_ohlcv(raw_df)
    clean_df = add_log_returns(clean_df)

    # ------------------------------------------------------------------
    # 3. Build features
    # ------------------------------------------------------------------
    pipeline = FeaturePipeline()
    feature_df = pipeline.build(clean_df)

    feature_cols = pipeline.feature_columns
    # Keep only features that exist in the built frame (lag features may
    # have been dropped during NaN elimination).
    available_feature_cols = [c for c in feature_cols if c in feature_df.columns]

    # ------------------------------------------------------------------
    # 4. Create supervised learning targets
    #    y[t] = close price *horizon* days after row t
    # ------------------------------------------------------------------
    target_col = "close"
    feature_df = feature_df.copy()
    feature_df["target"] = feature_df[target_col].shift(-horizon)

    # Drop the final *horizon* rows where target is NaN
    labelled_df = feature_df.dropna(subset=["target"])

    X: np.ndarray = labelled_df[available_feature_cols].values
    y: np.ndarray = labelled_df["target"].values

    logger.info(
        "[%s/%dd] Training on %d samples with %d features.",
        symbol,
        horizon,
        len(X),
        X.shape[1],
    )

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    forecaster = TimeSeriesForecaster(horizon=horizon)
    metrics = forecaster.train(X, y, feature_names=available_feature_cols)

    if save_data:
        model_path = MODEL_DIR / f"{symbol}_{horizon}d.joblib"
        forecaster.save(model_path)

    # ------------------------------------------------------------------
    # 6. Forecast on the latest available row
    # ------------------------------------------------------------------
    # The latest row in feature_df (before shifting) gives us "today's"
    # feature vector to forecast the future price.
    latest_features = feature_df[available_feature_cols].iloc[-1:].values
    predicted_price = float(forecaster.predict(latest_features)[0])

    result = ForecastResult(
        symbol=symbol,
        horizon=horizon,
        predicted_price=predicted_price,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        mape=metrics["mape"],
        feature_names=available_feature_cols,
    )

    logger.info("[%s/%dd] %r", symbol, horizon, result)
    return result


def run_all(
    symbols: list[str] | None = None,
    horizons: list[int] | None = None,
) -> list[ForecastResult]:
    """Run the pipeline for all *symbols* × *horizons* combinations.

    Args:
        symbols:  List of CoinGecko coin IDs. Defaults to
                  :data:`~src.config.settings.TIER1_SYMBOLS`.
        horizons: List of forecast horizons. Defaults to
                  :data:`~src.config.settings.FORECAST_HORIZONS`.

    Returns:
        List of :class:`~src.models.xgboost_model.ForecastResult` objects,
        one per (symbol, horizon) pair.
    """
    symbols = symbols or TIER1_SYMBOLS
    horizons = horizons or FORECAST_HORIZONS
    results: list[ForecastResult] = []

    for symbol in symbols:
        for horizon in horizons:
            try:
                result = run_pipeline(symbol, horizon)
                results.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Pipeline failed for %s/%dd: %s", symbol, horizon, exc
                )

    return results


if __name__ == "__main__":
    results = run_all(symbols=["bitcoin", "ethereum"])
    for r in results:
        print(r)
