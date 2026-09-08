"""Model manager and ForecastObject.

Provides a unified interface for generating predictions regardless
of which underlying AI model is selected.  The ``ModelManager``
routes requests to the appropriate model and returns a standardised
``ForecastObject`` that the rendering layer consumes.

Architecture
------------
This layer sits between the API route and the model implementations::

    GET /api/predict/{symbol}
           │
           ▼
    ModelManager.predict(symbol, timeframe, n_candles, model="auto")
           │
           ├── XGBoostForecaster   (Issue #1 — implemented)
           ├── LSTMForecaster      (future issue)
           ├── TransformerForecaster (future issue)
           └── EnsembleForecaster  (average of above)
           │
           ▼
    ForecastObject
           │
           ▼
    API response → Frontend chart renderer

Design principles
-----------------
* Open/Closed: adding a new model = adding a new class + one-line
  registration, with zero changes to existing code.
* Single Responsibility: ``ModelManager`` routes; individual
  forecasters predict; ``ForecastObject`` carries data.
* Dependency Inversion: the API layer depends on ``ModelManager``
  and ``ForecastObject``, not on ``XGBRegressor``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config.settings import HISTORY_DAYS, TIER1_SYMBOLS
from src.data.fetch_prices import CoinGeckoPriceRepository
from src.data.preprocessing import add_log_returns, clean_ohlcv
from src.features.feature_pipeline import FeaturePipeline
from src.models.xgboost_model import ForecastResult, TimeSeriesForecaster

logger = logging.getLogger(__name__)

# ── Mapping: Binance symbol → CoinGecko coin ID (for model training data) ──
_SYMBOL_TO_COINGECKO: dict[str, str] = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}

# ── Canonical horizon in calendar days per timeframe ───────────────────────
_TF_TO_DAYS: dict[str, int] = {
    "1m": 1,
    "5m": 1,
    "15m": 7,
    "30m": 7,
    "1H": 7,
    "4H": 30,
    "1D": 30,
    "1W": 30,
}

# Registered forecast horizons (days) — must be in FORECAST_HORIZONS
_HORIZON_MAP: dict[int, int] = {1: 1, 7: 7, 30: 30}


# ─────────────────────────────────────────────────────────────────────────────
# Data objects
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BandPoint:
    """A single (time, value) point on a confidence band."""

    time: int  # Unix seconds
    value: float


@dataclass(frozen=True)
class ForecastMetadata:
    """Structured prediction metadata for display in the UI."""

    direction: str          # "bullish" | "bearish"
    confidence: float       # 0.0–1.0
    expected_price: float
    expected_change_pct: float
    volatility_est: float
    model_name: str
    generated_at: str       # ISO-8601 timestamp


@dataclass
class ForecastObject:
    """The canonical data container for an AI price forecast.

    The rendering layer consumes this object via ``ForecastRenderer``.
    It knows nothing about XGBoost, LSTM, or any other model — it is
    pure data.

    Attributes:
        symbol:         Trading pair (e.g. ``"BTCUSDT"``).
        timeframe:      Candle period string (e.g. ``"1H"``).
        n_candles:      Number of projected future candles.
        future_prices:  Central predicted price at each future candle.
        upper_band:     Upper confidence interval prices.
        lower_band:     Lower confidence interval prices.
        timestamps:     Unix timestamps (seconds) for each projected point.
        metadata:       Structured metadata for UI display.
    """

    symbol: str
    timeframe: str
    n_candles: int
    future_prices: list[float]
    upper_band: list[float]
    lower_band: list[float]
    timestamps: list[int]
    metadata: ForecastMetadata
    anchor_time: int = 0        # timestamp of last historical candle
    anchor_price: float = 0.0   # close price of last historical candle
    prediction_id: str | None = None
    model_version: str = "1.0.0"
    historical_candles: list[dict] = field(default_factory=list)
    technical_indicators: dict = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    risk_level: str = "medium"
    market_regime: str = "sideways"
    reasoning_summary: str = ""

    def to_api_dict(self) -> dict:
        """Serialise to a JSON-serialisable dict for the API response."""
        candles = [
            {"time": t, "value": float(v)}
            for t, v in zip(self.timestamps, self.future_prices)
        ]
        upper = [
            {"time": t, "value": float(v)}
            for t, v in zip(self.timestamps, self.upper_band)
        ]
        lower = [
            {"time": t, "value": float(v)}
            for t, v in zip(self.timestamps, self.lower_band)
        ]
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_candles": self.n_candles,
            "prediction_id": self.prediction_id,
            "model_version": self.model_version,
            "anchor_time": self.anchor_time,
            "anchor_price": self.anchor_price,
            "candles": candles,
            "upper_band": upper,
            "lower_band": lower,
            "direction": self.metadata.direction,
            "confidence": self.metadata.confidence,
            "expected_price": self.metadata.expected_price,
            "expected_change_pct": self.metadata.expected_change_pct,
            "volatility_est": self.metadata.volatility_est,
            "model_name": self.metadata.model_name,
            "generated_at": self.metadata.generated_at,
            "risk_level": self.risk_level,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "market_regime": self.market_regime,
            "reasoning_summary": self.reasoning_summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Model Manager
# ─────────────────────────────────────────────────────────────────────────────


class ModelManager:
    """Routes prediction requests to the appropriate AI model.

    Currently wraps ``TimeSeriesForecaster`` (XGBoost).  Future models
    (LSTM, Transformer, TFT, LightGBM, Ensemble) are registered in
    ``_MODEL_REGISTRY`` with zero changes to this class.

    Args:
        default_model: Model name to use when ``model="auto"`` is
                       requested. Defaults to ``"xgboost"``.
    """

    # Registry: name → builder function (lazy to avoid loading all models)
    # Note: "auto" resolves to default_model at runtime.
    _MODEL_REGISTRY: dict[str, str] = {
        "xgboost": "xgboost",
        "auto": "xgboost",  # Auto currently selects XGBoost
        # "lstm":        "lstm",        # future
        # "transformer": "transformer", # future
        # "ensemble":    "ensemble",    # future
    }

    def __init__(self, default_model: str = "xgboost") -> None:
        self._default_model = default_model
        # Cache trained forecasters keyed by (coingecko_id, horizon_days)
        self._forecasters: dict[tuple[str, int], TimeSeriesForecaster] = {}
        self._pipeline = FeaturePipeline()

    @property
    def available_models(self) -> list[str]:
        """List of model names available for selection."""
        return list(self._MODEL_REGISTRY.keys())

    async def predict(
        self,
        symbol: str,
        timeframe: str,
        n_candles: int = 20,
        model: str = "auto",
    ) -> ForecastObject:
        """Generate a forecast and return a :class:`ForecastObject`.

        Args:
            symbol:    Trading pair (e.g. ``"BTCUSDT"``).
            timeframe: Candle period string.
            n_candles: Number of future candles to project.
            model:     Model selection key (``"auto"``, ``"xgboost"``, …).

        Returns:
            :class:`ForecastObject` with central prediction + confidence bands.

        Raises:
            ValueError: If *symbol* is not supported.
        """
        coin_id = _SYMBOL_TO_COINGECKO.get(symbol.upper())
        if coin_id is None:
            raise ValueError(
                f"Symbol '{symbol}' not supported. "
                f"Supported: {list(_SYMBOL_TO_COINGECKO.keys())}"
            )

        resolved_model = self._MODEL_REGISTRY.get(model, self._default_model)
        logger.info(
            "Generating %d-candle forecast for %s/%s using %s",
            n_candles,
            symbol,
            timeframe,
            resolved_model,
        )

        # Map timeframe → daily horizon for the model
        horizon_days = _TF_TO_DAYS.get(timeframe, 1)
        # Clamp to supported horizon
        if horizon_days not in _HORIZON_MAP:
            horizon_days = min(
                _HORIZON_MAP.keys(), key=lambda h: abs(h - horizon_days)
            )

        # Fetch data + build features
        repo = CoinGeckoPriceRepository()
        raw_df = repo.fetch(coin_id, HISTORY_DAYS)
        clean_df = add_log_returns(clean_ohlcv(raw_df))
        feature_df = self._pipeline.build(clean_df)

        feature_cols = self._pipeline.feature_columns
        X = feature_df[feature_cols].values
        y = feature_df["close"].shift(-horizon_days).dropna().values
        X_train = X[: len(y)]

        # Get or train forecaster
        cache_key = (coin_id, horizon_days)
        if cache_key not in self._forecasters:
            forecaster = TimeSeriesForecaster(horizon=horizon_days)
            forecaster.train(
                X_train,
                y,
                feature_names=feature_cols,
                eval_fraction=0.1,
            )
            self._forecasters[cache_key] = forecaster
        else:
            forecaster = self._forecasters[cache_key]

        # Current price (last close)
        current_price = float(feature_df["close"].iloc[-1])
        current_time = int(feature_df.index[-1].timestamp())

        # Predicted price at horizon
        X_latest = X[-1:].reshape(1, -1)
        predicted_price = float(forecaster.predict_latest(X_latest))

        # Build projected future prices (linear interpolation to predicted)
        future_prices = _interpolate_prices(
            start=current_price,
            end=predicted_price,
            n=n_candles,
        )

        # Estimate volatility from recent log returns
        vol = float(feature_df.get("volatility_14", pd.Series([0.02])).iloc[-1])
        vol = max(vol, 0.005)  # floor at 0.5% to avoid zero bands

        # Build confidence bands (widens with time)
        upper_band = [
            p * (1 + vol * np.sqrt(i + 1))
            for i, p in enumerate(future_prices)
        ]
        lower_band = [
            p * (1 - vol * np.sqrt(i + 1))
            for i, p in enumerate(future_prices)
        ]

        # Compute timestamps for projected candles
        interval_seconds = _timeframe_to_seconds(timeframe)
        timestamps = [
            current_time + (i + 1) * interval_seconds
            for i in range(n_candles)
        ]

        # Derive direction + confidence
        change_pct = (predicted_price - current_price) / current_price * 100
        direction = "bullish" if change_pct >= 0 else "bearish"

        # Proxy confidence from band width at horizon
        band_width_pct = (upper_band[-1] - lower_band[-1]) / predicted_price
        confidence = max(0.0, min(1.0, 1.0 - band_width_pct * 5))
        support_levels = _support_levels(feature_df["low"].tail(60).tolist())
        resistance_levels = _resistance_levels(feature_df["high"].tail(60).tolist())
        risk_level = _risk_level(vol)
        market_regime = _market_regime(feature_df)
        recent_candles = [
            {
                "time": int(idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            for idx, row in clean_df.tail(120).iterrows()
        ]
        latest_indicators = {
            col: float(feature_df[col].iloc[-1])
            for col in feature_cols
            if col in feature_df.columns and np.isfinite(feature_df[col].iloc[-1])
        }

        metadata = ForecastMetadata(
            direction=direction,
            confidence=round(confidence, 3),
            expected_price=round(predicted_price, 2),
            expected_change_pct=round(change_pct, 2),
            volatility_est=round(vol, 4),
            model_name=resolved_model,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        return ForecastObject(
            symbol=symbol.upper(),
            timeframe=timeframe,
            n_candles=n_candles,
            future_prices=[round(p, 2) for p in future_prices],
            upper_band=[round(p, 2) for p in upper_band],
            lower_band=[round(p, 2) for p in lower_band],
            timestamps=timestamps,
            metadata=metadata,
            anchor_time=current_time,
            anchor_price=current_price,
            model_version="1.0.0",
            historical_candles=recent_candles,
            technical_indicators=latest_indicators,
            feature_importance=forecaster.feature_importances,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            risk_level=risk_level,
            market_regime=market_regime,
            reasoning_summary=(
                f"{resolved_model} projects a {direction} move over {n_candles} "
                f"{timeframe} candles with {confidence:.1%} confidence in a "
                f"{market_regime} regime."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _interpolate_prices(
    start: float, end: float, n: int
) -> list[float]:
    """Linear interpolation from *start* to *end* over *n* steps."""
    if n <= 1:
        return [end]
    step = (end - start) / n
    return [start + step * (i + 1) for i in range(n)]


def _timeframe_to_seconds(tf: str) -> int:
    """Convert a canonical timeframe string to a second count."""
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1H": 3600,
        "4H": 14400,
        "1D": 86400,
        "1W": 604800,
    }
    return mapping.get(tf, 3600)


def _support_levels(lows: list[float]) -> list[float]:
    """Return nearby support levels from recent lows."""
    if not lows:
        return []
    return sorted({round(float(v), 2) for v in np.quantile(lows, [0.1, 0.25, 0.4])})


def _resistance_levels(highs: list[float]) -> list[float]:
    """Return nearby resistance levels from recent highs."""
    if not highs:
        return []
    return sorted({round(float(v), 2) for v in np.quantile(highs, [0.6, 0.75, 0.9])})


def _risk_level(volatility: float) -> str:
    if volatility >= 0.08:
        return "high"
    if volatility >= 0.035:
        return "medium"
    return "low"


def _market_regime(feature_df: pd.DataFrame) -> str:
    recent = feature_df["close"].tail(30)
    if len(recent) < 2:
        return "sideways"
    change = (float(recent.iloc[-1]) - float(recent.iloc[0])) / float(recent.iloc[0])
    if change >= 0.05:
        return "bull"
    if change <= -0.05:
        return "bear"
    return "sideways"
