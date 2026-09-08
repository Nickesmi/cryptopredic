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

Horizon semantics (read this before changing anything below)
--------------------------------------------------------------
The forecast horizon is **always** ``n_candles`` candles of the
**same timeframe** the caller requested, and the training target is
built on the **same candle series** that is displayed on the chart:

    target(t, n_candles) = close price n_candles * timeframe ahead of t

There used to be a second, hidden notion of "horizon" here
(``_TF_TO_DAYS`` / ``_HORIZON_MAP``) that silently retrained the model
to predict a price 1/7/30 *calendar days* ahead regardless of the
timeframe or ``n_candles`` the caller asked for, while the API still
labelled/expired the prediction as if it were an ``n_candles``-ahead
forecast at the requested timeframe. That is precisely why the system
could say "BTC will reach $X within 4 hours" (a 1H-timeframe, small
``n_candles`` request) while the model had actually been trained to
answer "what is BTC's price 30 days from now" — the two were
unrelated. See ``docs/TIMING_AUDIT_REPORT.md`` for the full writeup.
Do not reintroduce a timeframe -> horizon lookup table.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config.settings import (
    XGB_COLSAMPLE_BYTREE,
    XGB_LEARNING_RATE,
    XGB_MAX_DEPTH,
    XGB_N_ESTIMATORS,
    XGB_RANDOM_STATE,
    XGB_SUBSAMPLE,
)
from src.data.exchange.factory import ExchangeFactory
from src.data.preprocessing import add_log_returns, clean_ohlcv
from src.data.quality import enforce_quality_gate
from src.features.feature_pipeline import FeaturePipeline
from src.models.xgboost_model import ForecastResult, TimeSeriesForecaster
from src.utils.candles import bars_to_frame
from src.utils.regime import classify_regime
from src.utils.timeframes import timeframe_to_seconds

logger = logging.getLogger(__name__)

# ── Supported trading pairs ────────────────────────────────────────────────
_SUPPORTED_SYMBOLS: set[str] = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}

# Minimum number of *post-feature-engineering* rows required so that at
# least a handful of training samples remain once the horizon shift removes
# the trailing rows. This only guards against degenerate windows (e.g. a
# handful of candles); it intentionally does not impose a large minimum so
# that short backtest lookback windows remain usable.
_MIN_FEATURE_ROWS_OVER_HORIZON = 5


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
        n_candles:      Forecast horizon, expressed as a count of
                        ``timeframe`` candles. ``future_prices[-1]`` is
                        the model's actual prediction for
                        ``anchor_time + n_candles * timeframe``; every
                        other point on ``future_prices`` is an
                        interpolated visual guide, not an independent
                        model output (see ``path_is_interpolated``).
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
    target_timestamp: int = 0   # timestamp the model's prediction is actually for
    horizon_seconds: int = 0    # anchor_time -> target_timestamp, in seconds
    path_is_interpolated: bool = True
    prediction_id: str | None = None
    model_version: str = "1.0.0"
    feature_version: str = ""
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
            "feature_version": self.feature_version,
            "anchor_time": self.anchor_time,
            "anchor_price": self.anchor_price,
            "target_timestamp": self.target_timestamp,
            "horizon_seconds": self.horizon_seconds,
            "path_is_interpolated": self.path_is_interpolated,
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
        # Cache trained forecasters keyed by (symbol, timeframe, n_candles) —
        # each distinct (timeframe, horizon) combination gets its own model.
        # (Previously keyed by (coingecko_id, horizon_days), which meant two
        # *different* chart timeframes that happened to map to the same
        # bucketed horizon-in-days silently shared one cached model.)
        self._forecasters: dict[tuple[str, str, int], TimeSeriesForecaster] = {}
        # Provenance fingerprint for each cached forecaster -- see
        # _compute_model_version(). Kept alongside rather than bolted onto
        # TimeSeriesForecaster so the model class itself stays unaware of
        # ModelManager-level bookkeeping.
        self._model_versions: dict[tuple[str, str, int], str] = {}
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
        http_session=None,
    ) -> ForecastObject:
        """Generate a forecast and return a :class:`ForecastObject`.

        Fetches OHLCV candles at the *requested* timeframe from the same
        exchange (Binance) that powers the live chart, so the model is
        trained and evaluated on exactly the data the user sees — no
        second, mismatched data source and no hidden horizon remapping.

        Args:
            symbol:      Trading pair (e.g. ``"BTCUSDT"``).
            timeframe:   Candle period string.
            n_candles:   Forecast horizon, in candles of *timeframe*.
            model:       Model selection key (``"auto"``, ``"xgboost"``, …).
            http_session: Optional shared ``aiohttp.ClientSession``.

        Returns:
            :class:`ForecastObject` with central prediction + confidence bands.

        Raises:
            ValueError: If *symbol* is not supported.
        """
        symbol = symbol.upper()
        if symbol not in _SUPPORTED_SYMBOLS:
            raise ValueError(
                f"Symbol '{symbol}' not supported. Supported: {sorted(_SUPPORTED_SYMBOLS)}"
            )

        resolved_model = self._MODEL_REGISTRY.get(model, self._default_model)
        logger.info(
            "Generating %d-candle (%s) forecast for %s using %s",
            n_candles,
            timeframe,
            symbol,
            resolved_model,
        )

        adapter = ExchangeFactory.create("binance", session=http_session)
        bars = await adapter.get_candles(
            symbol, timeframe, limit=1000
        )
        # bars_to_frame() drops a still-forming trailing candle by default —
        # see its docstring. The anchor must always be the most recent
        # *closed* candle (timing-audit Phase 5): predicting from a close
        # price that is still changing tick-by-tick, or — for higher
        # timeframes — from an unclosed candle, is exactly the hazard that
        # section warns about.
        raw_df = bars_to_frame(bars)

        # A bad data point must never silently become a prediction: reject
        # duplicate/out-of-order/impossible-OHLC/stale data before it ever
        # reaches feature engineering. Only applied to the live path — the
        # backtester feeds already-known-good historical frames and must
        # stay deterministic with respect to wall-clock time, so it calls
        # predict_from_frame() directly rather than through here.
        enforce_quality_gate(raw_df, timeframe)

        return self.predict_from_frame(
            symbol=symbol,
            timeframe=timeframe,
            raw_df=raw_df,
            n_candles=n_candles,
            model=model,
        )

    def predict_from_frame(
        self,
        symbol: str,
        timeframe: str,
        raw_df: pd.DataFrame,
        n_candles: int = 20,
        model: str = "auto",
    ) -> ForecastObject:
        """Synchronous core of :meth:`predict`, operating on an in-memory OHLCV frame.

        This is the single code path used by both live inference
        (:meth:`predict`) and the backtester (``src/evaluation/backtest.py``).
        Using one function for both guarantees that feature engineering,
        target construction, and horizon semantics can never drift apart
        between training/backtesting and live inference.

        Args:
            symbol:     Trading pair, already validated/upper-cased by the caller.
            timeframe:  Candle period string.
            raw_df:     OHLCV DataFrame indexed by UTC timestamp, ascending,
                        columns ``[open, high, low, close, volume]``. Must
                        contain only data known as of the last row (the
                        caller is responsible for not leaking future rows
                        during backtesting).
            n_candles:  Forecast horizon, in candles of *timeframe*.
            model:      Model selection key.

        Returns:
            :class:`ForecastObject` with central prediction + confidence bands.
        """
        if n_candles < 1:
            raise ValueError("n_candles must be >= 1")

        resolved_model = self._MODEL_REGISTRY.get(model, self._default_model)
        interval_seconds = timeframe_to_seconds(timeframe)
        horizon_steps = n_candles

        clean_df = add_log_returns(clean_ohlcv(raw_df))
        feature_df = self._pipeline.build(clean_df)

        min_rows = horizon_steps + _MIN_FEATURE_ROWS_OVER_HORIZON
        if len(feature_df) < min_rows:
            raise ValueError(
                f"Insufficient history for {symbol}/{timeframe}: "
                f"{len(feature_df)} usable candles after feature engineering, "
                f"need at least {min_rows} (horizon={horizon_steps})."
            )

        feature_cols = self._pipeline.feature_columns
        X = feature_df[feature_cols].values
        y = feature_df["close"].shift(-horizon_steps).dropna().values
        X_train = X[: len(y)]

        # Get or train forecaster — cache key includes timeframe so a
        # 1H-horizon model can never be silently reused for a 4H request.
        cache_key = (symbol, timeframe, horizon_steps)
        if cache_key not in self._forecasters:
            forecaster = TimeSeriesForecaster(horizon=horizon_steps)
            forecaster.train(
                X_train,
                y,
                feature_names=feature_cols,
                eval_fraction=0.1,
            )
            self._forecasters[cache_key] = forecaster
            self._model_versions[cache_key] = _compute_model_version(
                feature_version=self._pipeline.feature_version,
                symbol=symbol,
                timeframe=timeframe,
                horizon_steps=horizon_steps,
                n_train_rows=len(X_train),
                train_start=feature_df.index[0],
                train_end=feature_df.index[len(X_train) - 1] if len(X_train) else feature_df.index[0],
            )
        else:
            forecaster = self._forecasters[cache_key]
        model_version = self._model_versions[cache_key]

        # Current price (last close)
        current_price = float(feature_df["close"].iloc[-1])
        current_time = int(feature_df.index[-1].timestamp())
        target_timestamp = current_time + horizon_steps * interval_seconds

        # Predicted price at horizon
        X_latest = X[-1:].reshape(1, -1)
        predicted_price = float(forecaster.predict_latest(X_latest))

        # Build projected future prices (linear interpolation to predicted).
        # Only the terminal point (index n_candles-1) is the model's actual
        # output — see ForecastObject.path_is_interpolated.
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
        timestamps = [
            current_time + (i + 1) * interval_seconds
            for i in range(n_candles)
        ]

        # Derive direction + confidence
        change_pct = (predicted_price - current_price) / current_price * 100
        direction = "bullish" if change_pct >= 0 else "bearish"

        # Proxy confidence from band width at horizon. NOTE: this is a
        # volatility-derived heuristic, not an empirically calibrated
        # probability — see src/evaluation/metrics.calibration_curve() to
        # check how well it tracks actual hit-rates before trusting it.
        band_width_pct = (upper_band[-1] - lower_band[-1]) / predicted_price
        confidence = max(0.0, min(1.0, 1.0 - band_width_pct * 5))
        support_levels = _support_levels(feature_df["low"].tail(60).tolist())
        resistance_levels = _resistance_levels(feature_df["high"].tail(60).tolist())
        risk_level = _risk_level(vol)
        market_regime = classify_regime(feature_df["close"])
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
            target_timestamp=target_timestamp,
            horizon_seconds=horizon_steps * interval_seconds,
            path_is_interpolated=n_candles > 1,
            model_version=model_version,
            feature_version=self._pipeline.feature_version,
            historical_candles=recent_candles,
            technical_indicators=latest_indicators,
            feature_importance=forecaster.feature_importances,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            risk_level=risk_level,
            market_regime=market_regime,
            reasoning_summary=(
                f"{resolved_model} projects a {direction} move over {n_candles} "
                f"{timeframe} candles ({horizon_steps * interval_seconds / 3600:.1f}h) "
                f"with {confidence:.1%} confidence in a {market_regime} regime."
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_model_version(
    feature_version: str,
    symbol: str,
    timeframe: str,
    horizon_steps: int,
    n_train_rows: int,
    train_start,
    train_end,
) -> str:
    """Deterministic provenance fingerprint for one trained forecaster instance.

    Two forecasters get the same ``model_version`` if and only if they were
    trained on the same feature-pipeline shape, the same (symbol,
    timeframe, horizon), the same number of rows, and the same training
    window -- i.e. they are, for all practical purposes, the same model.
    This is what "model versions are recorded" (Phase 17 / Phase-2
    production-safety checklist) actually requires: something that changes
    when the model changes and stays fixed when it doesn't, not a
    hand-maintained semantic-version string that never moves.
    """
    payload = {
        "feature_version": feature_version,
        "symbol": symbol,
        "timeframe": timeframe,
        "horizon_steps": horizon_steps,
        "n_train_rows": n_train_rows,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "hyperparams": {
            "n_estimators": XGB_N_ESTIMATORS,
            "max_depth": XGB_MAX_DEPTH,
            "learning_rate": XGB_LEARNING_RATE,
            "subsample": XGB_SUBSAMPLE,
            "colsample_bytree": XGB_COLSAMPLE_BYTREE,
            "random_state": XGB_RANDOM_STATE,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:12]


def _interpolate_prices(
    start: float, end: float, n: int
) -> list[float]:
    """Linear interpolation from *start* to *end* over *n* steps."""
    if n <= 1:
        return [end]
    step = (end - start) / n
    return [start + step * (i + 1) for i in range(n)]


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


