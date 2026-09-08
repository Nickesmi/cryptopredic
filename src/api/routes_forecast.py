"""Forecast, candle history, and WebSocket streaming routes.

Endpoints
---------
GET  /api/candles/{symbol}            Historical candles (REST)
WS   /api/ws/candles/{symbol}         Live candle stream (WebSocket)
GET  /api/predict/{symbol}            AI forecast with confidence bands
GET  /api/models                      List available AI models
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from src.data.exchange.factory import ExchangeFactory
from src.evaluation.prediction_store import PredictionStore
from src.models.model_manager import ModelManager
from src.utils.timeframes import timeframe_to_seconds

logger = logging.getLogger(__name__)

router = APIRouter(tags=["forecast"])

# Supported symbols and timeframes (validated on every request)
_SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
_SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W"}
_SUPPORTED_HORIZONS = {5, 10, 20, 50}

# Shared ModelManager instance (one per process — forecasters cached inside)
_model_manager = ModelManager()
_prediction_store = PredictionStore()


# ─────────────────────────────────────────────────────────────────────────────
# REST — Historical candles
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/candles/{symbol}")
async def get_candles(
    request: Request,
    symbol: str,
    tf: Annotated[str, Query(description="Timeframe: 1m|5m|15m|30m|1H|4H|1D|1W")] = "1H",
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> JSONResponse:
    """Return historical OHLCV candles for *symbol*.

    Args:
        symbol: Trading pair (e.g. ``BTCUSDT``).
        tf:     Candle timeframe.
        limit:  Number of candles to return (1–1000).

    Returns:
        JSON array of candle objects with ``time``, ``open``, ``high``,
        ``low``, ``close``, ``volume``.
    """
    symbol = symbol.upper()
    if symbol not in _SUPPORTED_SYMBOLS:
        return JSONResponse(
            {"error": f"Unsupported symbol '{symbol}'."},
            status_code=400,
        )
    if tf not in _SUPPORTED_TIMEFRAMES:
        return JSONResponse(
            {"error": f"Unsupported timeframe '{tf}'."},
            status_code=400,
        )

    session = request.app.state.http_session
    adapter = ExchangeFactory.create("binance", session=session)

    try:
        candles = await adapter.get_candles(symbol, tf, limit)
    except Exception as exc:
        logger.error("Failed to fetch candles for %s/%s: %s", symbol, tf, exc)
        return JSONResponse({"error": str(exc)}, status_code=502)

    data = [
        {
            "time": c.time,
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]
    return JSONResponse({"symbol": symbol, "timeframe": tf, "candles": data})


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — Live candle stream
# ─────────────────────────────────────────────────────────────────────────────


@router.websocket("/ws/candles/{symbol}")
async def ws_candles(
    websocket: WebSocket,
    symbol: str,
    tf: str = "1H",
) -> None:
    """Stream live candle updates to the client over WebSocket.

    Sends one JSON message per candle update.  Each message has::

        {
          "time": 1721900400,
          "open": 65234.5,
          "high": 65400.0,
          "low":  65100.0,
          "close": 65350.0,
          "volume": 120.45,
          "is_closed": true
        }

    The client should update the chart's live candle on every message
    and only shift the prediction forward when ``is_closed`` is ``true``.
    """
    symbol = symbol.upper()
    if symbol not in _SUPPORTED_SYMBOLS:
        await websocket.close(code=4000, reason=f"Unsupported symbol '{symbol}'")
        return
    if tf not in _SUPPORTED_TIMEFRAMES:
        await websocket.close(code=4001, reason=f"Unsupported timeframe '{tf}'")
        return

    await websocket.accept()
    logger.info("WS client connected: %s/%s", symbol, tf)

    session = websocket.app.state.http_session
    adapter = ExchangeFactory.create("binance", session=session)

    try:
        async for candle in adapter.stream_candles(symbol, tf):
            payload = {
                "time": candle.time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "is_closed": candle.is_closed,
            }
            await websocket.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        logger.info("WS client disconnected: %s/%s", symbol, tf)
    except Exception as exc:
        logger.error("WS error for %s/%s: %s", symbol, tf, exc)
        await websocket.close(code=1011, reason="Internal streaming error")


# ─────────────────────────────────────────────────────────────────────────────
# REST — AI prediction with confidence bands
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/predict/{symbol}")
async def predict(
    symbol: str,
    tf: Annotated[str, Query()] = "1H",
    n: Annotated[int, Query(ge=5, le=50, description="Number of future candles")] = 20,
    model: Annotated[str, Query(description="Model: auto|xgboost|lstm|ensemble")] = "auto",
) -> JSONResponse:
    """Generate an AI forecast with confidence bands for *symbol*.

    Returns a :class:`~src.models.model_manager.ForecastObject` as JSON,
    including:
    - ``candles``: central prediction line (time + value pairs)
    - ``upper_band``: upper confidence interval
    - ``lower_band``: lower confidence interval
    - ``direction``: ``"bullish"`` or ``"bearish"``
    - ``confidence``: float 0.0–1.0
    - ``expected_price``, ``expected_change_pct``, ``volatility_est``
    - ``model_name``, ``generated_at``
    - ``anchor_time``, ``anchor_price``: join point with historical data

    The frontend must draw historical candles up to ``anchor_time`` and
    the prediction from ``anchor_time`` onward — **never overlap them**.
    """
    symbol = symbol.upper()
    if symbol not in _SUPPORTED_SYMBOLS:
        return JSONResponse(
            {"error": f"Unsupported symbol '{symbol}'."},
            status_code=400,
        )

    n_candles = n
    if n_candles not in _SUPPORTED_HORIZONS:
        # Clamp to nearest supported value
        n_candles = min(_SUPPORTED_HORIZONS, key=lambda h: abs(h - n_candles))

    try:
        forecast = await _model_manager.predict(
            symbol=symbol,
            timeframe=tf,
            n_candles=n_candles,
            model=model,
            http_session=request.app.state.http_session,
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        logger.error("Prediction error for %s: %s", symbol, exc)
        return JSONResponse({"error": "Prediction failed."}, status_code=500)

    # The prediction expires exactly when the model's target timestamp is
    # reached — i.e. anchor_time + n_candles * timeframe, the same horizon
    # the model was actually trained to predict. Do NOT recompute this from
    # a different notion of "horizon"; forecast.target_timestamp is the one
    # source of truth (see src/models/model_manager.py's module docstring).
    expires_at = forecast.target_timestamp
    bullish_probability = (
        forecast.metadata.confidence
        if forecast.metadata.direction == "bullish"
        else 1.0 - forecast.metadata.confidence
    )
    prediction = _prediction_store.create_prediction(
        expires_at=expires_at,
        model_name=forecast.metadata.model_name,
        model_version=forecast.model_version,
        symbol=forecast.symbol,
        timeframe=forecast.timeframe,
        prediction_horizon=forecast.n_candles,
        anchor_time=forecast.anchor_time,
        anchor_price=forecast.anchor_price,
        predicted_price=forecast.metadata.expected_price,
        confidence=forecast.metadata.confidence,
        bullish_probability=bullish_probability,
        bearish_probability=1.0 - bullish_probability,
        risk_level=forecast.risk_level,
        expected_volatility=forecast.metadata.volatility_est,
        support_levels=forecast.support_levels,
        resistance_levels=forecast.resistance_levels,
        reasoning_summary=forecast.reasoning_summary,
        historical_candles=forecast.historical_candles,
        technical_indicators=forecast.technical_indicators,
        prediction_values=forecast.to_api_dict(),
        feature_importance=forecast.feature_importance,
        market_regime=forecast.market_regime,
    )
    forecast.prediction_id = prediction.prediction_id

    return JSONResponse(forecast.to_api_dict())


# ─────────────────────────────────────────────────────────────────────────────
# REST — Available models
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/models")
async def list_models() -> JSONResponse:
    """Return available AI model names for the model selector UI."""
    return JSONResponse({"models": _model_manager.available_models})
