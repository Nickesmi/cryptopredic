"""High-Potential Asset Discovery Engine — API surface.

Endpoints
---------
GET /api/rankings/opportunities   Scan a universe of candidates and return
                                   risk-adjusted opportunity rankings.

This is intentionally a separate router/subsystem from
``routes_forecast.py``: the opportunity scanner's job is to find which
assets are worth looking at, not to forecast the price of one asset
already chosen (see src/ranking/recommend.py's module docstring).
"""

from __future__ import annotations

import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from src.data.exchange.factory import ExchangeFactory
from src.ranking.liquidity_filter import LiquidityFilterConfig
from src.ranking.recommend import scan_opportunities
from src.utils.candles import bars_to_frame

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rankings"])

# A conservative default universe. In production this would come from an
# exchange's full symbol list; kept small and explicit here since this
# repository has no live symbol-discovery feed yet.
_DEFAULT_UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
]
_BENCHMARK_SYMBOL = "BTCUSDT"


@router.get("/rankings/opportunities")
async def get_opportunities(
    request: Request,
    symbols: Annotated[
        str | None,
        Query(description="Comma-separated symbols to scan; defaults to a built-in universe."),
    ] = None,
    tf: Annotated[str, Query(description="Timeframe for the scan, e.g. 1D")] = "1D",
    min_score: Annotated[float, Query(ge=0, le=100)] = 65.0,
) -> JSONResponse:
    """Scan candidate assets and return risk-adjusted opportunity rankings.

    Returns ``{"recommendations": [...], "excluded": {...}, "message": ...}``.
    ``message`` is set (and ``recommendations`` is empty) when nothing
    clears the quality/score bar — the scanner never forces a pick.
    """
    universe = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols
        else list(_DEFAULT_UNIVERSE)
    )

    session = request.app.state.http_session
    adapter = ExchangeFactory.create("binance", session=session)

    candidates: dict[str, pd.DataFrame] = {}
    fetch_errors: dict[str, str] = {}
    for symbol in universe:
        try:
            bars = await adapter.get_candles(symbol, tf, limit=500)
            candidates[symbol] = bars_to_frame(bars)
        except Exception as exc:  # noqa: BLE001 — surface as a per-symbol exclusion, not a 500
            fetch_errors[symbol] = str(exc)

    benchmark_df = candidates.get(_BENCHMARK_SYMBOL)
    if benchmark_df is None:
        try:
            bars = await adapter.get_candles(_BENCHMARK_SYMBOL, tf, limit=500)
            benchmark_df = bars_to_frame(bars)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fetch benchmark %s: %s", _BENCHMARK_SYMBOL, exc)
            benchmark_df = None

    result = scan_opportunities(
        candidates=candidates,
        benchmark_df=benchmark_df,
        timeframe=tf,
        min_opportunity_score=min_score,
        liquidity_config=LiquidityFilterConfig(),
    )
    payload = result.to_dict()
    if fetch_errors:
        payload["fetch_errors"] = fetch_errors
    return JSONResponse(payload)
