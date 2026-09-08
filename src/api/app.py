"""FastAPI application entry point.

Configures the FastAPI app with CORS, includes all routers, and
manages the aiohttp session lifecycle via the app lifespan.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import aiohttp
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes_forecast import router as forecast_router
from src.api.routes_evaluation import router as evaluation_router
from src.api.routes_health import router as health_router
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage shared resources across the app lifecycle."""
    logger.info("Starting Crypto Alpha Engine API…")
    app.state.http_session = aiohttp.ClientSession()
    yield
    logger.info("Shutting down Crypto Alpha Engine API…")
    await app.state.http_session.close()


app = FastAPI(
    title="Crypto Alpha Engine API",
    description=(
        "Real-time candlestick streaming, AI price forecasting, "
        "and multi-model prediction with confidence bands."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Next.js dev server + production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api")
app.include_router(forecast_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")
