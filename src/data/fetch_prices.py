"""Price data access layer — Repository pattern.

Defines the abstract ``PriceRepository`` interface and the concrete
``CoinGeckoPriceRepository`` implementation that fetches daily close
price + volume data from the CoinGecko public REST API (no API key
required).

Implementation note
-------------------
The CoinGecko **free** tier OHLC endpoint (``/coins/{id}/ohlc``) only
accepts a fixed set of ``days`` values (1, 7, 14, 30, 90, 180, 365).
Arbitrary ranges (e.g. 730) return a 400 Bad Request.

We therefore use the more flexible ``/coins/{id}/market_chart`` endpoint
which:
  * accepts any ``days`` value ≥ 1,
  * returns daily granularity automatically when ``days > 90``,
  * provides ``prices`` (close) and ``total_volumes`` in a single call.

We set ``open = high = low = close`` as a simplification; all our
feature-engineering uses the closing price only.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import pandas as pd
import requests

from src.config.settings import (
    COINGECKO_BASE_URL,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface (Dependency Inversion Principle)
# ---------------------------------------------------------------------------


class PriceRepository(ABC):
    """Abstract contract for fetching historical OHLCV price data."""

    @abstractmethod
    def fetch(self, symbol: str, days: int) -> pd.DataFrame:
        """Return a daily OHLCV DataFrame for *symbol* over the last *days*.

        Args:
            symbol: Exchange/data-source specific asset identifier.
            days:   Number of trailing calendar days of history to retrieve.

        Returns:
            DataFrame indexed by UTC date with columns:
            ``[open, high, low, close, volume]``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CoinGecko implementation
# ---------------------------------------------------------------------------


class CoinGeckoPriceRepository(PriceRepository):
    """Fetches daily price + volume data from the CoinGecko v3 public API.

    Uses the ``/coins/{id}/market_chart`` endpoint which supports any
    ``days`` value and returns daily granularity for ranges > 90 days.

    Args:
        base_url:    Override the API base URL (useful for testing).
        timeout:     HTTP request timeout in seconds.
        max_retries: Number of retry attempts on transient HTTP errors.
    """

    _MARKET_CHART_ENDPOINT = "/coins/{coin_id}/market_chart"

    def __init__(
        self,
        base_url: str = COINGECKO_BASE_URL,
        timeout: int = HTTP_TIMEOUT_SECONDS,
        max_retries: int = HTTP_MAX_RETRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch *days* of daily price history for *symbol*.

        Makes a single call to ``/market_chart`` and extracts daily
        close prices and volumes.  ``open``, ``high``, and ``low`` are
        set equal to ``close`` (our feature pipeline uses close only).

        Args:
            symbol: CoinGecko coin ID (e.g. ``"bitcoin"``).
            days:   Number of trailing calendar days (any positive int).

        Returns:
            DataFrame with UTC DatetimeIndex and columns
            ``[open, high, low, close, volume]``.
        """
        logger.info("Fetching %d days of price data for '%s'", days, symbol)

        url = self._base_url + self._MARKET_CHART_ENDPOINT.format(
            coin_id=symbol
        )
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        raw = self._get_with_retry(url, params)

        prices = raw.get("prices", [])
        volumes = raw.get("total_volumes", [])

        if not prices:
            raise ValueError(
                f"No price data returned by CoinGecko for '{symbol}'."
            )

        # Build close-price series
        close_df = self._to_series(prices, "close")

        # Build volume series (align to close index)
        if volumes:
            vol_df = self._to_series(volumes, "volume")
            df = close_df.to_frame().join(vol_df.to_frame(), how="left")
            df["volume"] = df["volume"].fillna(0.0)
        else:
            df = close_df.to_frame()
            df["volume"] = 0.0

        # Derive open/high/low from close (close-only dataset)
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]

        df = df[["open", "high", "low", "close", "volume"]]

        logger.info(
            "Fetched %d rows for '%s' (range: %s → %s)",
            len(df),
            symbol,
            df.index.min().date() if not df.empty else "N/A",
            df.index.max().date() if not df.empty else "N/A",
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_series(raw_list: list, name: str) -> pd.Series:
        """Convert a CoinGecko ``[[timestamp_ms, value], ...]`` list to a Series."""
        df = pd.DataFrame(raw_list, columns=["timestamp_ms", name])
        df["date"] = (
            pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            .dt.normalize()
        )
        df = df.set_index("date").drop(columns=["timestamp_ms"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df[name]

    def _get_with_retry(self, url: str, params: dict) -> dict | list:
        """HTTP GET with exponential back-off retries."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.get(
                    url, params=params, timeout=self._timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )
                if status == 429:
                    wait = 2 ** attempt
                    logger.warning(
                        "Rate-limited by CoinGecko (attempt %d/%d). "
                        "Waiting %ds before retry.",
                        attempt,
                        self._max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    last_error = exc
                    continue
                raise
            except requests.RequestException as exc:
                wait = 2 ** attempt
                logger.warning(
                    "HTTP error on attempt %d/%d: %s. Retrying in %ds.",
                    attempt,
                    self._max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                last_error = exc

        raise RuntimeError(
            f"Failed to fetch {url} after {self._max_retries} attempts."
        ) from last_error
