"""Price data access layer — Repository pattern.

Defines the abstract ``PriceRepository`` interface and the concrete
``CoinGeckoPriceRepository`` implementation that fetches daily OHLCV
data from the CoinGecko public REST API (no API key required).
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
            DataFrame indexed by date with columns:
            ``[open, high, low, close, volume]``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CoinGecko implementation
# ---------------------------------------------------------------------------


class CoinGeckoPriceRepository(PriceRepository):
    """Fetches daily OHLCV data from the CoinGecko v3 public API.

    CoinGecko's free tier returns daily granularity for ranges > 90 days
    which is exactly what we need for model training.

    Args:
        base_url: Override the API base URL (useful for testing).
        timeout:  HTTP request timeout in seconds.
        max_retries: Number of retry attempts on transient HTTP errors.
    """

    _OHLC_ENDPOINT = "/coins/{coin_id}/ohlc"
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
        """Fetch *days* of daily OHLCV history for *symbol*.

        Uses CoinGecko's OHLC endpoint which returns daily candles when
        *days* ≥ 2.  Falls back gracefully to close-only data if needed.

        Args:
            symbol: CoinGecko coin ID (e.g. ``"bitcoin"``).
            days:   Number of trailing calendar days.

        Returns:
            DataFrame with DatetimeIndex and columns
            ``[open, high, low, close, volume]``.
        """
        logger.info("Fetching %d days of OHLCV data for '%s'", days, symbol)
        ohlc_df = self._fetch_ohlc(symbol, days)
        volume_df = self._fetch_volume(symbol, days)

        # Merge volume into the OHLC frame on date
        df = ohlc_df.join(volume_df[["volume"]], how="left")
        df["volume"] = df["volume"].fillna(0.0)

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

    def _fetch_ohlc(self, coin_id: str, days: int) -> pd.DataFrame:
        """Return a DataFrame with open/high/low/close columns."""
        url = self._base_url + self._OHLC_ENDPOINT.format(coin_id=coin_id)
        params = {"vs_currency": "usd", "days": str(days)}
        raw = self._get_with_retry(url, params)

        # CoinGecko returns [[timestamp_ms, open, high, low, close], ...]
        df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close"])
        df["date"] = (
            pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.normalize()
        )
        df = df.set_index("date").drop(columns=["timestamp_ms"])
        # Keep last candle per day (CoinGecko may return multiple intra-day rows)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def _fetch_volume(self, coin_id: str, days: int) -> pd.DataFrame:
        """Return a DataFrame with a daily volume column."""
        url = self._base_url + self._MARKET_CHART_ENDPOINT.format(coin_id=coin_id)
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        raw = self._get_with_retry(url, params)

        # raw["total_volumes"] = [[timestamp_ms, volume], ...]
        volumes = raw.get("total_volumes", [])
        if not volumes:
            return pd.DataFrame(columns=["date", "volume"]).set_index("date")

        df = pd.DataFrame(volumes, columns=["timestamp_ms", "volume"])
        df["date"] = (
            pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.normalize()
        )
        df = df.set_index("date").drop(columns=["timestamp_ms"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def _get_with_retry(self, url: str, params: dict) -> dict | list:
        """HTTP GET with exponential back-off retries."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self._timeout)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
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
