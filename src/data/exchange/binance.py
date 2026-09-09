"""Binance market-data adapter.

Implements ``IExchangeAdapter`` for the Binance public REST and
WebSocket APIs.  No API key is required for market data.

REST base:  https://api.binance.com
WS base:    wss://stream.binance.com:9443/ws

All methods are async to integrate cleanly with FastAPI's event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

import aiohttp

from src.data.exchange import CandleBar
from src.data.exchange.base import IExchangeAdapter

logger = logging.getLogger(__name__)

_REST_BASE = "https://api.binance.com"
_WS_BASE = "wss://stream.binance.com:9443/ws"

# Map our canonical timeframe strings to Binance interval codes.
_TF_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
    "1W": "1w",
}


def _parse_kline(raw: list, now_ms: int | None = None) -> CandleBar:
    """Convert a Binance kline array to a :class:`CandleBar`.

    Binance REST kline format:
    [open_time, open, high, low, close, volume, close_time, ...]

    ``GET /api/v3/klines`` does **not** only return fully-elapsed candles —
    when called without an explicit ``endTime`` it includes the
    currently-forming candle as the last element, with a ``close`` that is
    still changing tick-by-tick. Every call site previously hard-coded
    ``is_closed=True`` regardless of this, which is exactly the "candle not
    fully closed yet" hazard called out in the timing audit (a prediction
    must never treat an in-progress candle as if it were finished data).
    We instead derive ``is_closed`` from the kline's own ``close_time``
    (index 6): the candle is closed only once that timestamp has passed.
    """
    close_time_ms = int(raw[6])
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    return CandleBar(
        time=int(raw[0]) // 1000,  # ms → seconds
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
        is_closed=close_time_ms <= now_ms,
    )


def _parse_ws_kline(data: dict) -> CandleBar:
    """Convert a Binance WebSocket kline payload to a :class:`CandleBar`.

    WS kline event structure: ``{"k": {t, o, h, l, c, v, x, ...}}``
    where ``x`` is True when the candle is closed.
    """
    k = data["k"]
    return CandleBar(
        time=int(k["t"]) // 1000,
        open=float(k["o"]),
        high=float(k["h"]),
        low=float(k["l"]),
        close=float(k["c"]),
        volume=float(k["v"]),
        is_closed=bool(k["x"]),
    )


class BinanceAdapter(IExchangeAdapter):
    """Binance market-data adapter (REST + WebSocket).

    Args:
        session: Optional ``aiohttp.ClientSession`` to reuse across
                 requests. If not provided, one is created per call
                 (not recommended for production; inject a shared
                 session via the FastAPI lifespan).
    """

    SYMBOL_ALIASES: dict[str, str] = {}  # Binance uses standard symbols

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None

    @property
    def name(self) -> str:
        return "Binance"

    @property
    def supported_timeframes(self) -> list[str]:
        return list(_TF_MAP.keys())

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        """Release the HTTP session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[CandleBar]:
        """Fetch historical klines from Binance REST API.

        Note: the **last** element of the response may be the
        currently-forming candle (Binance includes it whenever its open
        time is in the past, even though it hasn't closed yet) — check
        ``is_closed`` on the last returned bar before treating it as final
        data. Callers that must never use an incomplete candle (e.g.
        feature engineering / prediction anchoring) should drop any
        trailing bar with ``is_closed=False``.

        Args:
            symbol:    Binance symbol (e.g. ``"BTCUSDT"``).
            timeframe: Canonical timeframe string (``"1m"``–``"1W"``).
            limit:     Number of candles (max 1000).

        Returns:
            List of :class:`CandleBar` sorted oldest-first.

        Raises:
            ValueError: If *timeframe* is not supported.
            aiohttp.ClientError: On network or HTTP errors.
        """
        interval = _TF_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. "
                f"Supported: {list(_TF_MAP.keys())}"
            )

        url = f"{_REST_BASE}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000),
        }

        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        candles = [_parse_kline(row) for row in data]
        logger.info(
            "Fetched %d %s candles for %s from Binance",
            len(candles),
            timeframe,
            symbol,
        )
        return candles

    async def get_historical_klines(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        page_limit: int = 1000,
        request_delay_seconds: float = 0.25,
    ) -> list[CandleBar]:
        """Fetch a full historical range of klines, paginating past the 1000-per-call cap.

        ``get_candles`` above only returns the most recent *limit* (<=1000)
        candles — it cannot retrieve, say, three years of hourly history
        (>26,000 candles). This is the real capability gap Phase 6's data-
        availability audit found: nothing in this codebase could build a
        multi-year OHLCV dataset even with network access. This method
        closes it by repeatedly calling ``GET /api/v3/klines`` with
        ``startTime``/``endTime``, advancing ``startTime`` to one
        millisecond past the last returned candle's *open* time each round
        (Binance's ``startTime`` is inclusive), until the page's last candle
        reaches *end_ms* or a page returns fewer than *page_limit* rows
        (meaning history is exhausted).

        Args:
            symbol:      Binance symbol (e.g. ``"BTCUSDT"``).
            timeframe:   Canonical timeframe string.
            start_ms:    Range start, inclusive, Unix ms (UTC).
            end_ms:      Range end, inclusive, Unix ms (UTC).
            page_limit:  Candles per request (<=1000, Binance's own cap).
            request_delay_seconds: Delay between pages, a conservative
                         default to stay well under Binance's public rate
                         limit for an unauthenticated, multi-page pull.

        Returns:
            All :class:`CandleBar` in ``[start_ms, end_ms]``, ascending,
            de-duplicated at page boundaries.

        Raises:
            ValueError: If *timeframe* is unsupported or the range is empty/inverted.
            aiohttp.ClientError: On network or HTTP errors.
        """
        interval = _TF_MAP.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Unsupported timeframe '{timeframe}'. Supported: {list(_TF_MAP.keys())}"
            )
        if end_ms <= start_ms:
            raise ValueError(f"end_ms ({end_ms}) must be greater than start_ms ({start_ms}).")
        page_limit = min(page_limit, 1000)

        url = f"{_REST_BASE}/api/v3/klines"
        session = await self._get_session()
        all_bars: list[CandleBar] = []
        cursor_ms = start_ms
        seen_open_times: set[int] = set()

        while cursor_ms <= end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor_ms,
                "endTime": end_ms,
                "limit": page_limit,
            }
            async with session.get(url, params=params) as resp:
                resp.raise_for_status()
                page = await resp.json()

            if not page:
                break

            for raw in page:
                open_time_ms = int(raw[0])
                if open_time_ms in seen_open_times:
                    continue  # boundary candle repeated from the previous page's startTime
                seen_open_times.add(open_time_ms)
                # is_closed is judged against real wall-clock time, not the
                # query's end_ms boundary -- correct both for a purely
                # historical range (end_ms far in the past, all closed) and
                # a range whose end_ms is "now" (the trailing candle may
                # genuinely still be forming).
                all_bars.append(_parse_kline(raw))

            last_open_time_ms = int(page[-1][0])
            cursor_ms = last_open_time_ms + 1
            logger.info(
                "Fetched page of %d %s candles for %s (cursor now %d, %d total)",
                len(page), timeframe, symbol, cursor_ms, len(all_bars),
            )

            if len(page) < page_limit:
                break  # short page: exchange has no more data before endTime
            if request_delay_seconds > 0:
                await asyncio.sleep(request_delay_seconds)

        all_bars.sort(key=lambda b: b.time)
        return all_bars

    async def stream_candles(
        self,
        symbol: str,
        timeframe: str,
    ) -> AsyncIterator[CandleBar]:
        """Stream live kline updates from Binance WebSocket.

        Yields both live ticks and closed candles.  Reconnects
        automatically on connection drops with exponential back-off.

        Args:
            symbol:    Binance symbol (e.g. ``"BTCUSDT"``).
            timeframe: Canonical timeframe string.

        Yields:
            :class:`CandleBar` — one per WebSocket message.
        """
        interval = _TF_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe '{timeframe}'.")

        stream = f"{symbol.lower()}@kline_{interval}"
        url = f"{_WS_BASE}/{stream}"
        backoff = 1.0

        while True:
            try:
                session = await self._get_session()
                async with session.ws_connect(url) as ws:
                    logger.info("Connected to Binance WS stream: %s", stream)
                    backoff = 1.0  # reset on successful connection
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if "k" in data:
                                yield _parse_ws_kline(data)
                        elif msg.type in (
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            logger.warning(
                                "Binance WS closed/error — reconnecting in %gs",
                                backoff,
                            )
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Binance WS error: %s. Retry in %gs.", exc, backoff)

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)  # cap at 60s
