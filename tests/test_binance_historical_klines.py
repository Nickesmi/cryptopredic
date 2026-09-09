"""Tests for BinanceAdapter.get_historical_klines (Phase 6 real-data ingestion).

No real network access is used or required -- a fake aiohttp session
stands in for the exchange, proving the pagination/dedup/stop logic is
correct so the method is ready to run the moment real network access to
Binance is available (currently blocked by sandbox/org policy, per
docs/PHASE6_REAL_DATA_VALIDATION_REPORT.md Section 1).
"""

from __future__ import annotations

import pytest

from src.data.exchange.binance import BinanceAdapter

HOUR_MS = 3_600_000


def _make_kline(open_time_ms: int, close_offset_ms: int = HOUR_MS - 1) -> list:
    # Binance kline array: [open_time, open, high, low, close, volume, close_time, ...]
    return [
        open_time_ms, "100.0", "101.0", "99.0", "100.5", "10.0",
        open_time_ms + close_offset_ms, "1000.0", 5, "5.0", "500.0", "0",
    ]


class _FakeResponse:
    def __init__(self, payload: list) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> list:
        return self._payload


class _FakeSession:
    """Replays a fixed sequence of pages, one per call to .get(), and records params."""

    def __init__(self, pages: list[list[list]]) -> None:
        self._pages = pages
        self.calls: list[dict] = []
        self.closed = False

    def get(self, url: str, params: dict):
        self.calls.append(dict(params))
        page = self._pages[len(self.calls) - 1] if len(self.calls) <= len(self._pages) else []
        return _FakeResponse(page)


@pytest.mark.asyncio
async def test_paginates_across_a_full_page_boundary() -> None:
    # Page 1: 3 candles (a small "page_limit" for a fast, readable test).
    # Page 2: 2 candles (short page -> signals exhausted history).
    page1 = [_make_kline(i * HOUR_MS) for i in range(3)]
    page2 = [_make_kline(i * HOUR_MS) for i in range(3, 5)]
    session = _FakeSession([page1, page2])
    adapter = BinanceAdapter(session=session)

    bars = await adapter.get_historical_klines(
        "BTCUSDT", "1H", start_ms=0, end_ms=10 * HOUR_MS, page_limit=3, request_delay_seconds=0
    )

    assert [b.time for b in bars] == [i * HOUR_MS // 1000 for i in range(5)]
    assert len(session.calls) == 2
    # Second call's startTime must advance past the last candle of page 1.
    assert session.calls[1]["startTime"] == 2 * HOUR_MS + 1


@pytest.mark.asyncio
async def test_stops_when_a_short_page_is_returned() -> None:
    page1 = [_make_kline(i * HOUR_MS) for i in range(2)]  # shorter than page_limit
    session = _FakeSession([page1])
    adapter = BinanceAdapter(session=session)

    bars = await adapter.get_historical_klines(
        "BTCUSDT", "1H", start_ms=0, end_ms=100 * HOUR_MS, page_limit=5, request_delay_seconds=0
    )

    assert len(bars) == 2
    assert len(session.calls) == 1  # never asks for a page beyond exhausted history


@pytest.mark.asyncio
async def test_stops_when_a_page_is_empty() -> None:
    session = _FakeSession([[]])
    adapter = BinanceAdapter(session=session)

    bars = await adapter.get_historical_klines(
        "BTCUSDT", "1H", start_ms=0, end_ms=100 * HOUR_MS, page_limit=5, request_delay_seconds=0
    )

    assert bars == []


@pytest.mark.asyncio
async def test_deduplicates_boundary_candle_between_pages() -> None:
    # Simulate an exchange that (incorrectly, or via an off-by-zero) repeats
    # the boundary candle across two pages -- the adapter must not double-count it.
    page1 = [_make_kline(i * HOUR_MS) for i in range(3)]
    page2 = [_make_kline(i * HOUR_MS) for i in range(2, 5)]  # candle at index 2 repeated
    session = _FakeSession([page1, page2])
    adapter = BinanceAdapter(session=session)

    bars = await adapter.get_historical_klines(
        "BTCUSDT", "1H", start_ms=0, end_ms=10 * HOUR_MS, page_limit=3, request_delay_seconds=0
    )

    assert len(bars) == 5  # not 6
    assert len(set(b.time for b in bars)) == 5


@pytest.mark.asyncio
async def test_results_are_ascending_and_closed_for_a_historical_range() -> None:
    page1 = [_make_kline(i * HOUR_MS) for i in range(4)]
    session = _FakeSession([page1])
    adapter = BinanceAdapter(session=session)

    bars = await adapter.get_historical_klines(
        "BTCUSDT", "1H", start_ms=0, end_ms=3 * HOUR_MS, page_limit=10, request_delay_seconds=0
    )

    assert [b.time for b in bars] == sorted(b.time for b in bars)
    assert all(b.is_closed for b in bars)  # every candle's close_time is far in the past


@pytest.mark.asyncio
async def test_rejects_inverted_or_empty_range() -> None:
    adapter = BinanceAdapter(session=_FakeSession([]))
    with pytest.raises(ValueError):
        await adapter.get_historical_klines("BTCUSDT", "1H", start_ms=1000, end_ms=1000)
    with pytest.raises(ValueError):
        await adapter.get_historical_klines("BTCUSDT", "1H", start_ms=2000, end_ms=1000)


@pytest.mark.asyncio
async def test_rejects_unsupported_timeframe() -> None:
    adapter = BinanceAdapter(session=_FakeSession([]))
    with pytest.raises(ValueError):
        await adapter.get_historical_klines("BTCUSDT", "3H", start_ms=0, end_ms=1000)
