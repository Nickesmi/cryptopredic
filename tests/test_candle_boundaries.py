"""Regression tests for the "unclosed candle" bug found in Phase 2 of the
timing audit (see docs/TIMING_AUDIT_REPORT.md).

Binance's REST kline endpoint (``GET /api/v3/klines``) returns the
currently-forming candle as its last element whenever the interval's open
time has already passed — it does not wait for the candle to close.
``_parse_kline`` used to hard-code ``is_closed=True`` on every bar
regardless, and nothing downstream ever checked the flag, so a live
prediction could be anchored on a close price that was still changing
tick-by-tick, or — worse, for higher timeframes — could bake a
not-yet-final candle into feature engineering. This is exactly the
"10:15 must not use the unclosed 10:00-14:00 4H candle" hazard called out
in the task brief's Phase 5.
"""

from __future__ import annotations

import pytest

from src.data.exchange import CandleBar
from src.data.exchange.binance import _parse_kline
from src.utils.candles import bars_to_frame


def _raw_kline(open_time_ms: int, close_time_ms: int, close: float = 100.0) -> list:
    return [open_time_ms, close, close, close, close, 10.0, close_time_ms, 0, 0, 0, 0, 0]


class TestParseKlineClosedFlag:
    def test_kline_whose_close_time_has_passed_is_closed(self) -> None:
        now_ms = 1_000_000_000_000
        raw = _raw_kline(open_time_ms=now_ms - 3_600_000, close_time_ms=now_ms - 1)
        bar = _parse_kline(raw, now_ms=now_ms)
        assert bar.is_closed is True

    def test_kline_whose_close_time_is_in_the_future_is_not_closed(self) -> None:
        now_ms = 1_000_000_000_000
        # Still-forming candle: opened a moment ago, won't close until later.
        raw = _raw_kline(open_time_ms=now_ms - 60_000, close_time_ms=now_ms + 3_540_000)
        bar = _parse_kline(raw, now_ms=now_ms)
        assert bar.is_closed is False


class TestBarsToFrameDropsUnclosedTrailingCandle:
    def _bars(self, last_closed: bool) -> list[CandleBar]:
        closed_bars = [
            CandleBar(time=i * 3600, open=100 + i, high=101 + i, low=99 + i, close=100 + i, volume=10.0)
            for i in range(5)
        ]
        trailing = CandleBar(
            time=5 * 3600, open=105, high=106, low=104, close=999.0, volume=1.0, is_closed=last_closed
        )
        return closed_bars + [trailing]

    def test_drops_still_forming_last_candle_by_default(self) -> None:
        bars = self._bars(last_closed=False)
        df = bars_to_frame(bars)
        assert len(df) == 5
        assert 999.0 not in df["close"].values

    def test_keeps_last_candle_when_it_is_actually_closed(self) -> None:
        bars = self._bars(last_closed=True)
        df = bars_to_frame(bars)
        assert len(df) == 6
        assert 999.0 in df["close"].values

    def test_can_opt_out_of_dropping_for_non_model_consumers(self) -> None:
        bars = self._bars(last_closed=False)
        df = bars_to_frame(bars, drop_unclosed=False)
        assert len(df) == 6

    def test_raises_if_only_an_unclosed_candle_is_available(self) -> None:
        bars = [
            CandleBar(time=0, open=1, high=1, low=1, close=1, volume=1.0, is_closed=False)
        ]
        with pytest.raises(ValueError):
            bars_to_frame(bars)
