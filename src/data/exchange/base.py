"""Abstract exchange adapter interface.

Every supported exchange (Binance, Kraken, Coinbase, OKX, Bybit, …)
must implement ``IExchangeAdapter``.  The rest of the system depends
only on this interface — never on any concrete exchange class.

Design notes
------------
* Dependency Inversion: callers depend on the ABC, not on Binance.
* Single Responsibility: each adapter handles exactly one exchange.
* Open/Closed: adding Kraken means adding a new class, not modifying
  existing code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from src.data.exchange import CandleBar


class IExchangeAdapter(ABC):
    """Contract for all exchange market-data adapters.

    Concrete adapters must implement:
    - ``get_candles()``: historical REST fetch
    - ``stream_candles()``: real-time async generator

    They must NOT contain any business logic, prediction code, or
    direct references to the database or model layer.
    """

    # Canonical mapping from display symbol → exchange-specific symbol.
    # Subclasses override this if their convention differs.
    SYMBOL_ALIASES: dict[str, str] = {}

    @abstractmethod
    async def get_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[CandleBar]:
        """Return the last *limit* closed candles for *symbol*.

        Args:
            symbol:    Exchange symbol (e.g. ``"BTCUSDT"``).
            timeframe: Candle period string (e.g. ``"1m"``, ``"1h"``, ``"1d"``).
            limit:     Maximum number of candles to return (≤ 1000).

        Returns:
            List of :class:`CandleBar` sorted oldest-first.
        """
        raise NotImplementedError

    @abstractmethod
    async def stream_candles(
        self,
        symbol: str,
        timeframe: str,
    ) -> AsyncIterator[CandleBar]:
        """Yield live candle updates for *symbol* as they arrive.

        Yields both live ticks (``is_closed=False``) and closed candles
        (``is_closed=True``).  Callers must handle reconnection if the
        generator raises :class:`ConnectionError`.

        Args:
            symbol:    Exchange symbol.
            timeframe: Candle period string.

        Yields:
            :class:`CandleBar` — one per WebSocket message.
        """
        raise NotImplementedError

    def resolve_symbol(self, symbol: str) -> str:
        """Translate a canonical symbol to the exchange-specific format.

        Args:
            symbol: Canonical symbol (e.g. ``"BTCUSDT"``).

        Returns:
            Exchange-specific symbol string.
        """
        return self.SYMBOL_ALIASES.get(symbol, symbol)

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable exchange name (e.g. ``"Binance"``)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_timeframes(self) -> list[str]:
        """List of timeframe strings this adapter supports."""
        raise NotImplementedError
