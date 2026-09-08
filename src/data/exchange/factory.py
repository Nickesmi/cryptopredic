"""Exchange factory — create adapters by name.

Callers ask the factory for an adapter; they never instantiate
concrete adapter classes directly.  This makes swapping exchanges
trivial and keeps the rest of the codebase exchange-agnostic.

Example::

    adapter = ExchangeFactory.create("binance", session=http_session)
    candles = await adapter.get_candles("BTCUSDT", "1h", limit=200)
"""

from __future__ import annotations

import aiohttp

from src.data.exchange.base import IExchangeAdapter
from src.data.exchange.binance import BinanceAdapter

# Registry of supported exchanges.  To add Kraken, Coinbase, etc.,
# register them here — no other file needs to change (OCP).
_REGISTRY: dict[str, type[IExchangeAdapter]] = {
    "binance": BinanceAdapter,
}


class ExchangeFactory:
    """Factory for creating :class:`IExchangeAdapter` instances."""

    @staticmethod
    def create(
        exchange: str = "binance",
        session: aiohttp.ClientSession | None = None,
    ) -> IExchangeAdapter:
        """Instantiate an exchange adapter by name.

        Args:
            exchange: Lowercase exchange identifier
                      (``"binance"``, ``"kraken"``, …).
            session:  Shared ``aiohttp.ClientSession`` to inject.
                      If ``None``, the adapter creates its own.

        Returns:
            A concrete :class:`IExchangeAdapter` instance.

        Raises:
            ValueError: If *exchange* is not registered.
        """
        cls = _REGISTRY.get(exchange.lower())
        if cls is None:
            supported = list(_REGISTRY.keys())
            raise ValueError(
                f"Unknown exchange '{exchange}'. Supported: {supported}"
            )
        # Inject session if the adapter accepts it (all current adapters do)
        try:
            return cls(session=session)  # type: ignore[call-arg]
        except TypeError:
            return cls()

    @staticmethod
    def supported_exchanges() -> list[str]:
        """Return the names of all registered exchanges."""
        return list(_REGISTRY.keys())
