"""Input validation helpers for the Crypto Alpha Engine.

All validators are pure functions that raise ``ValueError`` on invalid
input so callers can decide how to handle errors.
"""

from __future__ import annotations

from src.config.settings import FORECAST_HORIZONS, TIER1_SYMBOLS


def validate_symbol(symbol: str) -> str:
    """Ensure *symbol* is a recognised CoinGecko coin ID.

    Args:
        symbol: Lowercase CoinGecko coin ID (e.g., ``"bitcoin"``).

    Returns:
        The validated *symbol* string.

    Raises:
        ValueError: If *symbol* is not in :data:`~src.config.settings.TIER1_SYMBOLS`.
    """
    if symbol not in TIER1_SYMBOLS:
        raise ValueError(
            f"Unknown symbol '{symbol}'. "
            f"Supported symbols: {TIER1_SYMBOLS}"
        )
    return symbol


def validate_horizon(horizon: int) -> int:
    """Ensure *horizon* is a supported forecast horizon in calendar days.

    Args:
        horizon: Number of calendar days to forecast ahead.

    Returns:
        The validated *horizon* integer.

    Raises:
        ValueError: If *horizon* is not in
            :data:`~src.config.settings.FORECAST_HORIZONS`.
    """
    if horizon not in FORECAST_HORIZONS:
        raise ValueError(
            f"Unsupported forecast horizon '{horizon}'. "
            f"Supported horizons: {FORECAST_HORIZONS}"
        )
    return horizon
