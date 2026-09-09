"""Tests for src/research/universes.py."""

from __future__ import annotations

import numpy as np

from src.research.universes import (
    ALL_SYMBOLS,
    N_CANDLES,
    STAGGERED_LISTINGS,
    build_null_universe,
    build_positive_control_universe,
    build_realistic_universe,
)


def test_all_universes_share_symbols_and_length() -> None:
    for builder in (build_positive_control_universe, build_null_universe, build_realistic_universe):
        universe = builder()
        assert set(universe.candidates) == set(ALL_SYMBOLS)
        for df in universe.candidates.values():
            assert len(df) == N_CANDLES


def test_staggered_listings_are_nan_before_listing_and_valid_after() -> None:
    universe = build_realistic_universe()
    for symbol, start in STAGGERED_LISTINGS.items():
        df = universe.candidates[symbol]
        assert df["close"].iloc[: start].isna().all()
        assert df["close"].iloc[start:].notna().all()


def test_non_staggered_assets_have_no_nan() -> None:
    universe = build_realistic_universe()
    for symbol in ALL_SYMBOLS:
        if symbol not in STAGGERED_LISTINGS:
            assert universe.candidates[symbol]["close"].notna().all()


def test_positive_control_is_reproducible_given_same_seed() -> None:
    u1 = build_positive_control_universe(seed=999)
    u2 = build_positive_control_universe(seed=999)
    for symbol in ALL_SYMBOLS:
        np.testing.assert_array_equal(
            u1.candidates[symbol]["close"].to_numpy(), u2.candidates[symbol]["close"].to_numpy()
        )


def test_null_universe_returns_have_near_zero_autocorrelation() -> None:
    universe = build_null_universe()
    close = universe.candidates["BTC"]["close"]
    r = close.pct_change().dropna()
    autocorr = r.autocorr(lag=1)
    assert abs(autocorr) < 0.15  # loose bound: iid noise should show no meaningful serial correlation


def test_positive_control_returns_show_meaningful_autocorrelation() -> None:
    # The planted trend is persistent (phi=0.995), which induces positive
    # serial correlation in realized returns relative to the null universe.
    universe = build_positive_control_universe()
    close = universe.candidates["BTC"]["close"]
    r = close.pct_change().dropna()
    autocorr = r.autocorr(lag=1)
    assert autocorr > 0.05


def test_realistic_universe_all_prices_positive() -> None:
    universe = build_realistic_universe()
    for df in universe.candidates.values():
        assert (df["close"].dropna() > 0).all()
