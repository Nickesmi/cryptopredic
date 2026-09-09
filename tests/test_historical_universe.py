"""Tests for src/research/historical_universe.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.historical_universe import (
    build_point_in_time_universe,
    eligible_symbols_at,
    infer_listing_registry_from_data,
    quantify_survivorship_gap,
)


def _ragged_histories() -> dict[str, pd.DataFrame]:
    """Three assets with genuinely different start dates and lengths (real-data-shaped, not padded)."""
    def _mk(start: str, n: int) -> pd.DataFrame:
        dates = pd.date_range(start, periods=n, freq="D")
        close = pd.Series(100 + np.arange(n, dtype=float), index=dates)
        return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=dates)

    return {
        "BTC": _mk("2020-01-01", 500),  # earliest, longest
        "ETH": _mk("2020-06-01", 350),
        "LATECOIN": _mk("2021-03-01", 100),  # lists well after the others
    }


def test_eligible_symbols_before_any_listing_is_empty() -> None:
    histories = _ragged_histories()
    result = eligible_symbols_at(pd.Timestamp("2019-01-01"), histories)
    assert result == []


def test_eligible_symbols_only_includes_already_listed_assets() -> None:
    histories = _ragged_histories()
    # 2020-07-01: BTC and ETH listed, LATECOIN not yet.
    result = eligible_symbols_at(pd.Timestamp("2020-07-01"), histories)
    assert result == ["BTC", "ETH"]
    assert "LATECOIN" not in result


def test_eligible_symbols_includes_all_after_every_listing() -> None:
    histories = _ragged_histories()
    result = eligible_symbols_at(pd.Timestamp("2021-06-01"), histories)
    assert result == ["BTC", "ETH", "LATECOIN"]


def test_build_point_in_time_universe_never_shows_a_future_listed_asset_early() -> None:
    histories = _ragged_histories()
    scan_points = pd.date_range("2020-01-01", "2021-12-01", freq="30D")
    universe = build_point_in_time_universe(histories, list(scan_points), listing_registry=None)

    latecoin_listing = histories["LATECOIN"].index.min()
    for ts, symbols in universe.items():
        if ts < latecoin_listing:
            assert "LATECOIN" not in symbols, f"LATECOIN appeared before its own listing at {ts}"
        else:
            assert "LATECOIN" in symbols


def test_registry_can_only_restrict_never_expand_eligibility() -> None:
    from src.research.historical_universe import AssetListingRecord

    histories = _ragged_histories()
    # Registry claims BTC wasn't really available until 2020-03-01, even
    # though the data has candles from 2020-01-01 -- eligibility must
    # respect the stricter (later) of the two.
    registry = {
        "BTC": AssetListingRecord(
            symbol="BTC", first_available_timestamp=pd.Timestamp("2020-03-01"),
            source="manual_override", is_proxy=False,
        )
    }
    assert "BTC" not in eligible_symbols_at(pd.Timestamp("2020-02-01"), histories, registry)
    assert "BTC" in eligible_symbols_at(pd.Timestamp("2020-04-01"), histories, registry)


def test_infer_listing_registry_marks_every_record_as_proxy() -> None:
    histories = _ragged_histories()
    registry = infer_listing_registry_from_data(histories)
    assert set(registry) == set(histories)
    assert all(record.is_proxy for record in registry.values())
    for symbol, record in registry.items():
        assert record.first_available_timestamp == histories[symbol].index.min()


def test_infer_listing_registry_skips_all_nan_assets() -> None:
    histories = _ragged_histories()
    histories["EMPTY"] = pd.DataFrame(
        {"open": [np.nan] * 5, "high": [np.nan] * 5, "low": [np.nan] * 5, "close": [np.nan] * 5, "volume": [np.nan] * 5},
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )
    registry = infer_listing_registry_from_data(histories)
    assert "EMPTY" not in registry


def test_quantify_survivorship_gap_without_registry_is_honest_about_not_knowing() -> None:
    report = quantify_survivorship_gap({"BTC", "ETH", "LATECOIN"}, delisted_registry=None)
    assert report.quantifiable is False
    assert report.n_known_delisted is None
    assert report.delisted_fraction is None
    assert "cannot be quantified" in report.message


def test_quantify_survivorship_gap_with_registry_computes_fraction() -> None:
    report = quantify_survivorship_gap({"BTC", "ETH"}, delisted_registry=["DEADCOIN", "RUGCOIN"])
    assert report.quantifiable is True
    assert report.n_known_delisted == 2
    assert report.delisted_fraction == 2 / 4
    assert set(report.delisted_symbols) == {"DEADCOIN", "RUGCOIN"}
