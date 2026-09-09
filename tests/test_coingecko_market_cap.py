"""Tests for CoinGeckoPriceRepository.fetch_with_market_cap (Phase 6).

Uses a mocked requests.Session (via monkeypatching the repository's
internal session) -- no real network access is used or required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.data.fetch_prices import CoinGeckoPriceRepository


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def _sample_payload() -> dict:
    return {
        "prices": [[1_700_000_000_000, 100.0], [1_700_086_400_000, 105.0], [1_700_172_800_000, 110.0]],
        "total_volumes": [[1_700_000_000_000, 1_000.0], [1_700_086_400_000, 1_200.0], [1_700_172_800_000, 1_500.0]],
        "market_caps": [[1_700_000_000_000, 2_000_000.0], [1_700_086_400_000, 2_100_000.0], [1_700_172_800_000, 2_200_000.0]],
    }


def test_fetch_with_market_cap_extracts_market_cap_column() -> None:
    repo = CoinGeckoPriceRepository()
    repo._session = MagicMock()
    repo._session.get.return_value = _mock_response(_sample_payload())

    df = repo.fetch_with_market_cap("bitcoin", days=3)

    assert list(df.columns) == ["open", "high", "low", "close", "volume", "market_cap"]
    assert len(df) == 3
    assert df["market_cap"].iloc[0] == pytest.approx(2_000_000.0)
    assert df["market_cap"].iloc[-1] == pytest.approx(2_200_000.0)


def test_fetch_with_market_cap_missing_series_gives_nan_column() -> None:
    payload = _sample_payload()
    del payload["market_caps"]
    repo = CoinGeckoPriceRepository()
    repo._session = MagicMock()
    repo._session.get.return_value = _mock_response(payload)

    df = repo.fetch_with_market_cap("bitcoin", days=3)

    assert df["market_cap"].isna().all()


def test_fetch_with_market_cap_raises_on_no_price_data() -> None:
    repo = CoinGeckoPriceRepository()
    repo._session = MagicMock()
    repo._session.get.return_value = _mock_response({"prices": [], "total_volumes": [], "market_caps": []})

    with pytest.raises(ValueError):
        repo.fetch_with_market_cap("bitcoin", days=3)


def test_fetch_unchanged_and_still_has_no_market_cap_column() -> None:
    """Regression guard: fetch() (the pre-existing method) must be untouched."""
    repo = CoinGeckoPriceRepository()
    repo._session = MagicMock()
    repo._session.get.return_value = _mock_response(_sample_payload())

    df = repo.fetch("bitcoin", days=3)

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
