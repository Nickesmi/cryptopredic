"""Tests for scanner/feature version provenance on recommendations (Phase 4,
Section 20: "every live recommendation should record ... scanner version").

Before this, OpportunityRecommendation had no version fields at all --
the exact gap Phase 2 found and fixed for the price forecaster's
model_version/feature_version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking.recommend import scan_candidate
from src.ranking.score_coins import scanner_version


def _make_ohlcv(n: int = 250, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.001, 0.02, n)), index=dates)
    return pd.DataFrame(
        {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 2_000_000.0},
        index=dates,
    )


def test_scanner_version_is_stable_for_the_same_weights() -> None:
    a = scanner_version({"trend_quality": 0.25, "momentum": 0.20, "relative_strength": 0.20, "volume_confirmation": 0.15, "volatility_adjusted": 0.20})
    b = scanner_version({"trend_quality": 0.25, "momentum": 0.20, "relative_strength": 0.20, "volume_confirmation": 0.15, "volatility_adjusted": 0.20})
    assert a == b
    assert len(a) == 12


def test_scanner_version_changes_with_different_weights() -> None:
    default = scanner_version({"trend_quality": 0.25, "momentum": 0.20, "relative_strength": 0.20, "volume_confirmation": 0.15, "volatility_adjusted": 0.20})
    changed = scanner_version({"trend_quality": 0.50, "momentum": 0.10, "relative_strength": 0.15, "volume_confirmation": 0.10, "volatility_adjusted": 0.15})
    assert default != changed


def test_recommendation_carries_non_empty_version_and_timestamp_fields() -> None:
    df = _make_ohlcv()
    btc = _make_ohlcv(seed=2)
    rec, _ = scan_candidate("AAA", df, btc, "1D")
    assert rec is not None
    assert rec.scanner_version
    assert len(rec.scanner_version) == 12
    assert rec.feature_version
    assert rec.scanned_at  # ISO timestamp, non-empty

    payload = rec.to_dict()
    assert payload["scanner_version"] == rec.scanner_version
    assert payload["feature_version"] == rec.feature_version
    assert payload["scanned_at"] == rec.scanned_at
