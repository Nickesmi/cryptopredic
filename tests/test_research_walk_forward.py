"""Tests for src/research/walk_forward.py: panel building, region splits, and the
no-future-listing guarantee (Phase 5, Section 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.signals import HORIZON_CANDLES, momentum
from src.research.walk_forward import (
    build_signal_panel,
    cross_sectional_relative_strength_panel,
    restrict_panel_to_range,
    split_design_validation_test,
)


def _candidates(n: int = 500, listing_start: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    listing_start = listing_start or {}
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    out = {}
    for i, symbol in enumerate(["A", "B", "C", "D", "E", "F"]):
        rng = np.random.default_rng(i)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n)), index=dates)
        df = pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1_000_000.0},
            index=dates,
        )
        start = listing_start.get(symbol, 0)
        if start > 0:
            df.iloc[:start] = np.nan
        out[symbol] = df
    return out


def test_split_design_validation_test_partitions_without_gap_or_overlap() -> None:
    split = split_design_validation_test(1000, design_frac=0.5, validation_frac=0.25)
    assert split.design == (0, 500)
    assert split.validation == (500, 750)
    assert split.test == (750, 1000)


def test_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError):
        split_design_validation_test(1000, design_frac=0.6, validation_frac=0.6)


def test_build_signal_panel_rejects_mismatched_lengths() -> None:
    candidates = _candidates(500)
    candidates["A"] = candidates["A"].iloc[:400]
    signal_series = {s: momentum(df["close"], 24) for s, df in candidates.items()}
    with pytest.raises(ValueError):
        build_signal_panel(candidates, signal_series, scan_every_candles=24, min_train_size=100)


def test_not_yet_listed_asset_excluded_before_its_listing_point() -> None:
    candidates = _candidates(500, listing_start={"F": 300})
    signal_series = {s: momentum(df["close"], HORIZON_CANDLES["1D"]) for s, df in candidates.items()}
    panel = build_signal_panel(candidates, signal_series, scan_every_candles=24, min_train_size=50)

    early_snaps = [snap for snap in panel if snap.scan_index < 300]
    late_snaps = [snap for snap in panel if snap.scan_index >= 340]  # 300 + warmup(24) + margin

    assert early_snaps and late_snaps
    assert all("F" not in snap.scores and "F" in snap.excluded for snap in early_snaps)
    assert all("F" in snap.scores for snap in late_snaps)


def test_restrict_panel_to_range_filters_by_scan_index() -> None:
    candidates = _candidates(500)
    signal_series = {s: momentum(df["close"], 24) for s, df in candidates.items()}
    panel = build_signal_panel(candidates, signal_series, scan_every_candles=24, min_train_size=50)
    restricted = restrict_panel_to_range(panel, 100, 200)
    assert all(100 <= snap.scan_index < 200 for snap in restricted)
    assert len(restricted) < len(panel)


def test_cross_sectional_relative_strength_panel_scores_relative_to_universe_mean() -> None:
    candidates = _candidates(500)
    panel = cross_sectional_relative_strength_panel(
        candidates, scan_every_candles=24, min_train_size=50, lookback=24
    )
    assert panel
    for snap in panel:
        if snap.scores:
            # scores are demeaned -- they must sum close to zero across the cross-section.
            assert abs(sum(snap.scores.values())) < 1e-6 * max(1, len(snap.scores))


def test_cross_sectional_relative_strength_panel_excludes_unlisted_assets() -> None:
    candidates = _candidates(500, listing_start={"F": 300})
    panel = cross_sectional_relative_strength_panel(
        candidates, scan_every_candles=24, min_train_size=50, lookback=24
    )
    early = [snap for snap in panel if snap.scan_index < 300]
    assert all("F" not in snap.scores for snap in early)
