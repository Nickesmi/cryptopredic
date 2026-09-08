"""Tests for expanding-window walk-forward validation (Phase 12 of the timing audit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.walk_forward import walk_forward_validate


def _oscillating_df(n: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    close = pd.Series(200 + 30 * np.sin(2 * np.pi * t / 30), index=dates, dtype=float)
    return pd.DataFrame(
        {"open": close - 0.1, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1000.0},
        index=dates,
    )


def test_folds_expand_and_never_overlap_with_their_own_test_block() -> None:
    df = _oscillating_df()
    report = walk_forward_validate(
        df, "BTCUSDT", "1D", horizon_candles=3, n_folds=3, min_train_size=200
    )

    assert len(report.folds) >= 2
    # Expanding window: each fold's training block is a strict superset in
    # size of the previous fold's.
    train_ends = [f.train_end for f in report.folds]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)

    for fold in report.folds:
        # Embargo gap: the test block must start strictly after train_end.
        assert fold.test_start >= fold.train_end
        assert fold.test_start - fold.train_end >= 3  # >= horizon_candles
        assert fold.n_predictions > 0

    assert report.overall["predictions"] == sum(f.n_predictions for f in report.folds)


def test_raises_when_insufficient_data() -> None:
    df = _oscillating_df(n=50)
    with pytest.raises(ValueError):
        walk_forward_validate(df, "BTCUSDT", "1D", horizon_candles=3, n_folds=3, min_train_size=200)


def test_report_serialises_to_dict() -> None:
    df = _oscillating_df()
    report = walk_forward_validate(
        df, "BTCUSDT", "1D", horizon_candles=3, n_folds=2, min_train_size=200
    )
    payload = report.to_dict()
    assert payload["symbol"] == "BTCUSDT"
    assert payload["horizon_candles"] == 3
    assert len(payload["folds"]) == len(report.folds)
    assert "directional_accuracy" in payload["overall"]
