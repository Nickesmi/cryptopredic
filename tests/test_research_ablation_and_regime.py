"""Tests for src/research/ablation.py and src/research/regime_breakdown.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.cross_sectional_analysis import CrossSectionalSnapshot
from src.research.ablation import combine_panels, zscore_snapshot_scores
from src.research.regime_breakdown import regime_breakdown_ic, summarize_regime_breakdown


def test_zscore_snapshot_scores_mean_zero_std_one() -> None:
    scores = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}
    z = zscore_snapshot_scores(scores)
    values = np.array(list(z.values()))
    assert values.mean() == pytest.approx(0.0, abs=1e-9)
    assert values.std(ddof=1) == pytest.approx(1.0, abs=1e-9)


def test_zscore_degenerate_cases() -> None:
    assert zscore_snapshot_scores({"A": 1.0}) == {"A": 0.0}
    assert zscore_snapshot_scores({"A": 5.0, "B": 5.0}) == {"A": 0.0, "B": 0.0}


def test_combine_panels_averages_zscored_components() -> None:
    panel_a = [CrossSectionalSnapshot(scan_index=10, scan_time=0, scores={"A": 1.0, "B": 2.0, "C": 3.0}, excluded=[])]
    panel_b = [CrossSectionalSnapshot(scan_index=10, scan_time=0, scores={"A": 3.0, "B": 2.0, "C": 1.0}, excluded=[])]
    combined = combine_panels([panel_a, panel_b])
    assert len(combined) == 1
    # A is highest in panel_a (z=+1.22) and lowest in panel_b (z=-1.22) -- symmetric inputs should
    # roughly cancel out for A and C, leaving B (the middle in both) closest to zero either way.
    scores = combined[0].scores
    assert set(scores) == {"A", "B", "C"}
    assert scores["B"] == pytest.approx(0.0, abs=1e-9)


def test_combine_panels_requires_matching_scan_index() -> None:
    panel_a = [CrossSectionalSnapshot(scan_index=10, scan_time=0, scores={"A": 1.0, "B": 2.0}, excluded=[])]
    panel_b = [CrossSectionalSnapshot(scan_index=20, scan_time=0, scores={"A": 1.0, "B": 2.0}, excluded=[])]
    with pytest.raises(ValueError):
        combine_panels([panel_a, panel_b])


def test_combine_panels_requires_same_length() -> None:
    panel_a = [CrossSectionalSnapshot(scan_index=10, scan_time=0, scores={"A": 1.0}, excluded=[])]
    panel_b = []
    with pytest.raises(ValueError):
        combine_panels([panel_a, panel_b])


def test_combine_panels_includes_asset_scored_by_only_one_component() -> None:
    panel_a = [CrossSectionalSnapshot(scan_index=10, scan_time=0, scores={"A": 1.0, "B": 2.0, "C": 3.0}, excluded=[])]
    panel_b = [CrossSectionalSnapshot(scan_index=10, scan_time=0, scores={"A": 1.0, "B": 2.0}, excluded=["C"])]
    combined = combine_panels([panel_a, panel_b])
    assert "C" in combined[0].scores  # scored by at least one component


def _candidates(n: int = 300) -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    out = {}
    for i, symbol in enumerate(["A", "B", "C", "D", "E", "F"]):
        rng = np.random.default_rng(i)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.01, n)), index=dates)
        out[symbol] = pd.DataFrame(
            {"open": close * 0.999, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1.0},
            index=dates,
        )
    return out


def test_regime_breakdown_ic_separates_by_label() -> None:
    candidates = _candidates()
    n = 300
    regime_labels = ["bull"] * 150 + ["bear"] * 150
    panel = [
        CrossSectionalSnapshot(
            scan_index=i, scan_time=0,
            scores={s: float(df["close"].iloc[i - 1]) for s, df in candidates.items()},
            excluded=[],
        )
        for i in range(50, n, 24)
    ]
    breakdown = regime_breakdown_ic(panel, candidates, regime_labels, horizon_candles=10)
    assert set(breakdown) <= {"bull", "bear"}
    summary = summarize_regime_breakdown(breakdown)
    assert "sign_reversal" in summary


def test_summarize_regime_breakdown_empty() -> None:
    summary = summarize_regime_breakdown({})
    assert summary["n_regimes_with_data"] == 0
