"""Regime/period stability breakdown for a signal's IC series (Phase 5, Section 11).

"A real candidate should not work only in one narrow period ... Look for
sign reversals, performance collapse, concentration, dependence on a
single regime." This module buckets a signal's per-snapshot IC by the
regime label active at that snapshot's anchor candle, and reports
``compute_ic_stats`` separately per bucket -- exactly what's needed to
see a sign flip or a collapse that an unconditional (pooled) IC would
average away and hide.
"""

from __future__ import annotations

from typing import Any

from src.evaluation.cross_sectional_analysis import (
    CrossSectionalSnapshot,
    forward_return,
    information_coefficient,
)
from src.research.statistics import ICStats, compute_ic_stats


def regime_breakdown_ic(
    panel: list[CrossSectionalSnapshot],
    candidates: dict,
    regime_labels: list[str],
    horizon_candles: int,
    method: str = "spearman",
) -> dict[str, ICStats]:
    """Per-regime IC summary. ``regime_labels`` must be indexed the same way as the candidates' DatetimeIndex."""
    ic_by_regime: dict[str, list[float]] = {}

    for snap in panel:
        anchor_idx = snap.scan_index - 1
        if anchor_idx < 0 or anchor_idx >= len(regime_labels):
            continue
        regime = regime_labels[anchor_idx]
        returns = {
            symbol: r
            for symbol in snap.scores
            if (r := forward_return(candidates[symbol], snap.scan_index, horizon_candles)) is not None
        }
        ic = information_coefficient(snap.scores, returns, method=method)
        if ic is not None:
            ic_by_regime.setdefault(regime, []).append(ic)

    return {regime: compute_ic_stats(values) for regime, values in ic_by_regime.items()}


def summarize_regime_breakdown(breakdown: dict[str, ICStats]) -> dict[str, Any]:
    """Flags sign reversal / concentration across the regime-conditional IC results."""
    means = {regime: stats.mean_ic for regime, stats in breakdown.items() if stats.mean_ic is not None}
    if not means:
        return {"n_regimes_with_data": 0, "sign_reversal": None, "means": {}}
    signs = {v > 0 for v in means.values()}
    return {
        "n_regimes_with_data": len(means),
        "sign_reversal": len(signs) > 1,
        "means": means,
        "max_abs_ic_regime": max(means, key=lambda r: abs(means[r])),
        "min_abs_ic_regime": min(means, key=lambda r: abs(means[r])),
    }
