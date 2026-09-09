"""Signal combination for ablation studies (Phase 5, Section 12).

"For the strongest candidate signals: test A alone, B alone, C alone,
A+B, A+C, B+C, A+B+C. Measure whether each added signal provides genuine
incremental information. Do NOT combine everything simply because the
combined backtest looks better."

Combining two signals that live on different natural scales (a momentum
return of ~0.01 and an RSI-like oscillator of ~50) by simply averaging
their raw values would let whichever happens to have larger magnitude
dominate for no principled reason. This module cross-sectionally
z-scores each signal's scores *within each snapshot* before combining --
the standard fix, and one that also makes the combination automatically
robust to a signal's arbitrary sign/scale convention.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.cross_sectional_analysis import CrossSectionalSnapshot


def zscore_snapshot_scores(scores: dict[str, float]) -> dict[str, float]:
    """Cross-sectional z-score of one snapshot's scores. Degenerate (std=0 or <2 assets) returns zeros."""
    if len(scores) < 2:
        return {s: 0.0 for s in scores}
    values = np.array(list(scores.values()))
    std = values.std(ddof=1)
    if std == 0:
        return {s: 0.0 for s in scores}
    mean = values.mean()
    return {s: float((v - mean) / std) for s, v in scores.items()}


def combine_panels(
    panels: list[list[CrossSectionalSnapshot]], weights: list[float] | None = None
) -> list[CrossSectionalSnapshot]:
    """Combine several signal panels (built over the SAME scan points) into one.

    Each panel's per-snapshot scores are cross-sectionally z-scored, then
    combined by a (weighted) average over the components that scored that
    asset at that snapshot. An asset must be scored by at least one
    component to appear in the combined snapshot; ``excluded`` on the
    combined snapshot is the union of every component's exclusions for
    assets never scored by any component.

    Requires every panel to have the same sequence of ``scan_index``
    values (i.e. built with the same ``scan_every_candles`` and
    ``min_train_size``) -- this is a precondition, not something this
    function tries to reconcile, because silently realigning mismatched
    scan grids would be exactly the kind of implicit behavior the audit's
    look-ahead discipline is built to avoid.
    """
    if not panels:
        raise ValueError("panels must be non-empty")
    n_snapshots = len(panels[0])
    for p in panels:
        if len(p) != n_snapshots:
            raise ValueError("All panels must have the same number of snapshots.")
    weights = weights or [1.0] * len(panels)
    if len(weights) != len(panels):
        raise ValueError("weights must match the number of panels.")

    combined: list[CrossSectionalSnapshot] = []
    for i in range(n_snapshots):
        snaps = [p[i] for p in panels]
        scan_indices = {s.scan_index for s in snaps}
        if len(scan_indices) != 1:
            raise ValueError(f"Panels are not aligned at position {i}: scan_index values differ {scan_indices}.")

        zscored = [zscore_snapshot_scores(s.scores) for s in snaps]
        all_symbols = set().union(*(z.keys() for z in zscored))
        combined_scores: dict[str, float] = {}
        for symbol in all_symbols:
            contributions = [(zscored[j][symbol], weights[j]) for j in range(len(panels)) if symbol in zscored[j]]
            if not contributions:
                continue
            total_weight = sum(w for _, w in contributions)
            combined_scores[symbol] = sum(v * w for v, w in contributions) / total_weight if total_weight else 0.0

        excluded = sorted(set().union(*(set(s.excluded) for s in snaps)) - set(combined_scores))
        combined.append(
            CrossSectionalSnapshot(
                scan_index=snaps[0].scan_index, scan_time=snaps[0].scan_time,
                scores=combined_scores, excluded=excluded,
            )
        )
    return combined
