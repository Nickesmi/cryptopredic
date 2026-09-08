"""Distribution-drift monitoring (Phase 15 of the Phase-2 validation brief).

Detects when a live population of values (predictions, features, targets,
errors, or confidence scores) has drifted away from a reference population
(typically the training-period or an earlier evaluation window), using the
Population Stability Index (PSI) — a standard, simple, well-understood
metric from credit-risk/ML-monitoring practice:

    PSI = sum_over_bins( (p_current - p_reference) * ln(p_current / p_reference) )

Conventional thresholds (used here): PSI < 0.1 -> no significant drift,
0.1-0.25 -> moderate drift (watch), > 0.25 -> significant drift (act).

This module only *detects and reports* drift. Per the brief ("Do not
automatically retrain unless there is a validated reason to do so"), it
deliberately does not trigger any retraining or auto-remediation itself —
that decision requires human judgement about *why* the drift happened
(regime change vs. genuine model decay vs. a data-quality issue upstream),
which this module cannot determine on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DriftReport:
    metric_name: str
    psi: float
    severity: str  # "none" | "moderate" | "significant"
    reference_n: int
    current_n: int
    reference_mean: float
    current_mean: float

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "psi": self.psi,
            "severity": self.severity,
            "reference_n": self.reference_n,
            "current_n": self.current_n,
            "reference_mean": self.reference_mean,
            "current_mean": self.current_mean,
        }


def _classify_psi(psi: float) -> str:
    if psi < 0.1:
        return "none"
    if psi < 0.25:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], n_bins: int = 10
) -> float:
    """Compute the PSI between two 1D samples.

    Bin edges are quantiles of the *reference* sample (so each reference
    bin holds ~1/n_bins of the reference population by construction);
    the current sample is then binned against those same fixed edges.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < n_bins or len(cur) == 0:
        raise ValueError("Not enough data to compute a stable PSI.")

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        # Degenerate reference distribution (near-constant) -- widen edges
        # slightly so both samples can still be binned meaningfully.
        edges = np.array([ref.min() - 1e-9, ref.mean(), ref.max() + 1e-9])

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_frac = ref_counts / ref_counts.sum()
    cur_frac = cur_counts / cur_counts.sum()

    # Laplace smoothing to avoid log(0) / division by zero in empty bins.
    epsilon = 1e-6
    ref_frac = np.where(ref_frac == 0, epsilon, ref_frac)
    cur_frac = np.where(cur_frac == 0, epsilon, cur_frac)

    psi = float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))
    return psi


def detect_drift(
    reference: Sequence[float], current: Sequence[float], metric_name: str, n_bins: int = 10
) -> DriftReport:
    """Compute a full :class:`DriftReport` for one named metric."""
    psi = population_stability_index(reference, current, n_bins=n_bins)
    ref_arr = np.asarray(reference, dtype=float)
    cur_arr = np.asarray(current, dtype=float)
    return DriftReport(
        metric_name=metric_name,
        psi=psi,
        severity=_classify_psi(psi),
        reference_n=int(np.isfinite(ref_arr).sum()),
        current_n=int(np.isfinite(cur_arr).sum()),
        reference_mean=float(np.nanmean(ref_arr)),
        current_mean=float(np.nanmean(cur_arr)),
    )


def monitor_prediction_pipeline_drift(
    reference_rows: Sequence[dict],
    current_rows: Sequence[dict],
    feature_columns: Sequence[str] | None = None,
    n_bins: int = 10,
) -> dict[str, DriftReport]:
    """Run the five drift checks the brief calls out, wherever the needed
    field is present in both row sets: prediction distribution, feature
    distribution (one report per feature column), target/actual-return
    distribution, error distribution, and confidence distribution.

    Row dicts are expected in the shape produced by
    ``PredictionStore.evaluated_rows()`` / a walk-forward prediction ledger:
    ``predicted_price``, ``anchor_price``, ``actual_price``,
    ``absolute_error`` (or ``price_error``), ``confidence``, and optionally
    feature columns matching *feature_columns*.

    Returns:
        ``{metric_name: DriftReport}`` for every check that had enough data
        in both row sets to compute (a check is silently skipped, not
        fabricated, if a required field is missing).
    """
    reports: dict[str, DriftReport] = {}

    def _try(name: str, ref_values: list[float], cur_values: list[float]) -> None:
        if len(ref_values) >= n_bins and len(cur_values) > 0:
            reports[name] = detect_drift(ref_values, cur_values, name, n_bins=n_bins)

    def _predicted_return(row: dict) -> float | None:
        anchor = row.get("anchor_price")
        predicted = row.get("predicted_price")
        if anchor in (None, 0) or predicted is None:
            return None
        return (predicted - anchor) / anchor

    def _actual_return(row: dict) -> float | None:
        anchor = row.get("anchor_price")
        actual = row.get("actual_price")
        if anchor in (None, 0) or actual is None:
            return None
        return (actual - anchor) / anchor

    pred_returns_ref = [r for r in map(_predicted_return, reference_rows) if r is not None]
    pred_returns_cur = [r for r in map(_predicted_return, current_rows) if r is not None]
    _try("prediction_distribution", pred_returns_ref, pred_returns_cur)

    actual_returns_ref = [r for r in map(_actual_return, reference_rows) if r is not None]
    actual_returns_cur = [r for r in map(_actual_return, current_rows) if r is not None]
    _try("target_distribution", actual_returns_ref, actual_returns_cur)

    error_field = "absolute_error" if any("absolute_error" in r for r in reference_rows) else "price_error"
    errors_ref = [float(r[error_field]) for r in reference_rows if error_field in r]
    errors_cur = [float(r[error_field]) for r in current_rows if error_field in r]
    _try("error_distribution", errors_ref, errors_cur)

    conf_ref = [float(r["confidence"]) for r in reference_rows if "confidence" in r]
    conf_cur = [float(r["confidence"]) for r in current_rows if "confidence" in r]
    _try("confidence_distribution", conf_ref, conf_cur)

    for col in feature_columns or []:
        ref_vals = [float(r[col]) for r in reference_rows if col in r and r[col] is not None]
        cur_vals = [float(r[col]) for r in current_rows if col in r and r[col] is not None]
        _try(f"feature:{col}", ref_vals, cur_vals)

    return reports


def summarise_drift(reports: dict[str, DriftReport]) -> dict:
    """Roll a set of drift reports up into an at-a-glance summary."""
    significant = [name for name, r in reports.items() if r.severity == "significant"]
    moderate = [name for name, r in reports.items() if r.severity == "moderate"]
    return {
        "n_metrics_checked": len(reports),
        "significant_drift": significant,
        "moderate_drift": moderate,
        "any_significant_drift": bool(significant),
        "details": {name: r.to_dict() for name, r in reports.items()},
    }
