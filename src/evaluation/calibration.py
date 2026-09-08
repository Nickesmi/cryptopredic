"""Confidence calibration — bucketed reliability analysis.

Phase 16/Phase-2 #8 of the audit: the system displays a "confidence" score
on every prediction (currently a volatility-band heuristic — see
``src/models/model_manager.py``'s ``confidence`` computation), but nothing
in the codebase checked whether that number means anything. Before this
module, ``src/evaluation/metrics.py::aggregate_metrics`` computed a single
scalar (``1 - |mean_confidence - accuracy|``) over an entire population,
which can look perfectly calibrated on average while being badly wrong in
every individual confidence bucket (e.g. always under-confident at 50-60%
and over-confident at 90-100%, cancelling out in the mean).

This module buckets predictions by their stated confidence (50-60%,
60-70%, ..., 90-100% by default) and reports, per bucket, how often the
prediction actually succeeded — the only way to answer "if the system says
80% confidence, do 80% of those actually come true?" honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass(frozen=True)
class CalibrationBucket:
    label: str
    lower: float
    upper: float
    n: int
    predicted_confidence_mean: float | None
    actual_success_rate: float | None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "range": [self.lower, self.upper],
            "n": self.n,
            "predicted_confidence_mean": self.predicted_confidence_mean,
            "actual_success_rate": self.actual_success_rate,
            "gap": (
                None
                if self.predicted_confidence_mean is None or self.actual_success_rate is None
                else self.predicted_confidence_mean - self.actual_success_rate
            ),
        }


@dataclass(frozen=True)
class CalibrationReport:
    buckets: list[CalibrationBucket]
    brier_score: float
    reliability: float  # Murphy decomposition's reliability term -- lower is better, 0 is perfect
    n_total: int

    def to_dict(self) -> dict:
        return {
            "buckets": [b.to_dict() for b in self.buckets],
            "brier_score": self.brier_score,
            "reliability": self.reliability,
            "n_total": self.n_total,
        }

    def is_well_calibrated(self, max_gap: float = 0.10, min_bucket_n: int = 20) -> bool:
        """A blunt pass/fail: every bucket with enough samples must be
        within ``max_gap`` of its own stated confidence. Buckets with too
        few samples to be statistically meaningful are ignored rather than
        allowed to silently pass — call ``n_total`` yourself to check
        overall sample size first.
        """
        checked = [b for b in self.buckets if b.n >= min_bucket_n]
        if not checked:
            return False
        return all(abs(b.to_dict()["gap"]) <= max_gap for b in checked)


DEFAULT_BUCKET_EDGES: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def compute_calibration(
    confidences: Sequence[float],
    successes: Sequence[bool],
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
) -> CalibrationReport:
    """Compute a bucketed calibration report.

    Args:
        confidences: Stated confidence for each prediction, in [0, 1].
        successes:   Whether each prediction actually succeeded, by
                     whichever success definition the caller chose (see
                     ``src/evaluation/success_criteria.py`` — directional,
                     price, target, or horizon success are all valid and
                     will give different calibration verdicts for the same
                     confidence scores, which is expected and worth
                     comparing).
        bucket_edges: Confidence bucket boundaries. Defaults to
                     ``[0-50%), [50-60%), [60-70%), [70-80%), [80-90%),
                     [90-100%]`` per the audit brief.

    Returns:
        A :class:`CalibrationReport`.
    """
    n = len(confidences)
    if n != len(successes):
        raise ValueError("confidences and successes must be the same length")
    if n == 0:
        raise ValueError("Cannot compute calibration over zero predictions.")

    outcomes = [1.0 if s else 0.0 for s in successes]
    brier_score = sum((c - o) ** 2 for c, o in zip(confidences, outcomes)) / n

    buckets: list[CalibrationBucket] = []
    reliability_sum = 0.0
    for lower, upper in zip(bucket_edges[:-1], bucket_edges[1:]):
        indices = [i for i in range(n) if lower <= confidences[i] < upper]
        bucket_n = len(indices)
        if bucket_n == 0:
            predicted_mean = None
            success_rate = None
        else:
            predicted_mean = sum(confidences[i] for i in indices) / bucket_n
            success_rate = sum(outcomes[i] for i in indices) / bucket_n
            reliability_sum += bucket_n * (predicted_mean - success_rate) ** 2

        label = f"{lower:.0%}-{min(upper, 1.0):.0%}"
        buckets.append(
            CalibrationBucket(
                label=label,
                lower=lower,
                upper=min(upper, 1.0),
                n=bucket_n,
                predicted_confidence_mean=predicted_mean,
                actual_success_rate=success_rate,
            )
        )

    reliability = reliability_sum / n

    return CalibrationReport(
        buckets=buckets,
        brier_score=brier_score,
        reliability=reliability,
        n_total=n,
    )


def calibration_from_rows(
    rows: Sequence[dict],
    confidence_field: str = "confidence",
    success_field: str = "direction_correct",
    bucket_edges: Sequence[float] = DEFAULT_BUCKET_EDGES,
) -> CalibrationReport:
    """Convenience wrapper: compute calibration from a list of row dicts
    (e.g. ``PredictionStore.evaluated_rows()`` output, or a walk-forward
    prediction ledger).
    """
    confidences = [float(r[confidence_field]) for r in rows]
    successes = [bool(r[success_field]) for r in rows]
    return compute_calibration(confidences, successes, bucket_edges=bucket_edges)
