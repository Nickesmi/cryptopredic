"""Opportunity scoring — the core of the High-Potential Asset Discovery Engine.

Combines transparent, individually-inspectable sub-scores into one
0-100 "Opportunity Score". Every sub-score and its weight is named
explicitly (see ``DEFAULT_WEIGHTS``) precisely so nothing about the
ranking is a black box — the audit's Phase 19 requirement is a
*transparent* score, not a fabricated single number.

Weights below are analyst-set defaults, not fit to historical data —
this codebase does not yet have a labelled "good opportunity" dataset to
optimise against, and fitting weights on the same history used to
validate them would be exactly the kind of overfitting the audit is
supposed to guard against. ``src/evaluation/walk_forward.py`` /
``benchmark.py`` are the tools to use to validate any future re-weighting
out-of-sample before trusting it — see docs/TIMING_AUDIT_REPORT.md.

Explicitly distinguishing "high potential" from "high volatility": every
sub-score here rewards *quality of trend/structure*, not magnitude of
recent price change. A coin that already went vertical scores worse on
``trend_quality`` (RSI overextension penalty) and ``volatility_adjusted``
(same move, more noise) than a coin grinding out a steady, orderly trend
of the same size — see ``_trend_quality`` and ``_volatility_adjusted``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.preprocessing import add_log_returns, clean_ohlcv
from src.features.feature_pipeline import FeaturePipeline

DEFAULT_WEIGHTS: dict[str, float] = {
    "trend_quality": 0.25,
    "momentum": 0.20,
    "relative_strength": 0.20,
    "volume_confirmation": 0.15,
    "volatility_adjusted": 0.20,
}

assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9

# Bump this whenever a sub-score's *formula* changes (not just its
# weight) -- score_coins.py has no fitted model to fingerprint the way
# src/models/model_manager.py does, so the scanner's "version" is a
# fingerprint of its configuration (weights) plus this manually-bumped
# formula identity. See scanner_version() below.
_RISK_PENALTY_FORMULA_VERSION = "risk_penalty_v1"


def scanner_version(weights: dict[str, float] = DEFAULT_WEIGHTS) -> str:
    """Deterministic provenance fingerprint for the opportunity-scoring configuration.

    Phase 4 audit, Section 20 ("every live recommendation should record
    ... scanner version"): before this, ``OpportunityRecommendation`` had
    no version field at all, mirroring the exact gap Phase 2 found and
    fixed for the price forecaster's ``model_version``/``feature_version``.
    Two scans get the same ``scanner_version`` iff they used the same
    sub-score weights and the same formula identity — changing a weight,
    or bumping ``_RISK_PENALTY_FORMULA_VERSION`` when a formula changes,
    changes this fingerprint.
    """
    payload = {"weights": weights, "risk_penalty_formula": _RISK_PENALTY_FORMULA_VERSION}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:12]


@dataclass
class SubScores:
    trend_quality: float
    momentum: float
    relative_strength: float
    volume_confirmation: float
    volatility_adjusted: float

    def weighted_total(self, weights: dict[str, float] = DEFAULT_WEIGHTS) -> float:
        return sum(getattr(self, key) * weight for key, weight in weights.items())


@dataclass
class OpportunityScore:
    raw_score: float          # 0-100, before the risk penalty
    risk_penalty: float       # 0-100, subtracted from raw_score
    final_score: float        # 0-100, clipped
    sub_scores: SubScores
    reasons: list[str] = field(default_factory=list)


def _rolling_return(closes: pd.Series, lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    return float(closes.iloc[-1] / closes.iloc[-lookback - 1] - 1.0)


def _trend_quality(feature_df: pd.DataFrame) -> tuple[float, list[str]]:
    """Reward a clean, orderly trend; penalise overextension (RSI > 80/ < 20)."""
    reasons = []
    close = feature_df["close"]
    sma_short = feature_df.get("sma_7", close)
    sma_long = feature_df.get("sma_30", close)

    above_short = float((close.tail(20) > sma_short.tail(20)).mean())
    trend_alignment = float(sma_short.iloc[-1] > sma_long.iloc[-1])

    rsi_col = [c for c in feature_df.columns if c.startswith("rsi_")]
    rsi = float(feature_df[rsi_col[0]].iloc[-1]) if rsi_col else 50.0

    score = 50.0 * above_short + 30.0 * trend_alignment
    if 45 <= rsi <= 68:
        score += 20.0  # healthy trend, room to run
        reasons.append(f"RSI at {rsi:.0f} — trending without being overextended")
    elif rsi > 80:
        score -= 25.0  # vertical / overextended — "already pumped", not "high potential"
        reasons.append(f"RSI at {rsi:.0f} — overextended, high odds of mean reversion")
    elif rsi < 30:
        score -= 10.0
        reasons.append(f"RSI at {rsi:.0f} — downtrend, no confirmed reversal yet")

    if trend_alignment and above_short > 0.6:
        reasons.append("Price holding above both short- and long-term moving averages")

    return float(np.clip(score, 0.0, 100.0)), reasons


def _momentum(closes: pd.Series) -> tuple[float, list[str]]:
    r7 = _rolling_return(closes, 7)
    r14 = _rolling_return(closes, 14)
    r30 = _rolling_return(closes, 30)
    # Reward positive, *accelerating* momentum (short-term > long-term),
    # not just any positive number — this is what separates early-stage
    # continuation from a move that is already decelerating.
    accelerating = r7 > r14 / 2 > 0
    score = 50.0 + 100.0 * np.clip((0.5 * r7 + 0.3 * r14 + 0.2 * r30), -0.5, 0.5)
    reasons = []
    if accelerating:
        score += 10.0
        reasons.append(f"Momentum accelerating: 7d return {r7:+.1%} vs 14d {r14:+.1%}")
    return float(np.clip(score, 0.0, 100.0)), reasons


def _relative_strength(closes: pd.Series, benchmark_closes: pd.Series | None) -> tuple[float, list[str]]:
    if benchmark_closes is None or len(benchmark_closes) < 15:
        return 50.0, []
    asset_return = _rolling_return(closes, 14)
    benchmark_return = _rolling_return(benchmark_closes, 14)
    excess = asset_return - benchmark_return
    # Cap the reward: wildly outperforming BTC by e.g. +300% in 14 candles
    # is the "moon coin" pattern the audit explicitly says to avoid
    # chasing, not a stronger signal than a solid, moderate outperformance.
    capped_excess = float(np.clip(excess, -0.5, 0.25))
    score = 50.0 + capped_excess * 150.0
    reasons = []
    if excess > 0.05:
        reasons.append(f"Outperforming BTC by {excess:+.1%} over the last 14 candles")
    elif excess < -0.05:
        reasons.append(f"Underperforming BTC by {excess:+.1%} over the last 14 candles")
    return float(np.clip(score, 0.0, 100.0)), reasons


def _volume_confirmation(volume: pd.Series) -> tuple[float, list[str]]:
    if len(volume) < 20:
        return 50.0, []
    recent = float(volume.tail(5).mean())
    baseline = float(volume.tail(30).mean()) or 1e-9
    ratio = recent / baseline
    reasons = []
    if 1.2 <= ratio <= 4.0:
        score = 50.0 + (ratio - 1.0) * 20.0
        reasons.append(f"Volume expanding {ratio:.1f}x vs its 30-candle average")
    elif ratio > 4.0:
        # Extreme, sudden volume spikes are exactly the "unusual volume"
        # pattern associated with manipulation/news shocks, not a
        # confirmed structural trend — reward it far more cautiously.
        score = 55.0
        reasons.append(f"Volume spike {ratio:.1f}x — treat as unconfirmed until it persists")
    else:
        score = 50.0 * ratio
    return float(np.clip(score, 0.0, 100.0)), reasons


def _volatility_adjusted(closes: pd.Series) -> tuple[float, list[str]]:
    log_returns = np.log(closes / closes.shift(1)).dropna()
    if len(log_returns) < 10:
        return 50.0, []
    mean_return = float(log_returns.tail(30).mean())
    vol = float(log_returns.tail(30).std()) or 1e-9
    sharpe_like = mean_return / vol
    score = 50.0 + np.clip(sharpe_like, -3.0, 3.0) * 15.0
    reasons = []
    if sharpe_like > 0.5:
        reasons.append("Smooth, low-noise trend relative to its own volatility")
    elif sharpe_like < -0.5:
        reasons.append("Choppy/noisy price action relative to its trend")
    return float(np.clip(score, 0.0, 100.0)), reasons


def compute_opportunity_score(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    risk_score: float,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> OpportunityScore:
    """Compute the transparent Opportunity Score for one asset.

    Args:
        df:            OHLCV history for the candidate asset, ascending,
                       already past the liquidity filter.
        benchmark_df:  OHLCV history for the reference asset (BTC), same
                       timeframe/alignment, or ``None`` to skip relative
                       strength (scored neutral).
        risk_score:    0-100 risk score from ``risk_score.assess_risk``,
                       subtracted from the raw opportunity score as a
                       penalty (never blended into the sub-scores).
        weights:       Sub-score weights, must sum to 1.0.
    """
    clean_df = add_log_returns(clean_ohlcv(df))
    feature_df = FeaturePipeline().build(clean_df)
    if feature_df.empty:
        raise ValueError("Not enough history to build features for opportunity scoring.")

    benchmark_closes = None
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_closes = clean_ohlcv(benchmark_df)["close"]

    reasons: list[str] = []

    trend_score, r = _trend_quality(feature_df)
    reasons += r
    momentum_score, r = _momentum(feature_df["close"])
    reasons += r
    rel_strength_score, r = _relative_strength(feature_df["close"], benchmark_closes)
    reasons += r
    volume_score, r = _volume_confirmation(clean_df["volume"])
    reasons += r
    vol_adjusted_score, r = _volatility_adjusted(feature_df["close"])
    reasons += r

    sub_scores = SubScores(
        trend_quality=trend_score,
        momentum=momentum_score,
        relative_strength=rel_strength_score,
        volume_confirmation=volume_score,
        volatility_adjusted=vol_adjusted_score,
    )
    raw_score = sub_scores.weighted_total(weights)
    # Risk is a penalty, capped so a merely "medium risk" asset isn't
    # automatically disqualified — only high risk meaningfully drags the
    # score down.
    risk_penalty = max(0.0, risk_score - 30.0) * 0.5
    final_score = float(np.clip(raw_score - risk_penalty, 0.0, 100.0))

    return OpportunityScore(
        raw_score=raw_score,
        risk_penalty=risk_penalty,
        final_score=final_score,
        sub_scores=sub_scores,
        reasons=reasons,
    )
