"""High-Potential Asset Discovery Engine — top-level orchestration.

Ties together the liquidity gate, risk assessment, opportunity scoring,
and the empirical time-to-target estimator into the single entry point
the API/dashboard calls: :func:`scan_opportunities`.

This is a *separate* subsystem from the per-symbol price forecaster in
``src/models/model_manager.py`` (Phase 19 of the timing audit is explicit
that discovery is not "predict the price of coins already selected") —
its job is to rank a universe of candidate assets by risk-adjusted
opportunity, and to say nothing at all when nothing qualifies rather
than force a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.preprocessing import clean_ohlcv
from src.evaluation.time_to_target import (
    DEFAULT_HORIZON_BUCKETS_SECONDS,
    most_likely_horizon,
    time_to_target_report,
)
from src.features.feature_pipeline import FeaturePipeline
from src.ranking.liquidity_filter import LiquidityFilterConfig, check_liquidity
from src.ranking.risk_score import assess_risk
from src.ranking.score_coins import DEFAULT_WEIGHTS, compute_opportunity_score
from src.ranking.score_coins import scanner_version as compute_scanner_version
from src.utils.regime import classify_regime
from src.utils.timeframes import timeframe_to_seconds

DEFAULT_MIN_OPPORTUNITY_SCORE = 65.0
NO_OPPORTUNITY_MESSAGE = "No high-confidence opportunity currently meets the system's criteria."


@dataclass
class OpportunityRecommendation:
    symbol: str
    current_price: float
    opportunity_score: float
    direction: str  # "bullish" | "bearish" | "neutral"
    potential_target: float
    expected_horizon_seconds: int | None
    probability_estimate: float | None
    risk_level: str
    market_regime: str = "sideways"       # the candidate's own recent regime
    benchmark_regime: str = "sideways"    # the reference asset's (e.g. BTC) regime
    reasons: list[str] = field(default_factory=list)
    invalidation_price: float = 0.0
    key_risks: list[str] = field(default_factory=list)
    scanner_version: str = ""             # fingerprint of the scoring weights/formula used
    feature_version: str = ""             # fingerprint of the FeaturePipeline config used
    scanned_at: str = ""                  # ISO-8601 UTC timestamp the scan was generated

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "current_price": round(self.current_price, 8),
            "opportunity_score": round(self.opportunity_score, 1),
            "direction": self.direction,
            "potential_target": round(self.potential_target, 8),
            "expected_horizon_seconds": self.expected_horizon_seconds,
            "probability_estimate": (
                None if self.probability_estimate is None else round(self.probability_estimate, 3)
            ),
            "risk_level": self.risk_level,
            "market_regime": self.market_regime,
            "benchmark_regime": self.benchmark_regime,
            "reasons": self.reasons,
            "invalidation_price": round(self.invalidation_price, 8),
            "key_risks": self.key_risks,
            "scanner_version": self.scanner_version,
            "feature_version": self.feature_version,
            "scanned_at": self.scanned_at,
        }


@dataclass
class ScanResult:
    recommendations: list[OpportunityRecommendation]
    excluded: dict[str, list[str]]  # symbol -> reasons excluded (liquidity/data-quality)
    message: str | None = None

    def to_dict(self) -> dict:
        return {
            "recommendations": [r.to_dict() for r in self.recommendations],
            "excluded": self.excluded,
            "message": self.message,
        }


def _direction_and_target(feature_close: pd.Series) -> tuple[str, float, float]:
    """Derive direction, a volatility-scaled target return, and a target price."""
    log_returns = np.log(feature_close / feature_close.shift(1)).dropna()
    vol = float(log_returns.tail(30).std()) if len(log_returns) > 5 else 0.02
    vol = max(vol, 0.005)
    recent_trend = float(feature_close.iloc[-1] / feature_close.iloc[-min(15, len(feature_close))] - 1.0)
    direction = "bullish" if recent_trend >= 0 else "bearish"

    # A "reasonable" near-term target: ~1.5 stdev move in the trend's
    # direction — large enough to be a meaningful opportunity, small
    # enough that reaching it is plausible rather than a moonshot.
    target_return = 1.5 * vol if direction == "bullish" else -1.5 * vol
    current_price = float(feature_close.iloc[-1])
    target_price = current_price * (1 + target_return)
    return direction, target_return, target_price


def _invalidation_price(df: pd.DataFrame, direction: str, lookback: int = 20) -> float:
    recent = df.tail(lookback)
    if direction == "bullish":
        return float(recent["low"].min())
    return float(recent["high"].max())


def scan_candidate(
    symbol: str,
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None,
    timeframe: str,
    liquidity_config: LiquidityFilterConfig = LiquidityFilterConfig(),
) -> tuple[OpportunityRecommendation | None, list[str]]:
    """Score a single candidate. Returns ``(recommendation_or_None, exclusion_reasons)``."""
    liquidity = check_liquidity(df, liquidity_config)
    if not liquidity.passed:
        return None, liquidity.reasons

    clean_df = clean_ohlcv(df)
    risk = assess_risk(clean_df, avg_quote_volume=liquidity.avg_quote_volume)

    try:
        opportunity = compute_opportunity_score(
            df, benchmark_df, risk_score=risk.risk_score, weights=DEFAULT_WEIGHTS
        )
    except ValueError as exc:
        return None, [f"scoring_failed: {exc}"]

    direction, target_return, target_price = _direction_and_target(clean_df["close"])
    invalidation = _invalidation_price(clean_df, direction)

    market_regime = classify_regime(clean_df["close"])
    benchmark_regime = (
        classify_regime(clean_ohlcv(benchmark_df)["close"])
        if benchmark_df is not None and not benchmark_df.empty
        else "sideways"
    )

    expected_horizon_seconds: int | None = None
    probability_estimate: float | None = None
    try:
        max_candles = min(len(clean_df) - 1, 24 * 30)  # cap look-ahead window used for estimation
        if max_candles > 20:
            report = time_to_target_report(
                clean_df,
                timeframe=timeframe,
                target_return=target_return,
                max_candles=max_candles,
                horizon_buckets_seconds=DEFAULT_HORIZON_BUCKETS_SECONDS,
            )
            expected_horizon_seconds = most_likely_horizon(report)
            if expected_horizon_seconds is not None:
                probability_estimate = report.probability_within[expected_horizon_seconds]
            else:
                probability_estimate = report.hit_rate
    except ValueError:
        pass  # not enough history for an empirical time-to-target estimate

    key_risks: list[str] = []
    if risk.risk_level in ("medium", "high"):
        key_risks.append(f"{risk.risk_level.title()} volatility ({risk.volatility:.1%} per candle)")
    if risk.max_drawdown <= -0.20:
        key_risks.append(f"Recent drawdown of {risk.max_drawdown:.0%}")
    if liquidity.spread_proxy > 0.03:
        key_risks.append(f"Wide high-low spread proxy ({liquidity.spread_proxy:.1%}) — slippage risk")
    if probability_estimate is not None and probability_estimate < 0.5:
        key_risks.append(
            f"Empirical hit-rate for this target is only {probability_estimate:.0%} "
            "within the estimated horizon"
        )
    if direction == "bullish" and benchmark_regime == "bear":
        key_risks.append(
            "Bullish pick while the reference market is in a bear regime — swimming against the tide"
        )
    elif direction == "bearish" and benchmark_regime == "bull":
        key_risks.append(
            "Bearish pick while the reference market is in a bull regime — swimming against the tide"
        )

    recommendation = OpportunityRecommendation(
        symbol=symbol,
        current_price=float(clean_df["close"].iloc[-1]),
        opportunity_score=opportunity.final_score,
        direction=direction,
        potential_target=target_price,
        expected_horizon_seconds=expected_horizon_seconds,
        probability_estimate=probability_estimate,
        risk_level=risk.risk_level,
        market_regime=market_regime,
        benchmark_regime=benchmark_regime,
        reasons=opportunity.reasons,
        invalidation_price=invalidation,
        key_risks=key_risks,
        scanner_version=compute_scanner_version(DEFAULT_WEIGHTS),
        feature_version=FeaturePipeline().feature_version,
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )
    return recommendation, []


def scan_opportunities(
    candidates: dict[str, pd.DataFrame],
    benchmark_df: pd.DataFrame | None,
    timeframe: str = "1D",
    min_opportunity_score: float = DEFAULT_MIN_OPPORTUNITY_SCORE,
    liquidity_config: LiquidityFilterConfig = LiquidityFilterConfig(),
    top_n: int = 10,
) -> ScanResult:
    """Scan a universe of candidate assets and rank the ones worth surfacing.

    Args:
        candidates:     ``{symbol: OHLCV dataframe}`` for every asset to
                        evaluate (ascending, same timeframe).
        benchmark_df:   OHLCV history for the reference asset (typically
                        BTC) used for relative-strength scoring.
        timeframe:      Candle period string shared by all inputs.
        min_opportunity_score: Minimum final opportunity score (0-100) an
                        asset must clear to be surfaced at all. Assets
                        below this line are computed but not recommended
                        — the scanner never forces a pick.
        liquidity_config: Hard liquidity/data-quality gate applied before
                        scoring.
        top_n:          Maximum number of recommendations to return.

    Returns:
        A :class:`ScanResult`. If no candidate clears
        ``min_opportunity_score``, ``recommendations`` is empty and
        ``message`` is set to ``NO_OPPORTUNITY_MESSAGE`` — callers must
        display that message rather than inventing a pick.
    """
    scored: list[OpportunityRecommendation] = []
    excluded: dict[str, list[str]] = {}

    for symbol, df in candidates.items():
        recommendation, exclusion_reasons = scan_candidate(
            symbol, df, benchmark_df, timeframe, liquidity_config
        )
        if recommendation is None:
            excluded[symbol] = exclusion_reasons
            continue
        scored.append(recommendation)

    qualifying = [r for r in scored if r.opportunity_score >= min_opportunity_score]
    qualifying.sort(key=lambda r: r.opportunity_score, reverse=True)
    top = qualifying[:top_n]

    message = None if top else NO_OPPORTUNITY_MESSAGE
    return ScanResult(recommendations=top, excluded=excluded, message=message)
