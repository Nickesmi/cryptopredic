"""Tests for the High-Potential Asset Discovery Engine (opportunity scanner).

Covers the anti-"moon coin" requirement (Phase 19 of the timing audit):
illiquid and abnormally-spiky assets must be excluded outright, and the
scanner must never force a recommendation when nothing qualifies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ranking.liquidity_filter import LiquidityFilterConfig, check_liquidity
from src.ranking.recommend import (
    NO_OPPORTUNITY_MESSAGE,
    scan_candidate,
    scan_opportunities,
)
from src.ranking.risk_score import assess_risk
from src.ranking.score_coins import compute_opportunity_score


def _make_ohlcv(close: np.ndarray, volume: float, start="2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(close), freq="D")
    close_s = pd.Series(close, index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close_s * 0.999,
            "high": close_s * 1.01,
            "low": close_s * 0.99,
            "close": close_s,
            "volume": volume,
        },
        index=dates,
    )


def _smooth_uptrend(n: int = 250) -> np.ndarray:
    t = np.arange(n)
    return 100 + 20 * np.sin(2 * np.pi * t / 30) + t * 0.15


def _btc_like_benchmark(n: int = 250) -> pd.DataFrame:
    t = np.arange(n)
    close = 40_000 + 4_000 * np.sin(2 * np.pi * t / 40) + t * 5
    return _make_ohlcv(close, volume=50_000_000.0)


class TestLiquidityFilter:
    def test_passes_healthy_market(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=1_000_000.0)
        verdict = check_liquidity(df)
        assert verdict.passed
        assert verdict.reasons == []

    def test_rejects_thin_volume(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=5.0)
        verdict = check_liquidity(df)
        assert not verdict.passed
        assert any("low_liquidity" in r for r in verdict.reasons)

    def test_rejects_short_history(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(n=10), volume=5_000_000.0)
        verdict = check_liquidity(df)
        assert not verdict.passed
        assert any("insufficient_history" in r for r in verdict.reasons)

    def test_rejects_vertical_pump(self) -> None:
        # Flat for a long stretch, then a single ~150% candle — the
        # "already pumped" / manipulation-like pattern the scanner must
        # not treat as a liquid, legitimate opportunity.
        close = np.concatenate([np.full(200, 10.0), np.full(50, 25.0)])
        df = _make_ohlcv(close, volume=5_000_000.0)
        verdict = check_liquidity(df, LiquidityFilterConfig())
        assert not verdict.passed
        assert any(
            "abnormal_price_action" in r or "wide_spread_proxy" in r for r in verdict.reasons
        )


class TestRiskScore:
    def test_low_volatility_series_scores_low_risk(self) -> None:
        close = _smooth_uptrend()
        # Dampen amplitude to make this genuinely low-volatility.
        close = 100 + (close - 100) * 0.05
        df = _make_ohlcv(close, volume=5_000_000.0)
        assessment = assess_risk(df, avg_quote_volume=5_000_000.0 * 100)
        assert assessment.risk_level == "low"

    def test_high_volatility_series_scores_higher_risk(self) -> None:
        rng = np.random.default_rng(7)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.12, 250))
        df = _make_ohlcv(close, volume=5_000_000.0)
        assessment = assess_risk(df, avg_quote_volume=5_000_000.0 * 100)
        assert assessment.risk_score > 30.0


class TestOpportunityScore:
    def test_sub_scores_bounded_0_to_100(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=2_000_000.0)
        btc = _btc_like_benchmark()
        score = compute_opportunity_score(df, btc, risk_score=20.0)
        for value in (
            score.sub_scores.trend_quality,
            score.sub_scores.momentum,
            score.sub_scores.relative_strength,
            score.sub_scores.volume_confirmation,
            score.sub_scores.volatility_adjusted,
            score.final_score,
        ):
            assert 0.0 <= value <= 100.0

    def test_risk_penalty_reduces_final_score(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=2_000_000.0)
        btc = _btc_like_benchmark()
        low_risk = compute_opportunity_score(df, btc, risk_score=10.0)
        high_risk = compute_opportunity_score(df, btc, risk_score=90.0)
        assert high_risk.final_score < low_risk.final_score
        assert low_risk.raw_score == pytest.approx(high_risk.raw_score)


class TestScanOpportunities:
    def test_no_forced_recommendation_when_bar_not_met(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=2_000_000.0)
        btc = _btc_like_benchmark()
        result = scan_opportunities(
            {"AAA": df}, benchmark_df=btc, timeframe="1D", min_opportunity_score=99.9
        )
        assert result.recommendations == []
        assert result.message == NO_OPPORTUNITY_MESSAGE

    def test_illiquid_candidate_is_excluded_not_scored_low(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=1.0)
        btc = _btc_like_benchmark()
        result = scan_opportunities({"THIN": df}, benchmark_df=btc, timeframe="1D")
        assert result.recommendations == []
        assert "THIN" in result.excluded
        assert any("low_liquidity" in r for r in result.excluded["THIN"])

    def test_qualifying_candidate_is_recommended_with_reasons(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=2_000_000.0)
        btc = _btc_like_benchmark()
        result = scan_opportunities(
            {"AAA": df}, benchmark_df=btc, timeframe="1D", min_opportunity_score=1.0
        )
        assert len(result.recommendations) == 1
        rec = result.recommendations[0]
        assert rec.symbol == "AAA"
        assert 0.0 <= rec.opportunity_score <= 100.0
        assert rec.direction in ("bullish", "bearish")
        assert rec.risk_level in ("low", "medium", "high")

    def test_scan_candidate_returns_none_and_reasons_when_excluded(self) -> None:
        df = _make_ohlcv(_smooth_uptrend(), volume=1.0)
        btc = _btc_like_benchmark()
        recommendation, reasons = scan_candidate("THIN", df, btc, "1D")
        assert recommendation is None
        assert reasons
