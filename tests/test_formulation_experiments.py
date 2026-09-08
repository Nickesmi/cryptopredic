"""Tests for the target-formulation experiment harness (Phase 3, Sections 1-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import FeaturePipeline
from src.evaluation.formulation_experiments import (
    build_target,
    run_formulation_walk_forward,
    run_quantile_walk_forward,
    split_design_and_frozen_test,
)


def _multi_regime_df(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    t = np.arange(n)
    # Oscillating + mild drift -- bounded enough that "price" formulation
    # isn't guaranteed to blow up as catastrophically as the pure-trend
    # case, but still non-trivial for the harness to exercise fully.
    close = pd.Series(
        40000 + 3000 * np.sin(2 * np.pi * t / 200) + t * 2 + rng.normal(0, 50, n),
        index=dates,
    )
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 20,
            "low": close - 20,
            "close": close,
            "volume": 1000.0,
        },
        index=dates,
    )


def test_split_design_and_frozen_test_is_chronological_and_disjoint() -> None:
    df = _multi_regime_df()
    design, frozen = split_design_and_frozen_test(df, test_fraction=0.2)
    assert len(design) + len(frozen) == len(df)
    assert design.index[-1] < frozen.index[0]
    assert len(frozen) == pytest.approx(len(df) * 0.2, abs=1)


class TestBuildTarget:
    def test_price_target_is_future_close(self) -> None:
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        y, ret = build_target(df, horizon_candles=2, target_type="price")
        assert y[0] == pytest.approx(102.0)
        assert np.isnan(y[-1]) and np.isnan(y[-2])

    def test_return_target_matches_pct_change(self) -> None:
        df = pd.DataFrame({"close": [100.0, 110.0, 90.0]})
        y, ret = build_target(df, horizon_candles=1, target_type="return")
        assert y[0] == pytest.approx(0.10)
        assert y[1] == pytest.approx(-0.1818181818, abs=1e-6)

    def test_direction_target_is_binary(self) -> None:
        df = pd.DataFrame({"close": [100.0, 110.0, 90.0]})
        y, ret = build_target(df, horizon_candles=1, target_type="direction")
        assert set(y[:-1]) <= {0.0, 1.0}
        assert y[0] == 1.0  # 100 -> 110 is up
        assert y[1] == 0.0  # 110 -> 90 is down

    def test_threshold_targets_respect_the_threshold(self) -> None:
        df = pd.DataFrame({"close": [100.0, 101.0, 105.0, 95.0]})
        y_up, _ = build_target(df, horizon_candles=1, target_type="threshold_up", threshold=0.03)
        # 100->101 (+1%) should NOT clear a 3% threshold; 101->105 (+3.96%) should.
        assert y_up[0] == 0.0
        assert y_up[1] == 1.0

    def test_unknown_target_type_raises(self) -> None:
        df = pd.DataFrame({"close": [100.0, 101.0]})
        with pytest.raises(ValueError):
            build_target(df, horizon_candles=1, target_type="not_a_real_target")  # type: ignore[arg-type]


class TestRunFormulationWalkForward:
    def test_price_and_return_formulations_produce_comparable_metrics(self) -> None:
        df = _multi_regime_df()
        design, _ = split_design_and_frozen_test(df, test_fraction=0.2)

        price_result = run_formulation_walk_forward(
            design, horizon_candles=12, target_type="price", n_folds=2, min_train_size=400, stride=15
        )
        return_result = run_formulation_walk_forward(
            design, horizon_candles=12, target_type="return", n_folds=2, min_train_size=400, stride=15
        )

        # Both formulations report metrics in the SAME implied-return units,
        # which is the entire point of the harness.
        assert "return_mae" in price_result.metrics
        assert "return_mae" in return_result.metrics
        assert price_result.n_predictions > 0
        assert return_result.n_predictions > 0

    def test_classification_target_produces_classification_metrics(self) -> None:
        df = _multi_regime_df()
        design, _ = split_design_and_frozen_test(df, test_fraction=0.2)
        result = run_formulation_walk_forward(
            design, horizon_candles=12, target_type="direction", n_folds=2, min_train_size=400, stride=15
        )
        for key in ("accuracy", "brier_score", "directional_accuracy"):
            assert key in result.metrics
        assert 0.0 <= result.metrics["brier_score"] <= 1.0

    def test_records_are_index_aligned_across_formulations_for_pairing(self) -> None:
        """Two formulations run with identical horizon/fold/stride config must
        produce records at the exact same test indices, so downstream paired
        bootstrap comparisons (Section 13) are valid."""
        df = _multi_regime_df()
        design, _ = split_design_and_frozen_test(df, test_fraction=0.2)
        kwargs = dict(horizon_candles=12, n_folds=2, min_train_size=400, stride=15)

        price_result = run_formulation_walk_forward(design, target_type="price", **kwargs)
        return_result = run_formulation_walk_forward(design, target_type="return", **kwargs)

        price_indices = [r["test_index"] for r in price_result.records]
        return_indices = [r["test_index"] for r in return_result.records]
        assert price_indices == return_indices

    def test_captures_per_fold_feature_importances(self) -> None:
        df = _multi_regime_df()
        design, _ = split_design_and_frozen_test(df, test_fraction=0.2)
        result = run_formulation_walk_forward(
            design, horizon_candles=12, target_type="return", n_folds=2, min_train_size=400, stride=15
        )
        assert len(result.fold_feature_importances) == 2
        for importances in result.fold_feature_importances:
            assert set(importances) == set(FeaturePipeline().feature_columns)

    def test_extra_features_are_selectable_via_feature_cols(self) -> None:
        df = _multi_regime_df()
        design, _ = split_design_and_frozen_test(df, test_fraction=0.2)
        extra = pd.DataFrame(
            {"volume_zscore_20": np.zeros(len(design))}, index=design.index
        )
        from src.features.feature_pipeline import FeaturePipeline

        cols = FeaturePipeline().feature_columns + ["volume_zscore_20"]
        result = run_formulation_walk_forward(
            design,
            horizon_candles=12,
            target_type="return",
            n_folds=1,
            min_train_size=400,
            stride=15,
            feature_cols=cols,
            extra_features=extra,
        )
        assert "volume_zscore_20" in result.fold_feature_importances[0]

    def test_raises_on_insufficient_data(self) -> None:
        df = _multi_regime_df(n=100)
        with pytest.raises(ValueError):
            run_formulation_walk_forward(
                df, horizon_candles=12, target_type="return", n_folds=2, min_train_size=400
            )


class TestRunQuantileWalkForward:
    def test_quantiles_are_monotonic_on_average(self) -> None:
        df = _multi_regime_df()
        design, _ = split_design_and_frozen_test(df, test_fraction=0.2)
        result = run_quantile_walk_forward(
            design, horizon_candles=12, quantiles=(0.1, 0.5, 0.9), n_folds=2, min_train_size=400, stride=15
        )
        assert result.n_predictions > 0
        mean_low = np.mean([r["predicted_quantiles"][0.1] for r in result.records])
        mean_mid = np.mean([r["predicted_quantiles"][0.5] for r in result.records])
        mean_high = np.mean([r["predicted_quantiles"][0.9] for r in result.records])
        assert mean_low < mean_mid < mean_high

    def test_coverage_reported_for_every_quantile(self) -> None:
        df = _multi_regime_df()
        design, _ = split_design_and_frozen_test(df, test_fraction=0.2)
        result = run_quantile_walk_forward(
            design, horizon_candles=12, quantiles=(0.1, 0.5, 0.9), n_folds=2, min_train_size=400, stride=15
        )
        assert set(result.coverage) == {0.1, 0.5, 0.9}
        for cov in result.coverage.values():
            assert 0.0 <= cov <= 1.0
        assert result.mean_interval_width >= 0.0

    def test_raises_on_insufficient_data(self) -> None:
        df = _multi_regime_df(n=100)
        with pytest.raises(ValueError):
            run_quantile_walk_forward(df, horizon_candles=12, n_folds=2, min_train_size=400)
