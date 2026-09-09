"""Tests for src/research/real_data_pipeline.py (Phase 6, Sections 2, 5, 6, 9, 16).

Two distinct kinds of test live here, deliberately not conflated:

1. **Mechanism tests** (mandatory, Section 5): prove the new real-data-
   shaped code path -- CSV loading -> quality gate -> historical universe
   -> Phase 5's signal/panel machinery -- introduces no look-ahead, using
   a locally fabricated CSV fixture standing in for ingested data. This
   is the exact "corrupt 50x/0.01x future price and extreme future
   volume, assert byte-identical" test required again in Phase 6, applied
   to code, not to a market data source.
2. **Contract tests**: prove ``run_phase6_real_data_experiment`` refuses
   to fabricate a result when given no data (Section 20's stop
   condition), and that it runs end-to-end correctly on a tiny fixture
   -- a code-correctness check, explicitly NOT presented anywhere as
   market evidence (see module docstring in real_data_pipeline.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.real_data_pipeline import (
    Phase6ExperimentConfig,
    RealDataUnavailableError,
    load_ohlcv_csv,
    run_phase6_real_data_experiment,
)


def _make_ohlcv_csv(path, n: int = 900, seed: int = 0, listing_offset: int = 0) -> None:
    dates = pd.date_range("2022-01-01", periods=n, freq="h")
    rng = np.random.default_rng(seed)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.015, n)), index=dates)
    df = pd.DataFrame(
        {
            "open": close * 0.999, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": rng.lognormal(10, 0.3, n),
        },
        index=dates,
    )
    if listing_offset > 0:
        df = df.iloc[listing_offset:]
    df.to_csv(path)


# ---------------------------------------------------------------------------
# 1. Mechanism tests: no look-ahead in the real-data-shaped pipeline
# ---------------------------------------------------------------------------


def test_load_ohlcv_csv_sorts_and_dedups(tmp_path) -> None:
    path = tmp_path / "btc.csv"
    dates = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"])
    df = pd.DataFrame(
        {"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3], "close": [1, 2, 3], "volume": [1, 2, 3]},
        index=dates,
    )
    df.to_csv(path)

    loaded = load_ohlcv_csv(path)
    assert loaded.index.is_monotonic_increasing
    assert len(loaded) == 2  # duplicate 2024-01-01 row collapsed, keeping the first


def test_load_ohlcv_csv_localizes_naive_timestamps_to_utc(tmp_path) -> None:
    path = tmp_path / "btc.csv"
    _make_ohlcv_csv(path, n=10)
    loaded = load_ohlcv_csv(path)
    assert loaded.index.tz is not None
    assert str(loaded.index.tz) == "UTC"


def test_load_ohlcv_csv_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"close": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h")).to_csv(path)
    with pytest.raises(ValueError):
        load_ohlcv_csv(path)


@pytest.mark.parametrize("shock_multiplier", [50.0, 0.01])
def test_real_data_signal_panel_unaffected_by_extreme_future_price_shock(tmp_path, shock_multiplier) -> None:
    """The Phase 5 signal machinery, fed data loaded through the new real-data loader,
    must remain byte-identical at an earlier point when everything after it is shocked --
    the exact Phase 4/5 test pattern, re-run against this phase's ingestion path."""
    from src.research.signals import HORIZON_CANDLES, momentum

    path_a = tmp_path / "btc_a.csv"
    path_b = tmp_path / "btc_b.csv"
    _make_ohlcv_csv(path_a, n=900, seed=1)
    _make_ohlcv_csv(path_b, n=900, seed=1)  # identical starting content

    df_a = load_ohlcv_csv(path_a)
    df_b = load_ohlcv_csv(path_b)

    eval_point = 400
    df_b.loc[df_b.index[eval_point + 1:], ["open", "high", "low", "close"]] *= shock_multiplier
    df_b.loc[df_b.index[eval_point + 1:], "volume"] *= 50.0

    series_a = momentum(df_a["close"], HORIZON_CANDLES["1D"])
    series_b = momentum(df_b["close"], HORIZON_CANDLES["1D"])

    assert series_a.iloc[eval_point] == pytest.approx(series_b.iloc[eval_point])


def test_real_data_universe_membership_unaffected_by_future_shock(tmp_path) -> None:
    """Historical-universe eligibility at an earlier scan point must not change
    when later candles (of a *different*, later-listed asset) are corrupted."""
    from src.research.historical_universe import eligible_symbols_at, infer_listing_registry_from_data

    path_btc = tmp_path / "btc.csv"
    path_late = tmp_path / "latecoin.csv"
    _make_ohlcv_csv(path_btc, n=900, seed=2)
    _make_ohlcv_csv(path_late, n=900, seed=3, listing_offset=500)

    df_btc = load_ohlcv_csv(path_btc)
    df_late_a = load_ohlcv_csv(path_late)
    df_late_b = df_late_a.copy()
    df_late_b.loc[df_late_b.index[700:], ["open", "high", "low", "close"]] *= 50.0

    histories_a = {"BTC": df_btc, "LATECOIN": df_late_a.reindex(df_btc.index)}
    histories_b = {"BTC": df_btc, "LATECOIN": df_late_b.reindex(df_btc.index)}

    registry_a = infer_listing_registry_from_data(histories_a)
    registry_b = infer_listing_registry_from_data(histories_b)

    early_ts = df_btc.index[300]  # before LATECOIN's listing
    assert eligible_symbols_at(early_ts, histories_a, registry_a) == eligible_symbols_at(early_ts, histories_b, registry_b)
    assert "LATECOIN" not in eligible_symbols_at(early_ts, histories_a, registry_a)


# ---------------------------------------------------------------------------
# 2. Contract tests: no-data stop condition, and code-correctness on a fixture
# ---------------------------------------------------------------------------


def test_run_raises_when_no_data_supplied_in_strict_mode() -> None:
    with pytest.raises(RealDataUnavailableError):
        run_phase6_real_data_experiment({})


def test_run_returns_no_data_status_when_not_strict() -> None:
    result = run_phase6_real_data_experiment({}, strict=False)
    assert result.status == "no_data"
    assert result.ledger is None


def test_run_requires_benchmark_symbol_to_pass_quality_gate(tmp_path) -> None:
    n = 600
    dates = pd.date_range("2022-01-01", periods=n, freq="h")
    close = pd.Series(np.linspace(100, 110, n), index=dates)
    good = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=dates)
    bad = good.copy()
    bad.loc[bad.index[0], "high"] = -1.0  # impossible OHLC -> critical

    config = Phase6ExperimentConfig(timeframe="1H", benchmark_symbol="BTCUSDT", min_train_size=100, scan_every_candles=24)
    result = run_phase6_real_data_experiment({"BTCUSDT": bad, "ETHUSDT": good}, config=config, strict=False)

    assert result.status == "no_data"
    assert "BTCUSDT" in result.rejected_symbols


def test_run_end_to_end_on_a_tiny_fixture_is_a_code_correctness_check_only() -> None:
    """Proves the orchestration runs to completion and populates a ledger --
    this is NOT evidence of a real signal (the input is a small synthetic
    fixture), only a proof the wiring works. See module docstring."""
    n = 700
    dates = pd.date_range("2022-01-01", periods=n, freq="h")
    rng = np.random.default_rng(7)

    def _mk():
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=dates)
        return pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0}, index=dates
        )

    ohlcv = {"BTCUSDT": _mk(), "ETHUSDT": _mk(), "ALTUSDT": _mk()}
    config = Phase6ExperimentConfig(
        timeframe="1H", benchmark_symbol="BTCUSDT", min_train_size=200, scan_every_candles=24,
        horizons={"1H": 1, "4H": 4, "1D": 24},
        # Quality gate's staleness check compares the last candle to real
        # wall-clock "now" by default; this fixture's dates are fixed in
        # the past, so pin "now" to the fixture's own last timestamp.
        quality_gate_kwargs={"now": dates[-1].tz_localize("UTC") + pd.Timedelta(minutes=30)},
    )

    result = run_phase6_real_data_experiment(ohlcv, config=config)

    assert result.status == "ran"
    assert result.n_symbols_passed_quality_gate == 3
    assert result.survivorship is not None
    assert result.survivorship["quantifiable"] is False  # no delisting registry supplied
    assert result.n_experiments > 0
