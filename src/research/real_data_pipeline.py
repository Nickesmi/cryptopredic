"""Real-data experiment orchestration (Phase 6, Sections 2, 6, 7, 9, 16).

Wires together, in order: CSV loading with a fixed timestamp convention,
the Phase 2 data-quality gate, Phase 6's historical universe
reconstruction, and Phase 5's *exact, unmodified* signal/statistics/
walk-forward/ledger framework plus its baselines. This module does not
redesign anything Phase 5 built — "Use the exact Phase 5 research
framework. Do not redesign it merely because the synthetic experiment
failed" — it only supplies the real-data-shaped plumbing in front of it
that Phase 5 (built for synthetic universes with a shared padded index)
does not need.

This module is infrastructure, not an empirical claim. Nothing in this
file is ever invoked against fabricated data and reported as a market
result — see ``docs/PHASE6_REAL_DATA_VALIDATION_REPORT.md`` Section 2.
``run_phase6_real_data_experiment`` raises :class:`RealDataUnavailableError`
by default when given no real data, rather than silently no-op'ing or
falling back to anything synthetic.

Canonical timestamp convention (Phase 6, Section 5), enforced by
``load_ohlcv_csv`` below:

- The DataFrame index is the candle's **open time**, UTC, left-labeled.
- A candle "closes" at ``open_time + interval_seconds``.
- Every downstream timestamp (feature timestamp, ranking timestamp, entry
  timestamp, forward-return start, exit timestamp) is expressed as an
  offset in *candles* from this same index — never as an independently
  computed wall-clock value — which is exactly what makes the Phase 4/5
  no-look-ahead pattern (``anchor_idx = scan_index - 1``,
  ``forward_return`` reading only ``> anchor_idx``) valid here too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.quality import DataQualityReport, validate_ohlcv
from src.research.experiment_ledger import ExperimentLedger
from src.research.experiment_runner import (
    apply_bh_within_families,
    confirmatory_stage,
    run_cross_sectional_relative_strength_screen,
    run_exploratory_screen,
    run_frozen_test_for_survivors,
)
from src.research.historical_universe import (
    build_point_in_time_universe,
    infer_listing_registry_from_data,
    quantify_survivorship_gap,
)
from src.research.signals import HORIZON_CANDLES, SIGNAL_REGISTRY
from src.research.walk_forward import split_design_validation_test


class RealDataUnavailableError(RuntimeError):
    """Raised by run_phase6_real_data_experiment(strict=True) when no real OHLCV data is supplied.

    This is deliberate and load-bearing: Phase 6 Section 20's stop
    condition requires that the empirical alpha claim be stopped, not
    quietly skipped, when real data cannot be obtained. A caller that
    wants a non-raising status object for programmatic use should pass
    ``strict=False`` and inspect ``Phase6ExperimentResult.status``.
    """


def load_ohlcv_csv(path: Path, assume_utc_if_naive: bool = True) -> pd.DataFrame:
    """Load one asset's OHLCV history from CSV under Phase 6's canonical timestamp convention.

    Expects a first column parseable as a datetime (the candle's open
    time) and columns ``open, high, low, close, volume``. Sorts ascending
    and de-duplicates identical timestamps (keeping the first) rather than
    silently trusting file order -- this is a loader, not a validator; the
    data-quality gate (``src/data/quality.py``) still runs after this and
    remains the source of truth for whether the frame is safe to use.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None and assume_utc_if_naive:
        df.index = df.index.tz_localize("UTC")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    return df[["open", "high", "low", "close", "volume"]]


@dataclass
class Phase6ExperimentConfig:
    timeframe: str = "1H"
    benchmark_symbol: str = "BTCUSDT"
    scan_every_candles: int = 24
    min_train_size: int = 500
    design_frac: float = 0.5
    validation_frac: float = 0.25
    horizons: dict[str, int] = field(default_factory=lambda: dict(HORIZON_CANDLES))
    quality_gate_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase6ExperimentResult:
    status: str  # "no_data" | "ran"
    n_symbols_supplied: int = 0
    n_symbols_passed_quality_gate: int = 0
    quality_reports: dict[str, DataQualityReport] = field(default_factory=dict)
    rejected_symbols: dict[str, list[str]] = field(default_factory=dict)  # symbol -> critical issue codes
    survivorship: dict[str, Any] | None = None
    universe_size_summary: dict[str, Any] | None = None
    ledger: ExperimentLedger | None = None
    n_experiments: int = 0
    exploratory_results: "pd.DataFrame | None" = None
    confirmatory_results: list[dict] = field(default_factory=list)
    n_confirmatory_significant: int = 0
    frozen_test_results: list[dict] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_symbols_supplied": self.n_symbols_supplied,
            "n_symbols_passed_quality_gate": self.n_symbols_passed_quality_gate,
            "rejected_symbols": self.rejected_symbols,
            "survivorship": self.survivorship,
            "n_experiments": self.n_experiments,
            "n_confirmatory_significant": self.n_confirmatory_significant,
            "n_frozen_test_evaluations": len(self.frozen_test_results),
            "message": self.message,
        }


def _apply_quality_gate(
    ohlcv_by_symbol: dict[str, pd.DataFrame], timeframe: str, quality_gate_kwargs: dict[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, DataQualityReport], dict[str, list[str]]]:
    """Run Phase 2's quality gate per symbol. Critical-issue symbols are excluded, not silently dropped.

    Section 4: "Do not automatically delete suspicious data without
    documenting the decision." Every rejection is recorded in the
    returned ``rejected`` dict with its exact critical issue codes, and
    every report (pass or fail) is returned in full.
    """
    passed: dict[str, pd.DataFrame] = {}
    reports: dict[str, DataQualityReport] = {}
    rejected: dict[str, list[str]] = {}

    for symbol, df in ohlcv_by_symbol.items():
        report = validate_ohlcv(df, timeframe, **quality_gate_kwargs)
        reports[symbol] = report
        if report.is_safe_to_use:
            passed[symbol] = df
        else:
            rejected[symbol] = [issue.code for issue in report.critical_issues]

    return passed, reports, rejected


def run_phase6_real_data_experiment(
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    config: Phase6ExperimentConfig | None = None,
    delisted_registry: list[str] | None = None,
    strict: bool = True,
) -> Phase6ExperimentResult:
    """Run Phase 5's exact framework against real (or, in tests only, fixture) OHLCV data.

    Args:
        ohlcv_by_symbol: ``{symbol: OHLCV DataFrame}``, real historical
            data. Empty/``None`` triggers the Section 20 stop condition.
        config:          Experiment parameters; defaults are the same
            shape as Phase 5's, per "do not redesign the framework."
        delisted_registry: Optional list of known-delisted symbols, for
            :func:`~src.research.historical_universe.quantify_survivorship_gap`.
            ``None`` (the expected case in this sandbox) yields an honest
            "unquantifiable" survivorship report, never a fabricated one.
        strict:          Raise :class:`RealDataUnavailableError` (default)
            vs. return a ``status="no_data"`` result, when no data is supplied.
    """
    config = config or Phase6ExperimentConfig()

    if not ohlcv_by_symbol:
        if strict:
            raise RealDataUnavailableError(
                "No real historical OHLCV data was supplied. Per Phase 6 Section 20, the "
                "empirical alpha claim is stopped rather than substituted with synthetic data. "
                "See docs/PHASE6_REAL_DATA_VALIDATION_REPORT.md for the required acquisition spec."
            )
        return Phase6ExperimentResult(status="no_data", message="No real historical OHLCV data was supplied.")

    passed, reports, rejected = _apply_quality_gate(
        ohlcv_by_symbol, config.timeframe, config.quality_gate_kwargs
    )

    if config.benchmark_symbol not in passed:
        return Phase6ExperimentResult(
            status="no_data",
            n_symbols_supplied=len(ohlcv_by_symbol),
            n_symbols_passed_quality_gate=len(passed),
            quality_reports=reports,
            rejected_symbols=rejected,
            message=(
                f"Benchmark symbol '{config.benchmark_symbol}' did not pass the data-quality "
                "gate (or was not supplied) -- cannot compute relative-strength signals or run "
                "the experiment without it."
            ),
        )

    survivorship = quantify_survivorship_gap(set(passed), delisted_registry).to_dict()

    # Align every passed asset onto one shared index (union of timestamps,
    # each asset's own pre-listing rows left as NaN) so build_signal_panel
    # (which requires a shared index length -- see its own docstring) can
    # run unmodified, exactly as it does in Phase 5.
    union_index = sorted(set().union(*(df.index for df in passed.values())))
    aligned = {symbol: df.reindex(union_index) for symbol, df in passed.items()}

    n = len(union_index)
    region_split = split_design_validation_test(n, config.design_frac, config.validation_frac)

    # Section 3's diagram (timestamp -> assets tradable at timestamp) made
    # concrete and recorded, using each asset's own first-available-candle
    # as a *proxy* listing date (Section 1/3 disclosure: no verified
    # exchange listing registry exists in this sandbox). This is the
    # record Section 3 requires ("reconstruct ... record"); Phase 5's
    # `build_signal_panel` (called below via experiment_runner) separately
    # enforces the same never-list-early guarantee at the signal level via
    # each series' own NaN warmup, which is what actually gates which
    # symbols contribute to each experiment cell -- this universe map is
    # the audit trail proving that gating matches what a proxy registry
    # would say, not the mechanism doing the gating itself.
    listing_registry = infer_listing_registry_from_data(aligned)
    scan_points = list(range(config.min_train_size, n, config.scan_every_candles))
    universe_by_scan = build_point_in_time_universe(
        aligned, [union_index[t] for t in scan_points], listing_registry
    )
    universe_sizes = [len(symbols) for symbols in universe_by_scan.values()]
    universe_size_summary = {
        "n_scan_points": len(universe_sizes),
        "min_universe_size": min(universe_sizes) if universe_sizes else 0,
        "max_universe_size": max(universe_sizes) if universe_sizes else 0,
        "mean_universe_size": (sum(universe_sizes) / len(universe_sizes)) if universe_sizes else 0.0,
    }

    ledger = ExperimentLedger()
    spec_by_name = {s.name: s for s in SIGNAL_REGISTRY}

    # Reuse Phase 5's exact exploratory-screen / cross-sectional-relative-
    # strength / BH-correction / confirmatory-stage / frozen-test functions
    # (src/research/experiment_runner.py, extracted unmodified from
    # scripts/run_phase5_alpha_research.py) -- "use the exact Phase 5
    # research framework, do not redesign it" taken literally: this is the
    # same code, not a re-implementation of the same idea.
    exploratory_df = run_exploratory_screen(
        aligned, config.benchmark_symbol, SIGNAL_REGISTRY, region_split, ledger, "real_data",
        scan_every_candles=config.scan_every_candles, min_train_size=config.min_train_size,
    )
    cross_sectional_df = run_cross_sectional_relative_strength_screen(
        aligned, region_split, ledger, "real_data",
        scan_every_candles=config.scan_every_candles, min_train_size=config.min_train_size,
    )
    exploratory_all = apply_bh_within_families(pd.concat([exploratory_df, cross_sectional_df], ignore_index=True))

    confirmatory = confirmatory_stage(
        aligned, config.benchmark_symbol, spec_by_name, exploratory_all, region_split, ledger, "real_data",
        min_train_size=config.min_train_size,
    )
    n_confirmatory_significant = sum(1 for r in confirmatory if r.get("bh_significant_confirmatory"))

    frozen_test_results = run_frozen_test_for_survivors(
        aligned, config.benchmark_symbol, spec_by_name, confirmatory, region_split,
        min_train_size=config.min_train_size,
    )

    return Phase6ExperimentResult(
        status="ran",
        n_symbols_supplied=len(ohlcv_by_symbol),
        n_symbols_passed_quality_gate=len(passed),
        quality_reports=reports,
        rejected_symbols=rejected,
        survivorship=survivorship,
        universe_size_summary=universe_size_summary,
        ledger=ledger,
        n_experiments=len(ledger),
        exploratory_results=exploratory_all,
        confirmatory_results=confirmatory,
        n_confirmatory_significant=n_confirmatory_significant,
        frozen_test_results=frozen_test_results,
        message=(
            f"Ran {len(ledger)} experiment cells across {len(passed)} symbols; "
            f"{n_confirmatory_significant} confirmatory-stage survivor(s); "
            f"{len(frozen_test_results)} frozen-test evaluation(s). "
            "This result is only meaningful if ohlcv_by_symbol is genuine historical market "
            "data -- see module docstring."
        ),
    )
