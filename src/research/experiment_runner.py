"""Shared exploratory/confirmatory/frozen-test experiment runner (Phase 5 + Phase 6).

Extracted, unmodified in behavior, from ``scripts/run_phase5_alpha_research.py``
so Phase 6 can satisfy "Use the exact Phase 5 research framework. Do not
redesign it merely because the synthetic experiment failed" *literally* --
by importing and calling the same functions against real data, rather
than re-describing the same procedure in new code that could drift from
what Phase 5 actually did. ``scripts/run_phase5_alpha_research.py`` now
imports from here too, so there is exactly one implementation of the
two-stage (exploratory screen -> confirmatory permutation test) procedure
for both phases.

Every function here operates on a plain ``{symbol: OHLCV DataFrame}``
dict and a ``benchmark_symbol`` string -- no dependency on
``src.research.universes`` or any synthetic-data concept, which is what
makes reuse against real data (Phase 6) possible without modification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.evaluation.cross_sectional_analysis import compute_ic_series, forward_return, quintile_analysis
from src.research.experiment_ledger import ExperimentLedger, ExperimentRecord
from src.research.signals import HORIZON_CANDLES, SignalSpec
from src.research.statistics import analyze_quantiles, benjamini_hochberg, permutation_test_ic
from src.research.walk_forward import (
    build_signal_panel,
    cross_sectional_relative_strength_panel,
    restrict_panel_to_range,
)

MIN_TRAIN_SIZE = 1300
SCAN_EVERY_CANDLES = 24
MIN_PERIODS_FOR_STATS = 20
CONTROL_SIGNAL_NAMES = {"momentum", "relative_strength"}


def analytic_p_value(
    mean_ic: float | None, std_ic: float | None, n: int, min_periods: int = MIN_PERIODS_FOR_STATS
) -> float | None:
    """Anti-conservative screening p-value (Phase 5 Section 5/7 caveat) -- see module docstring
    of scripts/run_phase5_alpha_research.py for why this is a screen, not confirmatory evidence."""
    if mean_ic is None or std_ic is None or std_ic == 0 or n < min_periods:
        return None
    z = mean_ic / (std_ic / np.sqrt(n))
    return float(2 * scipy_stats.norm.sf(abs(z)))


def run_exploratory_screen(
    candidates: dict[str, pd.DataFrame],
    benchmark_symbol: str,
    signal_specs: list[SignalSpec],
    region_split,
    ledger: ExperimentLedger,
    universe_tag: str,
    scan_every_candles: int = SCAN_EVERY_CANDLES,
    min_train_size: int = MIN_TRAIN_SIZE,
    min_periods_for_stats: int = MIN_PERIODS_FOR_STATS,
) -> pd.DataFrame:
    """Stage 1: every signal x own-lookback x forward-horizon x method x region, recorded in full."""
    benchmark_close = candidates[benchmark_symbol]["close"]
    rows = []

    for spec in signal_specs:
        own_lookbacks = list(HORIZON_CANDLES.items()) if spec.horizon_parameterized else [("combo", None)]
        for own_label, own_lb in own_lookbacks:
            signal_series = {
                symbol: spec.series(df, benchmark_close, own_lb if own_lb is not None else HORIZON_CANDLES["1D"])
                for symbol, df in candidates.items()
            }
            panel = build_signal_panel(
                candidates, signal_series, scan_every_candles=scan_every_candles, min_train_size=min_train_size
            )
            for region_name, bounds in (("design", region_split.design), ("validation", region_split.validation)):
                region_panel = restrict_panel_to_range(panel, *bounds)
                if not region_panel:
                    continue
                ic_by_horizon = compute_ic_series(region_panel, candidates, HORIZON_CANDLES, method="spearman")
                ic_by_horizon_pearson = compute_ic_series(
                    region_panel, candidates, {own_label: HORIZON_CANDLES[own_label]}, method="pearson"
                ) if own_label != "combo" else {}

                for fwd_label, fwd_candles in HORIZON_CANDLES.items():
                    methods = [("spearman", ic_by_horizon)]
                    if fwd_label == own_label and ic_by_horizon_pearson:
                        methods.append(("pearson", ic_by_horizon_pearson))
                    for method, ic_map in methods:
                        summary = ic_map[fwd_label].summary()
                        n = summary.get("n_periods", 0)
                        mean_ic = summary.get("mean_ic")
                        std_ic = summary.get("std_ic")
                        p_value = analytic_p_value(mean_ic, std_ic, n, min_periods_for_stats)

                        quint_result = None
                        if method == "spearman" and fwd_label == own_label:
                            buckets = quintile_analysis(region_panel, candidates, fwd_candles)
                            quint_result = analyze_quantiles(buckets)

                        record = ExperimentRecord(
                            experiment_id=f"{universe_tag}:{spec.name}:{own_label}:{fwd_label}:{method}:{region_name}",
                            family=spec.family,
                            signal_name=spec.name,
                            horizon_label=f"own={own_label}|fwd={fwd_label}",
                            universe=universe_tag,
                            region=region_name,
                            ic_method=method,
                            n_periods=n,
                            mean_ic=mean_ic,
                            median_ic=summary.get("median_ic"),
                            std_ic=std_ic,
                            ic_information_ratio=(mean_ic / std_ic) if (mean_ic and std_ic) else None,
                            ci_low=None,
                            ci_high=None,
                            hit_rate=summary.get("pct_periods_positive"),
                            quintile_monotonic=quint_result.monotonic if quint_result else None,
                            quintile_spearman=quint_result.monotonicity_spearman if quint_result else None,
                            top_minus_bottom=quint_result.top_minus_bottom if quint_result else None,
                            notes=f"analytic_p={p_value}",
                        )
                        ledger.add(record)
                        rows.append({
                            "universe": universe_tag, "family": spec.family, "signal": spec.name,
                            "own_horizon": own_label, "fwd_horizon": fwd_label, "method": method,
                            "region": region_name, "n_periods": n, "mean_ic": mean_ic, "std_ic": std_ic,
                            "analytic_p": p_value,
                        })

    return pd.DataFrame(rows)


def run_cross_sectional_relative_strength_screen(
    candidates: dict[str, pd.DataFrame],
    region_split,
    ledger: ExperimentLedger,
    universe_tag: str,
    scan_every_candles: int = SCAN_EVERY_CANDLES,
    min_train_size: int = MIN_TRAIN_SIZE,
    min_periods_for_stats: int = MIN_PERIODS_FOR_STATS,
) -> pd.DataFrame:
    rows = []
    for own_label, own_lb in HORIZON_CANDLES.items():
        panel = cross_sectional_relative_strength_panel(
            candidates, scan_every_candles=scan_every_candles, min_train_size=min_train_size, lookback=own_lb
        )
        for region_name, bounds in (("design", region_split.design), ("validation", region_split.validation)):
            region_panel = restrict_panel_to_range(panel, *bounds)
            if not region_panel:
                continue
            ic_by_horizon = compute_ic_series(region_panel, candidates, HORIZON_CANDLES, method="spearman")
            for fwd_label, fwd_candles in HORIZON_CANDLES.items():
                summary = ic_by_horizon[fwd_label].summary()
                n = summary.get("n_periods", 0)
                mean_ic = summary.get("mean_ic")
                std_ic = summary.get("std_ic")
                p_value = analytic_p_value(mean_ic, std_ic, n, min_periods_for_stats)

                quint_result = None
                if fwd_label == own_label:
                    buckets = quintile_analysis(region_panel, candidates, fwd_candles)
                    quint_result = analyze_quantiles(buckets)

                record = ExperimentRecord(
                    experiment_id=f"{universe_tag}:cross_sectional_relative_strength:{own_label}:{fwd_label}:spearman:{region_name}",
                    family="cross_sectional", signal_name="cross_sectional_relative_strength",
                    horizon_label=f"own={own_label}|fwd={fwd_label}", universe=universe_tag, region=region_name,
                    ic_method="spearman", n_periods=n, mean_ic=mean_ic, median_ic=summary.get("median_ic"),
                    std_ic=std_ic, ic_information_ratio=(mean_ic / std_ic) if (mean_ic and std_ic) else None,
                    ci_low=None, ci_high=None, hit_rate=summary.get("pct_periods_positive"),
                    quintile_monotonic=quint_result.monotonic if quint_result else None,
                    quintile_spearman=quint_result.monotonicity_spearman if quint_result else None,
                    top_minus_bottom=quint_result.top_minus_bottom if quint_result else None,
                    notes=f"analytic_p={p_value}",
                )
                ledger.add(record)
                rows.append({
                    "universe": universe_tag, "family": "cross_sectional", "signal": "cross_sectional_relative_strength",
                    "own_horizon": own_label, "fwd_horizon": fwd_label, "method": "spearman", "region": region_name,
                    "n_periods": n, "mean_ic": mean_ic, "std_ic": std_ic, "analytic_p": p_value,
                })
    return pd.DataFrame(rows)


def apply_bh_within_families(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bh_significant"] = False
    if df.empty:
        return df
    for (universe, region), group in df.groupby(["universe", "region"]):
        p_values = group["analytic_p"].fillna(1.0).to_list()
        flags = benjamini_hochberg(p_values, alpha=0.05)
        df.loc[group.index, "bh_significant"] = flags
    return df


def confirmatory_stage(
    candidates: dict[str, pd.DataFrame],
    benchmark_symbol: str,
    spec_by_name: dict[str, SignalSpec],
    candidates_df: pd.DataFrame,
    region_split,
    ledger: ExperimentLedger,
    universe_tag: str,
    min_train_size: int = MIN_TRAIN_SIZE,
    min_periods_for_stats: int = MIN_PERIODS_FOR_STATS,
    top_k: int = 10,
    n_perm: int = 300,
) -> list[dict]:
    """Stage 2: non-overlapping-stride permutation test on the top BH-surviving validation candidates."""
    if candidates_df.empty:
        return []
    survivors = candidates_df[
        (candidates_df["region"] == "validation") & (candidates_df["bh_significant"])
        & (candidates_df["n_periods"] >= min_periods_for_stats)
    ].copy()
    if survivors.empty:
        return []
    survivors["abs_ic"] = survivors["mean_ic"].abs()
    survivors = survivors.sort_values("abs_ic", ascending=False).drop_duplicates(subset=["signal", "own_horizon", "fwd_horizon"])
    top = survivors.head(top_k)

    benchmark_close = candidates[benchmark_symbol]["close"]
    results = []
    p_values = []
    for _, row in top.iterrows():
        fwd_candles = HORIZON_CANDLES[row["fwd_horizon"]]
        stride = fwd_candles  # non-overlapping: one snapshot per forward-return window

        if row["signal"] == "cross_sectional_relative_strength":
            panel = cross_sectional_relative_strength_panel(
                candidates, scan_every_candles=stride, min_train_size=min_train_size,
                lookback=HORIZON_CANDLES[row["own_horizon"]],
            )
        else:
            spec = spec_by_name[row["signal"]]
            own_lb = HORIZON_CANDLES.get(row["own_horizon"])
            signal_series = {
                symbol: spec.series(df, benchmark_close, own_lb if own_lb is not None else HORIZON_CANDLES["1D"])
                for symbol, df in candidates.items()
            }
            panel = build_signal_panel(candidates, signal_series, scan_every_candles=stride, min_train_size=min_train_size)

        region_panel = restrict_panel_to_range(panel, *region_split.validation)
        ic_series = compute_ic_series(region_panel, candidates, {row["fwd_horizon"]: fwd_candles}, method="spearman")
        summary = ic_series[row["fwd_horizon"]].summary()
        n = summary.get("n_periods", 0)
        if n < min_periods_for_stats:
            continue

        scores_by_snapshot = [snap.scores for snap in region_panel]
        returns_by_snapshot = [
            {s: r for s in snap.scores if (r := forward_return(candidates[s], snap.scan_index, fwd_candles)) is not None}
            for snap in region_panel
        ]
        p_value = permutation_test_ic(
            scores_by_snapshot, returns_by_snapshot, summary["mean_ic"], method="spearman", n_perm=n_perm, seed=42
        )
        p_values.append(p_value)
        results.append({
            "signal": row["signal"], "own_horizon": row["own_horizon"], "fwd_horizon": row["fwd_horizon"],
            "n_periods": n, "mean_ic": summary["mean_ic"], "std_ic": summary.get("std_ic"),
            "permutation_p": p_value, "stride": stride,
        })

    if p_values:
        flags = benjamini_hochberg(p_values, alpha=0.05)
        for r, flag in zip(results, flags):
            r["bh_significant_confirmatory"] = bool(flag)
            ledger.add(ExperimentRecord(
                experiment_id=f"{universe_tag}:CONFIRMATORY:{r['signal']}:{r['own_horizon']}:{r['fwd_horizon']}",
                family="confirmatory", signal_name=r["signal"], horizon_label=f"own={r['own_horizon']}|fwd={r['fwd_horizon']}",
                universe=universe_tag, region="validation", ic_method="spearman", n_periods=r["n_periods"],
                mean_ic=r["mean_ic"], median_ic=None, std_ic=r["std_ic"], ic_information_ratio=None,
                ci_low=None, ci_high=None, hit_rate=None, quintile_monotonic=None, quintile_spearman=None,
                top_minus_bottom=None, permutation_p_value=r["permutation_p"], bh_significant=flag,
                notes="non-overlapping-stride confirmatory permutation test",
            ))

    return results


def run_frozen_test_for_survivors(
    candidates: dict[str, pd.DataFrame],
    benchmark_symbol: str,
    spec_by_name: dict[str, SignalSpec],
    confirmatory_results: list[dict],
    region_split,
    min_train_size: int = MIN_TRAIN_SIZE,
    top_n: int = 3,
    cost_pct: float = 0.001,
    periods_per_year_base: float = 365 * 24,
    extra_baselines: dict[str, tuple] | None = None,
) -> list[dict]:
    """Stage 3: exactly-once frozen-test evaluation + economic backtest for confirmed survivors only.

    ``extra_baselines``: ``{name: (baseline_fn, args, kwargs)}`` for
    additional buy-and-hold-style baselines beyond the built-in ones
    (e.g. a second benchmark asset) -- kept optional so this function has
    no hard dependency on a specific universe having a second benchmark.
    """
    from src.evaluation.cross_sectional_analysis import rank_turnover
    from src.evaluation.portfolio_backtest import (
        backtest_top_n_portfolio,
        buy_and_hold_baseline,
        equal_weight_universe_baseline,
        momentum_ranking_baseline,
        random_selection_baseline,
    )
    from src.research.economic_simulation import cost_sensitivity_sweep, mfe_mae_report

    benchmark_close = candidates[benchmark_symbol]["close"]
    frozen_results = []

    for r in confirmatory_results:
        if not r.get("bh_significant_confirmatory"):
            continue
        fwd_candles = HORIZON_CANDLES[r["fwd_horizon"]]
        stride = fwd_candles
        if r["signal"] == "cross_sectional_relative_strength":
            panel = cross_sectional_relative_strength_panel(
                candidates, scan_every_candles=stride, min_train_size=min_train_size,
                lookback=HORIZON_CANDLES[r["own_horizon"]],
            )
        else:
            spec = spec_by_name[r["signal"]]
            own_lb = HORIZON_CANDLES.get(r["own_horizon"])
            signal_series = {
                symbol: spec.series(df, benchmark_close, own_lb if own_lb is not None else HORIZON_CANDLES["1D"])
                for symbol, df in candidates.items()
            }
            panel = build_signal_panel(candidates, signal_series, scan_every_candles=stride, min_train_size=min_train_size)

        test_panel = restrict_panel_to_range(panel, *region_split.test)
        ppy = periods_per_year_base / fwd_candles
        ic_series = compute_ic_series(test_panel, candidates, {r["fwd_horizon"]: fwd_candles}, method="spearman")
        test_summary = ic_series[r["fwd_horizon"]].summary()

        backtest = backtest_top_n_portfolio(
            test_panel, candidates, top_n=top_n, horizon_candles=fwd_candles, cost_pct=cost_pct, periods_per_year=ppy,
        )
        baselines = {
            "random": random_selection_baseline(candidates, test_panel, top_n=top_n, horizon_candles=fwd_candles, periods_per_year=ppy),
            "equal_weight_universe": equal_weight_universe_baseline(candidates, test_panel, horizon_candles=fwd_candles, periods_per_year=ppy),
            "buy_and_hold_benchmark": buy_and_hold_baseline(candidates, benchmark_symbol, test_panel, horizon_candles=fwd_candles, periods_per_year=ppy),
            "momentum_ranking": momentum_ranking_baseline(candidates, test_panel, top_n=top_n, horizon_candles=fwd_candles, periods_per_year=ppy),
        }
        for name, (fn, args, kwargs) in (extra_baselines or {}).items():
            baselines[name] = fn(*args, **kwargs)

        mfe_mae = mfe_mae_report(test_panel, candidates, top_n=top_n, horizon_candles=fwd_candles)
        cost_sweep = cost_sensitivity_sweep(test_panel, candidates, top_n=top_n, horizon_candles=fwd_candles)
        turnover = rank_turnover(test_panel, top_n=top_n)

        frozen_results.append({
            "signal": r["signal"], "own_horizon": r["own_horizon"], "fwd_horizon": r["fwd_horizon"],
            "frozen_test_ic": test_summary, "backtest_metrics": backtest.metrics, "baselines": baselines,
            "mfe_mae": mfe_mae.to_dict(), "cost_sensitivity": cost_sweep.to_dict(), "turnover": turnover,
        })

    return frozen_results
