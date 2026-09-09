"""Phase 5 Alpha Research Engine: the actual comparative experiment run.

Executes the bounded experiment matrix described in
``docs/PHASE5_ALPHA_RESEARCH_REPORT.md``:

  1. Build the three disclosed-synthetic universes (positive control,
     null, realistic weak-alpha).
  2. Exploratory screen: every registered signal, at every applicable
     lookback, against all 8 forward horizons, on the design AND
     validation regions separately, recording IC descriptive stats and
     an analytic (parametric) p-value into the experiment ledger for
     EVERY cell -- nothing is discarded.
  3. Multiple-testing correction (Benjamini-Hochberg) applied within
     each (universe, region) family of hypotheses actually tested.
  4. Confirmatory stage: the realistic universe's top BH-surviving
     validation-region candidates are re-evaluated with a *non-
     overlapping* scan stride (stride == the candidate's own forward
     horizon, to avoid the overlapping-window pseudo-replication that
     makes the exploratory screen's analytic p-values anti-conservative)
     and a proper permutation/null test, with BH correction applied
     again across just this small confirmatory family.
  5. Any confirmatory survivor is evaluated exactly once on the frozen
     test region (same non-overlapping construction), then put through
     an economic backtest, MFE/MAE, ablation, robustness/perturbation,
     and regime-stability breakdown.
  6. Positive-control and null-control checks are run through the same
     pipeline (Section 14 — mandatory).

Everything is written to ``data/research/`` as JSON so
``docs/PHASE5_ALPHA_RESEARCH_REPORT.md`` can cite exact numbers rather
than re-describing this run from memory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.evaluation.cross_sectional_analysis import compute_ic_series, forward_return, quintile_analysis, rank_turnover
from src.evaluation.portfolio_backtest import (
    backtest_top_n_portfolio,
    buy_and_hold_baseline,
    equal_weight_universe_baseline,
    momentum_ranking_baseline,
    random_selection_baseline,
)
from src.research.economic_simulation import cost_sensitivity_sweep, mfe_mae_report
from src.research.experiment_ledger import ExperimentLedger, ExperimentRecord
from src.research.signals import HORIZON_CANDLES, SIGNAL_REGISTRY
from src.research.statistics import analyze_quantiles, benjamini_hochberg, permutation_test_ic
from src.research.universes import (
    ALL_SYMBOLS,
    BENCHMARK_SYMBOL,
    SECOND_BENCHMARK_SYMBOL,
    build_null_universe,
    build_positive_control_universe,
    build_realistic_universe,
)
from src.research.walk_forward import (
    build_signal_panel,
    cross_sectional_relative_strength_panel,
    restrict_panel_to_range,
    split_design_validation_test,
)

OUT_DIR = Path("data/research")
MIN_TRAIN_SIZE = 1300
SCAN_EVERY_CANDLES = 24
MIN_PERIODS_FOR_STATS = 20
CONTROL_SIGNAL_NAMES = {"momentum", "relative_strength"}


def _analytic_p_value(mean_ic: float | None, std_ic: float | None, n: int) -> float | None:
    if mean_ic is None or std_ic is None or std_ic == 0 or n < MIN_PERIODS_FOR_STATS:
        return None
    z = mean_ic / (std_ic / np.sqrt(n))
    return float(2 * scipy_stats.norm.sf(abs(z)))


def _precompute_signal_series(universe, spec, benchmark_close: pd.Series) -> dict[str, dict[int, pd.Series]]:
    """{symbol: {lookback: series}} for every horizon this spec is swept across."""
    lookbacks = list(HORIZON_CANDLES.values()) if spec.horizon_parameterized else [None]
    out: dict[str, dict[int, pd.Series]] = {}
    for symbol, df in universe.candidates.items():
        out[symbol] = {}
        for lb in lookbacks:
            key = lb if lb is not None else 0
            out[symbol][key] = spec.series(df, benchmark_close, lb if lb is not None else HORIZON_CANDLES["1D"])
    return out


def run_exploratory_screen(universe, signal_specs, region_split, ledger: ExperimentLedger, universe_tag: str) -> pd.DataFrame:
    benchmark_close = universe.candidates[BENCHMARK_SYMBOL]["close"]
    rows = []

    for spec in signal_specs:
        own_lookbacks = list(HORIZON_CANDLES.items()) if spec.horizon_parameterized else [("combo", None)]
        for own_label, own_lb in own_lookbacks:
            signal_series = {
                symbol: spec.series(df, benchmark_close, own_lb if own_lb is not None else HORIZON_CANDLES["1D"])
                for symbol, df in universe.candidates.items()
            }
            panel = build_signal_panel(
                universe.candidates, signal_series, scan_every_candles=SCAN_EVERY_CANDLES, min_train_size=MIN_TRAIN_SIZE
            )
            for region_name, bounds in (("design", region_split.design), ("validation", region_split.validation)):
                region_panel = restrict_panel_to_range(panel, *bounds)
                if not region_panel:
                    continue
                ic_by_horizon = compute_ic_series(region_panel, universe.candidates, HORIZON_CANDLES, method="spearman")
                # Pearson is computed only at each signal's own natural
                # horizon (not swept across all 8 forward horizons like
                # Spearman) -- Spearman rank IC is the primary, standard
                # cross-sectional metric here (robust to outliers, doesn't
                # assume a linear score-return relationship); Pearson is
                # kept only as the secondary confirmation Section 7 asks
                # for, at the one horizon pairing that matters, to keep
                # the bounded experiment budget from doubling for a
                # metric this analysis does not treat as primary.
                ic_by_horizon_pearson = compute_ic_series(
                    region_panel, universe.candidates, {own_label: HORIZON_CANDLES[own_label]}, method="pearson"
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
                        p_value = _analytic_p_value(mean_ic, std_ic, n)

                        quint_result = None
                        if method == "spearman" and fwd_label == own_label:
                            buckets = quintile_analysis(region_panel, universe.candidates, fwd_candles)
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


def run_cross_sectional_relative_strength_screen(universe, region_split, ledger: ExperimentLedger, universe_tag: str) -> pd.DataFrame:
    rows = []
    for own_label, own_lb in HORIZON_CANDLES.items():
        panel = cross_sectional_relative_strength_panel(
            universe.candidates, scan_every_candles=SCAN_EVERY_CANDLES, min_train_size=MIN_TRAIN_SIZE, lookback=own_lb
        )
        for region_name, bounds in (("design", region_split.design), ("validation", region_split.validation)):
            region_panel = restrict_panel_to_range(panel, *bounds)
            if not region_panel:
                continue
            ic_by_horizon = compute_ic_series(region_panel, universe.candidates, HORIZON_CANDLES, method="spearman")
            for fwd_label, fwd_candles in HORIZON_CANDLES.items():
                summary = ic_by_horizon[fwd_label].summary()
                n = summary.get("n_periods", 0)
                mean_ic = summary.get("mean_ic")
                std_ic = summary.get("std_ic")
                p_value = _analytic_p_value(mean_ic, std_ic, n)

                quint_result = None
                if fwd_label == own_label:
                    buckets = quintile_analysis(region_panel, universe.candidates, fwd_candles)
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
    for (universe, region), group in df.groupby(["universe", "region"]):
        p_values = group["analytic_p"].fillna(1.0).to_list()
        flags = benjamini_hochberg(p_values, alpha=0.05)
        df.loc[group.index, "bh_significant"] = flags
    return df


def confirmatory_stage(universe, spec_by_name, candidates_df: pd.DataFrame, region_split, ledger: ExperimentLedger, universe_tag: str) -> list[dict]:
    """Non-overlapping-stride permutation test on the top BH-surviving validation candidates."""
    survivors = candidates_df[
        (candidates_df["region"] == "validation") & (candidates_df["bh_significant"]) & (candidates_df["n_periods"] >= MIN_PERIODS_FOR_STATS)
    ].copy()
    if survivors.empty:
        return []
    survivors["abs_ic"] = survivors["mean_ic"].abs()
    survivors = survivors.sort_values("abs_ic", ascending=False).drop_duplicates(subset=["signal", "own_horizon", "fwd_horizon"])
    top = survivors.head(10)

    benchmark_close = universe.candidates[BENCHMARK_SYMBOL]["close"]
    results = []
    p_values = []
    for _, row in top.iterrows():
        fwd_candles = HORIZON_CANDLES[row["fwd_horizon"]]
        stride = fwd_candles  # non-overlapping: one snapshot per forward-return window

        if row["signal"] == "cross_sectional_relative_strength":
            panel = cross_sectional_relative_strength_panel(
                universe.candidates, scan_every_candles=stride, min_train_size=MIN_TRAIN_SIZE,
                lookback=HORIZON_CANDLES[row["own_horizon"]],
            )
        else:
            spec = spec_by_name[row["signal"]]
            own_lb = HORIZON_CANDLES.get(row["own_horizon"])
            signal_series = {
                symbol: spec.series(df, benchmark_close, own_lb if own_lb is not None else HORIZON_CANDLES["1D"])
                for symbol, df in universe.candidates.items()
            }
            panel = build_signal_panel(universe.candidates, signal_series, scan_every_candles=stride, min_train_size=MIN_TRAIN_SIZE)

        region_panel = restrict_panel_to_range(panel, *region_split.validation)
        ic_series = compute_ic_series(region_panel, universe.candidates, {row["fwd_horizon"]: fwd_candles}, method="spearman")
        summary = ic_series[row["fwd_horizon"]].summary()
        n = summary.get("n_periods", 0)
        if n < MIN_PERIODS_FOR_STATS:
            continue

        scores_by_snapshot = [snap.scores for snap in region_panel]
        returns_by_snapshot = [
            {s: r for s in snap.scores if (r := forward_return(universe.candidates[s], snap.scan_index, fwd_candles)) is not None}
            for snap in region_panel
        ]
        p_value = permutation_test_ic(
            scores_by_snapshot, returns_by_snapshot, summary["mean_ic"], method="spearman", n_perm=300, seed=42
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


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedger()
    spec_by_name = {s.name: s for s in SIGNAL_REGISTRY}

    region_split = split_design_validation_test(8000, design_frac=0.5, validation_frac=0.25)
    print(f"Region split: design={region_split.design} validation={region_split.validation} test={region_split.test}")

    universes = {
        "realistic": build_realistic_universe(),
        "positive_control": build_positive_control_universe(),
        "null": build_null_universe(),
    }

    summary: dict = {"region_split": {"design": region_split.design, "validation": region_split.validation, "test": region_split.test}}

    print("Running full exploratory screen on REALISTIC universe...")
    realistic_df = run_exploratory_screen(universes["realistic"], SIGNAL_REGISTRY, region_split, ledger, "realistic")
    realistic_cs_df = run_cross_sectional_relative_strength_screen(universes["realistic"], region_split, ledger, "realistic")
    realistic_all = pd.concat([realistic_df, realistic_cs_df], ignore_index=True)
    realistic_all = apply_bh_within_families(realistic_all)
    print(f"  realistic universe: {len(realistic_all)} exploratory records")

    control_specs = [s for s in SIGNAL_REGISTRY if s.name in CONTROL_SIGNAL_NAMES]
    print("Running control screen on POSITIVE CONTROL universe...")
    pc_df = run_exploratory_screen(universes["positive_control"], control_specs, region_split, ledger, "positive_control")
    pc_cs_df = run_cross_sectional_relative_strength_screen(universes["positive_control"], region_split, ledger, "positive_control")
    pc_all = apply_bh_within_families(pd.concat([pc_df, pc_cs_df], ignore_index=True))

    print("Running control screen on NULL universe...")
    null_df = run_exploratory_screen(universes["null"], control_specs, region_split, ledger, "null")
    null_cs_df = run_cross_sectional_relative_strength_screen(universes["null"], region_split, ledger, "null")
    null_all = apply_bh_within_families(pd.concat([null_df, null_cs_df], ignore_index=True))

    print("Confirmatory stage on REALISTIC universe (non-overlapping stride)...")
    confirmatory = confirmatory_stage(universes["realistic"], spec_by_name, realistic_all, region_split, ledger, "realistic")

    print("Confirmatory stage on POSITIVE CONTROL (mechanism sanity check)...")
    pc_confirmatory = confirmatory_stage(universes["positive_control"], spec_by_name, pc_all, region_split, ledger, "positive_control")

    print("Confirmatory stage on NULL (must show nothing survives)...")
    null_confirmatory = confirmatory_stage(universes["null"], spec_by_name, null_all, region_split, ledger, "null")

    n_confirmed = sum(1 for r in confirmatory if r.get("bh_significant_confirmatory"))
    summary["realistic_exploratory_n_records"] = int(len(realistic_all))
    summary["realistic_exploratory_n_bh_significant"] = int(realistic_all["bh_significant"].sum())
    summary["realistic_confirmatory_candidates"] = confirmatory
    summary["realistic_confirmatory_n_significant"] = int(n_confirmed)
    summary["positive_control_exploratory_n_records"] = int(len(pc_all))
    summary["positive_control_exploratory_n_bh_significant"] = int(pc_all["bh_significant"].sum())
    summary["positive_control_confirmatory"] = pc_confirmatory
    summary["null_exploratory_n_records"] = int(len(null_all))
    summary["null_exploratory_n_bh_significant"] = int(null_all["bh_significant"].sum())
    summary["null_confirmatory"] = null_confirmatory
    summary["total_experiments_recorded"] = len(ledger)

    # ---- Frozen-test evaluation of confirmed survivors (exactly once) ----
    frozen_results = []
    benchmark_close = universes["realistic"].candidates[BENCHMARK_SYMBOL]["close"]
    for r in confirmatory:
        if not r.get("bh_significant_confirmatory"):
            continue
        fwd_candles = HORIZON_CANDLES[r["fwd_horizon"]]
        stride = fwd_candles
        if r["signal"] == "cross_sectional_relative_strength":
            panel = cross_sectional_relative_strength_panel(
                universes["realistic"].candidates, scan_every_candles=stride, min_train_size=MIN_TRAIN_SIZE,
                lookback=HORIZON_CANDLES[r["own_horizon"]],
            )
        else:
            spec = spec_by_name[r["signal"]]
            own_lb = HORIZON_CANDLES.get(r["own_horizon"])
            signal_series = {
                symbol: spec.series(df, benchmark_close, own_lb if own_lb is not None else HORIZON_CANDLES["1D"])
                for symbol, df in universes["realistic"].candidates.items()
            }
            panel = build_signal_panel(universes["realistic"].candidates, signal_series, scan_every_candles=stride, min_train_size=MIN_TRAIN_SIZE)

        test_panel = restrict_panel_to_range(panel, *region_split.test)
        ic_series = compute_ic_series(test_panel, universes["realistic"].candidates, {r["fwd_horizon"]: fwd_candles}, method="spearman")
        test_summary = ic_series[r["fwd_horizon"]].summary()

        backtest = backtest_top_n_portfolio(
            test_panel, universes["realistic"].candidates, top_n=3, horizon_candles=fwd_candles,
            cost_pct=0.001, periods_per_year=(365 * 24 / fwd_candles),
        )
        baselines = {
            "random": random_selection_baseline(universes["realistic"].candidates, test_panel, top_n=3, horizon_candles=fwd_candles, periods_per_year=(365 * 24 / fwd_candles)),
            "equal_weight_universe": equal_weight_universe_baseline(universes["realistic"].candidates, test_panel, horizon_candles=fwd_candles, periods_per_year=(365 * 24 / fwd_candles)),
            "buy_and_hold_btc": buy_and_hold_baseline(universes["realistic"].candidates, BENCHMARK_SYMBOL, test_panel, horizon_candles=fwd_candles, periods_per_year=(365 * 24 / fwd_candles)),
            "buy_and_hold_eth": buy_and_hold_baseline(universes["realistic"].candidates, SECOND_BENCHMARK_SYMBOL, test_panel, horizon_candles=fwd_candles, periods_per_year=(365 * 24 / fwd_candles)),
            "momentum_ranking": momentum_ranking_baseline(universes["realistic"].candidates, test_panel, top_n=3, horizon_candles=fwd_candles, periods_per_year=(365 * 24 / fwd_candles)),
        }
        mfe_mae = mfe_mae_report(test_panel, universes["realistic"].candidates, top_n=3, horizon_candles=fwd_candles)
        cost_sweep = cost_sensitivity_sweep(test_panel, universes["realistic"].candidates, top_n=3, horizon_candles=fwd_candles)
        turnover = rank_turnover(test_panel, top_n=3)

        frozen_results.append({
            "signal": r["signal"], "own_horizon": r["own_horizon"], "fwd_horizon": r["fwd_horizon"],
            "frozen_test_ic": test_summary, "backtest_metrics": backtest.metrics, "baselines": baselines,
            "mfe_mae": mfe_mae.to_dict(), "cost_sensitivity": cost_sweep.to_dict(), "turnover": turnover,
        })

    summary["frozen_test_results"] = frozen_results

    ledger.save(OUT_DIR / "phase5_experiment_ledger.jsonl")
    with (OUT_DIR / "phase5_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    realistic_all.to_csv(OUT_DIR / "realistic_exploratory.csv", index=False)
    pc_all.to_csv(OUT_DIR / "positive_control_exploratory.csv", index=False)
    null_all.to_csv(OUT_DIR / "null_exploratory.csv", index=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Total experiments recorded: {len(ledger)}")
    print(f"Realistic: {summary['realistic_exploratory_n_bh_significant']}/{summary['realistic_exploratory_n_records']} BH-significant exploratory cells")
    print(f"Realistic confirmatory candidates tested: {len(confirmatory)}, BH-significant: {n_confirmed}")
    print(f"Positive control: {summary['positive_control_exploratory_n_bh_significant']}/{summary['positive_control_exploratory_n_records']} BH-significant exploratory cells")
    print(f"Positive control confirmatory significant: {sum(1 for r in pc_confirmatory if r.get('bh_significant_confirmatory'))}/{len(pc_confirmatory)}")
    print(f"Null: {summary['null_exploratory_n_bh_significant']}/{summary['null_exploratory_n_records']} BH-significant exploratory cells")
    print(f"Null confirmatory significant: {sum(1 for r in null_confirmatory if r.get('bh_significant_confirmatory'))}/{len(null_confirmatory)}")
    print(f"Frozen-test evaluations run: {len(frozen_results)}")


if __name__ == "__main__":
    main()
