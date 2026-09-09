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

The exploratory/confirmatory/frozen-test procedure itself lives in
``src/research/experiment_runner.py`` (extracted post-hoc, Phase 6) so
Phase 6's real-data pipeline can reuse the *exact same* implementation
against real OHLCV data instead of re-describing the same procedure in
new code — "use the exact Phase 5 research framework, do not redesign
it" taken literally. This script is now a thin driver: build the
synthetic universes, call the shared runner, write the outputs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from src.research.experiment_runner import (
    CONTROL_SIGNAL_NAMES,
    apply_bh_within_families,
    confirmatory_stage,
    run_cross_sectional_relative_strength_screen,
    run_exploratory_screen,
    run_frozen_test_for_survivors,
)
from src.research.experiment_ledger import ExperimentLedger
from src.research.signals import SIGNAL_REGISTRY
from src.research.universes import (
    BENCHMARK_SYMBOL,
    SECOND_BENCHMARK_SYMBOL,
    build_null_universe,
    build_positive_control_universe,
    build_realistic_universe,
)
from src.research.walk_forward import split_design_validation_test
from src.evaluation.portfolio_backtest import buy_and_hold_baseline

OUT_DIR = Path("data/research")


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
    realistic_df = run_exploratory_screen(
        universes["realistic"].candidates, BENCHMARK_SYMBOL, SIGNAL_REGISTRY, region_split, ledger, "realistic"
    )
    realistic_cs_df = run_cross_sectional_relative_strength_screen(
        universes["realistic"].candidates, region_split, ledger, "realistic"
    )
    realistic_all = apply_bh_within_families(pd.concat([realistic_df, realistic_cs_df], ignore_index=True))
    print(f"  realistic universe: {len(realistic_all)} exploratory records")

    control_specs = [s for s in SIGNAL_REGISTRY if s.name in CONTROL_SIGNAL_NAMES]

    print("Running control screen on POSITIVE CONTROL universe...")
    pc_df = run_exploratory_screen(
        universes["positive_control"].candidates, BENCHMARK_SYMBOL, control_specs, region_split, ledger, "positive_control"
    )
    pc_cs_df = run_cross_sectional_relative_strength_screen(
        universes["positive_control"].candidates, region_split, ledger, "positive_control"
    )
    pc_all = apply_bh_within_families(pd.concat([pc_df, pc_cs_df], ignore_index=True))

    print("Running control screen on NULL universe...")
    null_df = run_exploratory_screen(
        universes["null"].candidates, BENCHMARK_SYMBOL, control_specs, region_split, ledger, "null"
    )
    null_cs_df = run_cross_sectional_relative_strength_screen(
        universes["null"].candidates, region_split, ledger, "null"
    )
    null_all = apply_bh_within_families(pd.concat([null_df, null_cs_df], ignore_index=True))

    print("Confirmatory stage on REALISTIC universe (non-overlapping stride)...")
    confirmatory = confirmatory_stage(
        universes["realistic"].candidates, BENCHMARK_SYMBOL, spec_by_name, realistic_all, region_split, ledger, "realistic"
    )

    print("Confirmatory stage on POSITIVE CONTROL (mechanism sanity check)...")
    pc_confirmatory = confirmatory_stage(
        universes["positive_control"].candidates, BENCHMARK_SYMBOL, spec_by_name, pc_all, region_split, ledger, "positive_control"
    )

    print("Confirmatory stage on NULL (must show nothing survives)...")
    null_confirmatory = confirmatory_stage(
        universes["null"].candidates, BENCHMARK_SYMBOL, spec_by_name, null_all, region_split, ledger, "null"
    )

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

    print("Frozen-test evaluation of confirmed survivors (exactly once)...")
    frozen_results = run_frozen_test_for_survivors(
        universes["realistic"].candidates, BENCHMARK_SYMBOL, spec_by_name, confirmatory, region_split,
        extra_baselines={
            "buy_and_hold_eth": (
                buy_and_hold_baseline,
                (universes["realistic"].candidates, SECOND_BENCHMARK_SYMBOL,),
                {},
            ),
        },
    )
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
