# Phase 5 — Alpha Research Engine: Is There Any Defensible Signal At All?

Phase 3 concluded the point-price/return forecasting engine is **FAILED**. Phase 4 concluded the
opportunity scanner is **WEAK** — its mechanism works (it detects an unambiguous planted signal
cleanly) but shows no statistically significant cross-sectional predictive power at any realistic
signal strength, is fragile to top-N concentration, and is not shadow- or paper-ready.

Phase 5 does not try to rescue either of those verdicts by tuning parameters or adding indicators
to the existing scanner. It asks a narrower, more fundamental question, from scratch, with a new
and independent research pipeline: **is there any statistically defensible, economically useful,
robust signal available at all in the information already accessible to this repository** —
momentum, mean reversion, volume/flow, volatility, relative strength, or some cross-sectional or
multi-timeframe combination of them — regardless of whether the existing scanner happens to use it?

**Data caveat, unchanged from Phases 3-4:** this sandbox has no outbound network access to
Binance/CoinGecko/any real market-data source (every attempt returns a proxy-level 403) and no
historical OHLCV file exists anywhere in this repository. Every quantitative result below is
computed on **disclosed, synthetic-but-controlled data** (`src/research/universes.py`) — a planted-
alpha positive control, a pure-noise null, and a "realistic" multi-regime universe designed to have
market-like statistical texture (volatility clustering, regime shifts, a weak per-asset return
autocorrelation near the edge of detectability). None of this is a claim about real crypto markets.
It is a claim about whether *this pipeline* can find a signal when one exists, correctly fail to
find one when it does not, and — on the "realistic" universe built to resemble the honest ambiguity
of a real market — whether anything survives out-of-sample, cost-aware, multiple-testing-corrected
scrutiny.

---

## 1. Frozen Baseline

Before any Phase 5 code was written, the Phase 4 scanner was frozen exactly as shipped. No line in
`src/ranking/liquidity_filter.py`, `risk_score.py`, `score_coins.py`, or `recommend.py` was touched
during Phase 5, and nothing in `src/research/` is imported by any of them — see Section 16.

**Frozen configuration** (`src/ranking/score_coins.py::DEFAULT_WEIGHTS`, unchanged):

```
trend_quality: 0.25, momentum: 0.20, relative_strength: 0.20,
volume_confirmation: 0.15, volatility_adjusted: 0.20
_RISK_PENALTY_FORMULA_VERSION = "risk_penalty_v1"
```

```
raw_score    = 0.25*trend_quality + 0.20*momentum + 0.20*relative_strength
             + 0.15*volume_confirmation + 0.20*volatility_adjusted
risk_penalty = max(0, risk_score - 30) * 0.5
final_score  = clip(raw_score - risk_penalty, 0, 100)
```

**Frozen Phase 4 metrics** (from `docs/PHASE4_SCANNER_AUDIT_REPORT.md`, reproduced verbatim as the
comparison baseline every Phase 5 "baseline" table below cites as "Phase 4 scanner"):

| Universe | Mean IC (7D) | Quintile 7D (bucket 0→4) | Mean top-3 turnover |
|---|---:|---|---:|
| Positive control | +0.535 | -2.1%, -1.9%, +0.9%, +1.9%, +3.0% (monotonic) | — |
| Realistic-alpha | -0.035 to +0.062 (horizon-dependent) | 1.1%, 0.9%, 0.3%, 0.9%, 0.4% (not monotonic) | 73% |
| Null | -0.064 to +0.019 (horizon-dependent) | 1.8%, 0.8%, 1.3%, 1.6%, -0.8% (inverted at top) | — |

| Strategy | Alpha universe total return | Alpha Sharpe-like | Null universe total return | Null Sharpe-like |
|---|---:|---:|---:|---:|
| Scanner top-3 | +9.9% | 0.46 | -16.6% | -0.47 |
| Scanner top-5 | +18.9% | 0.91 | -3.6% | -0.04 |
| BTC buy-and-hold | -22.7% | -0.74 | -22.7% | -0.74 |
| Random top-3 | +7.9% | 0.40 | +7.8% | 0.38 |

This table is not re-derived or re-run in Phase 5 — it is cited as-is, as the fixed point of
comparison. Phase 5's own baselines (random, equal-weight, buy-and-hold BTC/ETH, simple momentum,
persistence) are re-run fresh on Phase 5's own universes (different synthetic data, different
region split) specifically so a Phase 5 candidate signal is never compared against a baseline
computed under different conditions than itself.

**No optimization against a test set occurred here or anywhere in Phase 5**: Section 5 below fixes
the design/validation/frozen-test split once, before any signal was evaluated, and the frozen-test
region is touched at most once per surviving candidate (Section 9).

---

## 2. The Alpha Research Engine

`src/research/` is a new package, decoupled from `src/ranking/` by construction — nothing in it
imports `src/ranking/score_coins.py` or calls `scan_candidate()`. It has five parts:

- **`signals.py`** — every signal family the brief requests (momentum, mean reversion, volume/flow,
  volatility, relative strength, plus multi-horizon/multi-timeframe-confirmation combinations),
  each a pure, vectorised, causal (`rolling`/`shift`/`pct_change`/`ewm`) transform of an OHLCV
  series. `SIGNAL_REGISTRY` lists all 19 signal specs; `HORIZON_CANDLES` maps the 8 required
  horizons (1H-30D) to lookback-candle counts on the hourly-candle synthetic universes.
- **`universes.py`** — the three disclosed-synthetic universes (Section 4 below).
- **`walk_forward.py`** — a signal-agnostic cross-sectional panel builder
  (`build_signal_panel`), reusing (not duplicating) Phase 4's generic
  `CrossSectionalSnapshot`/`forward_return`/`compute_ic_series`/`quintile_analysis`/`rank_turnover`
  primitives, plus the design/validation/frozen-test index splitter
  (`split_design_validation_test`).
- **`statistics.py`** — IC descriptive statistics (mean/median/std/IR/CI/hit-rate), quantile
  monotonicity, a within-snapshot permutation/null test, and Benjamini-Hochberg multiple-testing
  correction.
- **`experiment_ledger.py`**, **`economic_simulation.py`**, **`ablation.py`**, **`regime_breakdown.py`**
  — the experiment record-keeping, transaction-cost-aware backtest + MFE/MAE + cost-sensitivity
  sweep, cross-sectional-z-scored signal combination for ablation, and per-regime IC breakdown.

`scripts/run_phase5_alpha_research.py` is the driver that actually executes the experiment matrix
described below and writes every result to `data/research/` — nothing in this report is invented;
every number cites a file under that directory.

---

## 3. Anti-Leakage Audit

Every function in `signals.py` is built exclusively from `pandas` `rolling().{mean,std,corr,rank}`,
`shift`, `pct_change`, and `ewm` — operations that read row `i` using only rows `<= i` by
construction, with no whole-series `.mean()`/`.std()`/min-max normalization step (the exact
forbidden pattern named in the brief) anywhere in the module. That is the engineering argument;
`tests/test_research_signals.py` is the empirical proof it was asked for:

- **`test_signal_unchanged_by_future_price_shock`**: every one of the 19 registered signals,
  evaluated at a fixed point, multiplies every future OHLCV row (price **and** volume) by **50×**
  and separately by **0.01×**, and asserts the signal's value at the evaluation point is
  bit-for-bit unchanged (or NaN in both cases). **38/38 parametrized cases pass** (19 signals × 2
  shock directions).
- **`test_signal_at_every_earlier_point_is_unaffected_by_one_future_shock`**: a single 50× shock
  far in the future must not change *any* earlier value, not just the one at the boundary —
  `pd.testing.assert_series_equal` over the entire pre-shock history. **19/19 pass.**
- `build_signal_panel`'s own no-future-listing guarantee (Section 4) is proven separately in
  `tests/test_research_walk_forward.py::test_not_yet_listed_asset_excluded_before_its_listing_point`.

**No look-ahead violation found** in any signal, in the panel builder, or in the forward-return
construction it reuses from Phase 4 (`forward_return` reads only `close[anchor + horizon]` where
`anchor = scan_index - 1`, strictly after the scoring point).

---

## 4. Survivorship Bias

Unchanged limitation from Phases 2-4, restated precisely: this sandbox has no reachable
listing/delisting registry and no historical exchange-symbol-status log, so it is **not possible**
to reconstruct which assets were actually tradable, at what liquidity, at each historical timestamp
— including assets that later failed or were delisted. That half of survivorship bias is disclosed
as a hard limitation of this environment, not something Phase 5 claims to have solved.

What Phase 5 **does** mechanically fix and empirically test is the other half: an asset that lists
*partway through* a sample window must never be scored or ranked before it exists.
`src/research/universes.py::STAGGERED_LISTINGS` gives 3 of the 20 assets in every universe an
all-NaN prefix (candle 0 up to their listing point) on a DatetimeIndex shared with every other
asset. `build_signal_panel` (Section 2) excludes any symbol with a NaN signal value at the scan
anchor — which a not-yet-listed asset always has — and
`tests/test_research_walk_forward.py::test_not_yet_listed_asset_excluded_before_its_listing_point`
proves the excluded/included boundary lands exactly at the listing candle. This is labeled for what
it is: a proof that the mechanical bookkeeping is correct, **not** a survivorship-bias-free result —
the universe composition itself (20 assets, none of which fail entirely) is still a present-day-like
selection, disclosed rather than hidden.

---

## 5. Experimental Design

**Regions** (`split_design_validation_test(8000, design_frac=0.5, validation_frac=0.25)`, fixed once
before any signal was scored):

| Region | Candle range | Use |
|---|---|---|
| Design | [0, 4000) | Signal construction, exploratory IC screening |
| Validation | [4000, 6000) | Candidate comparison, confirmatory permutation test + BH correction, ablation, robustness |
| Frozen test | [6000, 8000) | Touched at most once per confirmatory survivor |

**Universes**: positive control, null, realistic (Section 4/6 of `universes.py`'s docstring;
Section 6 below). All three share 20 assets (`BTC`, `ETH`, 18 synthetic alts), hourly candles,
8,000 candles (~333 days) each, generated with fixed seeds for reproducibility.

**Signal x horizon matrix**: 17 horizon-parameterized signals × 8 own-lookback horizons + 2
fixed-construction signals (multi-horizon momentum, multi-timeframe confirmation) + cross-sectional
relative strength × 8 lookbacks = **146 (signal, own-lookback) combinations**, each evaluated
against **all 8 forward-return horizons** (1H/4H/12H/1D/3D/7D/14D/30D) via Spearman IC (the primary
metric — a rank correlation, robust to outliers and non-linear score/return relationships) and, at
each signal's own natural horizon specifically, Pearson IC as the brief's secondary confirmation.
This is the bounded-but-real matrix Section 11 asks for: **every** cell is recorded in the
experiment ledger (Section 15), not just the interesting ones.

**Overlapping-window caveat, disclosed up front**: the exploratory screen above uses a 24-candle
scan stride shared across all 8 forward horizons for efficiency, which means forward-return windows
overlap heavily at longer horizons (e.g. a 720-candle/30D window advances by only 24 candles between
snapshots) — the per-snapshot IC values are therefore **not independent draws**, and the
exploratory stage's analytic p-values are anti-conservative (too easily "significant"). This is why
the exploratory stage is explicitly a **screen**, not evidence: every candidate that clears it is
re-evaluated in the **confirmatory stage** with a rebuilt panel using a **non-overlapping stride**
(stride = the candidate's own forward horizon) before any permutation test or BH-correction result
is treated as evidence of anything. See Section 7.

---

## 6. Baselines

Every backtest below (Section 10) compares the frozen-test performance of any confirmatory survivor
against, computed under identical cost/period assumptions on the same frozen-test region:

1. Random top-N selection (seeded, reproducible)
2. Equal-weight full universe (no ranking at all)
3. Market-cap-weighted universe — **not available**: this sandbox has no historical market-cap
   series (would require circulating-supply history per asset, not obtainable without real data
   access); disclosed as a gap rather than approximated with a proxy that would misrepresent it.
4. Buy-and-hold BTC
5. Buy-and-hold ETH
6. Simple trailing-momentum ranking (no signal-under-test involved)
7. The frozen Phase 4 scanner (Section 1's table, cited not re-run)
8. Persistence/drift-style baseline — inherited from `src/evaluation/benchmark.py`
   (`naive_persistence`/`naive_drift`), applicable to the point-forecast framing Phase 3 already
   falsified; not re-applicable to a cross-sectional ranking signal in the same form, so its
   cross-sectional analogue here is baseline #6 (momentum ranking) and #2 (equal-weight, the
   "assume no information" ranking analogue of persistence).

---

## 7. Statistical Methodology and the Two-Stage Test

Given the overlapping-window caveat in Section 5, Phase 5 uses a deliberate **two-stage** test
rather than a single p-value per cell:

**Stage 1 — Exploratory screen** (design + validation regions, `run_phase5_alpha_research.py`,
overlapping 24-candle stride): for every (signal, own-horizon, forward-horizon, method, universe,
region) cell, compute Spearman/Pearson IC descriptive statistics
(`src/research/statistics.py::compute_ic_stats` — mean, median, std, information ratio, hit rate)
and an **analytic** two-sided p-value from a normal approximation to the per-snapshot IC mean
(`z = mean_ic / (std_ic / sqrt(n))`), explicitly labeled anti-conservative per Section 5. Every
cell — including every non-significant, "failed" one — is recorded in the experiment ledger.
Benjamini-Hochberg FDR correction (`benjamini_hochberg`, α=0.05) is then applied **within each
(universe, region) family** of hypotheses actually tested (the full ~2,300+-hypothesis realistic-
universe family, and the smaller ~380-hypothesis control-universe families), not per-cell — this is
the multiple-testing discipline Section 15 requires: a raw p<0.05 out of hundreds of tested cells is
expected by chance alone and is never treated as a finding on its own.

**Stage 2 — Confirmatory test** (validation region only, non-overlapping stride = the candidate's
own forward horizon): every cell that survives Stage 1's BH correction in the realistic universe is
re-scored from scratch with a rebuilt panel at that non-overlapping stride (eliminating the
pseudo-replication that made Stage 1 anti-conservative), then subjected to
`permutation_test_ic` — a true empirical null built by shuffling the score-return pairing *within
each snapshot* 300 times and comparing the observed mean IC to that null distribution — followed by
a second, separate BH correction across just this small confirmatory family. **Only a signal that
survives both stages is eligible for frozen-test evaluation** (Section 9); surviving Stage 1 alone
is exploratory evidence, never confirmatory, per Section 15's explicit warning against presenting a
signal "significant only after trying dozens of variants" as independent evidence.

The mandatory positive-control and null-control checks (Section 14) are run through this identical
two-stage pipeline, using the same signal families (momentum, relative strength, cross-sectional
relative strength) as a representative subset, specifically so the controls validate the *pipeline*
end-to-end and not a separate, simpler procedure.

---

<!-- SECTION_8_EXPLORATORY_RESULTS -->

<!-- SECTION_9_CONFIRMATORY_AND_FROZEN_TEST -->

<!-- SECTION_10_ECONOMIC_BACKTEST -->

<!-- SECTION_11_MFE_MAE_AND_STABILITY -->

<!-- SECTION_12_ABLATION -->

<!-- SECTION_13_ROBUSTNESS -->

<!-- SECTION_14_CONTROLS -->

---

## 15. Multiple-Testing Discipline and the Experiment Ledger

Every experiment cell described in Sections 7-9 (Stage 1 and Stage 2, all three universes, both
regions) is written as one `ExperimentRecord` to `data/research/phase5_experiment_ledger.jsonl` —
nothing tested is discarded, including every non-significant and negative result. Each record
carries: signal family/name, own-horizon, forward-horizon, universe, region, IC method, n_periods,
mean/median/std IC, information ratio, hit rate, quintile monotonicity (computed at each signal's
own natural horizon), permutation p-value and BH-significance flag where Stage 2 was run, and free-
text notes carrying the Stage-1 analytic p-value.

<!-- SECTION_15_LEDGER_COUNTS -->

This ledger is the record required so that "future phases cannot accidentally treat a discovered
pattern as out-of-sample evidence after it has already been mined": any future extension of this
work must load this ledger and treat every cell in it as already-spent — re-testing a variant that
already appears here on the same data is not a new, independent test.

---

## 16. Production Separation

Five components, kept separate exactly as Phase 4 reaffirmed, with the Alpha Research Engine now
added as a fourth non-production role:

```
Forecasting Engine (src/models/)        -- Phase 3 verdict: NOT RELIABLE
        |  (separate — no shared imports)
Alpha Research Engine (src/research/)   -- Phase 5, this report
        |  (separate — no shared imports; feeds evidence, not code, into the next component)
Opportunity Scanner (src/ranking/)      -- Phase 4 verdict: WEAK (frozen baseline, Section 1)
        |  (separate — does not exist)
Risk Engine                             -- does not exist as a standalone component
        |  (separate — does not exist)
Execution Engine                        -- does not exist as a standalone component
```

Verified by import inspection: `grep -rn "from src.research" src/ranking/ src/models/ src/api/`
returns **no matches** — nothing in the production path imports anything from this phase's work.
Promoting a Phase 5 finding into the scanner is an explicit, human-driven next step (change
`DEFAULT_WEIGHTS`/add a sub-score in `score_coins.py`, then re-run a Phase-4-style audit on the
*changed* scanner) — it does not happen automatically, and nothing in this phase makes it happen
automatically.

---

## 17. Acceptance Criteria

| Classification | Requirement |
|---|---|
| FAILED | No statistically significant signal survives Stage 2 (confirmatory, non-overlapping, permutation-tested, BH-corrected) on the realistic universe. |
| WEAK | Some Stage-1 evidence exists but is unstable, economically insignificant, not robust to perturbation, or does not beat baselines after costs. |
| PROMISING | A Stage-2 survivor with monotonic quantiles, robustness across the perturbation grid (Section 13), incremental value over baselines (Section 6), economically meaningful after costs, acceptable drawdown, and no single-regime dependency. |
| SHADOW READY | PROMISING, plus frozen-test confirmation (Section 9), full provenance, no known leakage, stable under perturbation, realistic transaction costs. |
| PAPER READY | Only after shadow validation on genuinely new data without material degradation — **not achievable from a backtest alone**, and therefore not assignable in this report regardless of frozen-test results. |

<!-- SECTION_18_FINAL_DECISION -->

---

## 19. Deliverables

- `docs/PHASE5_ALPHA_RESEARCH_REPORT.md` — this report.
- `src/research/{__init__,signals,universes,walk_forward,statistics,experiment_ledger,economic_simulation,ablation,regime_breakdown}.py`
- `scripts/run_phase5_alpha_research.py` — the driver that produced every number in this report.
- `data/research/phase5_experiment_ledger.jsonl`, `phase5_summary.json`,
  `{realistic,positive_control,null}_exploratory.csv` — raw run outputs this report cites.
- `tests/test_research_{signals,walk_forward,statistics,universes,economic_simulation,experiment_ledger,ablation_and_regime}.py`
  — regression coverage for every module above, including the mandatory anti-leakage corruption
  tests (Section 3).
