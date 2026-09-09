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

**Result, in one line (Section 18 has the full answer):** 3,450 experiments recorded; the pipeline
correctly detects a planted signal (mean IC ≈0.97, 10/10 confirmatory-significant) and correctly
finds nothing in pure noise (0/416 null cells confirmed); on the realistic universe, **zero**
candidates produced even exploratory evidence at any horizon short enough to confirm, so **zero**
reached confirmatory testing, the frozen test, or an economic backtest. **Final classification:
FAILED.**

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

## 8. Exploratory Screen Results (Stage 1)

`scripts/run_phase5_alpha_research.py` ran to completion in 1,749.9s and recorded **3,450 total
experiments** to `data/research/phase5_experiment_ledger.jsonl`. The realistic-universe exploratory
screen alone produced **2,608 records** (146 signal×own-horizon combinations × up to 8 forward
horizons × up to 2 methods × 2 regions).

| Universe | Exploratory cells | BH-significant (α=0.05, within-family) |
|---|---:|---:|
| Realistic | 2,608 | **43 (1.6%)** |
| Positive control | 416 | **411 (98.8%)** |
| Null | 416 | 15 (3.6%) |

Two things stand out immediately, before any confirmatory testing:

**Where the realistic universe's 43 "hits" live.** Every single one is at a forward horizon of 7D,
14D, or 30D (20 at own+fwd=30D/30D, 9 more at other 14D/30D pairings, 6 at 7D) — **zero** of the 43
occur at 1H/4H/12H/1D/3D, the horizons short enough that the 24-candle exploratory stride does not
produce severe window overlap. The single largest hit, `price_volume_confirmation` at 30D/30D
(Pearson IC -0.186, analytic p≈1.5e-32, n=113 overlapping snapshots), is a textbook symptom of the
Section 5 caveat: at a 720-candle lookback and a 720-candle forward window advanced only 24 candles
per snapshot, consecutive "independent" observations share >96% of their underlying candles.

**The same pattern appears in the null universe** (6 of 15 significant cells at region=validation,
100% of those at 7D/14D — see `data/research/null_exploratory.csv`), and, in reverse, **in the
positive control**: of the positive control's only 5 non-significant cells (out of 416), all 5 are
at own=fwd=30D — the same starved-sample cell that inflates false "significance" elsewhere makes a
*real, enormous* effect (mean IC ≈0.97 everywhere else) occasionally fail to clear the p<0.05 bar
here, purely from estimation variance at n=83-113. The same cell is unreliable in both directions,
in all three universes — strong, self-consistent evidence that the exploratory screen's stated
anti-conservativeness at long horizons is real and is exactly where it was predicted to bite.

A further disclosed artifact, visible in the raw ledger: `short_term_reversal` and
`medium_term_reversal` are both implemented as `-momentum(lookback)` (Section 2 of `signals.py`);
swept at the *same* own-lookback as `momentum` itself, they are mathematically its negation, not
independent signals at that specific cell of the sweep — their appearance among the "significant"
cells (e.g. at own=30D) is the same underlying momentum-family result counted under three names, not
three corroborating discoveries. `relative_strength` and `cross_sectional_relative_strength` also
converge numerically with raw `momentum` at the 30D lookback in both the realistic and null
universes (identical mean IC to 6 decimal places) — at that timescale, idiosyncratic dispersion is
small relative to shared market-wide moves, so subtracting a benchmark or universe-mean return barely
changes the cross-sectional rank order. Both artifacts are disclosed here rather than left to look
like five independent corroborating hits.

**No signal shows exploratory evidence at any horizon the confirmatory stage could actually test**
(Section 9) in the realistic universe.

---

## 9. Confirmatory Stage (Stage 2) and Frozen Test

Confirmatory re-testing rebuilds each Stage-1 survivor's panel at a **non-overlapping** stride
(stride = the candidate's own forward horizon) restricted to the validation region alone (2,000
candles). This mechanically requires `n_periods = 2000 / stride ≥ 20` to even attempt a permutation
test — which immediately exposes a structural power limit disclosed here plainly: **a forward
horizon of 7D (stride 168) or longer cannot reach 20 non-overlapping validation-region
observations at all** (2000/168 ≈ 11 for 7D; 2000/336 ≈ 5 for 14D; 2000/720 ≈ 2 for 30D). Since
every one of the realistic universe's 43 Stage-1 hits sits at 7D/14D/30D (Section 8), **all 43 are
structurally unconfirmable with this study's validation-region length** — not "tested and failed",
but "the only stage-1 evidence that existed was at horizons long enough that no legitimate
non-overlapping confirmatory test of it was possible with 8,000 total candles." This is reported as
a power limitation of the experiment (see Section 18's data requirements), not papered over: **0
realistic-universe candidates were tested in Stage 2, and consequently 0 were BH-significant, and 0
frozen-test evaluations were run** — the frozen test region was **never touched** in this run,
exactly as the "at most once, and only for a confirmed survivor" rule requires.

**Positive control — mechanism validation (mandatory, Section 14):** all 10 Stage-1 survivors that
did reach a testable horizon (1H and 4H own/forward pairings, n=500-2,000 non-overlapping periods)
were re-confirmed: mean IC 0.968-0.970, permutation p-value **0.000** (0 of 300 shuffles matched or
exceeded the observed mean IC), **10/10 BH-significant** after correction. This is the pipeline
working exactly as required: an unambiguous, non-overlapping-tested, permutation-significant signal
is found when one is deliberately planted.

**Null — specificity validation (mandatory, Section 14):** 0 null-universe cells reached the
validation-region + n≥20 + BH-significant bar required to even enter Stage 2 — the null control
produced **zero** confirmatory-stage false positives, consistent with a correctly calibrated
procedure (its 6 validation-region Stage-1 "hits", like the realistic universe's, were entirely
concentrated at the unconfirmable 7D/14D horizons — see Section 8).

---

## 10-13. Economic Backtest, MFE/MAE, Ablation, and Robustness — Not Run, By Design

Sections 9-13 of the audit brief request these analyses **"for promising signals"**, and Section
17's acceptance ladder makes statistically significant, confirmatory-stage IC a *prerequisite* for
even reaching the PROMISING tier, let alone the tier where economic backtesting matters. Section 18
is explicit: **"If the answer is FAILED or WEAK, do NOT invent additional features simply to
continue development."** Zero candidates survived Stage 2 on the realistic universe (Section 9).
Running a transaction-cost-aware backtest, an MFE/MAE report, an ablation study, or a perturbation
sweep on a candidate that already failed the IC significance test it needed to pass first would not
be rigor — it would be exactly the kind of "keep trying things until something looks good" the
brief repeatedly prohibits. `src/research/economic_simulation.py`, `ablation.py`, and the
`parameter_sensitivity_sweep`/`cost_sensitivity_sweep` machinery all exist, are unit-tested, and are
ready to run **the moment Stage 2 produces a confirmed survivor** — in this run, on this data, it did
not, so they were not invoked on a non-survivor merely to fill in a report section.

---

## 14. Positive-Control and Null-Control Validation (Mandatory)

Both mandatory controls (Section 9's confirmatory results, repeated here for the record required by
Section 14):

| Control | Requirement | Result |
|---|---|---|
| Positive control | Planted alpha must be detected | **Detected**: mean IC 0.968-0.970 across all 10 testable (own,forward)-horizon pairs, permutation p=0.000 (0/300 shuffles matched or exceeded it), 10/10 BH-significant after correction, at both 1H and 4H horizons. |
| Null control | Pure noise must not produce significant alpha after multiple-testing correction | **Not produced**: 0 of the null universe's cells reached the validation-region + n≥20 + BH-significant bar needed to even enter confirmatory testing; the 6 Stage-1 "hits" it did show were, like the realistic universe's, entirely artifacts of the unconfirmable long-horizon overlapping-window cells (Section 8), not real alpha. |

Both mandatory conditions are satisfied: **the pipeline can find a real, strong signal when one
exists, and does not manufacture a false one out of pure noise after correction.** This validates
the Alpha Research Engine's mechanism itself — the realistic-universe FAILED result (Section 9, 18)
is therefore attributable to the absence of a confirmable signal in that universe's data, not to a
broken or miscalibrated testing pipeline.

---

## 15. Multiple-Testing Discipline and the Experiment Ledger

Every experiment cell described in Sections 7-9 (Stage 1 and Stage 2, all three universes, both
regions) is written as one `ExperimentRecord` to `data/research/phase5_experiment_ledger.jsonl` —
nothing tested is discarded, including every non-significant and negative result. Each record
carries: signal family/name, own-horizon, forward-horizon, universe, region, IC method, n_periods,
mean/median/std IC, information ratio, hit rate, quintile monotonicity (computed at each signal's
own natural horizon), permutation p-value and BH-significance flag where Stage 2 was run, and free-
text notes carrying the Stage-1 analytic p-value.

**3,450 total experiment records** across 3 universes: 2,608 (realistic exploratory) + 416
(positive-control exploratory) + 416 (null exploratory) + 10 (positive-control confirmatory) = 3,450.
0 realistic-universe and 0 null-universe candidates reached the confirmatory stage; both counts are
recorded in the ledger as zero, not omitted. Every exploratory cell — including all 2,565
non-significant realistic-universe cells — is a row in
`data/research/phase5_experiment_ledger.jsonl`, addressable by
`(universe, family, signal_name, own-horizon, forward-horizon, method, region)`, so a future phase
that wants to test, say, "momentum at 1D predicting 3D forward returns on the realistic universe"
can look up this exact result (mean IC, n, analytic p, BH-significance) before spending compute
re-deriving it.

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

---

## 18. Final Decision

> **Is there a statistically defensible crypto alpha signal in the currently available data that
> materially improves asset selection beyond random selection and simple baselines after costs?**

# **FAILED**

No candidate — across 19 signal specs, 8 own-lookback horizons, 8 forward-return horizons, 3
universes, 2 IC methods, 2 testing stages, and 3,450 recorded experiments — reached the
confirmatory, non-overlapping-stride, permutation-tested, multiple-testing-corrected bar on the
realistic universe. The result is not "borderline" or "mixed": at every forward horizon short
enough to test without the overlapping-window artifact documented in Sections 5, 8, and 9 (1H
through 3D), the realistic universe showed **zero** exploratory-stage evidence for **any** of the
momentum, mean-reversion, volume/flow, volatility, relative-strength, or cross-sectional signal
families. The only cells that ever looked significant lived exclusively at horizons (7D/14D/30D)
where this study's 8,000-candle sample cannot even construct a non-overlapping confirmatory test —
and the identical pattern of spurious "significance" at exactly that cell, in *both directions*,
appeared in the null universe (false positives) and the positive control (occasional false
negatives on a real, huge effect), which is the clearest possible demonstration that those specific
cells' apparent results are sampling artifacts of the exploratory stride, not evidence about markets.

This is not a verdict on the Alpha Research Engine itself: the mandatory controls (Section 14) show
the pipeline correctly detects an unambiguous planted signal (mean IC ≈0.97, permutation p=0.000,
10/10 significant) and correctly finds nothing in pure noise after correction (0/416 null cells
reached confirmatory testing). The pipeline works. It found nothing to confirm in the "realistic"
synthetic data because — by construction, and this is the honest caveat that must travel with a
FAILED verdict on synthetic data — that universe's weak AR(1) autocorrelation (|phi| ≤ 0.07) may
itself simply be too faint, or differently shaped, than whatever autocorrelation (if any) exists in
real crypto markets. **FAILED here means "this pipeline, run on this repository's best disclosed
synthetic approximation of a real market, found no confirmable signal" — it is evidence about the
method and about this synthetic universe, not a proof that no signal exists in real crypto data.**

**1. What was tested.** All of Section 5's 146 (signal, own-horizon) combinations × 8 forward
horizons × up to 2 IC methods, on 3 disclosed-synthetic universes, across 2 testing stages, with
Benjamini-Hochberg correction applied within each tested family — 3,450 recorded experiments, every
one written to the ledger regardless of outcome.

**2. What failed.** Every realistic-universe candidate failed to produce Stage-1 (exploratory)
evidence at any horizon short enough for Stage 2 (confirmatory, non-overlapping,
permutation-tested) to even attempt verification. No candidate reached Stage 2. No candidate
reached the frozen test. No candidate reached an economic backtest.

**3. What evidence is missing.** Real historical OHLCV data for the assets this system actually
trades. Everything in this report is necessarily computed on disclosed synthetic data (Section 4/6)
because this sandbox has no outbound network access to any real market-data source and no local
historical file exists in the repository — a limitation carried unchanged from Phases 2-4. A
FAILED verdict on synthetic data is evidence the *method* is sound (Section 14) and that *this
particular synthetic approximation* of a real market contains no confirmable signal at the tested
horizons; it is explicitly **not** evidence that real crypto markets contain no such signal. Also
missing: a real historical asset-listing/delisting registry (Section 4) and a real historical
market-cap series (Section 6, baseline #3) — neither obtainable without external data access.

**4. What data would be required to continue.** (a) Real historical OHLCV for a broad, liquid
crypto universe, at 1-hour or finer resolution, spanning multiple regimes (at least one full
bull/bear cycle, ideally 3+ years) — needed to re-run this exact pipeline (Sections 2, 5, 7-9)
against markets instead of a synthetic approximation. (b) A real historical listing/delisting log,
to replace the partial (listing-only) survivorship fix in Section 4 with a genuine survivorship-free
universe. (c) A real historical market-cap/circulating-supply series, to add baseline #3 (Section
6). (d) Ideally, a validation region long enough that 7D-30D horizons can also be confirmed
non-overlapping (Section 9's power limitation) — a longer real history would resolve this
automatically; on synthetic data it would require deliberately generating a longer sample.

**5. Should the project remain research-only?** **Yes.** Nothing in this report, or in Phases 3-4,
supports moving any signal — forecaster, scanner, or any Phase 5 candidate — beyond
`src/research/` and `src/ranking/`'s existing frozen, unpromoted state. The Alpha Research Engine
should remain exactly what Section 16 declares it: a research component that a human explicitly
promotes from, never a component anything in `src/api/` calls automatically. The correct next step
is not more synthetic experimentation on this codebase — the pipeline's mechanism is already proven
sound via the controls — but real historical data, per item 4 above.

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
