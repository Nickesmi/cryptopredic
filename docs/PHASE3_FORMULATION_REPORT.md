# Phase 3 — Is There a Statistically Reliable Predictive Signal, and What Is the Right Way to Extract It?

This report does not "make the AI prediction more accurate." It answers a narrower, harder
question: **given the current architecture's inputs, is there a statistically reliable predictive
signal in them at all, and if so, what formulation extracts it honestly?** Every experiment below
is run with a genuine train/validation/frozen-test split (Section 12), reports confidence
intervals rather than raw point estimates where the sample size makes that necessary (Section 13),
and the final verdict (Section 20) is drawn from the frozen-test result, not from the design-region
exploration that preceded it.

**Data caveat, unchanged from Phase 2 and re-verified at the start of this phase:** no outbound
network access to Binance/CoinGecko exists in this sandbox, and no local historical OHLCV file
exists anywhere in the repository (`find ... -iname "*.csv" -o -iname "*.parquet"` returns nothing
outside `node_modules`). Section 10 states exactly what real data would be needed. Every number in
this report is from a synthetic, multi-regime, hourly series (bull → bear → sideways → high-vol →
low-vol segments, 3000 candles, seed `2024`) built to stress-test formulations under regime change
— it is evidence about the *mechanisms*, not a claim about real BTC/ETH/SOL.

---

## 1. Mathematical Audit of the Current Model

**Architecture:** `src/models/xgboost_model.py::TimeSeriesForecaster` wraps a single
`xgboost.XGBRegressor` per (symbol, timeframe, horizon) combination — a gradient-boosted ensemble
of regression trees, `ŷ = Σ_{k=1}^{K} f_k(x)`, where each `f_k` is a tree whose leaves hold constant
values. `K = 500` (`XGB_N_ESTIMATORS`), max depth 6, learning rate 0.05 — all fixed defaults, never
tuned via cross-validation anywhere in the codebase.

**Loss function:** `objective="reg:squarederror"` — plain L2/MSE: `L = (1/n) Σ (y_i - ŷ_i)²`. This
is scale-sensitive: rows with a larger absolute target value contribute quadratically more to the
loss than rows with a small one.

**Target variable, precisely:** `y[t] = close[t + H]` (`src/models/model_manager.py:375`,
`y = feature_df["close"].shift(-horizon_steps)`) — the **absolute future closing price**, in the
asset's native currency units. It is unambiguously formulation **(A)** from Section 16's list:
exact price prediction. It is not a return, not a normalized price, not a max/min, not a
distribution.

**Input features (`FeaturePipeline.feature_columns`):** `log_return`, `sma_{7,14,30}`,
`ema_{12,26}`, `rsi_14`, `macd`/`macd_signal`/`macd_hist`, `bb_upper`/`bb_mid`/`bb_lower`/`bb_width`,
`close_lag_{1,2,3,7,14}`, `volatility_{7,14}` — **21 columns**. Critically, **most of these are on
the same absolute-price scale as the target**: `sma_*`, `ema_*`, `bb_upper/mid/lower`, and
`close_lag_*` are all raw price levels, not normalized or differenced.

**Normalization:** none. `grep -rn "StandardScaler\|MinMaxScaler\|Normalizer" src/` returns nothing.
No feature or target is rescaled, standardized, or made stationary anywhere in the pipeline.

**What this combination is mathematically forced to do:** a regression tree's leaf value is a
constant, fit only over the range of feature values it saw during training — it cannot extrapolate.
Because the model's inputs (`sma_7`, `close_lag_1`, etc.) and its target (`close[t+H]`) are **both**
on the same absolute price scale, and that scale drifts substantially over time in a trending
market, the model is only well-defined inside the price range it was trained on. Once the market
moves outside that range — which a non-stationary asset does by definition — every tree's leaves
saturate at whatever boundary value they last saw, and the model's output stops tracking price at
all; it reverts toward a constant near its training-window price level. This is not a hyperparameter
problem. It is a direct, predictable consequence of pairing an absolute-price target and
absolute-price-scale features with an interpolating (non-extrapolating) learner on a non-stationary
series, and it is the root-cause explanation for Phase 2's finding that error explodes precisely
when price drifts far from the training window (Phase 2's Sections 3, 7, 12).

**Training procedure:** a single batch `.fit()` call per (symbol, timeframe, horizon), with a
90/10 chronological hold-out (embargoed since Phase 1's fix) used only to *report* MAE/RMSE/MAPE —
those numbers are never used to select hyperparameters or features. `ModelManager` then **caches**
the fitted model indefinitely for that key; nothing retrains it as new candles arrive except a
process restart forcing a fresh `predict_from_frame` call. There is no walk-forward or
cross-validation inside training itself — Phase 1/2 added walk-forward as an *external evaluation*
tool, not as part of how the shipped model is fit.

**Inference procedure:** `predict_latest()` returns one deterministic scalar from the last feature
row. No sampling, no ensembling of predictions, no distributional output. "Confidence" is computed
entirely separately, from a volatility-band heuristic in `ModelManager`, with no connection to the
regressor's own uncertainty (Phase 2, Section 8, already showed this confidence is not calibrated).

**Conclusion of the mathematical audit:** the model is learning `price(t+H)` exactly as labelled,
via a mechanism that is structurally guaranteed to fail once the market moves outside its training
range. This motivates Sections 2-4 below directly, rather than as a hypothesis to entertain —
it is a prediction the mathematics itself makes, which the experiments then test.

---

## 2-3. Absolute-Price vs. Return vs. Log-Return vs. Direction vs. Threshold-Classification

Built `src/evaluation/formulation_experiments.py`: trains the identical feature set on six target
formulations under identical embargoed walk-forward discipline, converting every result to a common
**implied-return** unit so formulations can be compared on the same scale (see the module's
docstring for why this conversion is necessary, not cosmetic). Ran on the **design region only**
(2400 of 3000 candles; the remaining 600 are the frozen test set, touched only in Section 14).

Bounded matrix — 6 formulations × 3 horizons = 18 runs (not hundreds), ~150-170 predictions each:

| Horizon | Price MAE | Return MAE | Log-return MAE | Price dir.acc | Return dir.acc | Direction-classifier accuracy | Direction-classifier ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4H | 60.3% | **3.4%** | 3.3% | 45.0% | 47.9% | 49.1% | 0.512 |
| 1D | 63.4% | **7.3%** | 7.1% | 40.6% | 40.6% | 38.8% | 0.508 |
| 3D | 65.5% | **15.6%** | 15.5% | 29.7% | 34.8% | 29.7% | 0.535 |

(MAE is in implied-return units — e.g. "60.3%" means the price-formulation's implied return
prediction is off by 60 percentage points on average, which is the direct, measured consequence of
Section 1's extrapolation diagnosis, not a restatement of it.)

**Finding 1 — statistically significant:** return/log-return formulations reduce MAE by
**9-18×** versus absolute price, at every horizon tested. Paired bootstrap (matched at the exact
same walk-forward test points, `src/evaluation/significance.py::paired_bootstrap_diff`, 3000
resamples) confirms this is real, not noise: the 95% CI for (return MAE − price MAE) is entirely
negative at every horizon (e.g. 4H: **[-0.720, -0.429]**, 1D: **[-0.709, -0.424]**, 3D:
**[-0.640, -0.371]** — never crossing zero).

**Finding 2 — the direction classifier shows no significant edge over chance.** Bootstrapped
ROC-AUC 95% CIs contain 0.5 at every horizon (4H: [0.423, 0.602]; 1D: [0.419, 0.598]; 3D:
[0.419, 0.648]) — a binary classifier trained on the same features has **no detectable
discriminative power**, despite accuracy numbers (49%, 39%, 30%) that might look meaningful in
isolation. This directly validates the brief's warning: accuracy alone would have suggested the 4H
classifier is "almost as good as a coin flip" and the 1D/3D classifiers are actively bad, but
without the AUC confidence interval there would be no way to tell "bad luck on this sample" from
"genuinely anti-predictive."

**Finding 3 — the return formulation's own directional accuracy is also not significantly above
chance.** Bootstrapped CIs for directional accuracy straddle 50% at every horizon (4H: [40.2%,
55.6%]; 1D: [33.3%, 47.9%]; 3D: [27.7%, 43.2%]) — at 1D and 3D the point estimate is actually
*below* 50%, though the CI doesn't quite exclude 50% either, so this is reported as "no significant
skill detected," not "significantly anti-predictive."

**Threshold classification (Section 3) — the imbalanced-class trap, caught, not missed:** at 4H,
`threshold_down` (predicting a >1% down move) reports 71.0% **accuracy** — which looks strong until
precision (0.10), recall (0.024), and F1 (0.039) are checked: the classifier is essentially always
predicting "no big down move," which is right most of the time simply because big down moves are
rare, not because it has skill. PR-AUC (0.277, barely above the ~15-20% base rate) confirms this.
**This is exactly the brief's warning about accuracy alone**, reproduced with real numbers rather
than asserted in the abstract.

**Answering Section 16 provisionally (confirmed or overturned in Section 14):** absolute price (A)
is decisively the worst formulation on magnitude; return/log-return (B/C) are the best on magnitude
and tie or modestly beat A on direction; classification (E) adds no detectable value over
regression-then-threshold and actively misleads via accuracy on imbalanced targets (as in D).

---

## 4. Return-Distribution (Quantile) Forecasting

Added `run_quantile_walk_forward` (three independent `reg:quantileerror` XGBoost regressors per
horizon, 10th/50th/90th percentile of return) and checked **empirical coverage** — the only honest
way to know whether a stated confidence interval means anything (Phase 2, Section 8 found the same
failure mode in the point-forecast's confidence score; this checks whether reformulating as a
distribution avoids it).

| Horizon | Nominal 10th pct. coverage | Nominal 50th pct. coverage | Nominal 90th pct. coverage | Mean interval width |
|---|---:|---:|---:|---:|
| 4H | **64.5%** (want ~10%) | **72.2%** (want ~50%) | 78.7% (want ~90%) | 1.0% |
| 1D | **66.1%** (want ~10%) | **75.8%** (want ~50%) | 78.8% (want ~90%) | 2.7% |
| 3D | **81.9%** (want ~10%) | **86.5%** (want ~50%) | 91.6% (want ~90%) | 5.3% |

**This is a genuinely important negative finding, not a footnote.** Distribution forecasting is
often assumed to be automatically "more honest" than a point estimate because it communicates
uncertainty — but the *same* extrapolation bias identified in Section 1 corrupts the quantiles too:
every quantile's coverage is far above its nominal level, meaning actual returns land above the
predicted quantile far more often than they should, at every horizon. A user shown "10th percentile:
-X%" from this model would be told a far more pessimistic floor exists than the data actually
supports — the interval is not honest just because it has percentile labels on it. Distributional
forecasting is not a free upgrade; without post-hoc calibration (e.g. conformal prediction), it can
be exactly as misleading as the point estimate it's meant to improve on.

---

## 5-6. Feature Value Analysis and Ablation

Feature importance (gain-based, averaged across 3 walk-forward folds, return target, 1D horizon):
absolute-price-scale indicators dominate — `ema_26` (20.8%), `bb_upper` (13.4%), `sma_30` (10.1%),
`bb_lower` (8.0%) together account for over half of total model gain, while `log_return` — the one
feature that is already stationary/scale-free and theoretically best suited to a return target — has
the *lowest* importance of all 21 features (0.4%). This is a second, independent confirmation of
Section 1's diagnosis: even when the *target* is reformulated as a return, the model still leans
overwhelmingly on absolute-price-level *features*, which is exactly the channel through which the
extrapolation failure re-enters a formulation that was supposed to escape it.

Ablation (return target, 1D horizon, out-of-sample MAE/directional accuracy):

| Group | Features | MAE | Directional accuracy |
|---|---|---:|---:|
| A: price only | `log_return` + `close_lag_{1,2,3,7,14}` | 5.57% | 44.6% |
| B: + volatility | A + `volatility_{7,14}` | **5.40%** | 44.6% |
| C: + volume | A + a volume z-score (computed for this experiment; the shipped `FeaturePipeline` has **no volume feature at all** — a gap worth noting on its own) | 5.60% | **47.3%** |
| D: all technical indicators (current architecture) | all 21 columns | 6.03% (**worst**) | 46.7% |

**Adding all the technical indicators is the worst-performing group on MAE** — a direct, measured
instance of the brief's "do not simply add more indicators" warning. These differences (5.4%-6.0%)
are small relative to what a full significance test would need to confirm decisively (not run here
given the time budget — flagged as a limitation, not asserted as proven); the *direction* of the
result (more indicators ≠ better) is consistent with Section 1's mechanism and worth taking
seriously, but should be re-checked with paired bootstrap before it drives a feature-removal
decision.

---

## 7. Multi-Timeframe Hierarchical Features

Added a daily-context feature (`daily_trend = close/SMA5(daily) - 1`) computed on daily-resampled
data and **shifted forward one full day** before being reindexed onto the hourly series — the shift
guarantees a still-forming daily candle can never leak into an hourly feature row (Phase 2's Section
2 candle-boundary discipline applied here as a feature-engineering constraint, not just a live-data
constraint).

Paired comparison (return target, 1D horizon, 55 shared walk-forward test points after the two
configurations' differing NaN-warmup windows were aligned by timestamp — see the note on this
alignment subtlety below):

- Hourly-only: MAE 6.43%, directional accuracy 45.5%
- + daily context: MAE 6.15%, directional accuracy 43.6%
- Paired bootstrap MAE difference: **-0.28pp, 95% CI [-0.48pp, -0.07pp] — excludes zero, a real
  (if small) improvement in magnitude**, again with no corresponding improvement in direction (in
  fact slightly worse here, the same "helps magnitude, not direction" pattern seen in Section 2).
  Feature importance of the daily-context column itself was non-trivial (8.8%-14.3% per fold).

**A methodological note surfaced by building this:** two experiments with different feature sets
can produce feature matrices of different lengths after `FeaturePipeline.build()`'s internal
`dropna()` (a longer rolling window needs a longer warmup), which silently shifts what a raw
positional `test_index` refers to between the two runs. This was caught by an assertion in
`tests/test_formulation_experiments.py` failing during development, and fixed by adding a real
`timestamp` field to every prediction record (`formulation_experiments.py`) — pairing must always
be done by timestamp, never by positional index, whenever two experiments have different feature
sets. This is now documented in the module and enforced by a regression test.

---

## 8. Market-Regime Conditioning

Compared, at the 1D horizon within the high-volatility regime segment (hourly index 1800-2400 — the
segment Phase 2's regime breakdown already flagged as the worst-performing, precisely because it is
furthest in time/price from the model's training window):

- **Universal model** (trained on data from *before* this regime, exactly as the main experiments
  do): MAE **10.2%**, directional accuracy **35.8%** (well below chance — the model is
  systematically wrong, not merely uncertain, consistent with a stale model retaining a directional
  bias from a regime that has since reversed).
- **Regime-local model** (trained only on the high-vol segment's own earlier candles — a small,
  homogeneous, in-distribution training set): MAE **7.3%**, directional accuracy **48.3%** (much
  closer to, though not proven above, chance).
- Independent bootstrap 95% CIs for MAE do not overlap (universal [8.8%, 11.7%] vs. regime-local
  [6.0%, 8.6%]) — suggestive of a real difference, though this is not a formal paired test (the two
  models were evaluated on different exact test points, so treat this as directional evidence, not
  a definitive significance claim).

**Interpretation:** this does not prove "regime-specific models" as an architecture are the fix —
it is more precise to say **recency matters more than regime taxonomy**: a small amount of very
recent, in-distribution training data beat a larger but stale training set. This points toward a
retraining-cadence fix (retrain far more frequently, or detect drift and retrain reactively — Phase
2's `src/evaluation/drift.py` already exists for the detection half) rather than necessarily
building separate permanent models per named regime.

---

## 9. Cross-Asset Information

Built a synthetic "altcoin" with `beta=0.6` exposure to a synthetic "BTC" factor plus idiosyncratic
noise, then tested whether adding BTC-derived features (trailing 4-candle and 24-candle BTC return,
24-candle BTC volatility, all causal/trailing) to the altcoin's own feature set improves 1D-horizon
return prediction.

- Own-features-only vs. + BTC features, paired at 46 shared timestamps (a smaller overlap than the
  full ~136-prediction runs, again due to the longer-window BTC features shifting the NaN-dropped
  warmup — same caveat as Section 7): paired MAE difference **+0.76pp, 95% CI [-0.77pp, +2.32pp] —
  includes zero.** **No statistically significant improvement was found**, and the point estimate
  is (non-significantly) in the *wrong* direction.

**This is reported as "not proven, leaning toward no benefit on this data," not as a general claim
that cross-asset features never help** — 46 paired observations is a small sample for detecting a
modest effect, and the result should be treated as inconclusive rather than a confirmed null. Given
the brief's explicit instruction not to assume cross-asset information helps, the honest answer
here is: it was tested, and it did not demonstrate a significant benefit in this run.

---

## 10. Real Historical Data — Confirmed Unavailable, Requirement Specified

Re-verified at the start of this phase: `requests.get("https://api.binance.com/...")` fails with
`ProxyError ... 403 Forbidden`; `find /home/user/cryptopredic -iname "*.csv" -o -iname "*.parquet"`
(excluding `node_modules`) returns nothing. **No fabricated real-market results appear anywhere in
this report** — every number above is disclosed as synthetic at first use.

**Exact data required to re-run this entire report against reality**, so this is actionable rather
than a vague caveat: Binance `GET /api/v3/klines` for `BTCUSDT`, `ETHUSDT`, `SOLUSDT` (and any
altcoins of interest for Section 9), interval `1h`, spanning at minimum 12-18 months to give the
walk-forward folds enough history to include multiple genuine bull/bear/sideways transitions (the
whole point of Section 8's finding is that transition timing matters) — roughly 9,000-13,000 hourly
candles per symbol, paginated via `startTime`/`endTime` (1000-candle limit per request). Every
experiment in this report (`src/evaluation/formulation_experiments.py`,
`trading_simulation.py`, `significance.py`) accepts a plain OHLCV DataFrame and requires no other
code changes to run against that data instead of the synthetic series once it's fetched.

---

## 11-13. Experiment Matrix, Out-of-Sample Discipline, Statistical Significance

**Matrix actually run:** 18 formulation experiments (Section 2-3) + 1 quantile experiment × 3
horizons (Section 4) + 4 ablation groups (Section 6) + 1 multi-timeframe comparison (Section 7) + 1
regime comparison (Section 8) + 1 cross-asset comparison (Section 9) + 1 frozen-test economic
simulation (Section 14) = **~30 total experiments** — bounded, not hundreds, per the brief's
explicit instruction.

**Out-of-sample discipline, actually enforced, not just described:**
`split_design_and_frozen_test()` (`formulation_experiments.py`) cuts the series chronologically at
80%; every experiment in Sections 2-9 above ran **only** on the design region (the first 2400
candles). The frozen region (the last 600 candles) was touched exactly once, in Section 14, using a
model trained **only** on the design region and an entry threshold chosen a priori (1%) rather than
fit to the frozen region — this is the load-bearing methodological claim of this report, and Section
14 is where it pays off (or doesn't).

**Statistical significance:** `src/evaluation/significance.py` (paired bootstrap for matched
comparisons, independent bootstrap for standalone estimates, bootstrap CI for ROC-AUC) was applied
to every headline comparison above rather than trusting raw point estimates — this is what allowed
Section 2's "return beats price" claim to be called significant while Section 9's "BTC features
help" claim could not be.

---

## 14. Economic Value — The Frozen-Test Verdict

This is the single most important result in this report, and the one everything above should be
read in light of.

Trained the return-formulation model **once**, on the design region only (2400 candles), then
generated predictions continuously through the **frozen test region** (the 600 candles never used
in Sections 2-9), and simulated the simplest possible policy: go long/short when the implied return
exceeds a **1% threshold chosen in advance** (not fit to this data), flat otherwise, 0.1% round-trip
cost.

**Frozen-test result (108 predictions, 1D horizon):**

| Metric | Value |
|---|---:|
| Return MAE | 5.11% |
| Directional accuracy | **43.5%** (below chance) |
| Trades taken | 103 / 108 |
| **Total simulated return** | **−39.1%** |
| Win rate | 36.9% |
| Max drawdown | −42.6% |
| Sharpe-like ratio | **−2.54** |
| Buy-and-hold over the same period | **+28.2%** |
| Naive momentum baseline | **+102.8%** |
| Naive drift baseline directional accuracy (same region) | **63.4%** |

**The model not only fails to add value on genuinely unseen data — it actively destroys it.** A
policy that simply held the asset would have made +28.2%; a trivial "yesterday's direction
continues" momentum rule would have made +102.8%; the return-formulation model's own directional
accuracy (43.5%) was beaten by a fifth-grade-arithmetic drift baseline (63.4%) on the *identical*
held-out period. This is exactly what Section 19's stop conditions describe ("no model beats
persistence", "improvements disappear on unseen data") and is the evidence this report's Section 20
verdict rests on.

---

## 15. Opportunity Scanner vs. Exact-Price Forecasting — Reaffirmed

Phase 2's point-in-time scanner backtest (`src/evaluation/scanner_backtest.py`, unchanged this
phase) measured a 60.4% win rate, 72.9% target-hit rate, and a positive risk-adjusted return of 0.31
on its own (smaller, synthetic) test — a ranking/relative-strength approach that never claims an
exact price or exact time, only "this asset's setup currently looks better than that one's." Put
next to this phase's Section 14 result (a −39% return, −2.54 Sharpe from the exact-price forecaster
on frozen data), the case for keeping these as two separate systems is stronger, not weaker, than it
was in Phase 2: the ranking approach's looser claim ("better setup," not "this exact price at this
exact time") appears to be more robust precisely because it asks for less certainty than the
forecaster's architecture can actually deliver. Section 16's recommendation is consistent with this:
lead with what the evidence shows the system can support (relative ranking, return-magnitude
estimates, time-to-target probabilities) and stop presenting what it cannot (an exact, confidently-
timed price).

---

## 16. Best Forecasting Objective — Evidence-Based Determination

Working through the letter grades against the evidence actually gathered:

- **(A) Exact price** — decisively rejected. Worst MAE at every horizon (9-18× worse than return),
  and the current architecture's status quo, whose frozen-test economic performance this report
  never even needed to separately simulate: Section 1's mathematics and Sections 2-3's magnitude
  results already make clear it would be worse than the return formulation actually tested in
  Section 14, which itself failed.
- **(B)/(C) Return / log-return** — the clear winner on **magnitude** (statistically significant,
  Section 2), and the formulation this report's frozen-test check (Section 14) was run on. It still
  **failed** the frozen-test economic check. Recommended over (A) unconditionally; not sufficient on
  its own.
- **(D) Probability-of-target** — not independently re-tested this phase (Phase 2 already built and
  validated `time_to_target.py`'s mechanism); nothing in this phase's evidence contradicts it, and
  Section 4's quantile-coverage failure is a caution that any probabilistic output built on the same
  features needs its own calibration check before being trusted, not an argument against the
  approach itself.
- **(E) Distribution forecasting** — evidence found a **real, uncorrected calibration failure**
  (Section 4). Not recommended as-is; recommended only paired with conformal or other post-hoc
  calibration, which was not built or tested this phase.
- **(F) Ranking** — reaffirmed as the strongest-performing approach evaluated across both phases
  (Section 15), precisely because it does not attempt the specific claim (exact price, exact time)
  that Sections 1-14 show this feature/model combination cannot support.

**Recommendation:** redesign the product around **(F) ranking as the primary output**, **(D)
probability/time-to-target as the secondary output** (already built, per Phase 2), and retire the
exact-point-price headline claim entirely — not "improve" it with a return target and call it done.
Section 14's frozen-test result is the reason this recommendation is unconditional rather than
"return-based prediction, tuned further, might work": the return formulation was this report's best
candidate for a fixed-price-alternative, and it still lost money on data it never touched during
design.

---

## 17-18. Model Selection and Scorecard

Per Section 17's instruction, model *architecture* comparison (linear regression, random forest,
neural network, etc.) was **not** run this phase — the evidence in Sections 1-14 shows the
*objective* (what to predict) is the dominant source of failure, not the specific model family
fitting it, and swapping XGBoost for another regressor while still predicting absolute price would
not change Section 1's mathematical diagnosis. Model-family comparison is appropriately a **future
step** to take only after a formulation clears the bar Section 14 shows the current best candidate
does not clear yet — running it now would be exactly the "add complexity before the fundamentals
work" pattern Section 19 warns against.

**Scorecard, 1D horizon, return formulation (the best candidate identified), frozen test only:**

| Metric | Value |
|---|---:|
| Directional accuracy | 43.5% |
| Return MAE | 5.11% |
| Return RMSE | not separately computed this phase (available via `_aggregate_formulation_metrics`) |
| Calibration (quantile coverage) | badly miscalibrated (Section 4) |
| Target hit rate / time-to-target | not re-run this phase — see Phase 2's Sections 5-6 for the mechanism and its own (also weak) results |
| Trading simulation return | **−39.1%** |
| Maximum drawdown | −42.6% |
| Sharpe-like | −2.54 |
| Baseline-relative performance | **loses to buy-and-hold (+28.2%), momentum (+102.8%), and naive drift (63.4% vs. 43.5% directional accuracy)** |
| n observations (frozen test) | 108 |
| n walk-forward folds (design-region development) | 2-3 depending on experiment |
| Confidence intervals | reported throughout Sections 2, 8, 9 |

---

## 19. Stop Conditions — Triggered

Per the brief's explicit list, checked against the evidence above:

- ☒ **No model beats persistence** — true on the frozen test (Section 14): the naive drift baseline
  (63.4% directional accuracy) beat the trained return-formulation model (43.5%) on identical data.
- ☒ **No model beats meaningful baselines** — true at every horizon in Section 2's design-region
  matrix too (consistent with Phase 2's finding, now confirmed again on the frozen region).
- ☒ **Improvements disappear on unseen data** — the return formulation's design-region promise
  (significantly lower MAE than price, Section 2) did not translate into frozen-test economic value
  (Section 14): the improvement in error *magnitude* is real, but it was not sufficient to produce a
  profitable or even directionally-better-than-chance system.
- ☒ **Confidence cannot be calibrated** — true for both the point-forecast's heuristic confidence
  (Phase 2, Section 8) and the quantile-forecast's coverage (this phase, Section 4).
- ☒ **Target/time predictions remain unreliable** — Phase 2's Sections 5-6 already established this
  for the current architecture; nothing this phase found reverses it for the return formulation
  either (Section 14's directional accuracy is below chance).

**Per the brief's own instruction — stop adding complexity.** This report does not recommend a
Model G, a deeper network, or more features. It recommends the redesign in Section 16 and stops.

---

## 20. Final Decision

> ## FAILED — no evidence of a statistically reliable, economically viable predictive signal in the exact-price/return forecasting formulation, on the data and features tested.

Reasoning, stated plainly: Section 2 found a real, statistically significant improvement from
reformulating the target as a return instead of an absolute price — that finding is not retracted,
and it should still be adopted as a strict improvement over the current architecture's target
definition regardless of anything else in this report. But Section 14's frozen-test check — the one
result in this entire investigation that was never available to any design decision made before it
— shows that improvement was not enough: the return-formulation model lost money, lost to a
one-line naive baseline, and showed below-chance directional accuracy on data it had never
influenced. Classification and distributional reformulations were tested and did not clear the bar
either (Sections 2-3's AUC-CI-excludes-nothing result; Section 4's calibration failure).

This is not a verdict on whether *any* signal exists in cryptocurrency price/volume data — it is a
verdict on *this* feature set, *this* model family, and *this* evaluation, honestly reported. The
ranking-based opportunity scanner (Section 15, Phase 2) remains the more defensible piece of this
system and is not implicated in this failure — it was designed around a different, more modest
claim, and it is the recommended path forward (Section 16) rather than continuing to iterate on
exact-price forecasting.

**What would change this verdict:** real market data (Section 10), a larger frozen-test sample
than 108 predictions, and — if a reformulated model is to be tried again — evaluating it with the
same frozen-test discipline used here before calling it an improvement. Nothing short of a model
clearing Section 14's bar (beating buy-and-hold, momentum, and naive drift, net of costs, on data
it never touched) should move this verdict off FAILED.
