# Phase 4 — Auditing the Opportunity Scanner With Phase 3's Rigor

Phase 3 concluded the point-price/return forecasting engine is FAILED and should not be pursued
further. This phase does not assume the opportunity scanner fares better just because Phase 2's
narrower backtest was positive — it re-audits the scanner mathematically, for look-ahead, and for
cross-sectional predictive power, exactly as skeptically as Phase 3 treated the forecaster.

**Data caveat, unchanged:** no outbound network access to Binance/CoinGecko exists in this sandbox
and no local historical OHLCV file exists anywhere in the repository. Every number below is from
synthetic multi-asset panels built specifically to test the scanner's *mechanisms* — a "null"
universe (deliberately no embedded cross-sectional signal, to check for false positives), a
"realistic-alpha" universe (a modest, plausible quality/junk split among 19 assets), and a
"positive-control" universe (an extreme, unambiguous quality/junk split, to prove the mechanism can
detect signal at all when it is undeniably present). None of this is a claim about real crypto
markets — it is a claim about what this code does and does not detect, under conditions this audit
fully controls and discloses.

---

## 1-2. Mathematical Audit and Complete Look-Ahead Audit

Read `src/ranking/liquidity_filter.py`, `risk_score.py`, `score_coins.py`, `recommend.py` line by
line. Exact formulas, in order of the pipeline:

**Liquidity gate** (`check_liquidity`, hard pass/fail, computed only from `df.tail(≤30 or 60)`):
- `avg_quote_volume = mean(close * volume)` over the trailing 30 candles; fails if `< $1,000,000`.
- `spread_proxy = mean((high - low) / close)` over the trailing 30 candles; fails if `> 8%`.
- `max_single_candle_return = max(|pct_change(close)|)` over the trailing 60 candles; fails if
  `> 60%` (the anti-manipulation / bad-data check).
- Fails if fewer than 90 candles of history exist at all.

**Risk score** (`assess_risk`, 0-100, higher = riskier, trailing 60 candles):
```
volatility            = std(log(close / close.shift(1)))
max_drawdown          = min((close - cummax(close)) / cummax(close))
liquidity_score       = clip((log10(avg_quote_volume) - 5.0) / 3.0 * 100, 0, 100)
vol_component         = clip(volatility / 0.10, 0, 1) * 100
drawdown_component    = clip(|max_drawdown| / 0.50, 0, 1) * 100
illiquidity_component = 100 - liquidity_score
risk_score = clip(0.45*vol_component + 0.30*drawdown_component + 0.25*illiquidity_component, 0, 100)
```

**Opportunity score** (`compute_opportunity_score`, five sub-scores, each 0-100, trailing windows
only — `_trend_quality` uses `SMA_7`/`SMA_30`/`RSI_14` at the last row; `_momentum` uses 7/14/30-
candle trailing returns; `_relative_strength` uses a 14-candle trailing return vs. the benchmark,
excess capped at +25%; `_volume_confirmation` uses 5-candle vs. 30-candle trailing volume; `_volatility_adjusted`
is a 30-candle trailing Sharpe-like ratio):
```
raw_score    = 0.25*trend_quality + 0.20*momentum + 0.20*relative_strength
             + 0.15*volume_confirmation + 0.20*volatility_adjusted
risk_penalty = max(0, risk_score - 30) * 0.5
final_score  = clip(raw_score - risk_penalty, 0, 100)
```
Weights and the risk-penalty formula are fixed constants (`DEFAULT_WEIGHTS`,
`_RISK_PENALTY_FORMULA_VERSION`), not fit to any dataset — confirmed by reading, not merely by
the module's own docstring claiming it.

**Direction / target / invalidation** (`_direction_and_target`, `_invalidation_price`): direction
from the sign of a 15-candle trailing return; target = current price × (1 ± 1.5×trailing-30-candle
volatility); invalidation = 20-candle trailing low (bullish) or high (bearish).

**Look-ahead audit result: no violation found**, on two independent checks:

1. **By inspection.** Every calculation above reads only `.tail(N)` / `.shift(positive)` /
   `.cummax()` of a DataFrame that the caller has already point-in-time-sliced. Critically, **the
   scanner performs no cross-sectional normalization of any kind** — no z-scoring or percentile-
   ranking of a score against other assets' scores, and no min/max scaling against the historical
   dataset's own range. This means the specific forbidden pattern the audit brief names
   ("normalize asset_score using the min/max across the entire historical dataset") cannot occur
   here — there is no such normalization step to get wrong. (This is itself a finding, not just a
   clean bill of health — see Section 16.)
2. **The one path not previously verified, checked empirically this phase.** `scan_candidate()`
   calls `time_to_target_report()`, which internally runs a forward-scanning first-passage-time
   search over the candidate's *own* history to estimate `expected_horizon_seconds` /
   `probability_estimate`. This is legitimate retrospective statistics (every data point involved,
   both the base index and its forward window, lies within the point-in-time-sliced input) but had
   never been directly tested. `tests/test_rankings.py::test_full_recommendation_is_unchanged_by_corrupting_the_future`
   now does exactly what Phase 2's scanner-backtest-level test did *not* cover: scores an asset at
   `t`, multiplies every price after `t` by 50×, re-scores, and asserts every field — including
   `expected_horizon_seconds` and `probability_estimate` — is byte-for-byte identical. **Confirmed
   identical.**

**A caller-discipline note, not a bug:** `_relative_strength` and `classify_regime` depend on
whatever `benchmark_df` the caller passes in. The production API (`routes_rankings.py`) always
fetches a fresh benchmark for live use, which is correct for live scanning; any *historical*
reconstruction script must slice the benchmark to `.iloc[:t]` exactly like every other candidate.
Phase 2's `scanner_backtest.py` and this phase's `cross_sectional_analysis.py` both do this
correctly (`benchmark_df = point_in_time[benchmark_symbol]`) — verified by reading, and indirectly
proven by the no-lookahead test above, which uses a real benchmark throughout.

---

## 3. Survivorship Bias

Unchanged from Phase 2's disclosure, restated precisely: every backtest in this report (and Phase
2's) draws its candidate universe from a fixed, present-day `candidates: dict[str, DataFrame]`
supplied by the caller. It cannot select an asset that existed historically but is absent from that
dict, and it does not penalize a surviving asset for others having been delisted around it. This is
not fixable with synthetic data — it requires a real historical exchange-listing snapshot (which
symbols were actually tradable, at what liquidity, at each past timestamp, delisted ones included).
**Exact requirement to close this gap:** a historical listing/delisting log per exchange (e.g.
Binance's own symbol-status history) merged with historical OHLCV for every symbol that was ever
listed, not just those listed today. No such data source exists in this repository or this
environment; this limitation is disclosed on every relevant result rather than assumed away.

---

## 4-6. Historical Reconstruction and Forward-Return Analysis

Built `src/evaluation/cross_sectional_analysis.py`: at each historical scan point, scores **the
entire candidate universe** (not just the eventual top picks) via the exact production
`scan_candidate()` function, then computes forward returns at 1D/3D/7D/14D/30D (1H/4H/12H were not
separately run this phase given the daily-candle synthetic panels used — the mechanism accepts any
horizon in candles and was validated at daily resolution; sub-daily horizons need intraday
synthetic data or real intraday history to exercise meaningfully, noted as a follow-up). Every
snapshot records timestamp, per-asset score, and which assets were excluded and why — satisfying
the brief's storage requirement for reconstructed rankings.

---

## 7-8. Cross-Sectional Test and Information Coefficient

Three controlled universes, 18-25 scan points each, Spearman rank correlation between score and
forward return, computed **per snapshot then aggregated** (not pooled across snapshots, which would
hide whether signal is cross-sectional or just a shared market-wide move):

| Universe | Design | Mean IC (7D) | Mean IC (14D) | % periods IC > 0 (7D) | Approx. 95% CI excludes 0? |
|---|---|---:|---:|---:|---|
| **Positive control** | 5 assets with extreme, obvious planted uptrend vs. 5 with extreme planted downtrend vs. 5 neutral | **+0.535** | +0.450 | 100% | **Yes** — [0.42, 0.65] roughly |
| **Realistic-alpha** | 3 of 19 assets with a modest, plausible embedded quality edge (smooth +0.15%/day, low noise), 3 with embedded junk (-0.12%/day, high noise) | -0.035 to +0.062 across horizons | -0.011 | 33-67% (horizon-dependent, inconsistent) | **No**, at every horizon |
| **Null** | All 19 non-benchmark assets pure noise, no embedded signal | -0.064 to +0.019 across horizons | -0.018 | 33-61% (horizon-dependent, inconsistent) | **No**, at every horizon |

**Quintile analysis** (7D forward return, lowest-score bucket 0 → highest-score bucket 4):
- Positive control: **clean, monotonic staircase** — -2.1%, -1.9%, +0.9%, +1.9%, +3.0%. Exactly
  what real cross-sectional signal should look like.
- Realistic-alpha: **not monotonic** — 1.1%, 0.9%, 0.3%, 0.9%, 0.4%. The top-scored quintile does
  not have the highest forward return.
- Null: **not monotonic, and inverted at the top** — 1.8%, 0.8%, 1.3%, 1.6%, -0.8%. The
  highest-scored quintile has the *worst* forward return in this specific run.

**What this proves and what it doesn't:** the scanner's mechanism is not structurally broken — given
an obvious, unambiguous quality signal, it finds it cleanly and significantly (this is the important
positive-control result: it rules out "the scanner can never work," the harshest possible verdict).
But at every plausible-for-real-markets signal strength tested (realistic-alpha) and at zero signal
(null), the measured IC is **statistically indistinguishable from zero** at every horizon, and the
quintile buckets are **not** monotonic — meaning this audit found **no evidence of real
cross-sectional predictive power at a realistic signal-to-noise ratio**, only evidence that the
tool *could* detect signal several times stronger than anything remotely realistic.

## 9. Ranking Stability

`rank_turnover()` on a 60-day-stride panel: **mean top-3 turnover of 73%** between consecutive
rebalances — roughly 2 of every 3 top-3 slots change each rebalance. Combined with the weak/
inconsistent IC above, this is a second independent signal of a noisy, unstable score rather than a
persistent, tradable ranking.

---

## 10-13. Transaction-Cost-Aware Backtest, MFE/MAE, and Risk-Adjusted Performance

Built `src/evaluation/portfolio_backtest.py`: equal-weight top-N, 7-day holding, 0.1% round-trip
cost per position, rebalanced every 7 days, 60 rebalances, compared against BTC buy-and-hold,
equal-weight-all-alts, trivial-momentum-top-3 (ranked purely by trailing return, no scanner
involved), and random-top-3 (seeded, reproducible):

| Strategy | Alpha universe total return | Alpha Sharpe-like | Null universe total return | Null Sharpe-like |
|---|---:|---:|---:|---:|
| **Scanner top-3** | +9.9% | 0.46 | **-16.6%** | -0.47 |
| **Scanner top-5** | **+18.9%** | **0.91** | -3.6% | -0.04 |
| BTC buy-and-hold | -22.7% | -0.74 | -22.7% | -0.74 |
| Equal-weight universe | +4.5% | 0.43 | +2.8% | 0.28 |
| Momentum top-3 (no scanner) | +7.9% | 0.38 | -1.9% | 0.08 |
| Random top-3 (seeded) | +7.9% | 0.40 | **+7.8%** | 0.38 |

On the alpha universe (real signal present), the scanner's top-5 portfolio clearly leads every
baseline including trivial momentum — a genuine positive result when the underlying data actually
contains the kind of quality/junk separation the scanner is designed to detect. On the **null**
universe, however, scanner top-3 **underperforms literal random selection by 24 percentage points**
(-16.6% vs. +7.8%), and top-5 is roughly flat while random and equal-weight are both mildly
positive.

**Reconciling this with Section 8's near-zero average IC on the null universe (important, and not a
contradiction):** a separate, finer-grained ablation run (25 scan points instead of 18) found the
null universe's mean IC much closer to zero (-0.007, CI clearly spanning 0) with no individual
sub-score showing a significant bias either. The most defensible reading of both results together:
**the scanner carries no significant average cross-sectional bias on pure noise, but a concentrated
top-3 portfolio built from an inherently noisy score is itself high-variance, and can realize a
large gain or loss purely by chance even when the underlying ranking carries zero true edge** — a
distinct and important finding from "the scanner is systematically biased," and one directly
relevant to Section 19's robustness question.

**MFE/MAE and target/invalidation hit rates:** already built and validated in Phase 2
(`scanner_backtest.py`'s `ScannerTrade` records `max_favorable_excursion`, `max_adverse_excursion`,
`target_hit`/`target_hit_candles`, `invalidation_hit`/`invalidation_hit_candles`) — not re-run from
scratch this phase since the mechanism itself was not in question, only the cross-sectional
predictive power was. One acknowledged simplification carried over from Phase 2: "target hit" is
evaluated only within the fixed forward window, so "eventually hit but after the intended horizon"
and "never hit" are not currently distinguished — a real but minor gap, not re-engineered this phase
given the higher-priority finding above.

---

## 14. Regime Breakdown

Mean 7D IC by BTC regime, alpha universe (11-15 periods per regime — too few for significance
individually, but the *pattern* is informative):

| Regime | Mean IC |
|---|---:|
| Bull | -0.069 |
| Bear | -0.050 |
| Sideways | +0.048 |
| High volatility | +0.120 |
| Low volatility | +0.101 |

The sign of the cross-sectional signal **flips between trending (bull/bear) and non-trending
(sideways/high-vol/low-vol) regimes**, even though the planted quality/junk assets carry the same
constant structural edge throughout. This means the scanner's ability to detect a real, constant
edge is itself regime-dependent — consistent with the audit brief's concern that a scanner might
"only work in one particular environment." With this few periods per regime this is suggestive, not
proven, but it is a second, independent line of evidence (alongside Section 9's high turnover)
against treating the current scanner as regime-robust.

---

## 15. Asset-Class / Market-Cap Breakdown

**Not run — reported as a limitation, not skipped silently.** This requires real sector/category
labels (DeFi, L1, L2, meme, etc.) that do not exist anywhere in this codebase or in the synthetic
data built for this audit; fabricating a sector taxonomy for synthetic assets would not test
anything real. Real-data acquisition would need a source like CoinGecko's category endpoints
cross-referenced with the price history used elsewhere in this report.

---

## 16. Ablation Study

Isolated each sub-score (weight 1.0 on one, 0 on the rest) against the null universe, 25 scan
points, 7D horizon:

| Component | Mean IC | Approx. 95% CI |
|---|---:|---|
| Trend quality only | -0.005 | [-0.099, 0.088] |
| Momentum only | -0.019 | [-0.110, 0.071] |
| Relative strength only | +0.046 | [-0.059, 0.152] |
| Volume confirmation only | -0.050 | [-0.168, 0.069] |
| Volatility-adjusted only | -0.019 | [-0.128, 0.089] |
| Full scanner (all five, default weights) | -0.007 | [-0.109, 0.095] |

**No individual component, and no combination, shows a statistically significant edge on pure
noise** — reassuring in that no single sub-score is a systematic false-positive generator, but also
meaning **no component demonstrated real signal either**, on this synthetic panel. Sections A-F of
the brief's example table (liquidity-only, momentum-only, etc., "H/I/J: full scanner minus X") were
not all separately run given the time this phase had; the five single-component runs above are the
core of that request and were prioritized because they directly test whether any one piece is
carrying the (already-shown-to-be-weak) aggregate signal.

---

## 17. Score Weight Validation

The weights (`DEFAULT_WEIGHTS` in `score_coins.py`) remain exactly what Phase 2 documented:
analyst-set constants, never fit to any dataset. This phase did not run a formal train → validation
→ frozen-test weight optimization (that would require meaningfully more compute and, per the
brief's own caution, a real risk of overfitting weights to whatever synthetic panel was used) — the
ablation study in Section 16 is the closest thing to an empirical weight check performed this phase,
and it shows no individual component is clearly worth over- or under-weighting relative to the
others on the evidence available. **Recommendation, not yet executed:** if real market data becomes
available, run a proper walk-forward weight search (train/validation/frozen-test, exactly as Phase
3 did for the forecaster) before ever changing `DEFAULT_WEIGHTS` from their current, transparently-
documented, non-optimized values.

---

## 18. Statistical Significance

Every IC figure in Sections 7-9 and 16 is reported with an approximate 95% confidence interval
(normal approximation over the per-snapshot IC series; `n` = number of scan points, as listed).
**At no horizon, in either the null or the realistic-alpha universe, does the confidence interval
exclude zero.** Only the deliberately extreme positive-control universe produces a CI that clearly
excludes zero. This is the central, load-bearing statistical result of this report: Phase 2's
positive backtest and this phase's positive-control check both show the scanner *can* register a
strong effect — what has never been shown, at any point across two audit phases, is a *statistically
significant* effect at a signal strength plausible for real markets. No formal multiple-testing
correction (e.g. Bonferroni across the 5 horizons × 2 universes × 6 ablation components tested) was
applied on top of this — with none of those individual results reaching significance in the first
place, a correction would only push them further from significance, not create a false positive to
guard against.

---

## 19. Robustness

Two independent fragility signals, not one:
1. **Top-N sensitivity:** on the alpha universe, top-3 returned +9.9% while top-5 returned +18.9%
   — a 2× difference from changing one parameter by 2 assets. Per the brief's own standard ("if tiny
   changes destroy performance, classify the scanner as fragile"), this qualifies: the strategy's
   realized outcome is not stable to a small, defensible parameter choice.
2. **Rank turnover:** 73% mean top-3 turnover per rebalance (Section 9) means the *composition* of
   what gets traded is itself highly unstable, independent of whether the underlying score is any
   good — a practical concern for real transaction costs beyond what the flat 0.1%-per-position
   assumption in Section 10-13 captures (real turnover-driven costs would compound every rebalance
   the holdings actually changed, which this simulation already assumes happens on every
   rebalance regardless — a conservative, not optimistic, assumption, but still worth noting the
   *ranking itself* is this unstable underneath it).

Holding-period and fee-assumption sensitivity beyond the 7-day/0.1% figures used throughout were not
separately swept this phase given time constraints — flagged as an incomplete part of Section 19,
not a result.

---

## 20-21. Provenance and No-Forced-Recommendation

**Fixed this phase, mirroring Phase 2's forecaster fix:** `OpportunityRecommendation` previously
had no version fields at all. Added `scanner_version` (a content-hash fingerprint of the sub-score
weights and a manually-bumped formula-identity string — two scans get the same fingerprint iff they
used the same scoring configuration), `feature_version` (reusing `FeaturePipeline.feature_version`
from Phase 2), and `scanned_at` (generation timestamp). All three are now present in every
recommendation's `to_dict()` output and covered by `tests/test_scanner_versioning.py`.

**No-forced-recommendation:** already built and tested in Phase 2
(`NO_OPPORTUNITY_MESSAGE`) and re-confirmed this phase at the portfolio level —
`backtest_top_n_portfolio(..., min_score=99.99)` correctly holds an empty, zero-return position for
every period rather than forcing a pick (`tests/test_portfolio_backtest.py::test_min_score_can_exclude_every_period_and_holds_cash`).

---

## 22. Final Decision

> ## WEAK — small, inconsistent signal; the mechanism works but has not demonstrated a real edge at a realistic signal strength.

This is deliberately not FAILED and deliberately not PROMISING, and the distinction matters:

- **Not FAILED**, because the positive-control experiment (Section 7-8) rules out the harshest
  possible verdict — given an obvious, unambiguous cross-sectional quality signal, the scanner finds
  it cleanly (IC = 0.535, monotonic quintiles, all periods positive). The scoring mechanism, the
  liquidity gate, and the look-ahead discipline are all sound (Sections 1-2). This is a materially
  different position than Phase 3's forecaster, which failed even its own best formulation's frozen
  out-of-sample test outright.
- **Not PROMISING**, because "positive out-of-sample ranking signal" was never established at a
  realistic signal strength — every IC measured on the null and realistic-alpha universes, at every
  horizon, has a confidence interval spanning zero (Section 18), and the top-N portfolio backtest
  underperforms literal random selection on pure noise data by a wide margin (Section 10-13),
  reconciled as high concentration-driven variance rather than systematic bias, but still not
  evidence of edge.
- **WEAK** fits: "small/inconsistent signal" describes exactly what Sections 7-9, 14, and 16 show —
  point estimates that drift across zero depending on horizon, universe, and regime, with turnover
  and top-N sensitivity on top of that.

**What would move this off WEAK:** real market data (the standing limitation across all four
phases of this audit), a larger and longer historical panel (more assets, more independent
non-overlapping periods — 18-25 snapshots on ~20 assets is not enough statistical power to
distinguish "no edge" from "a modest, real edge" the way Section 7's numbers show), and — only if
real data supports it — a proper walk-forward weight optimization (Section 17) evaluated on a
frozen test region the way Phase 3 did for the forecaster.

---

## 23. Architecture: Keep the Four Responsibilities Separate

Restating and updating the split with this phase's evidence:

- **Forecasting Engine** ("what is the probabilistic future distribution?") — **NOT RELIABLE**
  (Phase 3's FAILED verdict stands unchanged).
- **Opportunity Scanner** ("which assets currently have the best risk-adjusted setup?") — **WEAK**
  per this phase; still the more defensible of the two research directions, and still the
  recommended primary research candidate, but not yet validated enough to act on.
- **Risk Engine** ("how much exposure is appropriate?") — **does not exist as a separate
  component.** `risk_score.py`'s per-asset risk assessment is embedded inside the scanner (used
  only as a penalty on the opportunity score); there is no portfolio-level risk manager governing
  aggregate exposure, correlation between concurrent positions, or position sizing beyond the
  backtest's own flat equal-weighting. This is a gap to build, not a finding about existing code.
  its own equal-weight assumption. This is a gap to build, not a finding about existing code.
- **Execution Engine** ("can this trade actually be executed at reasonable cost?") — **does not
  exist as a live component.** `trading_simulation.py` and `portfolio_backtest.py` are backtest
  research tools that assume a fixed cost percentage; there is no real order-execution simulator
  (order-book depth, realistic slippage curves, partial fills) anywhere in this codebase.

Per the brief's explicit instruction, none of these four should be collapsed into one model or one
service — this phase's evidence, if anything, reinforces that: the scanner's weak-but-not-absent
signal (Opportunity Scanner) and its total absence in the forecaster (Forecasting Engine) are very
different findings that would be impossible to state clearly if the two systems were not already
architecturally separate.
