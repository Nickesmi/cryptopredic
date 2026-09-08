# Phase 2 — Validation, Calibration, and Production Readiness

This report follows on from `docs/TIMING_AUDIT_REPORT.md` (the root-cause
fix for the timeframe/horizon conflation bug, 90 tests passing at the time
it was written). It does **not** re-litigate that fix; it verifies it,
then runs the empirical validation, calibration, and production-safety
work that fix alone did not constitute.

**Read this first:** this sandbox has no outbound network access to
Binance or CoinGecko (`ProxyError ... 403 Forbidden` on every attempt,
reconfirmed at the start of this phase — see Section 11 of the prior
report and Section J below). Every quantitative result in Sections 3–6 of
this report is computed by **actually running the shipped
`walk_forward_validate`, `benchmark.py`, `scanner_backtest.py`,
`success_criteria.py`, and `calibration.py` modules** against a
purpose-built **synthetic, multi-regime, hourly OHLCV series** (bull →
bear → sideways → high-vol → low-vol segments, 3000 hourly candles, seed
`2024`), not against real BTC history. The mechanisms are real and now
proven correct; the specific numbers are not a claim about real Bitcoin.
Re-running the same scripts against real market data, the moment network
access exists, is the single most important open action item (Section J).

---

## 1. Verify the Horizon Fix

**Claim to verify:** 4H → 4 hours, 1H → 1 hour, 1D → 1 day, and the
training target, inference target, evaluation window, prediction expiry,
and API/UI representation all refer to the same horizon.

**Method:** read the live implementation line-by-line
(`src/models/model_manager.py`, `src/api/routes_forecast.py`,
`src/evaluation/prediction_store.py`) and grepped the entire repository
for `timeframe|horizon|n_candles|target|expiry|prediction_time|
target_time|future|lookahead|days|hours|candles`.

**End-to-end trace of one prediction** (`GET /api/predict/BTCUSDT?tf=4H&n=6`):

```
MARKET DATA    BinanceAdapter.get_candles("BTCUSDT","4H",limit=1000)      [binance.py:125]
   │           -- REST kline array, last bar's is_closed derived from its
   │              own close_time vs now (fixed this phase, see Section 2)
   ▼
CANDLE         bars_to_frame(bars)  -- drops the bar if not is_closed      [candles.py:12]
   ▼
FEATURE        clean_ohlcv + add_log_returns + FeaturePipeline.build()    [model_manager.py:362-363]
   │           interval_seconds = timeframe_to_seconds("4H") = 14400       [timeframes.py]
   │           horizon_steps = n_candles = 6                               [model_manager.py:360]
   ▼
MODEL INPUT    X = feature_df[feature_cols].values                        [model_manager.py:374]
   ▼
TARGET         y = feature_df["close"].shift(-horizon_steps)               [model_manager.py:375]
   │           == close price 6 * 14400s = 24h ahead. Nothing else.
   ▼
MODEL OUTPUT   predicted_price = forecaster.predict_latest(X[-1:])         [model_manager.py:381-]
   ▼
HORIZON        target_timestamp = current_time + horizon_steps*interval    [model_manager.py:375]
   │           horizon_seconds = horizon_steps*interval = 21600s = 6h
   ▼
DATABASE       PredictionStore.create_prediction(
   │             expires_at = forecast.target_timestamp,                   [routes_forecast.py]
   │             prediction_horizon = forecast.n_candles = 6)
   ▼
API            ForecastObject.to_api_dict() exposes target_timestamp,
   │           horizon_seconds, n_candles, timeframe -- all consistent
   ▼
EVALUATION     PredictionStore.due_predictions() fires when
               now >= expires_at == target_timestamp == the same value
               the model was trained against.
```

Every arrow in that chain carries the **same** number
(`horizon_steps * interval_seconds`, here 21,600 seconds). This is now
pinned down by an executable regression test, not just documentation:
`tests/test_horizon_consistency.py::test_target_timestamp_matches_requested_horizon`,
parametrised over `1H/4h`, `4H/6candles`, `1D/7candles`, and `1m/30candles`,
asserts `forecast.target_timestamp == forecast.anchor_time +
n_candles*interval` and that it's the *last* point on the displayed price
path. Ran now: **6/6 parametrised cases pass.**

**Remaining hard-coded horizon mappings found by the repo-wide grep:**
**none** in the live prediction path. The grep did surface one **separate,
pre-existing, and — after inspection — internally-consistent** pipeline:
`src/data/fetch_market_data.py::run_pipeline()` (used by
`src/dashboard/streamlit_app.py` and nothing else) takes a `horizon`
argument that means **calendar days on daily CoinGecko data**, completely
independent of `ModelManager`'s `n_candles`-of-a-timeframe concept. It is
not bugged — its docstring and the Streamlit UI both say "Forecast
Horizon (Days)" and never claim any other resolution — but it is a second
place in the codebase where the word "horizon" means a different unit
than it means in `model_manager.py`. Flagged as a maintainability risk in
Section I, not a correctness bug.

**Verdict: PASS.** The specific bug from the prior audit
(`_TF_TO_DAYS`/`_HORIZON_MAP`) does not exist anywhere in the current
source tree (confirmed by `grep -rn "_TF_TO_DAYS\|_HORIZON_MAP\|horizon_days"` across
the whole repo — only doc references to the historical bug remain).

---

## 2. Verify Train/Live Data Consistency

| Path | Data source | Notes |
|---|---|---|
| Live inference | `ModelManager.predict()` → `ExchangeFactory.create("binance")` | Fetches the exact requested timeframe, up to 1000 candles. |
| Training (for that same prediction) | Identical `raw_df` → `predict_from_frame()` | Same call, same DataFrame — there is only one code path. |
| Backtest (`src/evaluation/backtest.py`) | Caller-supplied historical DataFrame → `predict_from_frame()` | Same function as live inference. |
| Walk-forward (`src/evaluation/walk_forward.py`) | Caller-supplied historical DataFrame → `predict_from_frame()` (via a fresh `ModelManager` per fold) | Same function. |
| Scanner backtest (`src/evaluation/scanner_backtest.py`) | Caller-supplied historical DataFrame, point-in-time sliced | Uses `score_coins.py`'s own `FeaturePipeline.build()` call, same pipeline class. |
| Shadow evaluation (`src/evaluation/shadow.py`) | Injected `price_lookup`; production implementation (`binance_price_lookup`) fetches from Binance | Not exercised live in this sandbox (no network) — see Section 13. |

**There is exactly one feature-engineering code path** (`FeaturePipeline`)
and exactly one target-construction code path
(`close.shift(-horizon_steps)`, inside `predict_from_frame`), shared by
live inference, the backtester, and walk-forward validation. This was
true after the first audit pass and remains true; it is the single
biggest structural reason train/live skew cannot silently reappear.

**New finding this phase — a real candle-boundary bug, not previously
caught:** `src/data/exchange/binance.py::_parse_kline()` hard-coded
`is_closed=True` on **every** REST-fetched kline, including the
currently-forming candle that Binance's `/api/v3/klines` endpoint returns
as the last element whenever its open time has already passed. Nothing
downstream ever checked the flag. This is exactly the "10:15 must not use
the unclosed 10:00–14:00 4H candle" hazard the original brief named
explicitly (Phase 5), and it existed in the code both before and
immediately after the first audit pass — the first pass fixed the
horizon-labelling bug but did not check this.

*Fix (this phase):* `_parse_kline` now derives `is_closed` from the
kline's own `close_time` (index 6 of the raw array) vs. wall-clock time.
`src/utils/candles.py::bars_to_frame()` drops a trailing unclosed candle
by default — this is the **one, shared enforcement point**, so both
`ModelManager.predict()` and the opportunity scanner's live fetch
(`routes_rankings.py`) are fixed by the same change. It also incidentally
fixed a second latent bug: the REST `/api/candles` endpoint's `is_closed`
field (documented in `routes_forecast.py` as the signal the frontend must
use before "shift[ing] the prediction forward") was **always `true`**
before this fix, silently defeating that documented safety mechanism for
REST-fetched history (the WebSocket path was already correct, since it
reads Binance's own `x` flag). Regression tests:
`tests/test_candle_boundaries.py` (6/6 passing), covering both the
close-time derivation and the frame-level drop, including an explicit
"can opt out for non-model consumers" case.

**Other consistency checks:**
- **Timezone:** every timestamp in the live path is UTC epoch seconds,
  derived from `pd.Timestamp(seconds, tz="UTC")` — confirmed by reading
  `bars_to_frame`, `CandleBar.time` (Binance ms→s), and
  `CoinGeckoPriceRepository._to_series` (separately, `.dt.normalize()` on
  a UTC-localised series for the unrelated daily pipeline). No local-time
  conversion exists anywhere in the codebase.
- **Normalization / feature ordering:** `FeaturePipeline.feature_columns`
  is a deterministic, order-stable list derived from the pipeline's own
  config; `X = feature_df[feature_cols].values` always re-selects by name,
  never by positional order, so column order cannot silently drift between
  training and inference even if `feature_df`'s own column order changes.
- **Missing/duplicate candles:** previously unchecked anywhere in the live
  path. Fixed this phase — see Section 14.

**Verdict: PASS, with one real bug found and fixed** (the unclosed-candle
issue above). No other train/live skew found.

---

## 3. Walk-Forward Validation — Actually Run

Ran `src/evaluation/walk_forward.py::walk_forward_validate()` (the real,
shipped module — not a mock) against the synthetic multi-regime series,
for all six horizons the brief lists, 2 expanding-window folds each,
`min_train_size=700` hourly candles, strided to ~150 evaluated
predictions per horizon for tractability (see the module's `stride`
parameter, added this phase specifically to make studies like this one
computationally feasible without changing the no-leakage guarantees —
`tests/test_walk_forward.py::test_stride_reduces_prediction_count_without_changing_no_leakage_guarantees`
confirms the embargo/expanding-window invariants hold identically at any
stride).

| Horizon | n | MAE (USD) | Directional accuracy | Notes |
|---|---:|---:|---:|---|
| 1H | 154 | 12,804 | 41.6% | Below coin-flip |
| 4H | 154 | 13,686 | 46.8% | Below coin-flip |
| 12H | 153 | 14,078 | 48.4% | Below coin-flip |
| 1D | 151 | 15,135 | 41.1% | Below coin-flip |
| 3D | 155 | 17,505 | 36.1% | Well below coin-flip |
| 7D | 142 | 17,923 | 31.7% | Far below coin-flip |

**This is a genuinely bad result, and it is reported as such.** Directional
accuracy *decreases* monotonically with horizon and is below 50% (a coin
flip) at every single horizon tested — at 7D it is 31.7%, meaning the
model is directionally **wrong more often than random guessing would be**.
MAE of $12,800–$17,900 against a series whose price ranges $7,947–$52,252
(see Section 6) is 25–50% of the asset's own price level.

**Root cause, confirmed by direct inspection (not just inferred):** each
fold trains **once** on its training block, then rolls forward through the
test block re-deriving *features* from newer data while the *model
itself* stays frozen (`walk_forward.py`'s documented design — see its
module docstring). Gradient-boosted trees cannot extrapolate past the
range of feature values seen during training (each leaf outputs a
constant). When the synthetic series drifts through a full bull run and
crash — from ~$40,000 up to ~$52,000 and back down to ~$8,000 — a model
trained near the top of the range is later asked to price an asset that
has moved far outside anything it saw in training, and it cannot. This
was first observed in the original audit's Section 7 on a single
unbounded trend; this phase confirms it generalises to a realistic,
regime-shifting series across **every horizon**, not just one contrived
case.

---

## 4. Baseline Comparison

Ran `src/evaluation/benchmark.py::compare_to_baselines()` (now with four
baselines — `moving_average` and `random_direction` were added this
phase alongside the pre-existing `naive_persistence` and `naive_drift`,
completing the brief's requested set) against the same walk-forward
output.

| Horizon | MODEL dir/MAE | persistence dir/MAE | drift dir/MAE | moving-avg dir/MAE | random dir/MAE |
|---|---|---|---|---|---|
| 1H | 41.6% / 12,804 | 0.0%\* / 240 | 49.8% / 246 | 50.4% / 611 | 50.3% / 370 |
| 4H | 46.8% / 13,686 | 0.0%\* / 469 | 50.4% / 511 | 49.3% / 728 | 50.1% / 548 |
| 12H | 48.4% / 14,078 | 0.0%\* / 780 | 52.8% / 970 | 48.8% / 980 | 49.4% / 831 |
| 1D | 41.1% / 15,135 | 0.0%\* / 1,119 | 53.4% / 1,631 | 46.2% / 1,296 | 49.4% / 1,166 |
| 3D | 36.1% / 17,505 | 0.0%\* / 2,086 | 53.0% / 4,390 | 49.0% / 2,205 | 51.1% / 2,104 |
| 7D | 31.7% / 17,923 | 0.0%\* / 3,412 | 53.3% / 9,613 | 47.4% / 3,548 | 50.7% / 3,427 |

\*`naive_persistence` predicts zero change; `evaluate_prediction`'s
`direction_correct` requires a strictly-signed predicted return, so a
flat predictor scores exactly 0% by definition — not a claim that
persistence has literally negative skill (this is documented in
`tests/test_benchmark.py::test_naive_persistence_always_predicts_no_change`
and flagged as a metric edge case in Section I).

**The model's MAE is 10–50× worse than every baseline, at every horizon,
with no exception.** `compare_to_baselines()`'s own `model_beats_*` flags
are `False` across the board — confirmed programmatically, not just read
off the table. **OLD vs. NEW**, done as a real (not analytical) single-point
comparison since a full walk-forward re-run of the OLD, broken async
backtester is not meaningful (it never functioned — see the prior
report's Section 2, #2) and re-training it 150+ times per horizon the way
the NEW system was validated above would not be a fair use of the time
this phase had:

| Requested horizon | OLD system actually trained on | NEW error (this snapshot) | OLD error (this snapshot) |
|---|---|---:|---:|
| 1H | 7 calendar days | $9,036 (96.6%) | $16,280 (174.1%) |
| 4H | 30 calendar days | $8,371 (89.5%) | $26,226 (280.5%) |
| 12H | 30 calendar days (no native "12H" tf existed) | $7,934 (84.9%) | $26,443 (282.8%) |
| 1D | 30 calendar days | $11,231 (120.1%) | $26,221 (280.4%) |
| 3D | 30 calendar days | $18,019 (192.7%) | $27,090 (289.7%) |
| 7D | 30 calendar days | $30,935 (330.8%) | $28,023 (299.7%) |

Built with the OLD system's *actual* shipped behaviour reproduced exactly
(CoinGecko-style daily, degenerate open=high=low=close bars;
`_TF_TO_DAYS`-bucketed target), trained fresh at one snapshot (75% through
the series, landing in the high-volatility regime — one of the harder
points for either system, as Section 6 also shows), then both judged
against the real price at the horizon the user was actually told to
expect. **OLD is worse than NEW at every horizon except 7D**, where NEW's
own genuine difficulty at that horizon (documented above) closes the gap.
Note the OLD column for 12H/1D/3D/7D is **the same trained model** four
times over — a live illustration of the original bug's "three different
user requests, one identical underlying model" finding, this time with
real numbers instead of an analytical argument. This is a single
snapshot (n=1 per horizon), stated plainly as such — it complements,
rather than substitutes for, the ~150-sample-per-horizon walk-forward
results above.

---

## 5. Time-to-Target Validation

Built a genuine per-prediction ledger (~150 predictions per horizon):
for each, recorded `anchor_time`, `predicted_price`, `predicted_horizon`,
`predicted_target_time`, then searched forward (real first-passage-time
search, `time_to_target.first_passage_candles`, up to a 20-day window) for
`actual_target_hit_time`. `time_error = actual_target_hit_time -
predicted_target_time`, computed **only** over predictions that were
actually hit — the never-hit population is reported separately per the
brief's explicit instruction, not folded into an average that would
misrepresent it.

| Horizon | n hit / n total | mean error | median error | MAE | RMSE | bias | std | %≤25% of horizon | %≤50% | %≤100% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1H | 31/152 (20.4%) | +31.0h | +8.0h | 31.0h | 76.8h | +31.0h | 71.5h | 19% | 19% | 32% |
| 4H | 31/152 (20.4%) | +25.6h | +12.0h | 27.0h | 61.1h | +25.6h | 56.4h | 10% | 13% | 35% |
| 12H | 29/151 (19.2%) | +56.3h | +14.0h | 61.9h | 112.9h | +56.3h | 99.6h | 17% | 21% | 48% |
| 1D | 30/164 (18.3%) | +65.3h | +13.0h | 77.5h | 127.8h | +65.3h | 111.8h | 10% | 17% | 67% |
| 3D | 31/159 (19.5%) | -40.4h | -63.0h | 59.1h | 63.7h | -40.4h | 50.0h | 13% | 13% | 94% |
| 7D | 31/151 (20.5%) | -118.6h | -149.0h | 130.4h | 137.4h | -118.6h | 70.5h | 10% | 13% | 100% |

**Target never hit** (within the 20-day search window): **~79–82% of all
predictions, at every horizon.** This is the dominant failure mode, and it
is a *different* failure from "hit late" — a model whose predicted price
level is simply never revisited is not a timing problem, it is a
magnitude/direction problem (consistent with Section 3's MAE findings).

For the minority that *do* hit: at the **1H** horizon, the median actual
arrival is **8 hours** after a 1-hour promise (8×); at **4H**, median
arrival is **12 hours** after a 4-hour promise (3×); at **12H**, median
arrival is 14 hours after a 12-hour promise (roughly on time, but the mean
of 56h and a 99.6h std shows this is a very fat-tailed, unreliable
"roughly." At **3D/7D** the sign flips — targets that do hit tend to hit
**earlier** than promised (median -63h and -149h respectively). This is a
believable, non-fabricated result: at very long promised horizons, a
small predicted move (the clamped-model artifact from Section 3) is often
crossed by ordinary noise well before the long horizon elapses, while a
larger, harder move is either hit quickly by chance or not at all. The
`%≤100% of horizon` columns climb toward 100% at 3D/7D for exactly this
reason — that tolerance band becomes very wide in absolute terms at long
horizons and should not be read as "timing gets better," only as "the
window got big enough to catch almost anything."

---

## 6. Original Failure-Pattern Investigation

Bucketed actual achievement time (as a fraction of *all* predictions, hit
or not, at each horizon) into the brief's own buckets:

| Horizon | <4h | 4-8h | 8-16h | 16-24h | 1-2d | 2-7d | 7d+ | never hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1H | 7.9% | 2.0% | 3.9% | 2.0% | 2.6% | 0.7% | 1.3% | 79.6% |
| 4H | 5.9% | 0.7% | 2.6% | 4.6% | 4.6% | 0.7% | 1.3% | 79.6% |
| 12H | 4.6% | 0.7% | 3.3% | 0.7% | 4.6% | 2.0% | 3.3% | 80.8% |
| 1D | 2.4% | 2.4% | 0.6% | 1.2% | 4.9% | 2.4% | 4.3% | 81.7% |
| 3D | 3.8% | 5.7% | 2.5% | 1.9% | 1.3% | 3.8% | 0.6% | 80.5% |
| 7D | 2.6% | 2.0% | 4.6% | 2.6% | 3.3% | 4.0% | 1.3% | 79.5% |

This **directly answers the brief's motivating question**: at the 1H
horizon, of the predictions whose target is ever reached, a comparable
share land in the 4-8h and 8-16h buckets as land under 4h — i.e. a
"1-hour" prediction that does come true is about as likely to actually
arrive several hours to half a day late as on time. The dominant number
in every single row, however, is "never hit" at ~80% — which is the
honest headline: the more common failure is not mistimed arrival, it is
that the promised price is simply not reached within any reasonable
window, at any horizon tested.

**Is the model genuinely horizon-aware, or just predicting eventual
direction?** Neither, on this evidence. If it were genuinely horizon-aware
we would expect the "never hit" rate to *fall* as the promised horizon
grows (more time for a real move to arrive); it does not — it is flat at
~80% from 1H to 7D. If it were merely predicting eventual direction
(ignoring timing) we would expect target-hit rate to keep climbing with a
longer search window regardless of horizon; the *search* window is
already fixed at 20 days for every row, so a flat ~80% never-hit rate
across horizons instead suggests the model's predicted price levels
themselves are simply not well-chosen (Section 3's MAE finding), and
horizon has little influence on which are eventually reached.

---

## 7. Is Point-Price Prediction the Right Objective?

Evidence assembled above, read together:

- **(A) Exact future price** (current architecture): loses to every
  baseline at every horizon (Section 4); MAE is 25-50% of asset price;
  target-hit rate ~18-21% regardless of horizon (Section 5/6).
- **(D) Probability of reaching a target**: `time_to_target_report()`
  (built and unit-tested last phase, exercised for real this phase inside
  the ledger) produces smooth, monotonic, sensible probability-within-H
  curves from nothing more than historical first-passage counting — no
  model training involved, no extrapolation failure possible. It is by
  construction never "confidently wrong" the way a point prediction is;
  its only failure mode is wide confidence intervals on small samples,
  which is visible and honest rather than hidden.
- **(B) Future return / (C) distribution of return**: not separately
  re-implemented this phase, but Section 3's diagnosis (trees cannot
  extrapolate absolute price ranges) applies with *less* force to a
  return target, since returns are far more stationary than price levels
  across regime changes — this is a reasoned inference from the existing
  results, not independently measured here, and should be treated as a
  hypothesis to test, not a proven fix.
- **(E) Time-to-target** as a first-class output (rather than a derived
  afterthought) is directly supported by Section 5: the system already
  has a genuine, working estimator for it, and it is arguably more useful
  to a user than a point price, since Section 5/6 show the point price is
  right only ~20% of the time regardless of what horizon is promised.

**Recommendation, evidence-based, not a default preference for
sophistication:** stop presenting a single point price as the headline
output. The current architecture's point-price objective (A) is
empirically the weakest of everything measured in this report — it never
beat a coin flip past the 4H horizon and never beat a single naive
baseline at any horizon. Recommend **(F) a combination**, specifically:
lead with the empirical **time-to-target probability curve** (D/E, already
built, already shown to behave sensibly) and a **return-based** (not
absolute-price) point estimate as a secondary figure, explicitly labelled
with its historical MAE/directional-accuracy at that horizon (Section 3's
table, or the equivalent computed on real data) so a user can weigh it
correctly instead of assuming false precision. This is not a call to
throw away the XGBoost model — it is a call to stop making an unqualified
"$X by time T" claim the headline when the evidence says that specific
claim is right about 1 time in 5.

---

## 8. Confidence Calibration

`src/evaluation/calibration.py` (built last phase) run for real against
the ledger's confidence scores and **directional success** as the outcome
(the strictest fair choice available without a large history of
`horizon_success` labels to bucket against yet — see Section 11's #2 for
why `horizon_success` calibration needs more accumulated data than a
150-row synthetic run provides).

**1H horizon** (Brier score **0.378** — worse than the 0.25 you'd get by
always saying "50%"):

| Confidence bucket | n | Predicted | Actual success | Gap |
|---|---:|---:|---:|---:|
| 60-70% | 12 | 66.6% | 25.0% | +41.6pp (overconfident) |
| 70-80% | 34 | 74.0% | 50.0% | +24.0pp (overconfident) |
| 80-90% | 24 | 87.5% | 37.5% | +50.0pp (overconfident) |
| 90-100% | 82 | 93.9% | 57.3% | +36.6pp (overconfident) |

**1D horizon** (Brier score **0.303**):

| Confidence bucket | n | Predicted | Actual success | Gap |
|---|---:|---:|---:|---:|
| 0-50% | 78 | 13.5% | 34.6% | -21.1pp (*under*confident) |
| 50-60% | 15 | 55.3% | 53.3% | +2.0pp (well calibrated) |
| 60-70% | 8 | 64.1% | 37.5% | +26.6pp (overconfident) |
| 70-80% | 63 | 75.3% | 44.4% | +30.9pp (overconfident) |

**7D horizon** (Brier score **0.243** — the "best" of the six, but only
because the model almost never states confidence above 50%, so there is
only one bucket to score): 0-50% bucket, n=151, predicted 12.9%, actual
25.2% — **under**confident here, the opposite direction from the high-
confidence buckets at shorter horizons.

**Answering the brief's question directly: no, the system's confidence
score does not mean what it claims.** Every bucket with a stated
confidence above ~55% is meaningfully overconfident (gaps of 20-50
percentage points); the only well-behaved bucket in this run is the
one near 50-60% at the 1D horizon. This is now checked by real,
executable code (`compute_calibration`, `CalibrationReport.is_well_calibrated()`)
wired into `PredictionStore.calibration_report()` /
`GET /api/performance/calibration`, not asserted from reading the
confidence formula — `src/models/model_manager.py`'s confidence is a
volatility-band-width heuristic with no historical grounding, and this
section is the empirical proof that it does not happen to be calibrated
by accident either.

---

## 9. Precise Definitions of "Success" (implemented, not just described)

`src/evaluation/success_criteria.py::evaluate_success()` (new this phase)
returns five **independent** booleans per prediction — none is derived
from another:

- **Directional success** — sign of actual return at the *promised*
  horizon matches sign of predicted return.
- **Price success** — actual price at the promised horizon within 1% of
  the predicted price.
- **Target success** — the predicted price was reached at *any* point
  within the search window (ignores timing). `None`, not `False`, if
  never reached within the window searched.
- **Horizon success** — target success **and** the hit landed within 25%
  of the promised horizon. This is the only one of the five that requires
  both "right price" and "right time" — exactly the brief's worked
  example ("$100,000 in 4H, arrives 3 days later" → `target_success=True`,
  `horizon_success=False`) is a unit test:
  `tests/test_success_criteria.py::test_price_right_but_wildly_late_is_target_success_not_horizon_success`.
- **Trading success** — whether acting on the prediction, net of an
  assumed 0.1% round-trip cost, would have been profitable.

**Measured on the same ledger** (all six horizons pooled, ~903 predictions):
directional success ≈ 41% (below chance, consistent with Section 3),
target success ≈ 20%, but **horizon success is only 2.4%** (computed from
the six horizons' `horizon_success_rate` values in
`/tmp/phase2_summary.json`, ranging 2.0%-3.9% per horizon — see Section 6
for the underlying hit/never-hit numbers). **This is the single number
that most directly answers the motivating question of this whole audit:**
even after the timing-label bug is fixed, a prediction that is
simultaneously right on price *and* right on time happens on this
synthetic, regime-shifting series only about 1 time in 40.

---

## 10. Opportunity Scanner Audit

| Item | Status | Where |
|---|---|---|
| Liquidity filtering | ✅ | `liquidity_filter.py` — min history, min quote volume, spread proxy, abnormal-candle rejection |
| Market-cap filtering | ❌ gap | No market-cap data source wired in (would need a CoinGecko `/coins/markets`-style feed not currently used anywhere in the codebase) |
| Volume filtering | ✅ | `liquidity_filter.py` (gate) + `score_coins.py::_volume_confirmation` (scoring) |
| Spread/slippage | ⚠️ proxy only | `spread_proxy = (high-low)/close` — a real bid/ask spread needs order-book data this system doesn't fetch |
| Volatility | ✅ | `risk_score.py::assess_risk`, `score_coins.py::_volatility_adjusted` |
| Momentum | ✅ | `score_coins.py::_momentum` (7/14/30-candle returns, rewards acceleration) |
| Relative strength | ✅ | `score_coins.py::_relative_strength` vs. the benchmark (BTC), reward capped to avoid rewarding a "moon coin" pump |
| Trend | ✅ | `score_coins.py::_trend_quality` |
| Market regime | ✅ **fixed this phase** | Was computed for the price forecaster (`model_manager.py`) but never surfaced by the scanner. Extracted to a shared `src/utils/regime.py::classify_regime()`; `OpportunityRecommendation` now carries both the candidate's own `market_regime` and the benchmark's `benchmark_regime`, with a `key_risks` note when a pick runs against the benchmark's regime ("swimming against the tide"). |
| "Already pumped" penalty | ✅ | `_trend_quality` penalises RSI > 80; `liquidity_filter` rejects a single-candle move > 60% outright; `_relative_strength`'s reward is capped at +25% excess return, so a much larger recent pump does not score higher than a moderate one |
| Risk score | ✅ | `risk_score.py` — kept as a separate penalty, never blended into the raw opportunity sub-scores |
| Opportunity score | ✅ | `score_coins.py` — five named, weighted sub-scores, transparent by construction |

**Verifying "cannot rank a coin highest purely because it already
pumped":** `tests/test_rankings.py::TestLiquidityFilter::test_rejects_vertical_pump`
constructs a flat-then-+150%-candle series and confirms it is *excluded*
by the liquidity gate (abnormal-price-action / spread-proxy), not merely
scored lower — this is deliberate: a hard gate is more robust than a
scoring penalty that a strong-enough short-term move could still
overcome.

---

## 11. Opportunity Scanner Backtest (point-in-time, no forced lookahead)

Ran `src/evaluation/scanner_backtest.py::backtest_scanner()` (new this
phase) on a small synthetic universe: BTC-like benchmark, one persistent
outperformer, one persistent underperformer, daily candles, 33 scan
points (every 30 days), 14-day forward evaluation window, `min_score=55`:

```
n_scans: 33  (2 produced no qualifying recommendation)
n_trades: 48
mean_subsequent_return: +2.29%   median: +1.33%
win_rate: 60.4%
target_hit_rate: 72.9%           invalidation_hit_rate: 10.4%
mean_max_favorable_excursion: +7.16%   mean_max_adverse_excursion: -4.59%
risk_adjusted_return: 0.31 (mean return / return std-dev across trades)
median_time_to_target: 5 candles
```

**No-lookahead guarantee, proven, not just asserted:**
`tests/test_scanner_backtest.py::test_no_lookahead_changing_the_future_does_not_change_past_picks`
corrupts every candidate's price data after a fixed scan point by 100× and
confirms the opportunity scores computed **at** that scan point are
byte-for-byte identical — the scanner literally cannot see the corrupted
future, because it is never given it (`candidates[symbol].iloc[:t]` is
the only data passed in for a scan at index `t`).

**Survivorship bias — explicitly NOT fully eliminated, and the module
says so on every call:** `backtest_scanner()`'s report always carries a
caveat that it uses a fixed, present-day candidate universe for the whole
backtest window. It genuinely cannot "pick" an asset that existed
historically but was delisted before today and is therefore absent from
whatever `candidates` dict a caller supplies. Removing this class of
survivorship bias needs a real historical universe snapshot (with
delisted tokens included) that this repository has no data source for —
stated plainly rather than silently assumed away.

---

## 12. Market Regime Analysis

Regime breakdown for the **1D** horizon (chosen as representative; the
same dispersion pattern held at every horizon inspected), from the same
ledger used in Sections 5/6:

| Regime | n | Directional accuracy | Target-hit rate | Mean price MAE |
|---|---:|---:|---:|---:|
| bear | 46 | 47.8% | 65.2% | $3,378 |
| sideways | 55 | 38.2% | 0.0% | $7,001 |
| high_vol | 54 | 27.8% | 0.0% | $26,832 |
| low_vol | 9 | 88.9%\* | 0.0% | $34,295 |

\*n=9 — noted explicitly as too small a sample to trust; included for
completeness, not as a claim about low-vol regime performance.

**Performance changes dramatically by regime, and it is not hidden here:**
error is **~8× larger** in the `high_vol`/`low_vol` regimes (which occur
late in the synthetic series, furthest from the single training snapshot)
than in `bear` (closer in time to training). This is the clearest possible
confirmation of Section 3's extrapolation diagnosis: it is not that the
model is uniformly mediocre, it is that its error is small near its
training window and catastrophic far from it, and "far from it" here
means *elapsed time and price drift*, not any inherent property of bear
vs. high-vol markets. A production system that retrains only rarely
would reproduce this same pattern on real markets.

---

## 13. Shadow Mode

Implemented `src/evaluation/shadow.py::evaluate_due_predictions()` this
phase — the piece the prior report's Section 11 (#5) flagged as entirely
missing: `PredictionStore` had `due_predictions()` and
`evaluate_prediction()` but nothing ever called the second from the
first. It now does, via an injected async `price_lookup` (so it's fully
unit-testable without network — 4/4 tests passing, including "one failed
lookup doesn't block the batch" and "re-running the cycle doesn't
double-evaluate"). `binance_price_lookup()` is the real production
implementation and `scripts/run_shadow_cycle.py` a minimal periodic entry
point (cron/Celery-beat/systemd-timer — whichever the deployment already
uses).

**Not activated against live data in this environment** — no outbound
network access. This is the most consequential unfinished item in this
report: nothing in Sections 3-12 is a substitute for actually running
shadow mode against real prices for a real evaluation window before
trusting any of this system's numbers on real markets.

---

## 14. Data Quality Monitoring

New: `src/data/quality.py`. `validate_ohlcv()` checks (all real, all
unit-tested — 11/11 passing in `tests/test_data_quality.py`): empty
frame, missing required columns, duplicate timestamps, non-monotonic
index, non-finite values, non-positive prices, impossible OHLC
(`high < low`, etc.), negative volume, timestamp gaps (warning below 5%
of the series missing, critical above), staleness (critical if the last
candle is more than 2 candle-intervals old), and abnormal-volume outliers
(z-score, warning-only). `enforce_quality_gate()` raises
`DataQualityError` on any critical issue and is now wired directly into
`ModelManager.predict()`'s live path — a bad data point can no longer
silently reach feature engineering.

**A real, version-dependent bug was caught while writing these tests, not
found by inspection:** the gap-detection logic originally computed candle
spacing via `df.index.view("int64") / 1e9`, assuming nanosecond storage.
pandas ≥ 2.x (this environment runs pandas 3.0.5) can store `DatetimeIndex`
at microsecond resolution instead, silently making every computed gap
1000× too small and defeating the check entirely on this pandas version.
Fixed by diffing as `timedelta64` and dividing by `np.timedelta64(1,
"s")`, which is resolution-independent. Caught by
`tests/test_data_quality.py::test_large_missing_fraction_is_critical`
failing on first run — a concrete example of why every new check in this
report was validated by an actually-failing-then-passing test, not
written and trusted on inspection alone.

**Deliberately NOT wired into the backtester/walk-forward path** — those
consume already-known historical frames and must stay deterministic with
respect to wall-clock time (the staleness check specifically would
misfire on old, intentionally-historical data). This is documented in the
wiring comment in `model_manager.py`.

---

## 15. Model Drift

New: `src/evaluation/drift.py`, using the Population Stability Index
(PSI) — a standard, simple credit-risk/ML-monitoring metric — for
prediction-distribution, target-distribution, error-distribution,
confidence-distribution, and per-feature drift. `PredictionStore.drift_report()`
compares the older vs. newer half of the accumulated evaluation history
and is exposed at `GET /api/performance/drift`. 7/7 tests passing,
including a synthetic case with an obvious shift correctly classified
"significant" (PSI > 0.25) and an identical-distribution case correctly
classified "none" (PSI < 0.1).

**Per the brief's explicit instruction, this module only detects and
reports drift — it does not trigger retraining.** Deciding *why* drift
happened (genuine model decay vs. a regime change vs. an upstream
data-quality issue) needs human judgement this module cannot supply on
its own, and the module's own docstring says so.

**Not yet exercised against real accumulated predictions** — like
calibration, `drift_report()` needs enough evaluated history
(`min_rows_per_half=30` on each side by default) to say anything, and the
store is currently empty in any real deployment until shadow mode
(Section 13) actually runs.

---

## 16. Tests

**Start of this phase:** 90 passed, 0 failed (end state of the prior
report).

**End of this phase:** **148 passed, 0 failed.** 58 new tests across 9 new
test files (`test_candle_boundaries.py`, `test_success_criteria.py`,
`test_calibration.py`, `test_scanner_backtest.py`, `test_data_quality.py`,
`test_drift.py`, `test_shadow.py`, `test_model_versioning.py`) plus
additions to `test_rankings.py` and `test_walk_forward.py`. No test was
weakened or removed; the only tests modified from the prior phase were
extended (new assertions added to existing passing tests for the
regime/versioning additions).

Every new module in this report followed the same discipline: write the
test first or alongside the implementation, run it, and treat a failure
as a bug to fix rather than a test to loosen — this caught two real,
independent bugs during this phase alone (the pandas datetime-resolution
gap-detection bug in Section 14, and the string-matching typo in
`test_scanner_backtest.py`'s own "no recommendation" assertion, fixed in
the test itself).

---

## 17. Production Safety

| Check | Status | Evidence |
|---|---|---|
| Live prediction cannot accidentally use future data | ✅ | Single shared code path (`predict_from_frame`); backtest/walk-forward slice `df.iloc[:i]` strictly; proven by `test_no_lookahead_changing_the_future_does_not_change_past_picks` for the scanner and the embargo tests for the forecaster. |
| Live prediction cannot use incomplete higher-timeframe candles | ✅ **fixed this phase** | Section 2 — `_parse_kline` + `bars_to_frame(drop_unclosed=True)`. |
| Model versions are recorded | ✅ **fixed this phase** | Was a hard-coded `"1.0.0"` string. Now a content-hash fingerprint (`_compute_model_version`) of the feature version, symbol, timeframe, horizon, training-row count, and training window bounds — two forecasters get the same version iff they are, for all practical purposes, the same trained model. `tests/test_model_versioning.py`, 6/6 passing, including "different training window ⇒ different version" and "different horizon ⇒ different version." |
| Feature versions are recorded | ✅ **fixed this phase** | `FeaturePipeline.feature_version` — a content hash of the pipeline's config + output column list, exposed on every `ForecastObject`. |
| Predictions are reproducible | ⚠️ partial | Reproducible *within* a `ModelManager`'s cached lifetime (fixed `random_state`, deterministic feature pipeline). **Not** reproducible across a process restart that retrains on a newer data window — this is inherent to "retrain on first use, cache" and is now at least *detectable* via `model_version` rather than silently assumed. |
| Failed data providers fail safely | ✅ mostly | `routes_forecast.py` catches fetch/prediction exceptions and returns a 5xx rather than a fabricated prediction; `routes_rankings.py` isolates per-symbol fetch failures into a `fetch_errors` map so one bad symbol doesn't fail the whole scan. |
| Stale data is rejected | ✅ **fixed this phase** | Section 14, `enforce_quality_gate` wired into the live path. |
| Confidence is not fabricated | ⚠️ not fabricated, but not calibrated | Confidence is a deterministic function of realised volatility, not random or invented — but Section 8 proves it is not empirically calibrated. The calibration-checking infrastructure now exists; the confidence-*generation* formula itself was not changed this phase. |
| Opportunity scanner can return "no opportunity" | ✅ | `NO_OPPORTUNITY_MESSAGE`, tested (`test_no_forced_recommendation_when_bar_not_met`). |

---

## 18. Final Report

### A. What was validated
The horizon-consistency fix (end-to-end, with a new regression test suite
protecting it), train/live data-source consistency (one bug found and
fixed: unclosed candles), the shipped walk-forward/benchmark/scanner-
backtest modules by actually running them on realistic multi-regime data,
confidence calibration (found badly miscalibrated), and a full production-
safety checklist.

### B. What failed
The forecasting model itself. At every horizon from 1H to 7D, on a
realistic regime-shifting series, it loses to naive persistence, naive
drift, a moving average, and a coin flip on directional accuracy; its
price predictions are reached at all only ~20% of the time regardless of
horizon; and even when reached, timing is wildly off in one direction at
short horizons and the other direction at long horizons. Confidence
scores are overconfident by 20-50 percentage points in every bucket above
~55% stated confidence. "Horizon success" (right price **and** right
time) occurs on roughly 1 in 40 predictions.

### C. What was fixed
Unclosed-candle usage in both the forecaster and the opportunity scanner
(and, incidentally, the frontend-facing `is_closed` flag on the REST
candles endpoint); missing model/feature version provenance (was a static
placeholder string); the opportunity scanner's total lack of market-regime
awareness; a pandas-version-dependent silent failure in gap detection;
the complete absence of a shadow-mode evaluation loop; the complete
absence of data-quality gating on live inference; the complete absence of
drift monitoring; and two missing baselines (moving-average,
random-direction) needed to fully answer Section 4.

### D. Historical performance
Section 3 (per-horizon MAE/directional accuracy) and Section 4 (vs. four
baselines, and vs. a faithfully-reproduced OLD system). All on synthetic
data — see the top-of-report caveat.

### E. Time-to-target performance
Section 5 — full mean/median/MAE/RMSE/bias/std/threshold-coverage table,
hit vs. never-hit separated, for all six horizons.

### F. Confidence calibration
Section 8 — bucketed, with Brier scores, for three representative
horizons; systematically overconfident above ~55% stated confidence.

### G. Baseline comparison
Section 4 — model loses to every baseline at every horizon; no exception.

### H. Opportunity scanner performance
Section 10 (checklist, one real gap found: market-cap filtering, plus a
market-regime gap that was fixed this phase) and Section 11 (point-in-time
backtest: 60.4% win rate, 72.9% target-hit rate, positive risk-adjusted
return of 0.31 on the small synthetic universe tested, with survivorship
bias explicitly not eliminated and disclosed on every report).

### I. Market-regime performance
Section 12 — up to 8× error dispersion between the best- and
worst-performing regimes at the same horizon, tied directly to elapsed
time/price distance from the model's one training snapshot rather than
any intrinsic property of a given regime.

Additional finding not tied to a single numbered section: two places in
the codebase use the word "horizon" with different units
(`model_manager.py`: candles of a timeframe; `fetch_market_data.py`:
calendar days). Neither is a bug — each is internally consistent and
documented — but the naming collision is a real risk for a future
contributor to introduce a new mixing bug. Recommend renaming one of them
(e.g. `horizon_days` in `fetch_market_data.py`) as low-priority cleanup.

### J. Remaining weaknesses (stated honestly)
1. **No live-market validation anywhere in this report or the prior one.**
   This is the single largest gap. Every number here is synthetic-data
   evidence that the *mechanisms* work correctly; none of it is evidence
   about real BTC/ETH/SOL behaviour.
2. Confidence-calibration and drift monitoring both need a real,
   accumulated evaluation history (via shadow mode, Section 13) to say
   anything about a live deployment — right now they can only be
   demonstrated on synthetic ledgers built for this report.
3. The forecasting model's core weakness (Sections 3, 6, 7, 12) is
   architectural — a "train once on a static window" gradient-boosted-tree
   regressor on absolute price cannot track a market that moves outside
   its training range, and no amount of hyperparameter tuning fixes an
   extrapolation limitation. Section 7's recommendation (lead with
   time-to-target probabilities, treat point price as secondary) is the
   evidence-based way forward, not a hyperparameter change.
4. Market-cap filtering and real bid/ask spread data remain unavailable
   to the opportunity scanner (Section 10).
5. Classic survivorship bias in the scanner backtest is not eliminated,
   only disclosed (Section 11).
6. Predictions are only reproducible within a server process's cached
   lifetime, not across restarts that retrain on newer data (Section 17)
   — now at least detectable via `model_version`, not fixed.

### K. Production readiness assessment

> **NOT READY** for anything involving real capital (production, paper
> trading with capital-allocation decisions riding on it, or unattended
> automated action of any kind).
>
> **READY FOR SHADOW MODE ONLY** — and only once network access exists to
> actually run it. The infrastructure for shadow mode, data-quality
> gating, calibration checking, and drift monitoring is now real and
> tested; what has never been checked is whether any of it holds on real
> market data. The forecasting model's own out-of-sample numbers in this
> report (Sections 3-9) are poor enough, even accounting for the synthetic
> data caveat, that the honest recommendation is: run shadow mode against
> real markets, accumulate real evaluated predictions, and revisit this
> verdict against *that* evidence — not against synthetic data, and not
> against the assumption that fixing the timing-label bug alone made the
> underlying forecasts trustworthy. It did not; it made them honestly
> labelled, which is necessary but nowhere close to sufficient.

This verdict is unchanged in spirit from what the evidence in this report
would support regardless of who is reading it: tests passing (148/148) is
proof of correctness of the *mechanisms*, not proof that the *forecasts*
are good. Those are different claims, and this report keeps them
separate on purpose.
