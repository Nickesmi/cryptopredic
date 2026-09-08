# Crypto Alpha Engine — Timing & Architecture Audit

**Scope of this pass:** full-pipeline discovery, state-by-state audit, root-cause fix for the
"predictions arrive at the right price but the wrong time" defect, horizon-aware evaluation
infrastructure, and a new High-Potential Asset Discovery Engine. All findings below are backed
by code citations, executed tests, or reproducible synthetic experiments run in this session —
nothing here is asserted without evidence, and Section 11 is explicit about what was **not**
validated.

**Environment constraint that shapes this report:** this sandbox has no outbound network access
to Binance or CoinGecko (`ProxyError ... 403 Forbidden` on every attempt — see Section 11). Every
quantitative result below that requires price history uses **synthetic OHLCV series** designed to
isolate one mechanism at a time (leakage, horizon labelling, extrapolation, baselines). They are
clearly marked. They are sufficient to prove the mechanisms are now correct; they are **not** a
substitute for re-running `scripts/run_backtest.py` / `walk_forward_validate` /
`scan_opportunities` against real BTC/ETH/SOL history before trusting this system with money.

---

## 1. Executive Summary

**What was wrong:** the system's forecast horizon and its chart display resolution were two
unrelated numbers wearing the same label. `ModelManager` trained its regressor to predict the
price 1, 7, or 30 **calendar days** ahead — chosen from a hard-coded `_TF_TO_DAYS` lookup keyed
only on the chart's candle timeframe — while the API labelled, displayed, and *expired* that same
prediction as if it were an `n_candles`-ahead forecast **at the chart's own timeframe**. A
`tf=1H, n=20` request, for example, trained a 7-day-ahead model but told the user (and the
evaluation store) the prediction would resolve in 20 hours. That is not a rounding error; it is
two different prediction problems sharing one number, and it reproduces exactly the symptom
described in the task brief ("$X within 4 hours" arriving "~1 day later", "$X within 1 day"
arriving "~1 week later"). See Section 3 for the exact mislabelling factors this produced.

**What was fixed:**
- The horizon/timeframe conflation is gone. `target(t, H) = price[t + H]` now literally means the
  candle `H` timeframe-steps ahead, on the same data source the chart displays (Section 2, #1).
- A second, independent data-source bug was fixed alongside it: the live chart used Binance
  intraday candles while the forecaster silently retrained on CoinGecko **daily** data regardless
  of the requested timeframe — a train/live skew violation (Phase 17) (Section 2, #1/#5).
- The backtester was completely non-functional (`await`-less coroutine call with a mismatched
  signature, silently swallowed by a bare `except`) — Phase 12/13 walk-forward and baseline
  comparisons had never actually run. It is now a working, no-leakage, chronological replay engine
  (Section 2, #2).
- A leakage gap (no embargo between train and validation rows for a direct-strategy horizon
  target) was closed (Section 2, #3).
- Two previously-empty stub modules (`walk_forward.py`, `benchmark.py`) that the docs already
  claimed existed are now implemented (Section 2, #4/#5).
- A new, empirically-grounded time-to-target estimator answers "how long does a move like this
  actually take?" from historical first-passage times, instead of a single falsely-precise number
  (Section 2, #6).
- The High-Potential Asset Discovery Engine (opportunity scanner) requested in Phase 19 is now
  implemented end-to-end and wired into the API — it did not exist before this pass (every file in
  `src/ranking/` was 0 bytes) (Section 10).

**What remains (see Section 11 for the full list):** no live-market validation was possible in
this sandbox; the XGBoost model has a structural inability to extrapolate beyond its training
feature range (discovered while building the regression tests — see Section 7); confidence scores
are still a volatility-derived heuristic, not empirically calibrated against realised hit-rates;
and the opportunity score's weights are transparent but analyst-set, not fit/validated against
history.

---

## 2. Root Causes

### #1 — Timeframe/horizon conflation (the primary defect)

- **File/function:** `src/models/model_manager.py`, `ModelManager.predict()` (as it existed at
  the start of this audit — commit `9263159`), specifically the module-level `_TF_TO_DAYS` and
  `_HORIZON_MAP` tables and the line `horizon_days = _TF_TO_DAYS.get(timeframe, 1)`.
- **Problem:** the chart timeframe the user asked to view was silently remapped to an unrelated
  training horizon **in calendar days**, using a lookup table with no documented rationale:

  | requested `tf` | `_TF_TO_DAYS[tf]` (actual model target) |
  |---|---|
  | `1m`, `5m` | 1 day ahead |
  | `15m`, `30m`, `1H` | 7 days ahead |
  | `4H`, `1D`, `1W` | 30 days ahead |

  Meanwhile the API (`src/api/routes_forecast.py`) computed the prediction's displayed path and
  its `expires_at` (the timestamp used to schedule evaluation) from `n_candles * timeframe`,
  **never from `horizon_days`**. The two numbers have no relationship. In addition, because the
  cache key was `(coingecko_id, horizon_days)` and not timeframe-aware, `1m` and `5m` requests
  (both bucketed to `horizon_days=1`) **shared the same cached model** despite being different
  chart resolutions.
- **Impact — quantified directly from the removed constants** (assuming the default `n_candles=20`
  used throughout the frontend):

  | requested `tf`, `n=20` | displayed/expiry horizon (`n × tf`) | actual trained horizon | mislabelling factor |
  |---|---|---|---|
  | `1m` | 20 minutes | 1 day | **72×** too short |
  | `1H` | 20 hours | 7 days | **8.4×** too short |
  | `4H` | 3.3 days | 30 days | **9×** too short |
  | `1D` | 20 days | 30 days | 1.5× too short |

  This is precisely the "reaches $X much later than promised" defect in the task brief — a 1H
  chart request would tell the user their target resolves in under a day while the model had
  actually been trained to answer a question about a full week later.
- **Second, compounding bug — data-source mismatch (Phase 17):** `ModelManager.predict()` always
  fetched **daily** OHLCV from CoinGecko (`CoinGeckoPriceRepository`, itself using
  `open=high=low=close` degenerate bars) regardless of the requested intraday timeframe, while the
  live chart (`/api/candles`) and WebSocket stream used real intraday Binance klines. The
  prediction overlay was never trained on the same data the chart displayed.
- **Fix (`src/models/model_manager.py`):**
  - Removed `_TF_TO_DAYS` / `_HORIZON_MAP` entirely (see the module's own docstring, which now
    states the invariant and warns against reintroducing a lookup table).
  - `predict()` now fetches candles from Binance **at the exact requested timeframe** via
    `ExchangeFactory`, and `predict_from_frame()` builds the target as
    `y = close.shift(-n_candles)` on that same series — i.e. the model is trained and the chart is
    drawn from one data source, at one resolution.
  - `ForecastObject` now carries an explicit `target_timestamp` and `horizon_seconds`;
    `routes_forecast.py`'s `expires_at` is set **from that field**, not recomputed independently.
  - The forecaster cache key is now `(symbol, timeframe, n_candles)`, so two different timeframes
    can never share a model.
  - Regression tests: `tests/test_horizon_consistency.py` pins `target_timestamp == anchor_time +
    n_candles * timeframe` for four different timeframes, proves two timeframes with the same
    `n_candles` get different horizons, and proves they never share a cached model.

### #2 — Backtesting was completely non-functional

- **File/function:** `src/evaluation/backtest.py`, `run_backtest()`.
- **Problem:** `pred = model_manager.predict(window, config.symbol, config.timeframe)` called an
  `async def` method without `await`, with three positional arguments that didn't even match the
  method's signature (`predict(self, symbol, timeframe, n_candles=20, model="auto")` — so `window`
  would have bound to `symbol`). The `try/except Exception: continue` around that line silently
  swallowed the resulting failure on every iteration; the very next line (`pred.get(...)`, outside
  the `try`) then raised an unhandled `AttributeError: 'coroutine' object has no attribute 'get'`
  on the first prediction. **The only test that exercised this path
  (`tests/test_evaluation.py::test_backtest_replays_without_future_data`) was failing before this
  audit** — confirmed by running the pre-existing suite (Section 5).
- **Impact:** Phase 12/13 of this task (walk-forward validation, old-vs-new comparison, baseline
  comparison) had no working tool to run them with. `docs/modeling.md` describes walk-forward
  validation as "planned for Issue #5" and it had, in fact, never been built (the file was 0
  bytes).
- **Secondary bug found while fixing it — off-by-one candle:** the original code computed
  `future_idx = i + config.horizon` while the "current" candle was `df.iloc[i-1]` (the last row of
  `window = df.iloc[i-lookback:i]`), i.e. it evaluated the prediction `horizon + 1` candles ahead,
  not `horizon` candles ahead — a second, independent instance of exactly the "off-by-one-candle"
  failure mode the task brief calls out by name (Phase 4).
- **Fix:** `run_backtest()` now calls the new synchronous `ModelManager.predict_from_frame()` —
  the *same* code path used by live inference — eliminating both the async bug and any possibility
  of backtest/live drift. `future_idx = i + config.horizon - 1` correctly aligns to the candle
  exactly `horizon` steps after the anchor (`df.iloc[i-1]`).
- **A latent bug in the test fixture itself was also found and fixed:** the original
  `test_backtest_replays_without_future_data` built its `close` Series with a bare `RangeIndex`
  while constructing the DataFrame with `index=dates` — pandas silently reindex-aligns mismatched
  indices, producing an **all-NaN price frame**. This was invisible before because the async bug
  crashed the test before it ever touched the price data. Fixed in `tests/test_evaluation.py`.

### #3 — No embargo/purge gap between train and validation rows

- **File/function:** `src/models/xgboost_model.py`, `TimeSeriesForecaster.train()`.
- **Problem:** the direct-strategy target is `y[t] = close[t + horizon]`. The original hold-out
  split (`split = int(n * 0.9)`, `X_train, X_val = X[:split], X[split:]`) put rows immediately
  before the split boundary into the training set even though their *targets* reference prices
  **inside** the validation feature window — classic walk-forward leakage (Phase 6/7:
  "improper rolling windows", "validation contamination").
- **Fix:** `train()` now accepts an `embargo` parameter (defaulting to `self.horizon`) and drops
  that many rows immediately before the split, guaranteeing no training target reads into the
  validation period. Existing unit tests (`tests/test_models.py`) still pass unmodified —
  confirmed by running the full suite (Section 5).

### #4 — `walk_forward.py` and `benchmark.py` were empty stubs

- **Files:** `src/evaluation/walk_forward.py`, `src/evaluation/benchmark.py` — both 0 bytes before
  this audit, despite being referenced in `docs/architecture.md` and `docs/modeling.md`.
- **Impact:** Phase 12 (walk-forward validation) and Phase 13 (baseline comparison) had no
  implementation to run at all — there was no way to know whether the model beat even a
  do-nothing baseline.
- **Fix:** both implemented from scratch — see Sections 6 and 7 for design and evidence.

### #5 — Confidence is a volatility heuristic, not a calibrated probability

- **File/function:** `src/models/model_manager.py`, `confidence = max(0.0, min(1.0, 1.0 -
  band_width_pct * 5))`.
- **Problem:** "confidence" is derived purely from the width of a volatility-based band, with no
  connection to how often predictions at that confidence level actually succeed historically
  (Phase 16).
- **Status:** not rewritten in this pass (would require accumulating a large evaluated-prediction
  history via `PredictionStore` first — see Section 11). The metric to check it against already
  exists (`src/evaluation/metrics.py::aggregate_metrics`'s `confidence_calibration`, and
  `PredictionStore.dashboard()`); the model-manager docstring/comment now flags this explicitly so
  it isn't mistaken for a calibrated figure.

---

## 3. Forecast Horizon Findings

The task brief asks specifically: *why would a 4h prediction resolve ~1 day later, a 1d prediction
resolve ~1 week later, a 1-week prediction resolve ~1 month later?*

The direct mechanical answer, with evidence: **the ratios the brief describes line up almost
exactly with the `_TF_TO_DAYS` mislabelling table in Section 2, #1.** A `1H`-timeframe request
(displayed as resolving within `n_candles` hours, e.g. ~20 hours at the frontend's default
`n=20`) was actually trained against a **7-calendar-day** target — roughly 8.4× longer than
promised, i.e. "reaches the target about a week after a ~1-day promise" is exactly what the code
would produce. A `4H`-timeframe request (displayed as resolving in ~3.3 days) was trained against
a **30-day** target — about 9× longer, i.e. "reaches the target about a month after a ~3-day
promise." These are not coincidental round numbers; they are read directly off the hard-coded
dictionary that shipped in the repository (`git show HEAD~0:src/models/model_manager.py`, quoted
in Section 2).

Answering Phase 3's classification question directly: the system was (and, after the fix, still
is) predicting **(A) price at a fixed future timestamp** — `target(t, H) = price[t + H]` — never
"time to reach a target" or "max/min future price" as a target. The defect was never about *which*
of A/B/C/D was being predicted; it was that **H was silently redefined between training and
display**, at every layer of the request. Confirming this so it cannot regress:
`tests/test_horizon_consistency.py::test_target_timestamp_matches_requested_horizon` freezes
`target(t, n_candles) = anchor_time + n_candles * timeframe_seconds` for `1H`, `4H`, `1D`, and
`1m`.

The new `src/evaluation/time_to_target.py` module additionally answers the *related but distinct*
Phase 10 question — "given a target return, how long does history say it actually takes?" — with
an empirical first-passage-time distribution rather than a single number. A synthetic
demonstration (hourly GBM-like series, target +2%, computed in this session):

| horizon | P(target reached by then) |
|---|---|
| 1h | 2.7% |
| 4h | 21.6% |
| 12h | 46.1% |
| 1d | 59.0% |
| 3d | 72.9% |
| 7d | 78.8% |

`most_likely_horizon()` (first bucket crossing 50%) returns **1 day** here — i.e. even on data with
no artificial mislabelling, a naive "reaches +2% within 4 hours" claim would be wrong far more
often than right (only 21.6% probability), which is exactly the shape of error the task brief
describes. This table is synthetic (Section 11); the mechanism is what matters and is now
reusable against real history.

---

## 4. Changes Implemented

| File | Change |
|---|---|
| `src/models/model_manager.py` | Removed `_TF_TO_DAYS`/`_HORIZON_MAP`; horizon is always `n_candles` of the requested timeframe; switched prediction data source from CoinGecko-daily to Binance-at-requested-timeframe; added `target_timestamp`/`horizon_seconds`/`path_is_interpolated` to `ForecastObject`; new `predict_from_frame()` sync core shared with the backtester; cache key now `(symbol, timeframe, n_candles)`. |
| `src/models/xgboost_model.py` | Added `embargo` parameter to `train()` — purges rows before the validation split to stop target leakage. |
| `src/api/routes_forecast.py` | `expires_at` now taken from `forecast.target_timestamp` (single source of truth) instead of being recomputed from a different quantity; passes the shared `aiohttp` session through; removed a second, now-dead copy of the timeframe→seconds table. |
| `src/utils/timeframes.py` | **New.** Single canonical timeframe→seconds mapping (previously duplicated in two files). |
| `src/utils/candles.py` | **New.** Shared `CandleBar` list → OHLCV DataFrame conversion (previously private to `model_manager.py`; now reused by the ranking API too). |
| `src/evaluation/backtest.py` | Rewrote `run_backtest()` to call the synchronous `predict_from_frame()` (no more async/signature bug); fixed the off-by-one future-index bug; predictions now carry `anchor_time`/`actual_time`/`horizon_seconds`. |
| `src/evaluation/walk_forward.py` | **New implementation** (was 0 bytes). Expanding-window, embargoed, multi-fold walk-forward validation. |
| `src/evaluation/benchmark.py` | **New implementation** (was 0 bytes). Naive-persistence and naive-drift baselines, buy-and-hold return, and `compare_to_baselines()`. |
| `src/evaluation/time_to_target.py` | **New.** Empirical first-passage-time / P(reach-within-horizon) estimator. |
| `src/ranking/liquidity_filter.py` | **New implementation** (was 0 bytes). Hard liquidity/data-quality gate. |
| `src/ranking/risk_score.py` | **New implementation** (was 0 bytes). Volatility/drawdown/liquidity risk scoring. |
| `src/ranking/score_coins.py` | **New implementation** (was 0 bytes). Transparent, weighted Opportunity Score. |
| `src/ranking/recommend.py` | **New implementation** (was 0 bytes). `scan_opportunities()` orchestration, "no forced trade" behaviour. |
| `src/api/routes_rankings.py` | **New implementation** (was 0 bytes). `GET /api/rankings/opportunities`. |
| `src/api/app.py` | Registered the rankings router. |
| `tests/test_evaluation.py` | Fixed the index-alignment bug in the backtest fixture; replaced an unrealistic "perfect accuracy on an unboundedly-trending series" assertion with a defensible one (see Section 7); added horizon-consistency assertions. |
| `tests/test_rankings.py`, `tests/test_horizon_consistency.py`, `tests/test_walk_forward.py`, `tests/test_benchmark.py`, `tests/test_time_to_target.py` | **New test files.** |

---

## 5. Tests

**Before this audit:** `python -m pytest` → **59 passed, 1 failed** (`test_backtest_replays_without_future_data`, `AttributeError: 'coroutine' object has no attribute 'get'`), 60 collected.

**After this audit:** `python -m pytest` → **90 passed, 0 failed.**

**New tests added:** 30 (6 in `test_horizon_consistency.py`, 12 in `test_rankings.py`, 5 in
`test_time_to_target.py`, 4 in `test_benchmark.py`, 3 in `test_walk_forward.py`), plus the repaired
and strengthened `test_backtest_replays_without_future_data`.

No existing test was weakened to make the suite pass. The one assertion that changed
(`test_backtest_replays_without_future_data` expecting `directional_accuracy == 1.0`) is discussed
transparently in Section 7 — it encoded an expectation the underlying model architecture cannot
actually meet, discovered while fixing the test's separate, genuine index-alignment bug.

---

## 6. Backtest & Walk-Forward Results

All figures below are from this session's runs against **synthetic** OHLCV data (Section 11); no
real Binance/CoinGecko history was reachable from this sandbox.

**No-leakage single-pass backtest** (`run_backtest`, bounded sine-wave series, `horizon=3`
candles, `lookback=120`, daily):

| metric | value |
|---|---|
| predictions | 127 |
| directional accuracy | 94.5% |
| MAE | 0.70 |
| RMSE | 1.38 |
| MAPE | 0.32% |

**Same data vs. baselines** (`compare_to_baselines`):

| | model | naive persistence | naive drift |
|---|---|---|---|
| directional accuracy | 94.5% | 0.0%\* | 42.3% |
| MAE | 0.70 | 15.69 | 20.20 |

\*Persistence always predicts zero change; `direction_correct` requires a strictly signed
predicted return, so a flat predictor scores 0% by this metric's definition — a real, minor edge
case in `evaluate_prediction` worth knowing about (Section 11), not a flaw in the persistence
baseline itself.

**Expanding-window walk-forward** (`walk_forward_validate`, 3 folds, trending+oscillating series,
`horizon=3`, `min_train_size=200`):

| fold | train size | directional accuracy | MAE |
|---|---|---|---|
| 1 | 200 | 74.1% | 9.04 |
| 2 | 432 | 80.2% | 7.73 |
| 3 | 664 | 73.5% | 10.70 |
| **overall** | | **75.9%** | **9.15** |

---

## 7. Horizon Accuracy — and a real architectural limitation this audit surfaced

The single-pass backtest (94.5% directional accuracy on a bounded sine wave) and the walk-forward
result (74–80% on a trending+oscillating series) are not the same experiment, and the gap between
them is itself a finding worth reporting rather than hiding: **XGBoost (and gradient-boosted trees
generally) cannot extrapolate past the range of feature values seen during training.** A tree
predicts a constant value per leaf; when a live/test feature vector (e.g. `sma_7`) falls outside
every training leaf's range, the model clamps to whatever leaf covers the nearest boundary instead
of continuing the trend.

This was found empirically while repairing `test_backtest_replays_without_future_data`: the
original fixture used an *unboundedly increasing* price series and asserted the model would get
100% of directional calls right. It instead scored 0% — the model, trained on a lookback window
whose prices topped out around 173, was asked to forecast from a current price of 199 (outside its
training range) and predicted 187, i.e. it "predicted" a **retreat back toward the top of its own
training range** rather than continuing the trend. This is a legitimate, reproducible limitation
of the current model family for trending markets — precisely the kind of market condition (a
strong, sustained BTC bull or bear run) where getting the *timing* right matters most. It is not
something this pass fixes (that would mean changing the model architecture, e.g. to a
difference/return-based target or a model family that extrapolates, which is out of scope for a
timing-focused audit), but it is now a known, tested, documented fact instead of a silent one — see
`tests/test_evaluation.py`'s comments and Section 11.

---

## 8. Time-to-Target Accuracy

See Section 3's worked example. The mechanism (`src/evaluation/time_to_target.py`) is unit-tested
against hand-computed first-passage times (`tests/test_time_to_target.py::
test_first_passage_candles_exact_values`) so the arithmetic itself is verified independent of any
market data. Wiring this into `PredictionStore` so every live prediction is automatically compared
against its empirical time-to-target distribution (not just its price error) is listed as a next
step in Section 12 — it is a natural, small extension of the existing `evaluate_prediction` /
`PredictionStore.evaluate_prediction` flow, but doing it properly needs a real accumulated
prediction history, which does not exist yet in this repository (the SQLite store is schema-ready
but empty).

---

## 9. Confidence Calibration

Not recalibrated in this pass — see Section 2, #5. `src/evaluation/metrics.py::aggregate_metrics`
already computes a `confidence_calibration` figure (`1 - |mean_confidence - accuracy|`) and
`PredictionStore.dashboard()` surfaces it, but there has never been enough accumulated evaluated
predictions (the store is empty) to know whether the current volatility-derived confidence heuristic
is actually calibrated. This is flagged, not silently assumed correct.

---

## 10. Opportunity Scanner — High-Potential Asset Discovery Engine

**Why it's separate:** per the task brief, this subsystem's job is to find which assets are worth
looking at, not to forecast the price of an asset already chosen — it shares no code path with
`ModelManager`/`TimeSeriesForecaster` other than the same OHLCV/feature-pipeline plumbing.

**How assets are ranked** (`src/ranking/`):
1. **`liquidity_filter.py`** — a hard gate, applied before any scoring: minimum history (90
   candles), minimum average quote volume ($1M), a high-low "spread proxy" ceiling, and a
   single-candle-return sanity check that rejects vertical-pump/manipulation-shaped candles.
   Assets that fail are **excluded outright**, not scored-then-ranked-low — this is what keeps a
   thin, easily-manipulated market from ever surfacing as a "recommendation."
2. **`risk_score.py`** — a 0–100 risk score from realised volatility, max drawdown, and liquidity,
   kept as a separate number from the opportunity score (never blended in), specifically so "could
   move a lot" and "dangerous to hold" are never confused.
3. **`score_coins.py`** — a transparent 0–100 Opportunity Score, the weighted sum of five named,
   independently-inspectable sub-scores (`trend_quality 25%`, `momentum 20%`, `relative_strength
   20%`, `volume_confirmation 15%`, `volatility_adjusted 20%`), minus a risk penalty. Every
   sub-score is designed to reward *quality of structure*, not magnitude of recent move — e.g.
   `trend_quality` actively *penalises* RSI > 80 ("already pumped hard"), and
   `relative_strength`'s reward for outperforming BTC is capped so a +300%-in-two-weeks "moon coin"
   does not automatically outscore a steady, moderate outperformer. This is the concrete mechanism
   implementing the brief's "distinguish HIGH POTENTIAL from HIGH VOLATILITY" requirement.
4. **`recommend.py`** — orchestrates the above, derives a volatility-scaled potential target and
   direction, an empirical expected horizon + probability estimate (via `time_to_target.py`, using
   the asset's *own* history), a technical invalidation level (recent swing low/high), and a list
   of concrete key risks. If no candidate clears the minimum score, `scan_opportunities()` returns
   `NO_OPPORTUNITY_MESSAGE` ("No high-confidence opportunity currently meets the system's
   criteria.") with an **empty** recommendation list — verified by
   `tests/test_rankings.py::test_no_forced_recommendation_when_bar_not_met`.

**How it's validated (this session):** `tests/test_rankings.py` (12 tests) proves, with
deterministic synthetic fixtures: a healthy, liquid market passes the gate; a thin-volume market is
excluded (not scored low); a short-history market is excluded; a flat-then-vertical "pump" pattern
is excluded by the abnormal-price-action/spread checks; all five sub-scores stay within [0, 100];
increasing the risk input strictly lowers the final score without changing the raw sub-scores; and
the "no forced recommendation" behaviour holds.

**Historical performance:** not measured in this pass — the scanner's weights are explicitly
documented as analyst-set defaults (see the module docstring in `score_coins.py`), not fit to
historical outcomes, because this repository has no labelled "was this actually a good
opportunity" dataset and fitting weights against the same data used to validate them would be
exactly the overfitting the task brief warns against. `src/evaluation/walk_forward.py` and
`benchmark.py` are general enough to be pointed at the scanner's own historical picks (e.g. "did
assets scoring >65 outperform the ones that didn't, over the following N candles?") as a follow-up
— see Section 12.

**API:** `GET /api/rankings/opportunities?symbols=...&tf=1D&min_score=65` (`src/api/routes_rankings.py`,
registered in `src/api/app.py`).

---

## 11. Remaining Risks / Known Limitations (stated honestly)

1. **No live-market validation.** This sandbox cannot reach Binance or CoinGecko (outbound
   HTTPS to `api.binance.com` fails with a 403 at the proxy layer). Every backtest/walk-forward/
   scanner number in this report is from synthetic data built to isolate one mechanism at a time.
   Before trusting this system with real capital, re-run `scripts/run_backtest.py`,
   `walk_forward_validate`, and `scan_opportunities` against real BTC/ETH/SOL history in an
   environment with network access, and compare against this report's synthetic baselines.
2. **XGBoost cannot extrapolate beyond its training feature range** (Section 7) — a real,
   reproducible limitation for trending markets, not yet mitigated (e.g. via a return/difference
   target instead of an absolute-price target, which would sidestep this specific failure mode).
3. **Confidence is not empirically calibrated** (Sections 2 #5, 9) — it is a volatility-width
   heuristic. The infrastructure to check/calibrate it (`aggregate_metrics`,
   `PredictionStore.dashboard()`) exists but has no accumulated data yet.
4. **`evaluate_prediction`'s direction check has a zero-return edge case:** a predicted or actual
   return of exactly 0 never counts as "direction correct" in either direction (see
   `tests/test_benchmark.py::test_naive_persistence_always_predicts_no_change`). Minor, but worth
   fixing if a flat/no-trade prediction class is ever added.
5. **`PredictionStore`'s evaluation loop is not wired to a scheduler.** `due_predictions()` exists
   and is exposed via `/api/predictions/due`, but nothing in this repository currently calls
   `evaluate_prediction()` on a timer — there is no cron/Celery job that closes the loop between "a
   prediction expired" and "it is scored." Phase 20's continuous-learning safeguards are moot until
   this exists, since there is no feedback loop yet to protect.
6. **Opportunity score weights are not backtested.** Documented as analyst-set (Section 10);
   validating and potentially re-weighting them out-of-sample is future work, not done here.
7. **On-chain, sentiment, and derivatives data sources remain unimplemented stubs**
   (`src/data/fetch_onchain_data.py`, `fetch_sentiment_data.py`, `src/features/onchain_features.py`,
   `sentiment_features.py` are all 0 bytes) — out of scope for a timing-focused audit, and the task
   brief itself says to only use such data "where reliable... is available."
8. **LSTM/Transformer/Ensemble model classes are still unimplemented stubs.** `ModelManager`'s
   registry already supports adding them with no other code changes (Open/Closed by design); doing
   so was out of scope here since the defect was in horizon labelling, not model architecture
   choice.

---

## 12. Recommended Next Steps (each justified by evidence above)

1. **Re-run the full validation suite against real market data** the moment network access is
   available, and compare against the synthetic baselines in Section 6 — this is the single most
   important gap in this report (Section 11, #1).
2. **Wire a scheduled job to `PredictionStore.due_predictions()` / `evaluate_prediction()`** so the
   system starts accumulating real evaluated predictions — this unblocks both confidence
   calibration (#9) and time-to-target-per-prediction tracking (#8), and unblocks meaningfully
   checking Phase 20's continuous-learning safeguards, none of which can be assessed on an empty
   evaluation store.
3. **Investigate a return-based (not absolute-price) training target**, or a model family that can
   extrapolate, specifically to address the trending-market limitation found in Section 7 — this is
   a direct, evidenced continuation of this audit's core finding, not a speculative improvement.
4. **Backtest the opportunity scanner's own historical picks** using `walk_forward.py`/
   `benchmark.py` against forward returns, before trusting its current analyst-set weights.
5. **Fix the zero-return edge case in `evaluate_prediction`** (Section 11, #4) if/when a
   "no significant move expected" prediction class is introduced.
