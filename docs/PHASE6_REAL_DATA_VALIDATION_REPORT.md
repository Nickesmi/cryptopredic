# Phase 6 — Real-Data Validation: Was Phase 5's FAILED Verdict the Data or the Signal?

Phase 3 concluded the point-price forecaster is **NOT RELIABLE**. Phase 4 concluded the opportunity
scanner is **WEAK**. Phase 5 concluded the Alpha Research Engine **FAILED** to find a confirmable
signal on a disclosed-synthetic "realistic" universe, while its mandatory controls proved the
*pipeline itself* works (it detects a planted signal cleanly and correctly rejects pure noise after
multiple-testing correction). Phase 6's question is narrower and more important than "try again":
**was Phase 5's failure caused by the synthetic data's specific statistical properties, or does it
reflect a genuine absence of signal that would also show up on real markets?** The only way to
answer that is to run the same, unmodified framework on genuinely historical data.

**Result, in one line (Section 21/final answer in full below):** this sandbox has no outbound
network access to any real market-data source and no local historical OHLCV file exists anywhere in
the repository — confirmed by a fresh, direct test at the start of this phase (Section 1), not
assumed from Phase 2-3's prior findings. Per the brief's explicit Section 20 stop condition, **the
real-data empirical experiment could not be run**. This phase instead delivers the infrastructure
Section 20 asks for in that case: real-data ingestion (with pagination past Binance's 1,000-candle
cap), a reproducible acquisition specification, data provenance, historical universe reconstruction,
and full integration with Phase 5's exact statistical framework — all built, tested, and proven
correct via mocked/fixture data, ready to run the moment real data becomes available. **Final
classification: FAILED** (see Section 18 for the precise, load-bearing caveat on what this FAILED
does and does not mean).

---

## 1. Data Availability Audit

A fresh, direct test was run at the start of this phase (not assumed from prior phases, since the
sandbox environment can change between sessions):

```
$ curl -m 10 https://api.binance.com/api/v3/klines?...     -> CONNECT tunnel failed, response 403
$ curl -m 10 https://api.binance.us/api/v3/klines?...      -> CONNECT tunnel failed, response 403
$ curl -m 10 https://api.coingecko.com/api/v3/ping         -> CONNECT tunnel failed, response 403
$ curl -m 10 https://api.kraken.com/0/public/OHLC?...      -> CONNECT tunnel failed, response 403
$ curl -m 10 https://example.com                            -> CONNECT tunnel failed, response 403
```

The sandbox's own egress-proxy status endpoint (`/__agentproxy/status`) confirms this is a **policy
denial**, not a transient network fault: `"kind":"connect_rejected","detail":"gateway answered 403 to
CONNECT (policy denial or upstream failure)"` for every host tried, including a plain, unrelated
`example.com` control (ruling out "Binance specifically is blocked" and confirming outbound HTTPS is
blocked wholesale by sandbox/org policy). No API keys, exchange credentials, or alternate
network paths were found in the environment (checked `env`, `/root/.claude/settings.json`,
`src/config/settings.py`).

Local filesystem inventory (every dataset type named in the brief):

| Dataset | Present? | Detail |
|---|---|---|
| OHLCV (historical) | **No** | No `.csv`/`.parquet`/`.feather`/`.h5` file with real OHLCV anywhere in the repo. `data/raw/` is empty prior to this phase. |
| Exchange / symbol / timeframe / timestamp / volume metadata | **No** | Only *code* that would fetch these fields (see Section 2) — no cached data. |
| Market cap | **No** | `CoinGeckoPriceRepository.fetch()` (pre-existing) called an endpoint that *includes* `market_caps` but discarded it — fixed in Section 2, still requires network to populate. |
| Circulating supply | **No** | Not fetched by any existing code path. |
| Listing / delisting dates | **No** | No registry file or fetcher exists. |
| BTC/ETH history | **No** | Same as OHLCV above — code exists to fetch it, no cached data exists. |
| Stablecoin information | **No** | Not referenced anywhere in the codebase. |
| Exchange availability (which exchanges list which symbols) | **No** | Only Binance is registered in `ExchangeFactory`; no cross-exchange availability data. |
| Order-book data | **No** | No depth/order-book fetcher exists in `src/data/`. |
| Funding rates | **No** | Not referenced anywhere in the codebase. |
| Open interest | **No** | Not referenced anywhere in the codebase. |
| On-chain data | **No** | `src/data/fetch_onchain_data.py` and `src/features/onchain_features.py` are both 0-byte stub files. |
| Derivatives data | **No** | Not referenced anywhere in the codebase. |

Two application databases exist (`crypto_trading.db`, `data/prediction_evaluation.sqlite3`) — both
were opened and every table queried: **all tables in both databases have 0 rows.** They are empty
application-state schemas (model registry, prediction ledger, etc.), not historical market data.

**Per-dataset record** (source / date range / timeframe / coverage / missingness / timestamp
convention / timezone / closed-candle status / survivorship / usable for historical research):

| Field | Value |
|---|---|
| Source | None available locally or over the network in this environment |
| Date range | N/A |
| Timeframe | N/A |
| Asset coverage | N/A |
| Missingness | N/A (nothing to measure) |
| Timestamp convention | N/A (defined prospectively in Section 5 for when data does arrive) |
| Timezone | N/A |
| Closed-candle status | N/A |
| Survivorship bias | N/A — see Section 3's mechanical-vs-registry distinction, which applies regardless of whether data exists |
| Usable for historical research? | **No — none exists to use.** |

---

## 2. Real Data Requirement — Infrastructure Built Instead

Per Section 2's explicit instruction ("If it is not available: DO NOT fabricate it... Instead:
document the exact missing datasets, implement the real-data ingestion/research pipeline, add
validation tests, create a reproducible acquisition specification, clearly state that the real-data
experiment cannot yet be completed"), this phase built:

1. **`BinanceAdapter.get_historical_klines()`** (`src/data/exchange/binance.py`) — the pre-existing
   `get_candles()` only returns the most recent ≤1,000 candles; nothing in the codebase could
   retrieve a multi-year history. This new method paginates via `startTime`/`endTime`, advancing
   the cursor past each page's last candle, de-duplicating boundary candles, and stopping on a
   short page (history exhausted) or reaching `end_ms`. Proven correct against a fake `aiohttp`
   session (7 tests, `tests/test_binance_historical_klines.py`) — no real network needed to verify
   the pagination logic itself.
2. **`CoinGeckoPriceRepository.fetch_with_market_cap()`** (`src/data/fetch_prices.py`) — the
   existing `fetch()` method already calls an endpoint (`/coins/{id}/market_chart`) whose response
   includes a `market_caps` series, silently discarded. This adds a second method that extracts it,
   directly closing the "market cap: not available" gap Phase 5's Section 6 baseline #3 hit. 4
   tests (`tests/test_coingecko_market_cap.py`), mocked HTTP, no real network.
3. **`src/research/data_provenance.py`** — `DatasetProvenance` (source, URL template, params, date
   range, retrieved-at, row count, SHA-256 content hash, git commit, license note),
   `AcquisitionSpec` (a precise, versioned, JSON-serializable fetch specification), and
   `save_with_provenance`/`load_provenance` sidecar-file helpers. 8 tests
   (`tests/test_data_provenance.py`).
4. **`src/research/historical_universe.py`** — point-in-time universe reconstruction from
   arbitrary-length, ragged real per-asset histories (Section 3, detailed below). 9 tests
   (`tests/test_historical_universe.py`).
5. **An explicit unclosed-candle gate** added to `src/data/quality.py::validate_ohlcv` (Section 4,
   detailed below). 5 tests (`tests/test_data_quality_unclosed_candle.py`).
6. **`src/research/experiment_runner.py`** — Phase 5's exploratory-screen / BH-correction /
   confirmatory-permutation-test / frozen-test procedure, extracted **unmodified** from
   `scripts/run_phase5_alpha_research.py` into an importable module so Phase 6 can call the *exact
   same code*, not a re-description of it (Section 7's "Use the exact Phase 5 research framework.
   Do not redesign it" taken literally). `scripts/run_phase5_alpha_research.py` was refactored to
   import from here too, so there is exactly one implementation for both phases.
7. **`src/research/real_data_pipeline.py`** — the top-level orchestration:
   CSV-load → quality gate → historical-universe reconstruction → Phase 5's exact exploratory /
   confirmatory / frozen-test procedure → economic backtest. `run_phase6_real_data_experiment()`
   raises `RealDataUnavailableError` when given no data (Section 20's stop condition, enforced in
   code, not just prose). 10 tests (`tests/test_real_data_pipeline.py`), including the mandatory
   future-corruption mechanism tests (Section 5/6, below).
8. **`scripts/fetch_real_historical_data.py`** — the runnable acquisition driver implementing
   `AcquisitionSpec` end to end (8 major-cap USDT pairs, 1H candles, 2021-01-01 to 2024-01-01).
   Actually executed against the real network during this phase (not just described): with
   `--dry-run` it builds and saves the spec with no network call; without it, it attempts the real
   fetch and — as expected given Section 1 — fails with the exact, informative error:

   ```
   FAILED to fetch BTCUSDT: 403, message='Forbidden', url='https://api.binance.com/api/v3/klines?
   symbol=BTCUSDT&interval=1h&startTime=1609459200000&endTime=1704067200000&limit=1000'
   ```

   The constructed URL (correct symbol, interval, startTime/endTime, limit) is proof the ingestion
   code itself is correct up to the exact point network policy blocks it — the failure is
   demonstrably at the network layer, not a bug in this phase's code. The spec file this run
   produced is committed at `data/research/phase6_acquisition_spec.json`.

**Explicit statement required by Section 2:** the real-data empirical experiment (Sections 7-16
below) **cannot yet be completed** in this environment. Every module above is built, unit-tested, and
ready; none of it has been run against genuine market data, and no result in this report claims
otherwise.

---

## 3. Historical Universe Reconstruction

`src/research/historical_universe.py` reconstructs, from arbitrary-length real per-asset OHLCV
series (not a shared padded index the way Phase 5's synthetic universes were built), exactly the
diagram Section 3 specifies:

```
timestamp -> assets actually tradable at timestamp -> historical OHLCV available by timestamp
```

`eligible_symbols_at(timestamp, asset_histories, listing_registry)` returns only symbols with
non-NaN data at or before `timestamp`, additionally restricted (never expanded) by an optional
registry entry. `build_point_in_time_universe()` applies this across a series of scan timestamps and
records the result — the "reconstruct... record" Section 3 asks for.

**Two-tier listing-date honesty, load-bearing throughout this phase:**

- `infer_listing_registry_from_data()` builds a registry from each asset's own first non-NaN OHLCV
  candle, and marks **every** record `is_proxy=True`. This is a lower bound, not a verified listing
  date — a real asset may have traded earlier than whatever a given data source happened to cache.
- No function anywhere in this phase upgrades a proxy into a verified fact, and no code path
  silently uses today's symbol list for a historical period — verified by
  `tests/test_historical_universe.py::test_build_point_in_time_universe_never_shows_a_future_listed_asset_early`
  and the future-corruption mechanism test in Section 5/6.

**Survivorship bias — `quantify_survivorship_gap()`:** called with no delisting registry (this
sandbox has none — Section 1), it returns `quantifiable=False` and an explicit message rather than a
number. This is intentional and tested
(`test_quantify_survivorship_gap_without_registry_is_honest_about_not_knowing`). If a real exchange
delisting log were ever supplied, the same function computes an exact
`n_known_delisted / n_total` fraction — the code path exists and is tested
(`test_quantify_survivorship_gap_with_registry_computes_fraction`), only the input data does not.

**Delisted assets cannot be reconstructed in this environment — explicitly quantified as
UNQUANTIFIABLE, not hidden**, per Section 3's explicit instruction.

---

## 4. Data Quality Gates

`src/data/quality.py::validate_ohlcv` (built in Phase 2, extended this phase) already covered, before
Phase 6 touched it: duplicate timestamps, unsorted/out-of-order candles, non-finite values, zero/
negative prices, impossible OHLC relationships (`high < low`, etc.), negative volume, missing-candle
gaps (warning below / critical above a configurable missing-fraction threshold), staleness, and
abnormal-volume outliers (z-score, warning-only).

**Added this phase: an explicit unclosed-candle gate.** `src/utils/candles.py::bars_to_frame` already
drops a *trailing* unclosed candle at the one call site that builds a frame directly from
`CandleBar` objects — but a frame reloaded from a provenance-tracked CSV, or built by concatenating
two fetched batches, carries no such metadata by the time it reaches `validate_ohlcv`. The gate now
accepts an optional `is_closed: pd.Series` aligned to the frame's index and flags **any** `False`
entry — not just a trailing one — as critical, plus a separate critical flag if the `is_closed`
series doesn't cover the whole frame (an alignment bug is exactly the kind of silent failure this
gate exists to catch). 5 new tests, all passing.

**Decisions, documented rather than silent (Section 4's explicit requirement):**
`run_phase6_real_data_experiment()` runs the quality gate per symbol and **excludes** (does not
silently delete rows from) any symbol with a critical issue, recording the exact issue codes in
`Phase6ExperimentResult.rejected_symbols` — proven by
`tests/test_real_data_pipeline.py::test_run_requires_benchmark_symbol_to_pass_quality_gate`, which
constructs a deliberately-bad benchmark frame (`high = -1.0`) and confirms it is rejected with the
`impossible_ohlc` code recorded, not silently dropped or imputed.

---

## 5. Time Alignment

**Canonical convention** (documented once here, enforced by `load_ohlcv_csv` and inherited unchanged
from every timestamp downstream — Phase 5's `anchor_idx = scan_index - 1` / `forward_return` reading
only `> anchor_idx` convention, reused verbatim):

| Timestamp | Convention |
|---|---|
| Candle open time | The DataFrame index value itself — UTC, left-labeled. |
| Candle close time | `open_time + interval_seconds` (not stored separately; derived when needed, e.g. the `is_closed` check). |
| Feature timestamp | Same index as the candle it's computed from — a rolling/shift/pct_change value at row `i` uses only rows `<= i` (Phase 5's `signals.py`, unchanged). |
| Ranking / scan timestamp | `scan_index`; the anchor (last visible candle) is `scan_index - 1`. |
| Entry timestamp | `close[scan_index - 1]` (Phase 4/5's convention, unchanged). |
| Forward-return start | Strictly `> scan_index - 1` (`forward_return`'s `start = anchor_idx + 1`). |
| Exit timestamp | `anchor_idx + horizon_candles`. |

**Mandatory extreme future-corruption test, re-run against the new real-data-shaped code path**
(not just Phase 5's synthetic-universe path): `tests/test_real_data_pipeline.py` corrupts every
candle after a fixed evaluation point by **50×** and **0.01×** on price, plus a **50× volume**
shock, loaded through the new CSV loader (`load_ohlcv_csv`) feeding directly into Phase 5's
`momentum()` signal function — the earlier value is asserted byte-identical
(`test_real_data_signal_panel_unaffected_by_extreme_future_price_shock`, both shock directions
pass). A second test corrupts a later-listed asset's future candles and confirms an *earlier*
asset's universe-membership determination at an *earlier* timestamp is unaffected
(`test_real_data_universe_membership_unaffected_by_future_shock`). **No look-ahead violation found**
in the new ingestion/universe/signal code path.

---

## 6. Real-Market Baselines

All nine required baselines are implemented and reused unmodified from Phase 4/5
(`src/evaluation/portfolio_backtest.py`, `src/research/experiment_runner.py`):

| # | Baseline | Status |
|---|---|---|
| 1 | Buy-and-hold BTC | Implemented (`buy_and_hold_baseline`), wired into `run_frozen_test_for_survivors` |
| 2 | Buy-and-hold ETH | Implemented, passed as an `extra_baselines` entry the same way |
| 3 | Equal-weight universe | Implemented (`equal_weight_universe_baseline`) |
| 4 | Market-cap-weighted universe | **Code path partially closed this phase** (Section 2's `fetch_with_market_cap`) but **not runnable without real data** — no market-cap series is cached anywhere |
| 5 | Random asset selection | Implemented (`random_selection_baseline`, seeded/reproducible) |
| 6 | Simple 1D momentum | Covered by `signals.py::momentum` at the `"1D"` horizon, already in the signal registry |
| 7 | Simple 7D momentum | Covered by `signals.py::momentum` at the `"7D"` horizon |
| 8 | Simple cross-sectional momentum | Covered by `momentum_ranking_baseline` (a pure trailing-return ranking, no scanner involved) |
| 9 | Existing Phase 4 scanner | Frozen and cited unchanged from `docs/PHASE4_SCANNER_AUDIT_REPORT.md` — not re-run against real data this phase since no real data exists to run it on |

None of these baselines has been *executed* against real data — the code is ready
(`tests/test_real_data_pipeline.py` exercises the full chain including baselines on a fixture) but
running it against fabricated data and calling the result a "baseline comparison" would misrepresent
exactly what Section 18 prohibits.

---

## 7-15. Real-Data Signal Research, Horizons, Walk-Forward, Statistics, Economics, Liquidity, Regimes, Survivorship, Real-vs-Synthetic Comparison — NOT RUN

Every mechanism Sections 7-15 ask for is implemented and already validated against Phase 5's
synthetic universes (`docs/PHASE5_ALPHA_RESEARCH_REPORT.md`) and against fixtures in this phase's
own test suite:

- **Section 7-8 (signal families, horizons):** `run_phase6_real_data_experiment()` calls
  `run_exploratory_screen` / `run_cross_sectional_relative_strength_screen` against **every** signal
  in `SIGNAL_REGISTRY` (momentum, mean reversion, volume/flow, volatility, relative strength,
  multi-horizon/multi-timeframe combinations) at all 8 forward horizons — the identical matrix
  Phase 5 ran. A horizon that cannot reach `MIN_PERIODS_FOR_STATS` (20) non-overlapping observations
  in the confirmatory stage is excluded from confirmatory testing exactly as Phase 5 defined
  (Section 9's `n < min_periods_for_stats: continue`) — this is the exact "mark UNTESTABLE rather
  than manufacture significance" behavior Section 8 requires, already proven necessary by Phase 5's
  own results (every one of its 43 "significant" exploratory cells lived at exactly the horizons
  this rule would exclude from confirmation).
- **Section 9 (walk-forward):** `split_design_validation_test` (chronological, no shuffling) +
  `confirmatory_stage`'s non-overlapping-stride re-evaluation, unchanged from Phase 5.
- **Section 10 (statistical significance):** Pearson/Spearman IC, IC information ratio, quantile
  spread/monotonicity, permutation p-value, and BH multiple-testing correction — all from
  `src/research/statistics.py`, unchanged, feeding the same `ExperimentLedger` schema Section 17
  requires.
- **Section 11 (economic validation):** `backtest_top_n_portfolio` (fees/slippage via `cost_pct`,
  turnover, concentration via top-N) plus `cost_sensitivity_sweep` and `mfe_mae_report` — wired into
  `run_frozen_test_for_survivors`, called only for a candidate that survives statistical screening
  (per Section 11's own "only candidates that survive statistical screening may enter economic
  simulation").
- **Section 12 (liquidity realism):** the liquidity/volume-bucket machinery this would use is
  `src/ranking/liquidity_filter.py` (Phase 4) — reusable but not wired into the Phase 6 pipeline,
  since it depends on real volume/spread data this phase does not have; documented as a gap, not
  silently skipped.
- **Section 13 (regime analysis):** `src/research/regime_breakdown.py` (Phase 5) is universe-
  agnostic and ready; it requires a real regime-labeled history (bull/bear/BTC-trending/etc.) to run
  against, which does not exist without real data.
- **Section 14 (survivorship, real vs. today's universe):** blocked on the same missing delisting
  registry as Section 3.
- **Section 15 (real vs. synthetic comparison):** cannot be performed — there is no real-data result
  to compare Phase 5's synthetic findings against. Stating this directly rather than comparing
  Phase 5 against itself and calling it "Section 15."

**None of Sections 7-15 produced a number in this report.** The infrastructure exists; the
data does not.

---

## 16. Frozen Test — Not Reached

`run_frozen_test_for_survivors()` (identical function Phase 5 used) touches the frozen-test region
at most once, and only for a candidate that already survived Stage 2 (confirmatory, non-overlapping,
permutation-tested, BH-corrected). With no real data, no candidate exists to survive Stage 2, so the
frozen-test region was never constructed, let alone touched. There is no frozen-test dataset hash or
experiment configuration to record for a run that did not happen.

---

## 17. Reproducibility

Every piece of provenance Section 17 requires is captured by
`src/research/data_provenance.py::DatasetProvenance` the moment real data is fetched: `git_commit`
(auto-captured via `git rev-parse HEAD`), `content_sha256` (deterministic row-hash of the fetched
frame), `source`/`url_template`/`params` (exact fetch parameters), `retrieved_at` (UTC timestamp),
and `row_count`. `AcquisitionSpec` separately captures the intended parameters (symbols, timeframe,
date range, pagination rule) *before* any fetch, so a future run reproduces the same intended
dataset even if this file's prose were lost. `ExperimentRecord` (Phase 5's ledger schema, unchanged)
already carries every experiment's exact parameters (signal, horizon, universe tag, region, method).
The one gap: `ExperimentRecord` does not yet carry a `data_provenance_hash` field linking a ledger
row back to the exact dataset version it ran against — noted here as a concrete follow-up for
whichever future phase actually has real data to link, rather than added speculatively now with
nothing to populate it.

---

## 18. Do Not Force a Positive Result — the Verdict

**No real data. No real-data experiment. No fabricated result.** Every "N/A" and "not run" in
Sections 7-16 is not an oversight to fill in later in this same report — it is the correct output of
Section 20's stop condition, executed rather than talked around.

**The precise scope of "FAILED" below, stated because conflating it with Phase 5's FAILED would be a
real error:**

- **Phase 5's FAILED** meant: real, executed statistical tests ran against disclosed-synthetic data,
  and no candidate survived confirmatory testing. That is evidence *about the pipeline and that
  specific synthetic universe*.
- **Phase 6's FAILED** means something narrower and weaker: **zero empirical evidence of a working
  real-market edge exists, because the empirical test could not be run at all.** This is not "tested
  on real data and found nothing" — it is "real data was not obtainable in this environment," which
  Section 20 explicitly requires reporting as a stop, not as silence, and not as an excuse to
  approximate with synthetic data and call it real-market validation.

Given the forced five-way classification (`FAILED`/`WEAK`/`PROMISING`/`SHADOW READY`/`PAPER READY`),
none of the latter four can be justified by definition — each requires some quantum of positive
statistical or economic evidence (Section 19's promotion criteria: "statistical significance,"
"positive IC," "improvement over baselines") that simply does not exist here, positive or negative.
`FAILED` is the only classification consistent with "do not force a positive result" when the honest
state is "untested," and it is reported with the caveat above attached everywhere this verdict is
cited, per Section 18's own instruction not to let a negative result get smoothed into something it
is not.

---

## Answer to Phase 6's Question

> Does the system show any robust, statistically defensible predictive signal when evaluated against
> genuinely historical crypto market data?

**No signal was found — because no genuinely historical crypto market data could be evaluated in
this environment.** The question "was Phase 5's failure caused by the data or by a genuine absence
of signal" is therefore **still open**, not answered by this phase, and cannot be answered without
real data. What this phase demonstrates is that the *entire remaining pipeline* — ingestion,
provenance, historical-universe reconstruction, quality gates, and Phase 5's exact statistical
framework — is built, tested, and reproducible, and needs only one thing to actually answer the
question: the dataset. See Section 2 for the exact `AcquisitionSpec` (`data/raw/phase6_real_data/
acquisition_spec.json`) a network-enabled environment can run today via
`python scripts/fetch_real_historical_data.py`.

---

## 19. Promotion Criteria — Not Met (Trivially)

No candidate exists to evaluate against the PROMISING/SHADOW READY/PAPER READY criteria (Section
19). Restated for completeness: PROMISING requires statistical significance after correction,
positive/stable IC, monotonic quantiles, improvement over baselines, economic value after costs,
acceptable drawdown, robustness, no leakage, no survivorship confound, and parameter robustness —
zero of these were evaluated because zero candidates were tested.

---

## 20. Stop Condition — Executed

Per Section 20: real data could not be obtained or reconstructed in this environment → **the
empirical alpha claim is stopped.** No synthetic data was substituted for it anywhere in Sections
7-16. What is delivered instead, per Section 20's own instruction, is exactly:

1. **The infrastructure required for real-data validation** — Sections 2-6, 16-17 above, all built
   and tested.
2. **A precise list of required datasets** — restated concisely:
   - Real historical OHLCV, 1H or finer, for a broad liquid crypto universe, spanning multiple
     market regimes (3+ years recommended, covering at least one full bull/bear cycle) — obtainable
     the moment this environment (or one like it) has real network access to Binance, via
     `scripts/fetch_real_historical_data.py` and the committed `AcquisitionSpec`.
   - A real historical exchange listing/delisting registry, to upgrade `historical_universe.py`'s
     proxy listing dates to verified ones and to make `quantify_survivorship_gap` return an actual
     number.
   - A real historical market-cap/circulating-supply series, to enable baseline #4 (market-cap-
     weighted universe) — `fetch_with_market_cap()` is ready to populate this the moment network
     access exists.
   - Real order-book/spread and volume data, for Section 12's liquidity-realism sensitivity tests
     beyond the OHLCV-derived proxies Phase 4's liquidity filter already uses.

---

## 21. Deliverables

- `docs/PHASE6_REAL_DATA_VALIDATION_REPORT.md` — this report.
- `src/data/exchange/binance.py` — added `get_historical_klines()` (pagination past the 1,000-candle cap).
- `src/data/fetch_prices.py` — added `CoinGeckoPriceRepository.fetch_with_market_cap()`.
- `src/data/quality.py` — added the `is_closed` unclosed-candle gate to `validate_ohlcv`.
- `src/research/{data_provenance,historical_universe,experiment_runner,real_data_pipeline}.py` — new modules.
- `scripts/run_phase5_alpha_research.py` — refactored to import the shared `experiment_runner` (no behavior change).
- `scripts/fetch_real_historical_data.py` — the runnable acquisition driver (dry-run verified; real fetch verified to fail exactly at the network layer, per Section 2).
- `data/research/phase6_acquisition_spec.json` — the exact, reproducible fetch specification produced by a real run of the driver above.
- `tests/test_{binance_historical_klines,coingecko_market_cap,data_provenance,historical_universe,data_quality_unclosed_candle,real_data_pipeline}.py` — 43 new tests, including the mandatory anti-leakage future-corruption tests (Section 5).

**Report summary fields, filled honestly:**

| Field | Value |
|---|---|
| Dataset coverage | None (no real data obtained) |
| Historical universe coverage | N/A |
| Assets | N/A |
| Date range | N/A |
| Timeframes | N/A (convention defined prospectively, Section 5) |
| Missingness | N/A |
| Survivorship limitations | Full — no delisting registry available; listing dates are proxy-only where infrastructure exists |
| Signals tested (on real data) | 0 |
| Horizons tested (on real data) | 0 |
| Total experiments (on real data) | 0 |
| Significant candidates | 0 |
| Multiple-testing-adjusted results | N/A |
| Best validation candidate | None |
| Frozen-test performance | Not reached |
| Baseline performance | Not run (code ready, Section 6) |
| Transaction-cost-adjusted performance | Not run |
| Drawdown | N/A |
| Regime stability | N/A |
| **Final classification** | **FAILED** (data-unavailability FAILED — see Section 18's caveat, not Phase 5's evidence-based FAILED) |

Full test suite: 359 passed, 0 failed (316 pre-Phase-6 + 43 new).
