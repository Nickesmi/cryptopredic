# Architecture

## Overview

The Crypto Alpha Engine follows **Clean Architecture** principles, organising code into
concentric layers where inner layers define interfaces and outer layers provide implementations.
Dependencies always point inward — no inner-layer module imports from an outer layer.

```
┌─────────────────────────────────────────────────────────┐
│  Infrastructure / Interface Adapters                    │
│  src/api/         — FastAPI route handlers              │
│  src/dashboard/   — Streamlit visualisation             │
│  src/data/        — CoinGecko repository implementation │
│  src/config/      — Environment-driven settings         │
│  src/utils/       — Logging, helpers, validators        │
├─────────────────────────────────────────────────────────┤
│  Use Cases / Application Logic                          │
│  src/features/    — Feature engineering pipeline        │
│  src/models/      — Model training & inference wrappers │
│  src/evaluation/  — Backtesting & walk-forward          │
│  src/ranking/     — Coin scoring & filtering            │
├─────────────────────────────────────────────────────────┤
│  Entities / Domain                                      │
│  Data contracts: ForecastResult, PriceRepository (ABC)  │
└─────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Layer | Responsibility |
|--------|-------|----------------|
| `src/config/settings.py` | Infrastructure | Env-driven config constants |
| `src/utils/logger.py` | Infrastructure | Structured logging factory |
| `src/utils/helpers.py` | Infrastructure | Parquet I/O & directory helpers |
| `src/utils/validators.py` | Infrastructure | Input validation (symbol, horizon) |
| `src/data/fetch_prices.py` | Infrastructure | `PriceRepository` ABC + `CoinGeckoPriceRepository` |
| `src/data/preprocessing.py` | Use Case | OHLCV cleaning, log-return computation |
| `src/data/fetch_market_data.py` | Use Case | End-to-end pipeline orchestration |
| `src/features/technical_indicators.py` | Use Case | Pure indicator functions (SMA, EMA, RSI, MACD, BB, lags, vol) |
| `src/features/feature_pipeline.py` | Use Case | `FeaturePipeline` orchestrator |
| `src/models/xgboost_model.py` | Interface Adapter | `TimeSeriesForecaster` & `ForecastResult` |

---

## SOLID Principles Applied

| Principle | Where |
|-----------|-------|
| **S**ingle Responsibility | Each module/class has exactly one reason to change |
| **O**pen/Closed | `FeaturePipeline._STEPS` list allows new indicators without modifying existing code |
| **L**iskov Substitution | `CoinGeckoPriceRepository` is fully substitutable for `PriceRepository` |
| **I**nterface Segregation | `PriceRepository` exposes only `fetch()` — no bloat |
| **D**ependency Inversion | `run_pipeline()` depends on `PriceRepository` ABC, not the concrete CoinGecko class |

---

## Data Flow (Issue #1 — Time-Series Forecasting)

```
CoinGecko API
     │
     ▼
CoinGeckoPriceRepository.fetch(symbol, days)
     │  returns raw OHLCV DataFrame
     ▼
preprocessing.clean_ohlcv()  +  add_log_returns()
     │  returns clean DataFrame
     ▼
FeaturePipeline.build()
     │  returns feature-complete DataFrame (SMA, EMA, RSI, MACD, BB, lags, vol)
     ▼
TimeSeriesForecaster.train(X, y)   [separate model per horizon: 1d / 7d / 30d]
     │  Direct Strategy: y[t] = close price horizon days ahead
     ▼
TimeSeriesForecaster.predict_latest(X)
     │  returns predicted closing price (USD)
     ▼
ForecastResult(symbol, horizon, predicted_price, mae, rmse, mape)
```

---

## Full System Pipeline (Separation of Concerns)

Each stage has a **single responsibility** and communicates only with its
immediate neighbours through defined interfaces. No stage skips a layer.

```
┌─────────────────────────────────────────────────┐
│  STAGE 1 — Market Data                          │
│  src/data/fetch_prices.py                       │
│  PriceRepository → CoinGeckoPriceRepository     │
│  Outputs: raw OHLCV DataFrame (UTC daily)       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2 — Feature Engineering                  │
│  src/data/preprocessing.py                      │
│  src/features/technical_indicators.py           │
│  src/features/sentiment_features.py  (future)   │
│  src/features/onchain_features.py    (future)   │
│  FeaturePipeline → feature matrix               │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3 — Forecast Models                      │
│  src/models/xgboost_model.py                    │
│  src/models/lstm_model.py           (future)    │
│  src/models/transformer_model.py    (future)    │
│  ForecastModel interface → predicted_price      │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STAGE 4 — Confidence Estimator                 │
│  src/evaluation/metrics.py                      │
│  src/evaluation/walk_forward.py                 │
│  Outputs: MAE, RMSE, MAPE, confidence intervals │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STAGE 5 — Risk Manager                         │
│  src/ranking/risk_score.py                      │
│  src/ranking/liquidity_filter.py                │
│  src/ranking/score_coins.py                     │
│  Outputs: Buy Candidate / Speculative Trap label │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STAGE 6 — LLM Reasoning Agent  (future)        │
│  src/agents/                                    │
│  Synthesises forecast + risk into plain-English  │
│  recommendations with cited evidence             │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  STAGE 7 — Frontend Visualisation               │
│  src/dashboard/streamlit_app.py                 │
│  frontend/  (Next.js — Issue #4)                │
│  Renders results; contains zero business logic  │
└─────────────────────────────────────────────────┘
```

**Key invariants:**
- A Forecast Model never makes a buy/sell decision (that is Stage 5's job).
- A Risk Manager never fetches raw market data (that is Stage 1's job).
- The Frontend never transforms data (that is Stages 1–5's job).

---

## Governance Documents

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE_RULES.md](../ARCHITECTURE_RULES.md) | Non-negotiable structural rules for all contributors |
| [CLAUDE_RULES.md](../CLAUDE_RULES.md) | AI coding constitution — rules for every AI-generated change |
| [docs/definition_of_done.md](definition_of_done.md) | Checklist that must pass before closing any issue |
