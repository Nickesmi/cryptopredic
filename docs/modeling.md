# Modeling Guide

## Issue #1 — Time-Series Forecasting for BTC/ETH

### Approach: Direct Multi-Step Forecasting with XGBoost

Rather than using a recursive or sequence-based method (which accumulate
prediction error over each step), we use the **Direct Strategy**:

> Train a **separate** `XGBRegressor` per forecast horizon.
> For horizon *h*, the target at training row *t* is:
>
> `y[t] = close_price[t + h]`

This means:

| Model | Target |
|-------|--------|
| `bitcoin_1d.joblib` | BTC closing price 1 calendar day ahead |
| `bitcoin_7d.joblib` | BTC closing price 7 calendar days ahead |
| `bitcoin_30d.joblib` | BTC closing price 30 calendar days ahead |
| `ethereum_1d.joblib` | ETH closing price 1 calendar day ahead |
| … | … |

---

### Feature Engineering

All features are computed from the daily closing price and OHLCV data:

| Feature Group | Columns | Purpose |
|---------------|---------|---------|
| Trend | `sma_7`, `sma_14`, `sma_30` | Capture medium/long-term trend |
| Trend | `ema_12`, `ema_26` | Faster trend signal |
| Momentum | `rsi_14` | Overbought / oversold signal |
| Momentum | `macd`, `macd_signal`, `macd_hist` | Trend crossover signal |
| Volatility | `bb_upper`, `bb_mid`, `bb_lower`, `bb_width` | Price channel & squeeze |
| Volatility | `volatility_7`, `volatility_14` | Realised vol (std of log returns) |
| Lag | `close_lag_1..14` | Explicit auto-regression features |
| Return | `log_return` | Stationary price change signal |

---

### Validation Strategy

A simple **temporal hold-out** split (last 10% of rows) is used to
compute out-of-sample validation metrics:

- **MAE** — Mean Absolute Error (USD)
- **RMSE** — Root Mean Squared Error (USD)
- **MAPE** — Mean Absolute Percentage Error (%)

> ⚠️ For rigorous walk-forward validation, see `src/evaluation/walk_forward.py`
> (planned for Issue #5).

---

### Hyperparameter Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 500 | Boosting rounds |
| `max_depth` | 6 | Tree depth |
| `learning_rate` | 0.05 | Step-size shrinkage |
| `subsample` | 0.8 | Row sampling per tree |
| `colsample_bytree` | 0.8 | Feature sampling per tree |

All parameters can be overridden via environment variables (see `.env.example`).

---

### Data Source

Daily OHLCV data is fetched from the **CoinGecko v3 public API** (no API key required).

- **Endpoint**: `GET /coins/{id}/ohlc?vs_currency=usd&days={N}`
- **Volume**: fetched separately from `/coins/{id}/market_chart`
- **Default history**: 730 days (configurable via `HISTORY_DAYS` env var)
- **Rate limiting**: handled with exponential back-off (up to 3 retries)

---

### Running the Pipeline

```bash
# Train and forecast all Tier-1 assets (BTC, ETH, SOL) × all horizons
python -m src.data.fetch_market_data

# Single asset / horizon (Python)
from src.data.fetch_market_data import run_pipeline
result = run_pipeline("bitcoin", horizon=7)
print(result)
```

Trained models are saved to `models/saved/` as `.joblib` files.
Raw OHLCV data is saved to `data/raw/` as `.parquet` files.
