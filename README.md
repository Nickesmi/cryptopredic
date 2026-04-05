# Crypto Alpha Engine 🚀

> Crypto Alpha Engine is a machine learning and quantitative research platform for forecasting cryptocurrency price movements, ranking high-potential digital assets, and evaluating risk-adjusted trading opportunities across multiple time horizons.

This is a research-grade platform. Instead of serving as a "magic coin picker", it employs empirical backtesting, defined confidence intervals, risk algorithms, and explicit anomaly calculations to surface probability scores.

## Modules

1. **Forecast Engine**: Generates 1d, 7d, and 30d regressions indicating exactly what the projected return metrics and interval confidence is for Tier-1 Assets like `BTC`, `ETH` and `SOL`.
2. **Coin Ranking Engine**: Ranks altcoins by momentum, volume growth, and on-chain metrics instead of arbitrary recommendations.
3. **Risk & Trust Engine**: Evaluates exchange liquidity footprints, drawdown traps, and market cap legitimacy to label an asset as `Buy Candidate` or `Speculative Trap`.
4. **Evaluation & Backtesting Dashboard**: Visually displays Walk-Forward testing algorithms to prove historical model efficacy against Buy & Hold standards.

## Tech Stack Overview

- **Storage / Data**: `Pandas`, `NumPy`, Daily Data Hooks mapped to `.parquet` blobs.
- **Machine Learning**: `Scikit-Learn`, `XGBoost`, `CatBoost`.
- **API Engine**: `FastAPI` logic gateways.
- **Reporting Dashboard**: `Streamlit` data-visualization framework.
- **Automation Pipeline**: `GitHub Actions` CI/CD processing schemas.

## Quick Start
1. Configure Environment: `python -m venv venv && source venv/bin/activate`
2. Install Engine Package: `pip install -e .` & `pip install -r requirements.txt`
3. Launch Streamlit MVP: `streamlit run src/dashboard/streamlit_app.py`
