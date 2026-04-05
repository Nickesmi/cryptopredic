# CryptoPredic 🚀

CryptoPredic is an advanced machine learning platform designed to predict cryptocurrency prices and uncover novel, high-potential tokens before they break out.

The platform is split into a robust **Python (FastAPI)** machine learning backend and a dynamic **Next.js (React)** frontend for analytics and visualization.

---

## 🗺️ Project Roadmap

Our development is structured into clear phases to ensure a robust, scalable, and highly accurate prediction platform.

### Phase 1: Foundation & Data pipeline (✅ Completed)
- Initialize project structure (FastAPI Backend + Next.js Frontend).
- Build automated data fetchers for Binance, CoinGecko, and CoinMarketCap.
- Establish database configuration (SQLite for dev, Time-series DB for prod).

### Phase 2: Core Machine Learning Development (⏳ Pending)
- **Time-Series Forecasting**: Build predictive models (LSTM, XGBoost) for top-tier coins (BTC, ETH) using candlestick data.
- **Novel Token Classification**: Develop scanning protocols to identify sudden volume spikes and social sentiment bursts for new cryptocurrencies.
- **API Integration**: Expose ML models via REST API endpoints for frontend consumption.

### Phase 3: Platform & Dashboard Integration (⏳ Pending)
- Design the Next.js frontend with dynamic charting (Lightweight-charts / Chart.js).
- Build the "High-Potential Recommendations" feed.
- Implement historical performance tracking to showcase AI accuracy over time.

### Phase 4: Production Deployment & Automation (⏳ Pending)
- Configure Docker and deploy to cloud architecture.
- Set up Celery/Redis task queues for daily retraining and real-time market data evaluation.

---

## 📋 Task List & Issue Tracker

To keep development organized, the following represents our active tasks to be turned into GitHub Issues:

### 🟢 Priority Issues / Tasks

*   [ ] **Issue #1: Implement Time-Series Models for BTC/ETH**
    *   **Description**: Train a time-series model (e.g., Prophet or XGBoost) to predict the closing price of Bitcoin and Ethereum for 1D, 1W, and 1M intervals.
    *   **Labels**: `machine-learning`, `backend`
*   [ ] **Issue #2: Develop Social Sentiment & Volume Spikers**
    *   **Description**: Identify new tokens with a "Probability of Spike" score by scanning CoinGecko trending and correlating with X (Twitter) API sentiment.
    *   **Labels**: `machine-learning`, `data-science`
*   [ ] **Issue #3: Setup Redis & Celery Background Tasks**
    *   **Description**: We need a background queue to fetch prices every minute and re-run predictions every hour without blocking the main API.
    *   **Labels**: `backend`, `infrastructure`
*   [ ] **Issue #4: Next.js Dashboard UI Build**
    *   **Description**: Build the web interface using Tailwind CSS. Must include a main chart view and a sidebar for high-potential alerts.
    *   **Labels**: `frontend`, `ui/ux`

---

## 🛠️ Tech Stack architecture

**Backend**:
- `FastAPI` (High-performance API routing)
- `SQLAlchemy` (ORM)
- `XGBoost`, `Prophet`, `Pandas` (Data processing and AI predictions)

**Frontend**:
- `Next.js 15` (App Router)
- `Tailwind CSS` (Styling & Aesthetics)
- `TypeScript` (Type safety)

## 🚀 Quick Start (Local Development)

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API docs available at: `http://localhost:8000/docs`

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Platform available at: `http://localhost:3000`
