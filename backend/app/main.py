from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="CryptoPredic API",
    description="API for accessing cryptocurrency price predictions and high-potential recommendations.",
    version="1.0.0"
)

# Configure CORS for our Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the CryptoPredic API"}

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

from app.services.crypto_api import crypto_fetcher

@app.get("/api/price/{symbol}")
def get_price(symbol: str):
    """Get the current price ticker for a popular coin like BTCUSDT or ETHUSDT."""
    return crypto_fetcher.get_binance_ticker(symbol.upper())

from app.services.sentiment_service import sentiment_service

@app.get("/api/high-potential")
def get_high_potential():
    """
    Get trending novel tokens from CoinGecko and process them through our
    NLP sentiment and on-chain volume logic to rank them by Spike Probability.
    """
    trending_raw = crypto_fetcher.get_coingecko_trending()
    if not isinstance(trending_raw, list) or len(trending_raw) == 0:
        return {"error": "Could not parse trending data"}
        
    scored_coins = sentiment_service.calculate_spike_probability(trending_raw)
    return {"high_potential_recommendations": scored_coins}

from app.services.prediction_service import predictor_service

@app.get("/api/predict/{symbol}")
def predict_crypto_price(symbol: str, days: int = 30):
    """
    Generate a price prediction forecast for the given symbol (e.g. BTCUSDT) 
    over a specified number of days (e.g. 1, 7, 30).
    """
    return predictor_service.predict_price(symbol.upper(), days_to_predict=days)
