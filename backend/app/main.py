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

@app.get("/api/high-potential")
def get_high_potential():
    """Get trending novel tokens from CoinGecko as a baseline for high potential."""
    trending = crypto_fetcher.get_coingecko_trending()
    return {"trending": trending}
