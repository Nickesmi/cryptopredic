import logging
import requests
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CryptoDataFetcher:
    """Service to fetch crypto data from multiple public APIs."""
    
    def __init__(self):
        self.binance_base_url = "https://api.binance.com/api/v3"
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
    
    def get_binance_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current price ticker from Binance."""
        try:
            url = f"{self.binance_base_url}/ticker/price"
            response = requests.get(url, params={"symbol": symbol})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from Binance: {e}")
            return {"error": str(e)}

    def get_coingecko_trending(self) -> List[Dict[str, Any]]:
        """Fetch trending search coins from CoinGecko (no auth required)."""
        try:
            url = f"{self.coingecko_base_url}/search/trending"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            # Extract basic coin info from the nested response
            coins = [item['item'] for item in data.get('coins', [])]
            return coins
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching trending from CoinGecko: {e}")
            return [{"error": str(e)}]
        
    def get_historical_klines(self, symbol: str, interval: str = "1d", limit: int = 30) -> List[List[Any]]:
        """
        Fetch historical candlestick data from Binance for ML training.
        interval: 1m, 1h, 1d, etc.
        """
        try:
            url = f"{self.binance_base_url}/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching klines from Binance: {e}")
            return []

crypto_fetcher = CryptoDataFetcher()
