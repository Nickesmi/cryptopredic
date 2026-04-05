import pandas as pd
from prophet import Prophet
import logging
from typing import Dict, Any, List
from datetime import datetime
from app.services.crypto_api import crypto_fetcher

logger = logging.getLogger(__name__)

class PricePredictor:
    """Service to handle time-series forecasting using Prophet."""
    
    def __init__(self):
        pass
        
    def _prepare_dataframe(self, kline_data: List[List[Any]]) -> pd.DataFrame:
        """
        Convert Binance kline data into a Pandas DataFrame formatted for Prophet.
        Binance Kline format:
        [
          [
            1499040000000,      // Open time
            "0.01633302",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            ...
          ]
        ]
        """
        if not kline_data or "error" in str(kline_data):
            raise ValueError("Invalid kline data passed to predictor")
            
        df = pd.DataFrame(kline_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'number_of_trades', 
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Prophet requires columns 'ds' (datetime) and 'y' (numeric target)
        df['ds'] = pd.to_datetime(df['close_time'], unit='ms')
        df['y'] = df['close'].astype(float)
        
        return df[['ds', 'y']]
        
    def predict_price(self, symbol: str, days_to_predict: int = 30) -> Dict[str, Any]:
        """
        Train a quick Prophet model on the last 1000 days and forecast into the future.
        """
        try:
            # 1. Fetch historical data (last 1000 days is a good sample for Prophet)
            kline_data = crypto_fetcher.get_historical_klines(symbol=symbol, interval="1d", limit=1000)
            
            # 2. Prepare Data
            df = self._prepare_dataframe(kline_data)
            
            # 3. Initialize and train model
            # Note: For production, models should be saved/loaded rather than trained per-request,
            # but Prophet trains very quickly for low-volume univariate data.
            model = Prophet(daily_seasonality=True)
            model.fit(df)
            
            # 4. Make Future Dataframe
            future = model.make_future_dataframe(periods=days_to_predict)
            forecast = model.predict(future)
            
            # Extract the actual forecast block we care about
            last_actual = df['y'].iloc[-1]
            future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_to_predict)
            
            # Prepare standard response
            forecast_results = []
            for _, row in future_forecast.iterrows():
                forecast_results.append({
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "predicted_price": round(row['yhat'], 2),
                    "lower_bound": round(row['yhat_lower'], 2),
                    "upper_bound": round(row['yhat_upper'], 2)
                })
                
            return {
                "symbol": symbol,
                "current_price": round(last_actual, 2),
                "forecast_days": days_to_predict,
                "predictions": forecast_results
            }
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return {"error": str(e)}

predictor_service = PricePredictor()
