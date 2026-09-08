import { predictCandles as olsPredict, type ForecastResult } from './ClientSidePredictor';
import type { CandleData } from '../binance/BinanceClient';

export class PredictionManager {
  /**
   * Fetches the prediction from the AI backend, and falls back to OLS
   * linear regression if the backend is unavailable or fails.
   */
  static async getPrediction(
    candles: CandleData[],
    nCandles: number,
    intervalSecs: number,
    symbol: string,
    timeframe: string,
    modelName: string = 'auto'
  ): Promise<ForecastResult> {
    if (candles.length < 10) throw new Error('Not enough candles for prediction');
    
    // 1. Try fetching from the backend API
    try {
      const url = new URL('http://localhost:8000/api/predict'); // Or relative if proxied
      url.searchParams.set('symbol', symbol.toLowerCase().replace('usdt', ''));
      url.searchParams.set('timeframe', timeframe);
      url.searchParams.set('horizon', nCandles.toString());
      url.searchParams.set('model', modelName);
      
      // Use a timeout so we don't hang the UI
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);
      
      const res = await fetch(url.toString(), {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      
      if (res.ok) {
        const aiData = await res.json();
        // Assuming AI backend returns a structure that maps to ForecastResult
        if (aiData && aiData.expected_price) {
           return {
             anchorTime: aiData.anchor_time,
             anchorPrice: aiData.anchor_price,
             line: aiData.candles,
             upper: aiData.upper_band,
             lower: aiData.lower_band,
             direction: aiData.direction,
             confidence: aiData.confidence,
             expectedPrice: aiData.expected_price,
             expectedChangePct: aiData.expected_change_pct,
             volatilityEst: aiData.volatility_est,
           };
        }
      }
    } catch (err) {
      console.warn('[PredictionManager] AI backend unavailable or timed out. Falling back to OLS.', err);
    }
    
    // 2. Fallback to ClientSidePredictor (OLS)
    return olsPredict(candles, nCandles, intervalSecs);
  }
}
