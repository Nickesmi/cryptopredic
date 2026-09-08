/**
 * Domain types shared across the entire frontend.
 *
 * These are the single source of truth for all data shapes. The rendering
 * layer, WebSocket client, and React components all import from here —
 * never from each other.
 */

// ── Candle data ──────────────────────────────────────────────────────────────

/** A single OHLCV candlestick bar. */
export interface CandleBar {
  /** Unix timestamp in seconds (UTC). */
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  /** False while the candle period is still open (live tick). */
  is_closed?: boolean;
}

// ── Timeframes + Symbols ─────────────────────────────────────────────────────

export type Timeframe = '1m' | '5m' | '15m' | '30m' | '1H' | '4H' | '1D' | '1W';
export type SupportedSymbol = 'BTCUSDT' | 'ETHUSDT' | 'SOLUSDT';
export type PredictionHorizon = 5 | 10 | 20 | 50;
export type ModelName = 'auto' | 'xgboost' | 'lstm' | 'ensemble';

export const TIMEFRAMES: Timeframe[] = ['1m', '5m', '15m', '30m', '1H', '4H', '1D', '1W'];
export const SUPPORTED_SYMBOLS: SupportedSymbol[] = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'];
export const PREDICTION_HORIZONS: PredictionHorizon[] = [5, 10, 20, 50];

// ── Forecast / Prediction ────────────────────────────────────────────────────

/** A single point on the prediction line or confidence band. */
export interface PredictionPoint {
  time: number;
  value: number;
}

/**
 * The canonical data object returned by the prediction API.
 * The rendering layer consumes this; it knows nothing about AI models.
 */
export interface ForecastObject {
  symbol: string;
  timeframe: Timeframe;
  n_candles: number;
  /** Unix timestamp of the last historical candle (join point). */
  anchor_time: number;
  /** Close price of the last historical candle. */
  anchor_price: number;
  /** Central prediction line. */
  candles: PredictionPoint[];
  /** Upper confidence interval. */
  upper_band: PredictionPoint[];
  /** Lower confidence interval. */
  lower_band: PredictionPoint[];
  direction: 'bullish' | 'bearish';
  /** 0.0 – 1.0 */
  confidence: number;
  expected_price: number;
  expected_change_pct: number;
  volatility_est: number;
  model_name: string;
  generated_at: string;
}

// ── Chart state ──────────────────────────────────────────────────────────────

export interface ChartState {
  symbol: SupportedSymbol;
  timeframe: Timeframe;
  predictionHorizon: PredictionHorizon;
  modelName: ModelName;
  forecast: ForecastObject | null;
  isLoadingHistory: boolean;
  isLoadingForecast: boolean;
  error: string | null;
}
