/**
 * Client-side price predictor.
 *
 * Uses linear regression on the last N closing prices to project
 * future candles. Confidence bands widen with √(time) to reflect
 * compounding uncertainty — the standard deviation cone used by
 * options markets and professional forecasting tools.
 *
 * This runs entirely in the browser — no backend call required.
 */

import type { CandleData } from '../binance/BinanceClient';

export interface ForecastPoint {
  time: number;
  value: number;
}

export interface ForecastResult {
  anchorTime: number;    // Unix seconds of last historical candle
  anchorPrice: number;   // Close price of last historical candle
  line: ForecastPoint[];
  upper: ForecastPoint[];
  lower: ForecastPoint[];
  direction: 'bullish' | 'bearish';
  /** 0.0–1.0 */
  confidence: number;
  expectedPrice: number;
  expectedChangePct: number;
  volatilityEst: number;
}

/** Timeframe → interval seconds */
export function timeframeToSeconds(tf: string): number {
  const map: Record<string, number> = {
    '1m': 60, '5m': 300, '15m': 900, '30m': 1_800,
    '1H': 3_600, '4H': 14_400, '1D': 86_400, '1W': 604_800,
  };
  return map[tf] ?? 3_600;
}

/** Simple ordinary least-squares on y = [y₀, y₁, …, yₙ₋₁] */
function olsSlope(y: number[]): { slope: number; intercept: number } {
  const n = y.length;
  if (n < 2) return { slope: 0, intercept: y[0] ?? 0 };
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX  += i;
    sumY  += y[i];
    sumXY += i * y[i];
    sumXX += i * i;
  }
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) return { slope: 0, intercept: sumY / n };
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

/** Daily close-to-close log-return volatility (annualised → per-candle) */
function estimateVolatility(candles: CandleData[]): number {
  if (candles.length < 3) return 0.02;
  const returns: number[] = [];
  for (let i = 1; i < candles.length; i++) {
    const r = Math.log(candles[i].close / candles[i - 1].close);
    if (isFinite(r)) returns.push(r);
  }
  if (returns.length < 2) return 0.02;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance =
    returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / returns.length;
  return Math.max(Math.sqrt(variance), 0.003);
}

/**
 * Generate a price forecast for the next `nCandles` periods.
 *
 * @param candles       Historical OHLCV data (sorted oldest-first).
 * @param nCandles      Number of future candles to project.
 * @param intervalSecs  Seconds per candle period.
 */
export function predictCandles(
  candles: CandleData[],
  nCandles: number,
  intervalSecs: number,
): ForecastResult {
  // Use at most the last 100 candles for regression stability
  const sample = candles.slice(-100);
  const closes = sample.map(c => c.close);
  const vol = estimateVolatility(sample.slice(-30));

  const { slope, intercept } = olsSlope(closes);

  const lastCandle = candles[candles.length - 1];
  const anchorPrice = lastCandle.close;
  const anchorTime  = lastCandle.time;
  const baseIdx     = closes.length;

  const line:  ForecastPoint[] = [];
  const upper: ForecastPoint[] = [];
  const lower: ForecastPoint[] = [];

  for (let i = 1; i <= nCandles; i++) {
    const time  = anchorTime + i * intervalSecs;
    // Raw OLS projection
    const raw   = intercept + slope * (baseIdx + i - 1);
    // Blend toward anchor price to prevent runaway extrapolation
    const blend = 0.7;
    const value = Math.max(raw * blend + anchorPrice * (1 - blend), anchorPrice * 0.5);
    // Confidence band: ±1σ × √steps
    const band  = anchorPrice * vol * Math.sqrt(i);

    line.push({ time, value });
    upper.push({ time, value: value + band });
    lower.push({ time, value: Math.max(value - band, value * 0.1) });
  }

  const expectedPrice    = line[line.length - 1].value;
  const expectedChangePct = ((expectedPrice - anchorPrice) / anchorPrice) * 100;
  const direction         = slope >= 0 ? 'bullish' : 'bearish';

  // Confidence ≈ tightness of the band relative to price
  const finalBandPct =
    (upper[upper.length - 1].value - lower[upper.length - 1].value) / expectedPrice;
  const confidence = Math.max(0.05, Math.min(0.97, 1 - finalBandPct * 2));

  return {
    anchorTime,
    anchorPrice,
    line,
    upper,
    lower,
    direction,
    confidence,
    expectedPrice,
    expectedChangePct,
    volatilityEst: vol,
  };
}
