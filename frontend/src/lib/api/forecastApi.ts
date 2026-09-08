/**
 * Typed REST API client for the Crypto Alpha Engine backend.
 *
 * All functions are pure async — no side effects, no state.
 * The chart layer calls these; it never uses fetch directly.
 */

import type { CandleBar, ForecastObject, Timeframe, ModelName } from '../../types/chart';

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

/** Generic JSON fetcher with typed return. */
async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/** Fetch historical candles for a symbol + timeframe. */
export async function fetchCandles(
  symbol: string,
  timeframe: Timeframe,
  limit = 500,
): Promise<CandleBar[]> {
  const data = await apiFetch<{ candles: CandleBar[] }>(
    `/api/candles/${symbol}?tf=${timeframe}&limit=${limit}`,
  );
  return data.candles;
}

/** Fetch an AI forecast with confidence bands. */
export async function fetchForecast(
  symbol: string,
  timeframe: Timeframe,
  nCandles: number,
  model: ModelName = 'auto',
): Promise<ForecastObject> {
  return apiFetch<ForecastObject>(
    `/api/predict/${symbol}?tf=${timeframe}&n=${nCandles}&model=${model}`,
  );
}

/** Fetch list of available AI model names. */
export async function fetchAvailableModels(): Promise<string[]> {
  const data = await apiFetch<{ models: string[] }>('/api/models');
  return data.models;
}

/** Check API health. */
export async function checkHealth(): Promise<boolean> {
  try {
    const data = await apiFetch<{ status: string }>('/api/health');
    return data.status === 'ok';
  } catch {
    return false;
  }
}
