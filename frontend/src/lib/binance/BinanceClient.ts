/**
 * Binance candle client — calls the local Next.js proxy route.
 *
 * The browser cannot call api.binance.com directly (CORS).
 * Instead, it calls /api/klines which fetches server-side.
 * WebSocket still connects directly — no CORS for WSS.
 */


export interface CandleData {
  time: number;   // Unix seconds (UTC)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

/**
 * Fetch historical OHLCV candles via the Next.js server proxy.
 *
 * @param symbol    e.g. "BTCUSDT"
 * @param timeframe e.g. "1H"
 * @param limit     number of candles (max 1000)
 */
export async function fetchBinanceCandles(
  symbol: string,
  timeframe: string,
  limit = 500,
): Promise<CandleData[]> {
  if (!timeframe) throw new Error('Timeframe is required');

  const url = new URL('/api/klines', window.location.origin);
  url.searchParams.set('symbol',    symbol.toUpperCase());
  url.searchParams.set('timeframe', timeframe);    // canonical e.g. "1H" — proxy maps to Bybit
  url.searchParams.set('limit',     String(Math.min(limit, 1000)));

  const res = await fetch(url.toString());
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Klines proxy error ${res.status}: ${body}`);
  }

  // Binance klines format: [openTime, open, high, low, close, volume, ...]
  const raw = (await res.json()) as Array<[number, string, string, string, string, string, ...unknown[]]>;

  return raw.map(([openTime, open, high, low, close, volume]) => ({
    time:   Math.floor(openTime / 1000),
    open:   parseFloat(open),
    high:   parseFloat(high),
    low:    parseFloat(low),
    close:  parseFloat(close),
    volume: parseFloat(volume),
  }));
}
