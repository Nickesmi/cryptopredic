/**
 * Next.js API proxy for Binance US klines (OHLCV).
 *
 * Binance.us is accessible without geo-restrictions and uses the
 * exact same API format as Binance global.
 *
 * GET /api/klines?symbol=BTCUSDT&timeframe=1H&limit=500
 *
 * Timeframe → Binance interval mapping:
 *   1m→1m, 5m→5m, 15m→15m, 30m→30m, 1H→1h, 4H→4h, 1D→1d, 1W→1w
 */

import { type NextRequest, NextResponse } from 'next/server';

const BINANCE_US_BASE = 'https://api.binance.us';

const TF_MAP: Record<string, string> = {
  '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
  '1H': '1h', '4H': '4h', '1D': '1d', '1W': '1w',
};

export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl;
  const symbol    = searchParams.get('symbol');
  const timeframe = searchParams.get('timeframe');
  const limit     = parseInt(searchParams.get('limit') ?? '500', 10);

  if (!symbol || !timeframe) {
    return NextResponse.json(
      { error: 'Missing required params: symbol, timeframe' },
      { status: 400 },
    );
  }

  const interval = TF_MAP[timeframe];
  if (!interval) {
    return NextResponse.json(
      { error: `Unsupported timeframe "${timeframe}". Supported: ${Object.keys(TF_MAP).join(', ')}` },
      { status: 400 },
    );
  }

  const url = new URL(`${BINANCE_US_BASE}/api/v3/klines`);
  url.searchParams.set('symbol',   symbol.toUpperCase());
  url.searchParams.set('interval', interval);
  url.searchParams.set('limit',    String(Math.min(limit, 1000)));

  try {
    const res = await fetch(url.toString(), { cache: 'no-store' });
    if (!res.ok) {
      const body = await res.text();
      return NextResponse.json(
        { error: `Binance US error ${res.status}: ${body}` },
        { status: res.status },
      );
    }

    // Binance format: [openTime_ms, open, high, low, close, volume, ...]
    // Return array as-is — the client parses it the same way
    const data = await res.json();
    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'public, max-age=5, stale-while-revalidate=10' },
    });
  } catch (err) {
    console.error('[/api/klines] Binance US proxy error:', err);
    return NextResponse.json({ error: 'Failed to reach Binance US API' }, { status: 502 });
  }
}
