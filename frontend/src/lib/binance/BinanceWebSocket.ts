/**
 * Binance US WebSocket client for live candle streams.
 *
 * Connects to wss://stream.binance.us:9443 — accessible without geo-restrictions.
 * Uses the exact same Binance kline stream format as Binance global.
 *
 * Stream: {symbol}@kline_{interval}
 * Message shape: { k: { t, o, h, l, c, v, x } }
 *   t=openTime(ms), o=open, h=high, l=low, c=close, v=volume, x=isClosed
 *
 * Auto-reconnects with exponential back-off (1s → 60s cap).
 */

const WS_BASE = 'wss://stream.binance.us:9443/ws';

const TF_MAP: Record<string, string> = {
  '1m':  '1m',
  '5m':  '5m',
  '15m': '15m',
  '30m': '30m',
  '1H':  '1h',
  '4H':  '4h',
  '1D':  '1d',
  '1W':  '1w',
};

export interface LiveCandle {
  time: number;    // Unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  isClosed: boolean;
}

type CandleCallback = (candle: LiveCandle) => void;
type StatusCallback = (status: 'connecting' | 'connected' | 'disconnected') => void;

export class BinanceWebSocket {
  private ws: WebSocket | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private backoff = 1_000;
  private destroyed = false;

  private candleCallbacks: CandleCallback[] = [];
  private statusCallbacks: StatusCallback[] = [];

  constructor(
    private readonly symbol: string,
    private readonly timeframe: string,
  ) {}

  onCandle(cb: CandleCallback): this {
    this.candleCallbacks.push(cb);
    return this;
  }

  onStatus(cb: StatusCallback): this {
    this.statusCallbacks.push(cb);
    return this;
  }

  connect(): this {
    if (!this.destroyed) this._open();
    return this;
  }

  disconnect(): void {
    this.destroyed = true;
    if (this.timer) clearTimeout(this.timer);
    if (this.ws) {
      this.ws.onclose = null; // prevent reconnect on intentional close
      this.ws.close(1000, 'Client disconnected');
      this.ws = null;
    }
    this.candleCallbacks = [];
    this.statusCallbacks = [];
  }

  private _emit(status: 'connecting' | 'connected' | 'disconnected') {
    this.statusCallbacks.forEach(cb => cb(status));
  }

  private _open(): void {
    const interval = TF_MAP[this.timeframe];
    if (!interval) return;

    const stream = `${this.symbol.toLowerCase()}@kline_${interval}`;
    const url = `${WS_BASE}/${stream}`;

    this._emit('connecting');

    try {
      this.ws = new WebSocket(url);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this.backoff = 1_000;
      this._emit('connected');
    };

    this.ws.onmessage = (event: MessageEvent<string>) => {
      try {
        const msg = JSON.parse(event.data) as {
          k: {
            t: number; o: string; h: string; l: string;
            c: string; v: string; x: boolean;
          };
        };
        if (!msg.k) return;
        const k = msg.k;
        const candle: LiveCandle = {
          time:     Math.floor(k.t / 1000),
          open:     parseFloat(k.o),
          high:     parseFloat(k.h),
          low:      parseFloat(k.l),
          close:    parseFloat(k.c),
          volume:   parseFloat(k.v),
          isClosed: Boolean(k.x),
        };
        this.candleCallbacks.forEach(cb => cb(candle));
      } catch { /* ignore malformed message */ }
    };

    this.ws.onerror = () => {
      // onerror always followed by onclose — handle reconnect there
    };

    this.ws.onclose = (event: CloseEvent) => {
      this._emit('disconnected');
      if (!this.destroyed && event.code !== 1000) {
        this._scheduleReconnect();
      }
    };
  }

  private _scheduleReconnect(): void {
    if (this.destroyed) return;
    this.timer = setTimeout(() => {
      this.backoff = Math.min(this.backoff * 2, 60_000);
      this._open();
    }, this.backoff);
  }
}
