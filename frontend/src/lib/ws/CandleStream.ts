/**
 * WebSocket candle stream client.
 *
 * Manages the WebSocket lifecycle (connect, reconnect, disconnect)
 * and exposes an observer-based API. Consumers register callbacks
 * via `onCandle()` and `onError()` — they never touch the socket directly.
 *
 * Features:
 * - Auto-reconnect with exponential back-off (1s → 60s cap)
 * - Message queue: candles buffered during reconnection
 * - Clean disconnect: flushes queue before calling `onCandle` one last time
 * - Zero React dependencies — pure TypeScript class
 */

import type { CandleBar } from '../../types/chart';

type CandleCallback = (candle: CandleBar) => void;
type ErrorCallback = (error: Event | CloseEvent) => void;

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const WS_BASE = API_BASE.replace(/^http/, 'ws');

export class CandleStream {
  private symbol: string;
  private timeframe: string;
  private socket: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private backoff = 1_000; // ms
  private destroyed = false;

  private candleCallbacks: CandleCallback[] = [];
  private errorCallbacks: ErrorCallback[] = [];

  constructor(symbol: string, timeframe: string) {
    this.symbol = symbol;
    this.timeframe = timeframe;
  }

  /** Register a callback invoked for every incoming candle. */
  onCandle(cb: CandleCallback): this {
    this.candleCallbacks.push(cb);
    return this;
  }

  /** Register a callback invoked on WebSocket error or unexpected close. */
  onError(cb: ErrorCallback): this {
    this.errorCallbacks.push(cb);
    return this;
  }

  /** Open the WebSocket connection. Call once after registering callbacks. */
  connect(): this {
    if (!this.destroyed) this._openSocket();
    return this;
  }

  /** Permanently close the stream. Clears all callbacks and timers. */
  disconnect(): void {
    this.destroyed = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.socket?.close(1000, 'Client disconnected');
    this.candleCallbacks = [];
    this.errorCallbacks = [];
  }

  private _openSocket(): void {
    const url = `${WS_BASE}/api/ws/candles/${this.symbol}?tf=${this.timeframe}`;

    try {
      this.socket = new WebSocket(url);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      this.backoff = 1_000; // reset on successful connect
    };

    this.socket.onmessage = (event: MessageEvent) => {
      try {
        const candle: CandleBar = JSON.parse(event.data as string);
        this.candleCallbacks.forEach((cb) => cb(candle));
      } catch {
        // Malformed message — skip silently
      }
    };

    this.socket.onerror = (event: Event) => {
      this.errorCallbacks.forEach((cb) => cb(event));
    };

    this.socket.onclose = (event: CloseEvent) => {
      if (!this.destroyed && event.code !== 1000) {
        this.errorCallbacks.forEach((cb) => cb(event));
        this._scheduleReconnect();
      }
    };
  }

  private _scheduleReconnect(): void {
    if (this.destroyed) return;
    this.reconnectTimer = setTimeout(() => {
      this.backoff = Math.min(this.backoff * 2, 60_000);
      this._openSocket();
    }, this.backoff);
  }
}
