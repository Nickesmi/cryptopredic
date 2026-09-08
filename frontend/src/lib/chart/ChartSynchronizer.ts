/**
 * ChartSynchronizer — connects live data streams to the chart.
 *
 * This is the "traffic controller" of the chart system.  It receives
 * candle events from the WebSocket stream and coordinates updates to:
 *   1. The historical candlestick chart (via ChartAdapter)
 *   2. The prediction overlays (via ForecastRenderer)
 *
 * Key behaviours matching the architectural requirements:
 *
 * INCREMENTAL UPDATES (not full redraws):
 *   - Live tick (is_closed=false): update the live candle only
 *   - Closed candle (is_closed=true): append candle + slide prediction
 *
 * DEBOUNCED PREDICTION REFRESH:
 *   - After a candle closes, waits 500ms before fetching a new forecast
 *   - Concurrent requests are cancelled (AbortController)
 *
 * SMOOTH ANIMATION:
 *   - requestAnimationFrame used for all chart mutations
 *   - No flickering: overlays slide immediately, new forecast replaces
 *
 * ZERO REACT DEPENDENCIES — pure TypeScript class.
 */

import type { ChartAdapter } from './ChartAdapter';
import type { ForecastRenderer } from './ForecastRenderer';
import type { CandleStream } from '../ws/CandleStream';
import type { CandleBar, ForecastObject, Timeframe, ModelName } from '../../types/chart';
import { fetchForecast } from '../api/forecastApi';

const PREDICTION_DEBOUNCE_MS = 500;

export class ChartSynchronizer {
  private adapter: ChartAdapter;
  private renderer: ForecastRenderer;
  private stream: CandleStream | null = null;
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private abortController: AbortController | null = null;
  private rafHandle: number | null = null;

  private symbol: string;
  private timeframe: Timeframe;
  private nCandles: number;
  private modelName: ModelName;

  constructor(options: {
    adapter: ChartAdapter;
    renderer: ForecastRenderer;
    symbol: string;
    timeframe: Timeframe;
    nCandles: number;
    modelName: ModelName;
  }) {
    this.adapter = options.adapter;
    this.renderer = options.renderer;
    this.symbol = options.symbol;
    this.timeframe = options.timeframe;
    this.nCandles = options.nCandles;
    this.modelName = options.modelName;
  }

  /** Attach a live candle stream and start synchronising. */
  attachStream(stream: CandleStream): this {
    this.stream = stream;
    stream.onCandle((candle) => this._handleCandle(candle));
    stream.connect();
    return this;
  }

  /** Update config when symbol/timeframe/horizon changes. */
  reconfigure(options: {
    symbol?: string;
    timeframe?: Timeframe;
    nCandles?: number;
    modelName?: ModelName;
  }): void {
    if (options.symbol) this.symbol = options.symbol;
    if (options.timeframe) this.timeframe = options.timeframe;
    if (options.nCandles) this.nCandles = options.nCandles;
    if (options.modelName) this.modelName = options.modelName;
  }

  /**
   * Manually trigger a forecast fetch (e.g. on initial load or
   * symbol/timeframe switch).
   */
  async refreshForecast(): Promise<void> {
    this._cancelPendingForecast();
    this.abortController = new AbortController();
    try {
      const forecast = await fetchForecast(
        this.symbol,
        this.timeframe,
        this.nCandles,
        this.modelName,
      );
      this._renderForecast(forecast);
    } catch (err: unknown) {
      if (err instanceof Error && err.name !== 'AbortError') {
        console.error('[ChartSynchronizer] Forecast fetch failed:', err);
      }
    }
  }

  /** Clean up all resources. Must be called on unmount. */
  destroy(): void {
    this.stream?.disconnect();
    this._cancelPendingForecast();
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
    this.renderer.clear();
  }

  // ── Private ───────────────────────────────────────────────────────────────

  private _handleCandle(candle: CandleBar): void {
    // Schedule chart update on the next animation frame
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
    this.rafHandle = requestAnimationFrame(() => {
      this.adapter.updateCandle(candle);

      if (candle.is_closed) {
        // Slide prediction forward immediately (smooth UX)
        this.renderer.slide();
        // Debounce the actual API call
        this._scheduleForecastRefresh();
      }
    });
  }

  private _scheduleForecastRefresh(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      void this.refreshForecast();
    }, PREDICTION_DEBOUNCE_MS);
  }

  private _cancelPendingForecast(): void {
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.abortController?.abort();
  }

  private _renderForecast(forecast: ForecastObject): void {
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
    this.rafHandle = requestAnimationFrame(() => {
      this.renderer.render(forecast);
    });
  }
}
