/**
 * ChartAdapter — wraps the lightweight-charts library.
 *
 * Provides a typed facade over the raw `createChart()` API so that
 * the rest of the chart layer never imports from 'lightweight-charts'
 * directly. Swapping the charting library requires changing only this file.
 *
 * Responsibilities:
 * - Create and configure the chart instance (dark theme)
 * - Manage the candlestick series (historical data)
 * - Handle responsive resizing via ResizeObserver
 * - Expose typed methods for adding series (used by overlays)
 */

import {
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickSeriesOptions,
  type LineSeriesOptions,
  type AreaSeriesOptions,
  type DeepPartial,
  type ChartOptions,
  ColorType,
} from 'lightweight-charts';

import type { CandleBar } from '../../types/chart';

/** Dark theme matching the app's design system. */
const DARK_THEME: DeepPartial<ChartOptions> = {
  layout: {
    background: { type: ColorType.Solid, color: '#0A0B0F' },
    textColor: '#A6B0C3',
  },
  grid: {
    vertLines: { color: 'rgba(255,255,255,0.04)' },
    horzLines: { color: 'rgba(255,255,255,0.04)' },
  },
  crosshair: {
    vertLine: { color: 'rgba(124,92,255,0.5)', width: 1, style: 1 },
    horzLine: { color: 'rgba(124,92,255,0.5)', width: 1, style: 1 },
  },
  rightPriceScale: {
    borderColor: 'rgba(255,255,255,0.08)',
    textColor: '#6C7486',
  },
  timeScale: {
    borderColor: 'rgba(255,255,255,0.08)',
    timeVisible: true,
    secondsVisible: false,
  },
  handleScroll: { mouseWheel: true, pressedMouseMove: true },
  handleScale: { mouseWheel: true, pinch: true },
};

/** Candlestick colour palette. */
const CANDLE_OPTS: DeepPartial<CandlestickSeriesOptions> = {
  upColor: '#00C389',
  downColor: '#FF5D73',
  borderUpColor: '#00C389',
  borderDownColor: '#FF5D73',
  wickUpColor: '#00C389',
  wickDownColor: '#FF5D73',
};

export class ChartAdapter {
  private chart: IChartApi;
  private candleSeries: ISeriesApi<'Candlestick'>;
  private resizeObserver: ResizeObserver;
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
    this.chart = createChart(container, {
      ...DARK_THEME,
      width: container.clientWidth,
      height: container.clientHeight,
      autoSize: false,
    });

    this.candleSeries = this.chart.addCandlestickSeries(CANDLE_OPTS);

    // Responsive resize
    this.resizeObserver = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      this.chart.resize(width, height);
    });
    this.resizeObserver.observe(container);
  }

  // ── Historical candles ─────────────────────────────────────────────────────

  /** Replace the entire historical dataset. */
  setCandles(candles: CandleBar[]): void {
    const data = candles.map((c) => ({
      time: c.time as unknown as import('lightweight-charts').Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    this.candleSeries.setData(data);
    this.chart.timeScale().fitContent();
  }

  /**
   * Update or append a single candle (live tick or closed candle).
   * lightweight-charts handles deduplication by timestamp.
   */
  updateCandle(candle: CandleBar): void {
    this.candleSeries.update({
      time: candle.time as unknown as import('lightweight-charts').Time,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    });
  }

  // ── Series factory methods (used by overlays) ──────────────────────────────

  addLineSeries(options?: DeepPartial<LineSeriesOptions>) {
    return this.chart.addLineSeries(options);
  }

  addAreaSeries(options?: DeepPartial<AreaSeriesOptions>) {
    return this.chart.addAreaSeries(options);
  }

  removeSeries(series: ISeriesApi<'Line'> | ISeriesApi<'Area'>): void {
    this.chart.removeSeries(series);
  }

  // ── Chart controls ─────────────────────────────────────────────────────────

  scrollToRealtime(): void {
    this.chart.timeScale().scrollToRealTime();
  }

  fitContent(): void {
    this.chart.timeScale().fitContent();
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────

  destroy(): void {
    this.resizeObserver.disconnect();
    this.chart.remove();
  }
}
