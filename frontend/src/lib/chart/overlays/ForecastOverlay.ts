/**
 * ForecastOverlay — renders the central AI prediction line.
 *
 * Implements IChartOverlay.
 *
 * Visual design:
 * - Starts exactly at the anchor (last historical candle close)
 * - Dashed purple/gradient line extending into the future
 * - Opacity gradient: full at anchor → 50% at furthest projection
 * - Clearly distinguishable from historical candlesticks
 *
 * The overlay never modifies historical candles.
 */

import type { ISeriesApi } from 'lightweight-charts';
import type { IChartOverlay } from './IChartOverlay';
import type { ChartAdapter } from '../ChartAdapter';
import type { ForecastObject } from '../../../types/chart';

// Bullish purple prediction line
const BULLISH_COLOR = 'rgba(124, 92, 255, 0.9)';
const BEARISH_COLOR = 'rgba(255, 93, 115, 0.9)';

export class ForecastOverlay implements IChartOverlay {
  readonly id = 'forecast-line';
  readonly displayName = 'AI Forecast';
  isVisible = true;

  private adapter: ChartAdapter;
  private lineSeries: ISeriesApi<'Line'> | null = null;
  private slideOffset = 0;
  private lastForecast: ForecastObject | null = null;

  constructor(adapter: ChartAdapter) {
    this.adapter = adapter;
  }

  render(forecast: ForecastObject): void {
    this.lastForecast = forecast;
    this.slideOffset = 0;
    this._draw(forecast);
  }

  slide(): void {
    this.slideOffset += 1;
    if (this.lastForecast) this._draw(this.lastForecast);
  }

  clear(): void {
    if (this.lineSeries) {
      this.adapter.removeSeries(this.lineSeries);
      this.lineSeries = null;
    }
  }

  setVisible(visible: boolean): void {
    this.isVisible = visible;
    if (!visible) {
      this.clear();
    } else if (this.lastForecast) {
      this.render(this.lastForecast);
    }
  }

  private _draw(forecast: ForecastObject): void {
    if (!this.isVisible) return;

    const color =
      forecast.direction === 'bullish' ? BULLISH_COLOR : BEARISH_COLOR;

    if (!this.lineSeries) {
      this.lineSeries = this.adapter.addLineSeries({
        color,
        lineWidth: 2,
        lineStyle: 2, // dashed
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: color,
        crosshairMarkerBackgroundColor: '#0A0B0F',
      });
    } else {
      // Update colour if direction changed
      this.lineSeries.applyOptions({ color });
    }

    const offset = this.slideOffset;
    const points = forecast.candles.slice(offset);
    if (!points.length) return;

    // Anchor point connects the prediction to the last historical candle
    const data = [
      {
        time: forecast.anchor_time as unknown as import('lightweight-charts').Time,
        value: forecast.anchor_price,
      },
      ...points.map((p) => ({
        time: p.time as unknown as import('lightweight-charts').Time,
        value: p.value,
      })),
    ];

    this.lineSeries.setData(data);
  }
}
