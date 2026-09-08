/**
 * ConfidenceBandOverlay — renders the upper and lower CI bands.
 *
 * Implements IChartOverlay.
 *
 * Visual design:
 * - Upper band: faint purple dashed line
 * - Lower band: faint purple dashed line
 * - Fill between them: semi-transparent purple area
 *
 * The fill is achieved by stacking two AreaSeries:
 * 1. Upper area series (fills down from upper band)
 * 2. A "mask" area that fills the lower portion white — creating the
 *    appearance of a shaded region between the two bands.
 *
 * Note: lightweight-charts v4 does not natively support "fill between
 * two series", so we use a transparent topColor + visible line approach
 * on both series to simulate the band fill.
 */

import type { ISeriesApi } from 'lightweight-charts';
import type { IChartOverlay } from './IChartOverlay';
import type { ChartAdapter } from '../ChartAdapter';
import type { ForecastObject } from '../../../types/chart';

// Purple with low opacity for band fill
const BAND_FILL = 'rgba(124, 92, 255, 0.08)';
const BAND_LINE = 'rgba(124, 92, 255, 0.35)';

export class ConfidenceBandOverlay implements IChartOverlay {
  readonly id = 'confidence-band';
  readonly displayName = 'Confidence Band';
  isVisible = true;

  private adapter: ChartAdapter;
  private upperSeries: ISeriesApi<'Area'> | null = null;
  private lowerSeries: ISeriesApi<'Area'> | null = null;
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
    if (this.upperSeries) {
      this.adapter.removeSeries(this.upperSeries);
      this.upperSeries = null;
    }
    if (this.lowerSeries) {
      this.adapter.removeSeries(this.lowerSeries);
      this.lowerSeries = null;
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

    const offset = this.slideOffset;
    const upper = forecast.upper_band.slice(offset);
    const lower = forecast.lower_band.slice(offset);
    if (!upper.length || !lower.length) return;

    // Anchor point — connect band to historical data join
    const anchor = {
      time: forecast.anchor_time as unknown as import('lightweight-charts').Time,
      value: forecast.anchor_price,
    };

    const toPoint = (p: { time: number; value: number }) => ({
      time: p.time as unknown as import('lightweight-charts').Time,
      value: p.value,
      topColor: BAND_FILL,
      bottomColor: 'transparent',
      lineColor: BAND_LINE,
    });

    // Create or reuse upper area series
    if (!this.upperSeries) {
      this.upperSeries = this.adapter.addAreaSeries({
        lineWidth: 1,
        lineStyle: 2, // dashed
        topColor: BAND_FILL,
        bottomColor: 'transparent',
        lineColor: BAND_LINE,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      });
    }
    this.upperSeries.setData([anchor, ...upper.map(toPoint)]);

    // Lower band as a line series
    if (!this.lowerSeries) {
      this.lowerSeries = this.adapter.addAreaSeries({
        lineWidth: 1,
        lineStyle: 2,
        topColor: 'transparent',
        bottomColor: 'transparent',
        lineColor: BAND_LINE,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      });
    }
    this.lowerSeries.setData([
      anchor,
      ...lower.map((p) => ({
        time: p.time as unknown as import('lightweight-charts').Time,
        value: p.value,
        topColor: 'transparent',
        bottomColor: 'transparent',
        lineColor: BAND_LINE,
      })),
    ]);
  }
}
