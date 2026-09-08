/**
 * IChartOverlay — the interface all chart overlays must implement.
 *
 * Adding a new overlay type (buy zone, support/resistance, risk area,
 * stop-loss, take-profit, AI label) is a matter of implementing this
 * interface and registering the overlay — zero changes to existing code
 * (Open/Closed Principle).
 */

import type { ForecastObject } from '../../../types/chart';

export interface IChartOverlay {
  /** Unique identifier for this overlay instance. */
  readonly id: string;

  /** Human-readable display name shown in the overlay controls UI. */
  readonly displayName: string;

  /**
   * Render or update the overlay using the latest forecast data.
   * Called whenever the prediction changes.
   */
  render(forecast: ForecastObject): void;

  /**
   * Slide the overlay forward by one candle period.
   * Called when a new closed candle arrives — shifts prediction forward
   * WITHOUT regenerating it (until the debounced refresh fires).
   */
  slide(): void;

  /**
   * Remove all series this overlay added to the chart.
   * Must leave the chart in a clean state.
   */
  clear(): void;

  /** Whether this overlay is currently visible. */
  isVisible: boolean;

  /** Show or hide this overlay. */
  setVisible(visible: boolean): void;
}
