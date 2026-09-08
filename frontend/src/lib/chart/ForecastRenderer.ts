/**
 * ForecastRenderer — orchestrates all IChartOverlay instances.
 *
 * This is the single entry point for rendering a ForecastObject.
 * It delegates to registered overlays; it knows nothing about AI models,
 * candle streams, or React components.
 *
 * Responsibilities:
 * - Maintain the registry of active overlays
 * - Expose `render(forecast)`: atomic update of all overlays
 * - Expose `slide()`: shift all overlays forward by one candle
 * - Expose `clear()`: remove all overlays from the chart
 * - Enable/disable individual overlays by ID
 *
 * Adding a new overlay:
 *   1. Create a class implementing IChartOverlay
 *   2. Call renderer.registerOverlay(new MyOverlay(adapter))
 *   → Zero changes to this file (OCP)
 */

import type { IChartOverlay } from './overlays/IChartOverlay';
import type { ForecastObject } from '../../types/chart';

export class ForecastRenderer {
  private overlays = new Map<string, IChartOverlay>();

  /** Register an overlay. Replaces any existing overlay with the same ID. */
  registerOverlay(overlay: IChartOverlay): this {
    this.overlays.set(overlay.id, overlay);
    return this;
  }

  /** Remove a specific overlay from the chart and the registry. */
  unregisterOverlay(id: string): void {
    const overlay = this.overlays.get(id);
    if (overlay) {
      overlay.clear();
      this.overlays.delete(id);
    }
  }

  /**
   * Atomically render all overlays with the new forecast.
   * Clears all existing overlay data first, then redraws.
   */
  render(forecast: ForecastObject): void {
    for (const overlay of this.overlays.values()) {
      if (overlay.isVisible) overlay.render(forecast);
    }
  }

  /**
   * Slide all overlays forward by one candle.
   * Called on every closed candle before the new forecast arrives.
   */
  slide(): void {
    for (const overlay of this.overlays.values()) {
      if (overlay.isVisible) overlay.slide();
    }
  }

  /** Remove all overlay series from the chart. */
  clear(): void {
    for (const overlay of this.overlays.values()) {
      overlay.clear();
    }
  }

  /** Enable or disable a specific overlay by ID. */
  setOverlayVisible(id: string, visible: boolean): void {
    this.overlays.get(id)?.setVisible(visible);
  }

  /** List all registered overlays with their visibility state. */
  getOverlays(): Array<{ id: string; displayName: string; isVisible: boolean }> {
    return Array.from(this.overlays.values()).map((o) => ({
      id: o.id,
      displayName: o.displayName,
      isVisible: o.isVisible,
    }));
  }
}
