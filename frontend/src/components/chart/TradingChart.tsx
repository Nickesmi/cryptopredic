'use client';

/**
 * TradingChart — professional TradingView Lightweight Chart.
 *
 * Self-contained: fetches real Binance candles, streams live updates
 * via Binance WebSocket, runs client-side linear-regression forecasts,
 * and renders everything in a single lightweight-charts instance.
 *
 * Zero backend required — works immediately with `npm run dev`.
 *
 * What is rendered:
 *   1. Candlestick series  — real historical OHLCV from Binance
 *   2. Volume histogram    — coloured green/red per candle direction
 *   3. Forecast line       — dashed purple line starting at last real candle
 *   4. Upper CI band       — dotted line above forecast
 *   5. Lower CI band       — dotted line below forecast
 */

import { useEffect, useRef, useCallback } from 'react';
import {
  createChart,
  CrosshairMode,
  LineStyle,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type Time,
  type CandlestickData,
  type HistogramData,
  type LineData,
} from 'lightweight-charts';

import {
  fetchBinanceCandles,
  type CandleData,
} from '../../lib/binance/BinanceClient';
import {
  BinanceWebSocket,
  type LiveCandle,
} from '../../lib/binance/BinanceWebSocket';
import { PredictionManager } from '../../lib/prediction/PredictionManager';
import { timeframeToSeconds, type ForecastResult } from '../../lib/prediction/ClientSidePredictor';
import type { Timeframe, SupportedSymbol, PredictionHorizon } from '../../types/chart';

// ─── Props ─────────────────────────────────────────────────────────────────

export interface TradingChartProps {
  symbol: SupportedSymbol;
  timeframe: Timeframe;
  predictionHorizon: PredictionHorizon;
  onForecastUpdate: (forecast: ForecastResult | null) => void;
  onStatusChange?: (status: 'loading' | 'live' | 'disconnected') => void;
}

// ─── Design tokens ──────────────────────────────────────────────────────────

const UP_COLOR   = '#00C389';
const DOWN_COLOR = '#FF5D73';
const FORECAST_COLOR = 'rgba(124, 92, 255, 0.95)';
const BAND_COLOR     = 'rgba(124, 92, 255, 0.40)';
const BG_COLOR   = '#0A0B0F';
const TEXT_COLOR = '#A6B0C3';
const MUTED      = '#6C7486';
const GRID       = 'rgba(255,255,255,0.04)';
const XHAIR      = 'rgba(124,92,255,0.50)';

// ─── Component ──────────────────────────────────────────────────────────────

export function TradingChart({
  symbol,
  timeframe,
  predictionHorizon,
  onForecastUpdate,
  onStatusChange,
}: TradingChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Chart object refs — mutations never trigger React re-renders
  const chartRef        = useRef<IChartApi | null>(null);
  const candleSerRef    = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSerRef    = useRef<ISeriesApi<'Histogram'> | null>(null);
  const fcastLineRef    = useRef<ISeriesApi<'Line'> | null>(null);
  const upperBandRef    = useRef<ISeriesApi<'Line'> | null>(null);
  const lowerBandRef    = useRef<ISeriesApi<'Line'> | null>(null);

  // Live state refs — readable inside callbacks without stale closure
  const candlesRef   = useRef<CandleData[]>([]);
  const wsRef        = useRef<BinanceWebSocket | null>(null);
  const rafRef       = useRef<number | null>(null);
  const activeRef    = useRef(false);
  const horizonRef   = useRef(predictionHorizon);
  const tfRef        = useRef(timeframe);
  const cbRef        = useRef(onForecastUpdate);
  const statusCbRef  = useRef(onStatusChange);

  // Keep callback refs current without re-creating effects
  useEffect(() => { horizonRef.current  = predictionHorizon; }, [predictionHorizon]);
  useEffect(() => { tfRef.current       = timeframe; },        [timeframe]);
  useEffect(() => { cbRef.current       = onForecastUpdate; }, [onForecastUpdate]);
  useEffect(() => { statusCbRef.current = onStatusChange; },   [onStatusChange]);

  const isPredictingRef = useRef(false);

  // ── Prediction runner (stable — uses only refs) ─────────────────────────
  const runPrediction = useCallback(async (candles: CandleData[]) => {
    if (candles.length < 10) return;
    if (isPredictingRef.current) return;
    
    try {
      isPredictingRef.current = true;
      const fc = await PredictionManager.getPrediction(
        candles,
        horizonRef.current,
        timeframeToSeconds(tfRef.current),
        symbol,
        tfRef.current,
        'auto'
      );

      if (!activeRef.current) return;

      const anchor: LineData = {
        time: fc.anchorTime as Time,
        value: fc.anchorPrice,
      };

      fcastLineRef.current?.setData([
        anchor,
        ...fc.line.map(p => ({ time: p.time as Time, value: p.value })),
      ]);
      upperBandRef.current?.setData([
        anchor,
        ...fc.upper.map(p => ({ time: p.time as Time, value: p.value })),
      ]);
      lowerBandRef.current?.setData([
        anchor,
        ...fc.lower.map(p => ({ time: p.time as Time, value: p.value })),
      ]);

      cbRef.current(fc);
    } catch (e) {
      console.error('[TradingChart] prediction error:', e);
    } finally {
      isPredictingRef.current = false;
    }
  }, [symbol]); // stable — all other deps via refs

  // ── Re-forecast when horizon changes without rebuilding chart ───────────
  useEffect(() => {
    if (candlesRef.current.length > 10) {
      runPrediction(candlesRef.current);
    }
  }, [predictionHorizon, runPrediction]);

  // ── Main effect: build chart, load data, connect stream ─────────────────
  // Recreates when symbol or timeframe changes (correct behaviour).
  useEffect(() => {
    if (!containerRef.current) return;

    activeRef.current = true;
    statusCbRef.current?.('loading');

    // ── 1. Create chart ───────────────────────────────────────────────────
    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: BG_COLOR },
        textColor: TEXT_COLOR,
        fontSize: 12,
        fontFamily: "'Inter', system-ui, sans-serif",
      },
      grid: {
        vertLines: { color: GRID },
        horzLines: { color: GRID },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: XHAIR,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#7C5CFF',
        },
        horzLine: {
          color: XHAIR,
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: '#7C5CFF',
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        textColor: MUTED,
        scaleMargins: { top: 0.06, bottom: 0.22 },
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.06)',
        timeVisible: true,
        secondsVisible: false,
        fixRightEdge: false,
        fixLeftEdge: false,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        mouseWheel: true,
        pinch: true,
        axisPressedMouseMove: { time: true, price: true },
      },
    });
    chartRef.current = chart;

    // ── 2. Candlestick series ─────────────────────────────────────────────
    const candleSer = chart.addCandlestickSeries({
      upColor:        UP_COLOR,
      downColor:      DOWN_COLOR,
      borderUpColor:  UP_COLOR,
      borderDownColor: DOWN_COLOR,
      wickUpColor:    UP_COLOR,
      wickDownColor:  DOWN_COLOR,
      priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
    });
    candleSerRef.current = candleSer;

    // ── 3. Volume histogram (occupies bottom 20%) ─────────────────────────
    const volumeSer = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: 'vol',
    });
    chart.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.80, bottom: 0 },
      visible: false,
    });
    volumeSerRef.current = volumeSer;

    // ── 4. Forecast line (dashed purple) ──────────────────────────────────
    const fcastLine = chart.addLineSeries({
      color: FORECAST_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerRadius: 5,
      crosshairMarkerBorderColor: FORECAST_COLOR,
      crosshairMarkerBackgroundColor: BG_COLOR,
    });
    fcastLineRef.current = fcastLine;

    // ── 5. Upper confidence band ──────────────────────────────────────────
    const upperBand = chart.addLineSeries({
      color: BAND_COLOR,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    upperBandRef.current = upperBand;

    // ── 6. Lower confidence band ──────────────────────────────────────────
    const lowerBand = chart.addLineSeries({
      color: BAND_COLOR,
      lineWidth: 1,
      lineStyle: LineStyle.Dotted,
      lastValueVisible: false,
      priceLineVisible: false,
      crosshairMarkerVisible: false,
    });
    lowerBandRef.current = lowerBand;

    // ── 7. Load history + stream ──────────────────────────────────────────
    let ws: BinanceWebSocket | null = null;

    const toCandle = (c: CandleData): CandlestickData => ({
      time:  c.time as Time,
      open:  c.open,
      high:  c.high,
      low:   c.low,
      close: c.close,
    });

    const toVolume = (c: CandleData): HistogramData => ({
      time:  c.time as Time,
      value: c.volume,
      color: c.close >= c.open
        ? 'rgba(0,195,137,0.45)'
        : 'rgba(255,93,115,0.45)',
    });

    const init = async () => {
      try {
        const candles = await fetchBinanceCandles(symbol, timeframe, 500);
        if (!activeRef.current) return;

        candlesRef.current = candles;
        candleSer.setData(candles.map(toCandle));
        volumeSer.setData(candles.map(toVolume));
        chart.timeScale().scrollToRealTime();

        runPrediction(candles);

        // Live stream
        ws = new BinanceWebSocket(symbol, timeframe);
        wsRef.current = ws;

        ws.onStatus(s => {
          if (!activeRef.current) return;
          statusCbRef.current?.(
            s === 'connected' ? 'live' : s === 'connecting' ? 'loading' : 'disconnected',
          );
        });

        ws.onCandle((live: LiveCandle) => {
          if (!activeRef.current) return;
          if (rafRef.current) cancelAnimationFrame(rafRef.current);

          rafRef.current = requestAnimationFrame(() => {
            // Update/append live candle
            candleSerRef.current?.update({
              time:  live.time as Time,
              open:  live.open,
              high:  live.high,
              low:   live.low,
              close: live.close,
            });
            volumeSerRef.current?.update({
              time:  live.time as Time,
              value: live.volume,
              color: live.close >= live.open
                ? 'rgba(0,195,137,0.45)'
                : 'rgba(255,93,115,0.45)',
            });

            if (live.isClosed) {
              // Merge closed candle into history
              const arr = candlesRef.current;
              const idx = arr.findIndex(c => c.time === live.time);
              const updated: CandleData = {
                time:   live.time,
                open:   live.open,
                high:   live.high,
                low:    live.low,
                close:  live.close,
                volume: live.volume,
              };
              if (idx >= 0) {
                arr[idx] = updated;
              } else {
                candlesRef.current = [...arr, updated];
              }
              runPrediction(candlesRef.current);
            }
          });
        });

        ws.connect();
      } catch (err) {
        console.error('[TradingChart] init error:', err);
        statusCbRef.current?.('disconnected');
      }
    };

    void init();

    // ── Cleanup ───────────────────────────────────────────────────────────
    return () => {
      activeRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      ws?.disconnect();
      wsRef.current     = null;
      candlesRef.current = [];
      // Null out series refs before removing chart
      candleSerRef.current  = null;
      volumeSerRef.current  = null;
      fcastLineRef.current  = null;
      upperBandRef.current  = null;
      lowerBandRef.current  = null;
      chart.remove();
      chartRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbol, timeframe]); // intentional: runPrediction is stable via useCallback

  return (
    <div
      ref={containerRef}
      id="trading-chart-container"
      className="w-full h-full"
      aria-label={`${symbol} ${timeframe} live candlestick chart with AI prediction overlay`}
    />
  );
}
