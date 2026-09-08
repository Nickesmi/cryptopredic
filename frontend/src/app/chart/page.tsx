'use client';

/**
 * Live Chart page — the primary visualization in the application.
 *
 * Dynamic imports TradingChart with ssr:false because lightweight-charts
 * uses browser DOM APIs at import time, which would fail during SSR.
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │  Toolbar: symbol | timeframes | horizon | refresh | status  │
 *   ├────────────────────────────────────────────┬─────────────────┤
 *   │                                            │                 │
 *   │          TradingChart (fills space)        │  Prediction     │
 *   │                                            │  Panel          │
 *   │                                            │                 │
 *   └────────────────────────────────────────────┴─────────────────┘
 */

import dynamic from 'next/dynamic';
import { useState, useCallback } from 'react';
import { RefreshCw, Activity, WifiOff, Loader2 } from 'lucide-react';

import { TimeframeSelector }       from '../../components/chart/TimeframeSelector';
import { SymbolSelector }          from '../../components/chart/SymbolSelector';
import { PredictionHorizonSelector } from '../../components/chart/PredictionHorizonSelector';
import { PredictionPanel }         from '../../components/chart/PredictionPanel';

import type {
  Timeframe,
  SupportedSymbol,
  PredictionHorizon,
} from '../../types/chart';
import type { ForecastResult }     from '../../lib/prediction/ClientSidePredictor';
import type { TradingChartProps }  from '../../components/chart/TradingChart';

// ── Dynamic import — no SSR ─────────────────────────────────────────────────
const TradingChart = dynamic<TradingChartProps>(
  () =>
    import('../../components/chart/TradingChart').then(
      m => ({ default: m.TradingChart }),
    ),
  {
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center w-full h-full bg-[#0A0B0F]">
        <div className="flex flex-col items-center gap-3 text-muted">
          <Loader2 className="animate-spin text-premium" size={32} />
          <span className="text-sm font-medium tracking-wider uppercase">
            Loading Chart Engine…
          </span>
        </div>
      </div>
    ),
  },
);

// ─────────────────────────────────────────────────────────────────────────────

type ConnectionStatus = 'loading' | 'live' | 'disconnected';

export default function ChartPage() {
  const [symbol,    setSymbol]    = useState<SupportedSymbol>('BTCUSDT');
  const [timeframe, setTimeframe] = useState<Timeframe>('1H');
  const [horizon,   setHorizon]   = useState<PredictionHorizon>(20);
  const [forecast,  setForecast]  = useState<ForecastResult | null>(null);
  const [status,    setStatus]    = useState<ConnectionStatus>('loading');
  const [chartKey,  setChartKey]  = useState(0); // bump to force remount

  // Stable callbacks (don't change on re-render)
  const handleForecastUpdate = useCallback(
    (fc: ForecastResult | null) => setForecast(fc),
    [],
  );
  const handleStatusChange = useCallback(
    (s: ConnectionStatus) => setStatus(s),
    [],
  );

  const handleRefresh = () => {
    setForecast(null);
    setStatus('loading');
    setChartKey(k => k + 1);
  };

  const handleSymbolChange = (s: SupportedSymbol) => {
    setSymbol(s);
    setForecast(null);
    setStatus('loading');
  };

  const handleTimeframeChange = (tf: Timeframe) => {
    setTimeframe(tf);
    setForecast(null);
    setStatus('loading');
  };

  return (
    <div className="flex flex-col h-full gap-3 animate-in fade-in duration-500">

      {/* ── Toolbar ─────────────────────────────────────────────────────── */}
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-3 bg-surface border border-[#ffffff0f] rounded-xl px-4 py-3">
        {/* Left: symbol + timeframes */}
        <div className="flex items-center gap-3 flex-wrap">
          <SymbolSelector selected={symbol} onSelect={handleSymbolChange} />
          <div className="h-5 w-px bg-[#ffffff15] hidden sm:block" />
          <TimeframeSelector selected={timeframe} onSelect={handleTimeframeChange} />
        </div>

        {/* Right: horizon + refresh + status */}
        <div className="flex items-center gap-3 flex-wrap">
          <PredictionHorizonSelector selected={horizon} onSelect={setHorizon} />
          <div className="h-5 w-px bg-[#ffffff15] hidden sm:block" />

          {/* Refresh */}
          <button
            id="chart-refresh-btn"
            onClick={handleRefresh}
            title="Reload chart and prediction"
            className="flex items-center gap-1.5 text-xs font-semibold text-secondary hover:text-primary transition-colors px-3 py-1.5 bg-[#ffffff05] border border-[#ffffff0a] rounded-lg hover:border-premium/30"
          >
            <RefreshCw size={13} />
            <span className="hidden sm:inline">Refresh</span>
          </button>

          {/* Live status pill */}
          <StatusPill status={status} />
        </div>
      </div>

      {/* ── Main workspace ───────────────────────────────────────────────── */}
      <div className="flex flex-1 min-h-0 gap-3">

        {/* Chart (fills remaining width) */}
        <div className="flex-1 min-w-0 bg-[#0A0B0F] border border-[#ffffff0f] rounded-xl overflow-hidden relative shadow-2xl">

          {/* Symbol watermark */}
          <div className="absolute top-3 left-4 z-10 pointer-events-none select-none">
            <span className="text-xs font-bold text-[#ffffff10] tracking-widest uppercase">
              {symbol} · {timeframe}
            </span>
          </div>

          {/* Prediction label overlaid on chart */}
          {forecast && (
            <div
              className="absolute top-3 right-4 z-10 flex items-center gap-2 px-2.5 py-1 rounded-md bg-premium/15 border border-premium/30 text-[11px] font-bold text-premium pointer-events-none select-none"
              aria-label="AI prediction active"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-premium animate-pulse" />
              AI Forecast Active
            </div>
          )}

          <TradingChart
            key={`${chartKey}-${symbol}-${timeframe}`}
            symbol={symbol}
            timeframe={timeframe}
            predictionHorizon={horizon}
            onForecastUpdate={handleForecastUpdate}
            onStatusChange={handleStatusChange}
          />
        </div>

        {/* Right sidebar — prediction analytics */}
        <div className="w-72 shrink-0 flex flex-col gap-3 overflow-y-auto">
          <div className="luxury-card !p-5 border-premium/20 bg-premium/5">
            <h3 className="text-xs font-bold text-muted uppercase tracking-widest mb-4 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-premium shadow-[0_0_8px_rgba(124,92,255,0.8)]" />
              AI Prediction Engine
            </h3>
            <PredictionPanel
              forecast={forecast}
              isLoading={status === 'loading'}
              connectionStatus={status}
            />
          </div>

          {/* Static market context card */}
          <div className="luxury-card !p-5">
            <h3 className="text-xs font-bold text-muted uppercase tracking-widest mb-3">
              Chart Legend
            </h3>
            <div className="space-y-2.5 text-xs">
              <LegendItem color="bg-bullish" label="Bullish candle" />
              <LegendItem color="bg-bearish" label="Bearish candle" />
              <LegendItem color="bg-premium" dashed label="AI Forecast line" />
              <LegendItem
                color="bg-premium opacity-40"
                dotted
                label="Confidence band"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────────────

function StatusPill({ status }: { status: ConnectionStatus }) {
  if (status === 'live') {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-bullish/10 border border-bullish/25 rounded-full text-xs font-bold text-bullish">
        <Activity size={12} className="animate-pulse" />
        <span>LIVE</span>
      </div>
    );
  }
  if (status === 'loading') {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-neutral/10 border border-neutral/25 rounded-full text-xs font-bold text-neutral">
        <Loader2 size={12} className="animate-spin" />
        <span>Connecting</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-1.5 px-3 py-1.5 bg-bearish/10 border border-bearish/25 rounded-full text-xs font-bold text-bearish">
      <WifiOff size={12} />
      <span>Offline</span>
    </div>
  );
}

function LegendItem({
  color,
  label,
  dashed,
  dotted,
}: {
  color: string;
  label: string;
  dashed?: boolean;
  dotted?: boolean;
}) {
  return (
    <div className="flex items-center gap-2.5 text-secondary">
      <div className="w-10 flex items-center justify-center shrink-0">
        {dashed || dotted ? (
          <div
            className={`w-full h-px border-t-2 ${
              dashed ? 'border-dashed' : 'border-dotted'
            } border-premium`}
          />
        ) : (
          <div className={`w-full h-2 rounded-sm ${color}`} />
        )}
      </div>
      <span>{label}</span>
    </div>
  );
}
