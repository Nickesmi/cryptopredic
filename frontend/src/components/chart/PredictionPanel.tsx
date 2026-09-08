'use client';

import { TrendingUp, TrendingDown, Clock, Zap, Activity, AlertCircle } from 'lucide-react';
import type { ForecastResult } from '../../lib/prediction/ClientSidePredictor';

interface PredictionPanelProps {
  forecast: ForecastResult | null;
  isLoading: boolean;
  connectionStatus: 'loading' | 'live' | 'disconnected';
}

export function PredictionPanel({
  forecast,
  isLoading,
  connectionStatus,
}: PredictionPanelProps) {
  // ── Loading skeleton ───────────────────────────────────────────────────
  if (isLoading && !forecast) {
    return (
      <div className="space-y-4 animate-pulse">
        <div className="h-16 bg-[#ffffff08] rounded-xl" />
        <div className="h-2 bg-[#ffffff06] rounded-full" />
        <div className="grid grid-cols-2 gap-3">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-14 bg-[#ffffff06] rounded-lg" />
          ))}
        </div>
        <div className="h-8 bg-[#ffffff05] rounded-lg" />
      </div>
    );
  }

  // ── No forecast yet ────────────────────────────────────────────────────
  if (!forecast) {
    return (
      <div className="flex flex-col items-center justify-center h-32 gap-2 text-muted">
        <AlertCircle size={20} />
        <span className="text-sm">Awaiting forecast…</span>
      </div>
    );
  }

  const isBullish      = forecast.direction === 'bullish';
  const signalColor    = isBullish ? 'text-bullish' : 'text-bearish';
  const signalBg       = isBullish
    ? 'bg-bullish/10 border-bullish/25'
    : 'bg-bearish/10 border-bearish/25';
  const signalBarBg    = isBullish ? 'bg-bullish' : 'bg-bearish';
  const changeSign     = forecast.expectedChangePct >= 0 ? '+' : '';
  const confPct        = Math.round(forecast.confidence * 100);
  const changeAbs      = Math.abs(forecast.expectedChangePct).toFixed(2);

  return (
    <div className="space-y-4">
      {/* Direction + confidence badge */}
      <div className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${signalBg}`}>
        <div className={`${signalColor} shrink-0`}>
          {isBullish
            ? <TrendingUp size={24} />
            : <TrendingDown size={24} />
          }
        </div>
        <div>
          <div className={`text-lg font-extrabold capitalize ${signalColor}`}>
            {forecast.direction}
          </div>
          <div className="text-[10px] text-muted uppercase tracking-widest font-bold">
            AI Signal
          </div>
        </div>
        <div className="ml-auto text-right shrink-0">
          <div className="text-xl font-extrabold text-primary">{confPct}%</div>
          <div className="text-[10px] text-muted uppercase tracking-widest">Confidence</div>
        </div>
      </div>

      {/* Confidence bar */}
      <div>
        <div className="flex justify-between text-[11px] text-muted mb-1.5 font-medium">
          <span>Model Confidence</span>
          <span>{confPct}%</span>
        </div>
        <div className="w-full h-1.5 bg-[#ffffff08] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${signalBarBg}`}
            style={{ width: `${confPct}%` }}
          />
        </div>
      </div>

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-2.5">
        <MetricCell
          label="Target Price"
          value={`$${forecast.expectedPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
          accent={signalColor}
        />
        <MetricCell
          label="Expected Δ"
          value={`${changeSign}${changeAbs}%`}
          accent={signalColor}
        />
        <MetricCell
          label="Volatility"
          value={`${(forecast.volatilityEst * 100).toFixed(2)}%`}
          accent="text-neutral"
        />
        <MetricCell
          label="Horizon"
          value="Client ML"
          accent="text-premium"
        />
      </div>

      {/* Anchor info */}
      <div className="px-3 py-2.5 rounded-lg bg-[#ffffff04] border border-[#ffffff08] text-xs">
        <div className="flex justify-between items-center text-muted">
          <span>Anchor Price</span>
          <span className="font-bold text-secondary">
            ${forecast.anchorPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </span>
        </div>
      </div>

      {/* Status footer */}
      <div className="flex items-center gap-2 text-xs border-t border-[#ffffff08] pt-3">
        <Clock size={12} className="text-muted shrink-0" />
        <span className="text-muted flex-1 truncate">
          {new Date(forecast.anchorTime * 1000).toLocaleTimeString()}
        </span>
        <ConnectionBadge status={connectionStatus} />
      </div>
    </div>
  );
}

function MetricCell({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div className="p-3 rounded-lg bg-[#ffffff04] border border-[#ffffff08]">
      <div className="text-[10px] text-muted uppercase tracking-wider mb-1 font-bold">
        {label}
      </div>
      <div className={`text-sm font-bold truncate ${accent}`}>{value}</div>
    </div>
  );
}

function ConnectionBadge({ status }: { status: 'loading' | 'live' | 'disconnected' }) {
  if (status === 'live') {
    return (
      <div className="flex items-center gap-1.5 text-bullish shrink-0">
        <Activity size={12} className="animate-pulse" />
        <span className="font-bold">LIVE</span>
      </div>
    );
  }
  if (status === 'loading') {
    return (
      <div className="flex items-center gap-1.5 text-neutral shrink-0">
        <Zap size={12} className="animate-bounce" />
        <span className="font-bold">Connecting…</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-1.5 text-bearish shrink-0">
      <AlertCircle size={12} />
      <span className="font-bold">Offline</span>
    </div>
  );
}
