'use client';

import { useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, BarChart3, BrainCircuit, Gauge, RefreshCw, Target } from 'lucide-react';

interface MetricBlock {
  predictions: number;
  directional_accuracy: number;
  hit_rate: number;
  mae: number;
  rmse: number;
  mape: number;
  sharpe_ratio: number;
  confidence_calibration: number;
}

interface LeaderboardRow {
  model: string;
  version: string;
  predictions: number;
  directional_accuracy: number;
  mae: number;
  rmse: number;
  mape: number;
  sharpe: number;
  confidence_calibration: number;
  last_updated: string;
}

interface DashboardPayload {
  overall: MetricBlock;
  last_100: MetricBlock;
  last_500: MetricBlock;
  last_1000: MetricBlock;
  by_symbol: Record<string, MetricBlock>;
  by_timeframe: Record<string, MetricBlock>;
  by_model: Record<string, MetricBlock>;
  by_market_regime: Record<string, MetricBlock>;
  bull_market: MetricBlock;
  bear_market: MetricBlock;
  failure_patterns: { pattern: string; count: number; recommendation: string }[];
  recommendations: string[];
  leaderboard: LeaderboardRow[];
}

const emptyMetrics: MetricBlock = {
  predictions: 0,
  directional_accuracy: 0,
  hit_rate: 0,
  mae: 0,
  rmse: 0,
  mape: 0,
  sharpe_ratio: 0,
  confidence_calibration: 0,
};

export default function PerformanceDashboard() {
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/api/evaluation/dashboard', { cache: 'no-store' });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      setData(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load performance metrics');
      setData(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const payload = data ?? {
    overall: emptyMetrics,
    last_100: emptyMetrics,
    last_500: emptyMetrics,
    last_1000: emptyMetrics,
    by_symbol: {},
    by_timeframe: {},
    by_model: {},
    by_market_regime: {},
    bull_market: emptyMetrics,
    bear_market: emptyMetrics,
    failure_patterns: [],
    recommendations: [],
    leaderboard: [],
  };

  const symbolRows = useMemo(() => Object.entries(payload.by_symbol), [payload.by_symbol]);
  const modelRows = useMemo(() => Object.entries(payload.by_model), [payload.by_model]);

  return (
    <div className="h-full overflow-y-auto pb-10 space-y-6 animate-in fade-in duration-300">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-primary">Prediction Performance</h1>
          <p className="text-secondary mt-1">Objective forecast validation, model ranking, and continuous learning signals.</p>
        </div>
        <button
          onClick={load}
          className="inline-flex h-10 items-center gap-2 rounded border border-[#ffffff14] bg-card px-4 text-sm font-semibold text-primary hover:border-premium/50"
        >
          <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </header>

      {error && (
        <div className="rounded border border-bearish/30 bg-bearish/10 p-4 text-sm text-primary flex items-center gap-3">
          <AlertTriangle size={18} className="text-bearish" />
          {error}
        </div>
      )}

      <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        <MetricCard icon={<Target />} label="Overall Accuracy" value={pct(payload.overall.directional_accuracy)} sub={`${count(payload.overall.predictions)} evaluated`} />
        <MetricCard icon={<Gauge />} label="Last 100 Accuracy" value={pct(payload.last_100.directional_accuracy)} sub={`MAE ${num(payload.last_100.mae)}`} />
        <MetricCard icon={<BarChart3 />} label="Last 500 Accuracy" value={pct(payload.last_500.directional_accuracy)} sub={`RMSE ${num(payload.last_500.rmse)}`} />
        <MetricCard icon={<Activity />} label="Last 1000 Accuracy" value={pct(payload.last_1000.directional_accuracy)} sub={`Sharpe ${num(payload.last_1000.sharpe_ratio)}`} />
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <Panel title="Model Scoreboard" className="xl:col-span-2">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-muted uppercase text-xs">
                <tr className="border-b border-[#ffffff12]">
                  <th className="text-left py-3">Model</th>
                  <th className="text-right py-3">Predictions</th>
                  <th className="text-right py-3">Accuracy</th>
                  <th className="text-right py-3">MAE</th>
                  <th className="text-right py-3">RMSE</th>
                  <th className="text-right py-3">MAPE</th>
                  <th className="text-right py-3">Sharpe</th>
                  <th className="text-right py-3">Calibration</th>
                </tr>
              </thead>
              <tbody>
                {payload.leaderboard.map((row) => (
                  <tr key={`${row.model}-${row.version}`} className="border-b border-[#ffffff0a]">
                    <td className="py-3 text-primary font-semibold">{row.model} <span className="text-muted">v{row.version}</span></td>
                    <td className="py-3 text-right text-secondary">{count(row.predictions)}</td>
                    <td className="py-3 text-right text-bullish">{pct(row.directional_accuracy)}</td>
                    <td className="py-3 text-right text-secondary">{num(row.mae)}</td>
                    <td className="py-3 text-right text-secondary">{num(row.rmse)}</td>
                    <td className="py-3 text-right text-secondary">{num(row.mape)}%</td>
                    <td className="py-3 text-right text-secondary">{num(row.sharpe)}</td>
                    <td className="py-3 text-right text-secondary">{num(row.confidence_calibration)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {payload.leaderboard.length === 0 && <EmptyState text="No evaluated predictions yet." />}
          </div>
        </Panel>

        <Panel title="Continuous Learning">
          <div className="space-y-3">
            {payload.recommendations.map((item) => (
              <div key={item} className="rounded border border-[#ffffff10] bg-[#ffffff05] p-3 text-sm text-primary flex gap-3">
                <BrainCircuit size={18} className="text-premium shrink-0 mt-0.5" />
                <span>{item}</span>
              </div>
            ))}
            {payload.recommendations.length === 0 && <EmptyState text="Recommendations will appear after failures are evaluated." />}
          </div>
        </Panel>
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6">
        <Breakdown title="Accuracy by Symbol" rows={symbolRows} />
        <Breakdown title="Accuracy by Model" rows={modelRows} />
        <Breakdown title="Bull Market" rows={[['bull', payload.bull_market]]} />
        <Breakdown title="Bear Market" rows={[['bear', payload.bear_market]]} />
      </section>

      <Panel title="Recurring Failure Patterns">
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {payload.failure_patterns.map((item) => (
            <div key={item.pattern} className="rounded border border-[#ffffff10] bg-[#ffffff05] p-4">
              <div className="flex items-center justify-between">
                <span className="text-primary font-semibold">{labelize(item.pattern)}</span>
                <span className="text-bearish font-bold">{item.count}</span>
              </div>
              <p className="text-secondary text-sm mt-2">{item.recommendation}</p>
            </div>
          ))}
        </div>
        {payload.failure_patterns.length === 0 && <EmptyState text="No recurring failure pattern has been detected." />}
      </Panel>
    </div>
  );
}

function MetricCard({ icon, label, value, sub }: { icon: React.ReactNode; label: string; value: string; sub: string }) {
  return (
    <div className="rounded border border-[#ffffff10] bg-card p-5">
      <div className="flex items-center justify-between">
        <div className="text-secondary text-sm font-semibold">{label}</div>
        <div className="text-premium">{icon}</div>
      </div>
      <div className="text-3xl font-bold text-primary mt-4">{value}</div>
      <div className="text-muted text-sm mt-1">{sub}</div>
    </div>
  );
}

function Panel({ title, children, className = '' }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <section className={`rounded border border-[#ffffff10] bg-card p-5 ${className}`}>
      <h2 className="text-lg font-bold text-primary mb-4">{title}</h2>
      {children}
    </section>
  );
}

function Breakdown({ title, rows }: { title: string; rows: [string, MetricBlock][] }) {
  return (
    <Panel title={title}>
      <div className="space-y-3">
        {rows.map(([key, metrics]) => (
          <div key={key} className="flex items-center justify-between gap-4">
            <span className="text-secondary text-sm truncate">{labelize(key)}</span>
            <span className="text-primary font-bold">{pct(metrics.directional_accuracy)}</span>
          </div>
        ))}
        {rows.length === 0 && <EmptyState text="No data yet." />}
      </div>
    </Panel>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="rounded border border-dashed border-[#ffffff14] p-5 text-center text-sm text-muted">{text}</div>;
}

function pct(value: number) {
  return `${Math.round((value || 0) * 100)}%`;
}

function num(value: number) {
  return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function count(value: number) {
  return Math.round(value || 0).toLocaleString();
}

function labelize(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}
