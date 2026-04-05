import Link from 'next/link';
import { ArrowUpRight, TrendingUp, ShieldAlert, Cpu } from 'lucide-react';

export default function Dashboard() {
  return (
    <div className="space-y-8 pb-12 animate-in fade-in zoom-in duration-500">
      
      {/* Hero Section */}
      <section className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 pb-4">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight text-primary mb-2">
            AI-Powered Forecasting
          </h1>
          <p className="text-secondary tracking-wide text-lg max-w-2xl">
            Track price forecasts, discover high-potential coins, and evaluate risk-adjusted opportunities.
          </p>
        </div>
        <div className="luxury-card !py-3 !px-5 flex items-center gap-4 border-bullish/30 bg-bullish/5 flex-shrink-0">
           <Cpu className="text-bullish" size={24} />
           <div>
             <div className="text-xs uppercase tracking-widest text-secondary font-bold">Total Sentiment</div>
             <div className="text-xl font-bold text-bullish">Moderate Risk-On</div>
           </div>
        </div>
      </section>

      {/* Market Overview Strip */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MarketCard symbol="BTC" price="$94,210" change="+2.4%" signal="Bullish" bullish={true} />
        <MarketCard symbol="ETH" price="$3,842" change="+1.2%" signal="Bullish" bullish={true} />
        <MarketCard symbol="SOL" price="$210.4" change="-0.8%" signal="Neutral" bullish={false} />
        <MarketCard symbol="ACT" price="$0.12" change="+14.2%" signal="Strong Buy" bullish={true} />
      </section>

      {/* Top AI Opportunities */}
      <section>
        <div className="flex items-center justify-between mb-4 mt-6">
          <h2 className="text-2xl font-bold text-primary flex items-center gap-2">
            <TrendingUp className="text-premium" />
            Top High-Potential Opportunities
          </h2>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <OpportunityCard 
            coin="ACT" 
            prob="89%" 
            upside="+18.4%" 
            risk="Medium"
            label="Strong Buy Candidate" 
            glow="bullish"
          />
          <OpportunityCard 
            coin="SUI" 
            prob="72%" 
            upside="+8.1%" 
            risk="High Volatility"
            label="Momentum Watchlist" 
            glow="premium"
          />
          <OpportunityCard 
            coin="TAO" 
            prob="68%" 
            upside="+12.0%" 
            risk="High"
            label="Speculative Swing" 
            glow="neutral"
          />
        </div>
      </section>

      {/* Narrative Summary */}
      <section className="luxury-card border-premium/30 bg-premium/5 mt-8">
        <h3 className="text-lg font-bold text-premium mb-2 uppercase tracking-widest flex items-center gap-2">
          <ShieldAlert size={18} /> Today's AI Market Interpretation
        </h3>
        <p className="text-primary leading-relaxed text-lg">
          The market is currently in a structurally bullish regime. Core primitives (BTC/ETH) show volume expansion supporting continuation. However, momentum shifts indicate rotation towards select mid-caps. Our ensemble models warn of transient drawdown risks in highly leveraged protocols. Tighten stop-losses on speculative allocations.
        </p>
      </section>
      
    </div>
  );
}

function MarketCard({ symbol, price, change, signal, bullish }: { symbol: string, price: string, change: string, signal: string, bullish: boolean }) {
  return (
    <div className={`luxury-card border-t-2 ${bullish ? 'border-t-bullish group' : 'border-t-neutral'}`}>
      <div className="text-secondary font-bold text-sm mb-1">{symbol} / USD</div>
      <div className="text-2xl font-bold text-primary mb-2 group-hover:glow-text transition-all">{price}</div>
      <div className="flex justify-between items-center text-sm">
        <span className={change.startsWith('+') ? 'text-bullish font-medium' : 'text-bearish font-medium'}>{change}</span>
        <span className="px-2 py-1 rounded bg-[#ffffff0a] text-xs font-semibold tracking-wider text-secondary">{signal}</span>
      </div>
    </div>
  );
}

function OpportunityCard({ coin, prob, upside, risk, label, glow }: any) {
  const colorMap: any = {
    bullish: 'text-bullish',
    premium: 'text-premium',
    neutral: 'text-neutral'
  };
  const colorClass = colorMap[glow];

  return (
    <div className="luxury-card flex flex-col justify-between">
      <div className="flex justify-between items-start mb-6">
        <div>
          <div className="text-2xl font-extrabold text-primary">{coin}</div>
          <div className="text-sm font-semibold tracking-wide text-secondary uppercase mt-1">7-Day Horizon</div>
        </div>
        <div className="w-14 h-14 rounded-full border-4 border-[#ffffff10] flex items-center justify-center relative">
          <svg className="absolute top-0 left-0 w-full h-full -rotate-90">
             <circle className="text-[#ffffff10]" strokeWidth="4" stroke="currentColor" fill="transparent" r="26" cx="28" cy="28" />
             <circle className={colorClass} strokeWidth="4" strokeDasharray="160" strokeDashoffset="40" strokeLinecap="round" stroke="currentColor" fill="transparent" r="26" cx="28" cy="28" />
          </svg>
          <span className="font-bold text-sm text-primary z-10">{prob}</span>
        </div>
      </div>

      <div className="space-y-4">
         <div className="flex justify-between border-b border-[#ffffff0f] pb-2">
           <span className="text-muted">Expected Upside</span>
           <span className="font-bold text-bullish">{upside}</span>
         </div>
         <div className="flex justify-between border-b border-[#ffffff0f] pb-2">
           <span className="text-muted">Downside Risk</span>
           <span className="font-medium text-primary">{risk}</span>
         </div>
      </div>

      <div className="mt-6 flex items-center justify-between">
         <div className={`px-3 py-1 rounded-sm bg-[#ffffff0a] text-xs font-bold uppercase tracking-wider ${colorClass}`}>
           {label}
         </div>
         <ArrowUpRight className="text-secondary hover:text-primary cursor-pointer transition-colors" />
      </div>
    </div>
  );
}
