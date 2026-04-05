import Link from 'next/link';
import { ArrowLeft, Target, Shield, TrendingUp, Activity, BarChart2, CheckCircle2, AlertTriangle, Zap } from 'lucide-react';
import { RechartsChart } from '../../../components/RechartsChart';

export default function CoinAnalysisPage({ params }: { params: { symbol: string }}) {
  const symbol = params.symbol || 'ACT';
  
  return (
    <div className="space-y-6 pb-12 animate-in fade-in zoom-in duration-500">
       <Link href="/" className="text-secondary hover:text-primary flex items-center gap-2 text-sm font-medium transition-colors mb-6">
          <ArrowLeft size={16} /> Back to Dashboard
       </Link>

       {/* Top Header Panel */}
       <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6">
         <div className="flex items-center gap-4">
           <div className="w-16 h-16 rounded-full bg-surface border border-[#ffffff1a] flex items-center justify-center text-2xl font-bold text-primary">
             {symbol}
           </div>
           <div>
             <h1 className="text-3xl font-extrabold text-primary flex items-center gap-3">
               {symbol} <span className="text-xl text-muted font-normal">/ USD</span>
             </h1>
             <div className="flex gap-3 text-sm font-medium mt-1">
               <span className="text-secondary">Rank #42</span>
               <span className="text-[#ffffff2a]">•</span>
               <span className="text-secondary">Market Cap: $1.2B</span>
             </div>
           </div>
         </div>

         <div className="flex items-center gap-6 bg-surface px-6 py-4 rounded-xl border border-[#ffffff1a] shadow-lg">
           <Metric title="Probability Up" value="89%" emphasize="text-bullish" />
           <Metric title="Expected Return" value="+18.4%" emphasize="text-bullish" />
           <Metric title="Confidence" value="High" emphasize="text-primary" />
           <Metric title="Risk Level" value="Medium" emphasize="text-neutral" />
         </div>
       </header>

       <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
         
         {/* Main Chart Section (2/3 width) */}
         <section className="lg:col-span-2 space-y-6">
            <div className="luxury-card pb-8 h-[500px] flex flex-col">
              <div className="flex justify-between items-center mb-6">
                 <h2 className="text-lg font-bold text-primary flex items-center gap-2">
                   <Target className="text-premium" size={20} /> Forecast Trajectory & Confidence Cone
                 </h2>
                 <div className="flex gap-2 text-xs font-semibold">
                   <div className="px-3 py-1 rounded bg-premium/20 text-premium border border-premium/30 cursor-pointer">7D</div>
                   <div className="px-3 py-1 rounded bg-surface text-secondary hover:text-primary cursor-pointer border border-transparent">30D</div>
                 </div>
              </div>
              <div className="flex-1 relative">
                 <RechartsChart />
              </div>
            </div>

            {/* AI Reasoning Panel */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
               <ReasonCard icon={<Activity />} title="Momentum Breakout Detected" detail="Price structurally broke massive overhead resistance causing velocity expansion." />
               <ReasonCard icon={<BarChart2 />} title="Volume Expansion +38%" detail="Significant structural buying from large cohort tracking identified on chain." />
               <ReasonCard icon={<AlertTriangle />} title="Volatility Compressed" detail="Pre-breakout volatility registered at 6-month lows indicating heavy kinetic build-up." />
               <ReasonCard icon={<Zap />} title="Alt-Friendly Regime" detail="BTC Dominance is artificially weakened shifting beta exposure onto micro-caps." />
            </div>
         </section>

         {/* Right Rail Data Intelligence (1/3 width) */}
         <aside className="space-y-6">
            {/* Forecast Matrix Table */}
            <div className="luxury-card p-0 overflow-hidden">
               <div className="p-5 border-b border-[#ffffff0f] bg-[#ffffff02]">
                  <h3 className="font-bold text-primary uppercase tracking-wider text-sm flex items-center gap-2">
                     <TrendingUp size={16} className="text-premium"/> Multi-Timeframe Forecast Matrix
                  </h3>
               </div>
               <div className="p-5">
                 <table className="w-full text-sm text-left">
                   <thead>
                     <tr className="text-muted border-b border-[#ffffff0f]">
                       <th className="pb-3 font-medium">Horizon</th>
                       <th className="pb-3 font-medium">Target</th>
                       <th className="pb-3 font-medium">Prob</th>
                       <th className="pb-3 font-medium text-right">Signal</th>
                     </tr>
                   </thead>
                   <tbody className="divide-y divide-[#ffffff0a]">
                     <TableRow horizon="24H" target="+2.1%" prob="68%" signal="Bullish" color="bullish" />
                     <TableRow horizon="7D" target="+18.4%" prob="89%" signal="Strong Buy" color="bullish" />
                     <TableRow horizon="30D" target="+41.2%" prob="71%" signal="Bullish" color="bullish" />
                     <TableRow horizon="1Y" target="+120%" prob="42%" signal="Speculative" color="neutral" />
                   </tbody>
                 </table>
               </div>
            </div>

            {/* Risk Intelligence */}
            <div className="luxury-card border-neutral/30">
               <h3 className="font-bold text-neutral mb-6 uppercase tracking-wider text-sm flex items-center gap-2">
                  <Shield size={16}/> Risk Intelligence
               </h3>
               
               <div className="flex items-center gap-6 mb-6 pb-6 border-b border-[#ffffff0f]">
                  <div className="relative w-20 h-20 flex items-center justify-center">
                    <svg className="w-full h-full -rotate-90">
                      <circle className="text-[#ffffff10]" strokeWidth="6" stroke="currentColor" fill="transparent" r="36" cx="40" cy="40" />
                      <circle className="text-neutral" strokeWidth="6" strokeDasharray="226" strokeDashoffset="100" strokeLinecap="round" stroke="currentColor" fill="transparent" r="36" cx="40" cy="40" />
                    </svg>
                    <span className="absolute font-bold text-xl text-primary">54</span>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-primary mb-1">Medium Risk</div>
                    <div className="text-xs text-secondary tracking-wide uppercase">Aggregate Score</div>
                  </div>
               </div>

               <div className="space-y-4">
                  <RiskBar label="Volatility Risk" value={80} color="bg-bearish" />
                  <RiskBar label="Liquidity Risk" value={30} color="bg-bullish" />
                  <RiskBar label="Drawdown Trap" value={45} color="bg-neutral" />
               </div>
            </div>
         </aside>

       </div>
    </div>
  );
}

function Metric({ title, value, emphasize }: { title: string, value: string, emphasize: string }) {
  return (
    <div className="flex flex-col">
       <span className="text-xs tracking-wider text-muted font-bold uppercase mb-1">{title}</span>
       <span className={`text-2xl font-extrabold ${emphasize}`}>{value}</span>
    </div>
  );
}

function ReasonCard({ icon, title, detail }: { icon: any, title: string, detail: string }) {
  return (
    <div className="p-4 rounded-xl border border-[#ffffff0f] bg-card hover:border-premium/40 transition-colors cursor-crosshair group">
      <div className="flex items-center gap-3 mb-2">
         <div className="text-premium group-hover:scale-110 transition-transform">{icon}</div>
         <div className="font-bold text-primary tracking-wide text-sm">{title}</div>
      </div>
      <p className="text-secondary text-xs leading-relaxed">{detail}</p>
    </div>
  );
}

function TableRow({ horizon, target, prob, signal, color }: any) {
  const cMap: any = { bullish: 'text-bullish', neutral: 'text-neutral', bearish: 'text-bearish' };
  return (
    <tr>
      <td className="py-3 font-bold text-primary">{horizon}</td>
      <td className={`py-3 font-semibold ${cMap[color]}`}>{target}</td>
      <td className="py-3 font-bold text-primary">{prob}</td>
      <td className="py-3 text-right">
        <span className={`px-2 py-1 bg-[#ffffff0a] text-[10px] font-bold uppercase tracking-wider rounded ${cMap[color]}`}>{signal}</span>
      </td>
    </tr>
  );
}

function RiskBar({ label, value, color }: { label: string, value: number, color: string }) {
  return (
    <div>
      <div className="flex justify-between text-xs font-semibold mb-1 text-secondary">
        <span>{label}</span>
        <span>{value}/100</span>
      </div>
      <div className="w-full h-1.5 bg-[#ffffff0a] rounded-full overflow-hidden">
        <div className={`h-full ${color}`} style={{ width: `${value}%` }}></div>
      </div>
    </div>
  );
}
