import Link from 'next/link';
import { Filter, Search, ArrowUpRight, ArrowDownRight, Activity } from 'lucide-react';

const mockScreenerData = [
  { coin: 'ACT', name: 'Act I', score: 94, prob: '89%', upside: '+18.4%', risk: 'Medium', confidence: 'High', signal: 'Strong Buy' },
  { coin: 'SUI', name: 'Sui Network', score: 82, prob: '72%', upside: '+8.1%', risk: 'High', confidence: 'Medium', signal: 'Watchlist' },
  { coin: 'TAO', name: 'Bittensor', score: 79, prob: '68%', upside: '+12.0%', risk: 'High', confidence: 'Low', signal: 'Speculative' },
  { coin: 'PRO', name: 'Propy', score: 71, prob: '62%', upside: '+5.4%', risk: 'Low', confidence: 'High', signal: 'Bullish' },
  { coin: 'PEPE', name: 'Pepe', score: 45, prob: '38%', upside: '-12.0%', risk: 'Extreme', confidence: 'High', signal: 'Avoid' }
];

export default function ScreenerPage() {
  return (
    <div className="space-y-6 pb-12 animate-in fade-in zoom-in duration-500 h-full flex flex-col">
       
       {/* Screener Header */}
       <header className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-2">
         <div>
           <h1 className="text-3xl font-extrabold text-primary flex items-center gap-3">
             <Activity className="text-premium" size={28} /> High-Potential Screener
           </h1>
           <p className="text-secondary mt-2 max-w-xl">
             Discover and filter the market mathematically based on forecasted upside probability, algorithmic confidence, and structural risk.
           </p>
         </div>
       </header>

       {/* Toolbar */}
       <div className="luxury-card !py-3 !px-5 flex flex-wrap gap-4 items-center justify-between border-[#ffffff0f]">
          <div className="flex items-center gap-4 w-full md:w-auto">
             <div className="relative w-full md:w-64">
                <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                <input 
                  type="text" 
                  placeholder="Search assets..." 
                  className="w-full bg-core border border-[#ffffff10] rounded-lg py-1.5 pl-9 pr-3 text-sm text-primary focus:outline-none focus:border-premium/50"
                />
             </div>
             <button className="flex items-center gap-2 text-sm text-secondary hover:text-primary transition-colors bg-[#ffffff05] px-3 py-1.5 rounded-lg border border-[#ffffff0a]">
                <Filter size={16} /> Filters
             </button>
          </div>
          <div className="flex gap-2">
             <div className="text-xs font-semibold px-3 py-1 bg-surface text-secondary border border-transparent rounded cursor-pointer hover:border-[#ffffff10]">24H</div>
             <div className="text-xs font-semibold px-3 py-1 bg-premium/20 text-premium border border-premium/30 rounded cursor-pointer">7D</div>
             <div className="text-xs font-semibold px-3 py-1 bg-surface text-secondary border border-transparent rounded cursor-pointer hover:border-[#ffffff10]">30D</div>
          </div>
       </div>

       {/* Premium Screener Table */}
       <div className="luxury-card p-0 flex-1 overflow-x-auto">
         <table className="w-full text-left whitespace-nowrap min-w-[800px]">
           <thead className="bg-[#ffffff02]">
             <tr>
               <th className="px-6 py-4 text-xs font-bold text-muted uppercase tracking-wider border-b border-[#ffffff0f]">Asset</th>
               <th className="px-6 py-4 text-xs font-bold text-muted uppercase tracking-wider border-b border-[#ffffff0f]">AI Score</th>
               <th className="px-6 py-4 text-xs font-bold text-muted uppercase tracking-wider border-b border-[#ffffff0f]">Prob Up</th>
               <th className="px-6 py-4 text-xs font-bold text-muted uppercase tracking-wider border-b border-[#ffffff0f]">Exp. Return</th>
               <th className="px-6 py-4 text-xs font-bold text-muted uppercase tracking-wider border-b border-[#ffffff0f]">Risk Level</th>
               <th className="px-6 py-4 text-xs font-bold text-muted uppercase tracking-wider border-b border-[#ffffff0f]">Confidence</th>
               <th className="px-6 py-4 text-xs font-bold text-muted uppercase tracking-wider text-right border-b border-[#ffffff0f]">Signal</th>
             </tr>
           </thead>
           <tbody className="divide-y divide-[#ffffff0a]">
             {mockScreenerData.map((row, idx) => (
                <tr key={idx} className="hover:bg-[#ffffff02] transition-colors cursor-pointer group">
                  <td className="px-6 py-4">
                     <Link href={`/coin/${row.coin}`} className="flex items-center gap-3">
                       <div className="w-8 h-8 rounded-full bg-core border border-[#ffffff1a] flex items-center justify-center font-bold text-xs text-primary">{row.coin[0]}</div>
                       <div>
                         <div className="font-bold text-primary group-hover:text-premium transition-colors">{row.coin}</div>
                         <div className="text-xs text-secondary">{row.name}</div>
                       </div>
                     </Link>
                  </td>
                  <td className="px-6 py-4 font-bold text-primary">{row.score}</td>
                  <td className="px-6 py-4 font-bold text-primary">{row.prob}</td>
                  <td className={`px-6 py-4 font-semibold flex items-center gap-1 ${row.upside.startsWith('+') ? 'text-bullish' : 'text-bearish'}`}>
                    {row.upside.startsWith('+') ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />} {row.upside}
                  </td>
                  <td className="px-6 py-4 text-sm text-secondary">{row.risk}</td>
                  <td className="px-6 py-4 text-sm text-primary">{row.confidence}</td>
                  <td className="px-6 py-4 text-right">
                    <span className={`px-2 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider bg-[#ffffff0a] 
                      ${row.signal === 'Strong Buy' ? 'text-bullish' : row.signal === 'Avoid' ? 'text-bearish' : 'text-neutral'}`}>
                      {row.signal}
                    </span>
                  </td>
                </tr>
             ))}
           </tbody>
         </table>
       </div>
    </div>
  );
}
