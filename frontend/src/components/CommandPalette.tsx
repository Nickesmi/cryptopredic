"use client";
import { useState, useEffect } from 'react';
import { Search, Command } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

const DUMMY_DB = [
  { symbol: 'BTC', name: 'Bitcoin', signal: 'Bullish' },
  { symbol: 'ETH', name: 'Ethereum', signal: 'Bullish' },
  { symbol: 'SOL', name: 'Solana', signal: 'Neutral' },
  { symbol: 'ACT', name: 'Act I', signal: 'Strong Buy' },
  { symbol: 'SUI', name: 'Sui Network', signal: 'Watchlist' },
  { symbol: 'TAO', name: 'Bittensor', signal: 'Speculative' },
  { symbol: 'PRO', name: 'Propy', signal: 'Bullish' },
];

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const router = useRouter();

  // Keyboard shortcut listener
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((open) => !open);
      }
    };
    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, []);

  const results = query === '' 
    ? DUMMY_DB.slice(0, 4) 
    : DUMMY_DB.filter((coin) => 
        coin.symbol.toLowerCase().includes(query.toLowerCase()) || 
        coin.name.toLowerCase().includes(query.toLowerCase())
      );

  return (
    <div className="relative w-64 md:w-96 z-50">
      <div 
        className="w-full bg-[#ffffff05] border border-[#ffffff10] rounded-lg py-2 pl-9 pr-3 flex items-center justify-between cursor-text text-sm text-secondary hover:border-[#ffffff2a] transition-colors"
        onClick={() => setOpen(true)}
      >
        <Search size={16} className="absolute left-3 text-muted" />
        <span>Universal asset search...</span>
        <kbd className="hidden md:inline-flex items-center gap-1 px-2 py-0.5 rounded bg-[#ffffff0a] text-xs font-mono text-muted border border-[#ffffff0f]">
          <Command size={10} /> K
        </kbd>
      </div>

      {open && (
        <>
          <div className="fixed inset-0 bg-core/80 backdrop-blur-sm z-40" onClick={() => setOpen(false)} />
          <div className="absolute top-12 left-0 w-full md:w-[500px] bg-surface border border-[#ffffff1a] rounded-xl shadow-2xl z-50 overflow-hidden animate-in slide-in-from-top-2 duration-200">
            <div className="p-3 border-b border-[#ffffff0f]">
               <input 
                 autoFocus
                 value={query}
                 onChange={(e) => setQuery(e.target.value)}
                 placeholder="Search by token or symbol..."
                 className="w-full bg-transparent border-none text-primary focus:outline-none focus:ring-0 placeholder-muted text-lg"
               />
            </div>
            <div className="max-h-80 overflow-y-auto">
               {results.length === 0 ? (
                 <div className="p-6 text-center text-muted">No assets found matching "{query}"</div>
               ) : (
                 <div className="p-2 space-y-1">
                   {results.map((coin) => (
                     <div 
                       key={coin.symbol} 
                       onClick={() => {
                         setOpen(false);
                         setQuery('');
                         router.push(`/coin/${coin.symbol}`);
                       }}
                       className="flex items-center justify-between p-3 rounded-lg hover:bg-[#ffffff0a] cursor-pointer group transition-colors"
                     >
                       <div className="flex items-center gap-3">
                         <div className="w-8 h-8 rounded-full bg-core border border-[#ffffff10] text-xs font-bold flex items-center justify-center text-primary group-hover:border-premium/50 transition-colors">
                           {coin.symbol[0]}
                         </div>
                         <div>
                           <div className="font-bold text-primary group-hover:text-premium transition-colors">{coin.symbol}</div>
                           <div className="text-secondary text-xs">{coin.name}</div>
                         </div>
                       </div>
                       <div className="text-[10px] font-bold uppercase tracking-widest px-2 py-1 rounded bg-[#ffffff05] text-muted border border-[#ffffff0a] group-hover:border-[#ffffff20]">
                         {coin.signal}
                       </div>
                     </div>
                   ))}
                 </div>
               )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
