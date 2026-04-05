import type { Metadata } from 'next';
import './globals.css';
import { Activity, BarChart2, Compass, Layers, Settings, Shield, Search } from 'lucide-react';
import Link from 'next/link';
import { CommandPalette } from '../components/CommandPalette';

export const metadata: Metadata = {
  title: 'Crypto Alpha Engine | Quant Terminal',
  description: 'AI-Powered Crypto Forecasting & Opportunity Intelligence',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="flex h-screen overflow-hidden bg-core">
        {/* Deep Graphite Sidebar Structure */}
        <aside className="w-64 bg-surface border-r border-[#ffffff0f] flex flex-col justify-between hidden md:flex z-10 transition-all duration-300">
          <div>
            <div className="p-6 border-b border-[#ffffff0f]">
              <h1 className="text-xl font-bold tracking-tight text-primary flex items-center gap-2">
                <Activity className="text-premium" />
                Alpha <span className="text-muted font-light">Engine</span>
              </h1>
            </div>
            <nav className="p-4 space-y-2">
              <SidebarItem icon={<Compass />} label="Dashboard" href="/" />
              <SidebarItem icon={<BarChart2 />} label="Screener" href="/screener" />
              <SidebarItem icon={<Layers />} label="Coin Analysis" href="/coin/BTC" />
              <SidebarItem icon={<Shield />} label="Forecast Lab" href="#" />
            </nav>
          </div>
          <div className="p-4 border-t border-[#ffffff0f]">
            <SidebarItem icon={<Settings />} label="Terminal Settings" />
          </div>
        </aside>

        {/* Global Dashboard Shell */}
        <div className="flex-1 flex flex-col relative z-0">
          <header className="h-16 border-b border-[#ffffff0f] bg-core/80 backdrop-blur-md flex items-center justify-between px-6">
             <CommandPalette />
             <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-3 py-1 bg-bullish/10 border border-bullish/20 rounded-full">
                  <div className="w-2 h-2 rounded-full bg-bullish animate-pulse"></div>
                  <span className="text-xs font-semibold text-bullish tracking-wider">Models Live</span>
                </div>
             </div>
          </header>
          
          <main className="flex-1 overflow-y-auto p-8 relative">
             {/* Cyber glow background accent */}
             <div className="absolute top-0 left-1/2 -ml-[400px] w-[800px] h-[300px] bg-premium/10 blur-[120px] rounded-full pointer-events-none -z-10"></div>
             {children}
          </main>
        </div>
      </body>
    </html>
  );
}

function SidebarItem({ icon, label, href = "#" }: { icon: React.ReactNode, label: string, href?: string }) {
  return (
    <Link href={href} className="flex items-center gap-3 px-4 py-3 rounded-lg cursor-pointer transition-all duration-200 text-secondary hover:bg-card hover:text-primary border border-transparent">
      {icon}
      <span>{label}</span>
    </Link>
  );
}
