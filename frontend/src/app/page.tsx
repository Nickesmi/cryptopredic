import PriceChart from '../components/PriceChart';
import HighPotentialFeed from '../components/HighPotentialFeed';
import { Activity } from 'lucide-react';

export default function Home() {
  return (
    <main className="app-container">
      <header className="header">
        <h1>
          <Activity color="var(--accent-cyan)" size={32} />
          CryptoPredic <span style={{ fontWeight: 300, color: 'var(--text-secondary)' }}>AI</span>
        </h1>
        <div style={{ display: 'flex', gap: '16px' }}>
          <div className="badge-live" style={{ borderColor: 'var(--positive-green)', color: 'var(--positive-green)' }}>
            System Online
          </div>
        </div>
      </header>

      {/* Main Chart Section */}
      <section style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <PriceChart />
      </section>

      {/* Sidebar: High Potential Recommendations */}
      <aside>
        <HighPotentialFeed />
      </aside>
    </main>
  );
}
