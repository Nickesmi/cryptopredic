'use client'

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Sparkles, TrendingUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface CoinScoring {
  id: string;
  symbol: string;
  name: string;
  volume_surge_1hr: string;
  probability_of_spike: number;
}

export default function HighPotentialFeed() {
  const [coins, setCoins] = useState<CoinScoring[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFeed = async () => {
      try {
        const res = await axios.get('http://localhost:8000/api/high-potential');
        if (res.data && res.data.high_potential_recommendations) {
          setCoins(res.data.high_potential_recommendations.slice(0, 5)); // show top 5 gems
        }
      } catch (err) {
        console.error("Error fetching gems", err);
      } finally {
        setLoading(false);
      }
    };
    fetchFeed();
  }, []);

  return (
    <div className="glass-panel" style={{ height: '100%' }}>
      <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
        <Sparkles color="var(--accent-purple)" />
        <span className="gradient-text">High-Potential Gems</span>
      </h2>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
        AI Anomaly Score based on Social Sentiment & Volume
      </p>

      <div className="crypto-list">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="crypto-card shimmer-bg" style={{ height: '72px' }} />
          ))
        ) : (
          <AnimatePresence>
            {coins.map((coin, index) => (
              <motion.div 
                key={coin.id}
                className="crypto-card"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
              >
                <div>
                  <div className="crypto-symbol">{coin.symbol.toUpperCase()}</div>
                  <div className="crypto-name">{coin.name}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.75rem', textAlign: 'right' }}>
                    Sent:<br/>
                    <span style={{ color: 'var(--accent-cyan)' }}>{coin.volume_surge_1hr} Vol</span>
                  </div>
                  <div className="spike-probability">
                    <div className="spike-value" style={{ 
                      color: coin.probability_of_spike > 80 ? 'var(--positive-green)' : 'var(--text-primary)' 
                    }}>
                      {coin.probability_of_spike}%
                    </div>
                    <div className="spike-label">Spike Prob</div>
                  </div>
                  <TrendingUp color={coin.probability_of_spike > 80 ? 'var(--positive-green)' : 'var(--text-secondary)'} size={20} />
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}
