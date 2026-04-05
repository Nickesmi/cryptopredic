'use client'

import React, { useState, useEffect } from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import axios from 'axios';

interface DataPoint {
  date: string;
  predicted_price: number;
  lower_bound: number;
  upper_bound: number;
}

export default function PriceChart() {
  const [data, setData] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [symbol, setSymbol] = useState("BTCUSDT");

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        setLoading(true);
        // Using a mock URL for build, but hits our python fastAPI
        const res = await axios.get(`http://localhost:8000/api/predict/${symbol}?days=30`);
        if (res.data && res.data.predictions) {
          setData(res.data.predictions);
        }
      } catch (err) {
        console.error("Error fetching predictions", err);
      } finally {
        setLoading(false);
      }
    };
    fetchPrediction();
  }, [symbol]);

  return (
    <div className="glass-panel">
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
        <h2><span className="gradient-text">BTC</span> Time-Series Forecast</h2>
        <div className="badge-live">Live ML Active</div>
      </div>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '24px' }}>
        Prophet Model Projection (Next 30 Days)
      </p>

      {loading ? (
        <div className="chart-container shimmer-bg" style={{ borderRadius: '12px' }} />
      ) : (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <XAxis dataKey="date" stroke="var(--text-secondary)" tick={{fontSize: 12}} />
              <YAxis domain={['auto', 'auto']} stroke="var(--text-secondary)" tick={{fontSize: 12}} />
              <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" vertical={false} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(11, 15, 25, 0.9)', 
                  border: '1px solid var(--glass-border)',
                  borderRadius: '8px'
                }} 
              />
              <Area type="monotone" dataKey="predicted_price" stroke="var(--accent-cyan)" strokeWidth={3} fillOpacity={1} fill="url(#colorPrice)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
