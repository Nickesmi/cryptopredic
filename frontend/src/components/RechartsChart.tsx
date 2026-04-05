"use client";
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';

const dummyData = [
  { date: 'Oct 01', base: 0.08, upper: 0.08, lower: 0.08 },
  { date: 'Oct 02', base: 0.085, upper: 0.088, lower: 0.082 },
  { date: 'Oct 03', base: 0.09, upper: 0.095, lower: 0.085 },
  { date: 'Oct 04', base: 0.10, upper: 0.11, lower: 0.09 },
  { date: 'Oct 05', base: 0.115, upper: 0.13, lower: 0.10 },
  { date: 'Oct 06', base: 0.12, upper: 0.14, lower: 0.11 },
  // Future forecast cone starts here
  { date: 'Oct 07', base: 0.13, upper: 0.16, lower: 0.115, isForecast: true },
  { date: 'Oct 08', base: 0.135, upper: 0.175, lower: 0.11, isForecast: true },
  { date: 'Oct 09', base: 0.142, upper: 0.19, lower: 0.105, isForecast: true },
];

export function RechartsChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={dummyData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="var(--accent-premium)" stopOpacity={0.3}/>
            <stop offset="95%" stopColor="var(--accent-premium)" stopOpacity={0}/>
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
        <XAxis dataKey="date" stroke="#6C7486" tick={{ fill: '#6C7486', fontSize: 12 }} />
        <YAxis stroke="#6C7486" tick={{ fill: '#6C7486', fontSize: 12 }} />
        <Tooltip 
          contentStyle={{ backgroundColor: '#11131A', borderColor: '#ffffff1a', borderRadius: '8px' }}
          itemStyle={{ color: '#F5F7FA' }}
        />
        
        {/* Confidence Cone representing uncertainty */}
        <Area type="monotone" dataKey="upper" stroke="none" fill="url(#colorForecast)" fillOpacity={0.4} />
        <Area type="monotone" dataKey="lower" stroke="none" fill="var(--bg-card)" fillOpacity={1} />
        
        {/* Central Core AI Trajectory */}
        <Area type="monotone" dataKey="base" stroke="var(--accent-premium)" strokeWidth={3} fill="none" />
        
        {/* Present Day Marker */}
        <ReferenceLine x="Oct 06" stroke="var(--signal-bullish)" strokeDasharray="3 3" label={{ position: 'top', value: 'Present', fill: '#00C389', fontSize: 12 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
