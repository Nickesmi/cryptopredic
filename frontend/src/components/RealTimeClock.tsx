'use client';

import { useState, useEffect } from 'react';
import { Clock } from 'lucide-react';

export function RealTimeClock() {
  const [time, setTime] = useState<string>('');

  useEffect(() => {
    // Set initial time
    setTime(new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' UTC');

    const timer = setInterval(() => {
      // Use UTC time to be standard across the platform
      const now = new Date();
      
      const hours = String(now.getUTCHours()).padStart(2, '0');
      const minutes = String(now.getUTCMinutes()).padStart(2, '0');
      const seconds = String(now.getUTCSeconds()).padStart(2, '0');
      
      setTime(`${hours}:${minutes}:${seconds} UTC`);
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Avoid hydration mismatch by rendering a placeholder on the server
  if (!time) {
    return (
      <div className="flex items-center gap-2 text-secondary text-sm font-medium w-24">
        <Clock size={14} className="text-muted" />
        <span className="opacity-0">00:00:00</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 text-secondary text-sm font-medium border border-[#ffffff0f] bg-surface px-3 py-1.5 rounded-md shadow-sm">
      <Clock size={14} className="text-premium" />
      <span className="tracking-wider tabular-nums">{time}</span>
    </div>
  );
}
