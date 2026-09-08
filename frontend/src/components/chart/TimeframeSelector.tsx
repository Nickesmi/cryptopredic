'use client';
import { TIMEFRAMES, type Timeframe } from '../../types/chart';

interface TimeframeSelectorProps {
  selected: Timeframe;
  onSelect: (tf: Timeframe) => void;
}

export function TimeframeSelector({ selected, onSelect }: TimeframeSelectorProps) {
  return (
    <div className="flex items-center gap-1" role="group" aria-label="Timeframe selector">
      {TIMEFRAMES.map((tf) => (
        <button
          key={tf}
          id={`tf-btn-${tf}`}
          onClick={() => onSelect(tf)}
          aria-pressed={selected === tf}
          className={`
            px-2.5 py-1 rounded text-xs font-bold tracking-wider transition-all duration-150
            ${selected === tf
              ? 'bg-premium/20 text-premium border border-premium/40'
              : 'text-secondary hover:text-primary bg-transparent border border-transparent hover:border-[#ffffff10]'
            }
          `}
        >
          {tf}
        </button>
      ))}
    </div>
  );
}
