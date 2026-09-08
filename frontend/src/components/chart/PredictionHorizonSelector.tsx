'use client';
import { PREDICTION_HORIZONS, type PredictionHorizon } from '../../types/chart';

interface PredictionHorizonSelectorProps {
  selected: PredictionHorizon;
  onSelect: (n: PredictionHorizon) => void;
}

export function PredictionHorizonSelector({ selected, onSelect }: PredictionHorizonSelectorProps) {
  return (
    <div className="flex items-center gap-1" role="group" aria-label="Prediction horizon selector">
      <span className="text-xs text-muted font-medium mr-1 uppercase tracking-wider">Forecast</span>
      {PREDICTION_HORIZONS.map((n) => (
        <button
          key={n}
          id={`horizon-btn-${n}`}
          onClick={() => onSelect(n)}
          aria-pressed={selected === n}
          className={`
            px-2 py-1 rounded text-xs font-bold transition-all duration-150
            ${selected === n
              ? 'bg-bullish/20 text-bullish border border-bullish/40'
              : 'text-muted hover:text-secondary border border-transparent hover:border-[#ffffff10]'
            }
          `}
        >
          {n}
        </button>
      ))}
      <span className="text-xs text-muted ml-1">candles</span>
    </div>
  );
}
