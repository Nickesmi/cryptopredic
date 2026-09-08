'use client';
import { SUPPORTED_SYMBOLS, type SupportedSymbol } from '../../types/chart';

interface SymbolSelectorProps {
  selected: SupportedSymbol;
  onSelect: (symbol: SupportedSymbol) => void;
}

export function SymbolSelector({ selected, onSelect }: SymbolSelectorProps) {
  return (
    <select
      id="symbol-selector"
      value={selected}
      onChange={(e) => onSelect(e.target.value as SupportedSymbol)}
      aria-label="Asset selector"
      className="bg-card border border-[#ffffff15] text-primary font-bold text-sm rounded-lg px-3 py-1.5 focus:outline-none focus:border-premium/50 cursor-pointer"
    >
      {SUPPORTED_SYMBOLS.map((s) => (
        <option key={s} value={s}>
          {s}
        </option>
      ))}
    </select>
  );
}
