'use client';

interface ModelSelectorProps {
  selected: string;
  models: string[];
  onSelect: (model: string) => void;
}

export function ModelSelector({ selected, models, onSelect }: ModelSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted uppercase tracking-wider font-medium">Model</span>
      <select
        id="model-selector"
        value={selected}
        onChange={(e) => onSelect(e.target.value)}
        className="bg-card border border-[#ffffff15] text-secondary text-xs font-semibold rounded px-2 py-1 focus:outline-none focus:border-premium/50 cursor-pointer"
      >
        {models.map((m) => (
          <option key={m} value={m}>
            {m.charAt(0).toUpperCase() + m.slice(1)}
          </option>
        ))}
      </select>
    </div>
  );
}
