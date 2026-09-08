'use client';

import { Layers } from 'lucide-react';

interface Overlay {
  id: string;
  displayName: string;
  isVisible: boolean;
}

interface OverlayControlsProps {
  overlays: Overlay[];
  onToggle: (id: string, visible: boolean) => void;
}

export function OverlayControls({ overlays, onToggle }: OverlayControlsProps) {
  if (!overlays || overlays.length === 0) return null;

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1.5 text-xs text-muted font-medium uppercase tracking-wider">
        <Layers size={14} />
        <span>Overlays</span>
      </div>
      <div className="flex bg-[#ffffff05] rounded-lg p-1 border border-[#ffffff0a]">
        {overlays.map((overlay) => (
          <button
            key={overlay.id}
            onClick={() => onToggle(overlay.id, !overlay.isVisible)}
            className={`
              px-3 py-1 text-xs font-bold rounded-md transition-all duration-200
              ${overlay.isVisible
                ? 'bg-[#ffffff15] text-primary shadow-sm'
                : 'text-muted hover:text-secondary'
              }
            `}
          >
            {overlay.displayName}
          </button>
        ))}
      </div>
    </div>
  );
}
