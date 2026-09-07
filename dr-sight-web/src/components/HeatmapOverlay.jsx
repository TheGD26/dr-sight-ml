import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "../lib/cn.js";

/**
 * Explainability overlay.
 *  - `heatmap`  : base64 PNG (Grad-CAM overlay from the model). Blended over the
 *                 scan with an opacity control.
 *  - `regions`  : optional [{x,y,radius,intensity,label}] — rendered as hoverable
 *                 hotspots when no heatmap is available (matches the original UI).
 */
export default function HeatmapOverlay({
  imageUrl,
  heatmap,
  regions = [],
  className,
}) {
  const [hover, setHover] = useState(null);
  const [opacity, setOpacity] = useState(0.75);
  const heatmapSrc = heatmap
    ? heatmap.startsWith("data:")
      ? heatmap
      : `data:image/png;base64,${heatmap}`
    : null;

  return (
    <div className={className}>
      <div className="relative w-full overflow-hidden rounded-2xl ring-1 ring-border bg-muted">
        <img
          src={imageUrl}
          alt="Retinal scan"
          className="w-full h-full object-cover select-none"
          draggable={false}
        />

        {heatmapSrc && (
          <img
            src={heatmapSrc}
            alt="Grad-CAM heatmap — regions that most influenced the grade"
            className="pointer-events-none absolute inset-0 w-full h-full object-cover transition-opacity"
            style={{ opacity }}
            draggable={false}
          />
        )}

        {!heatmapSrc && regions.length > 0 && (
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="absolute inset-0 w-full h-full"
          >
            <defs>
              <radialGradient id="heat" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="rgba(255,235,59,0.85)" />
                <stop offset="55%" stopColor="rgba(255,87,34,0.55)" />
                <stop offset="100%" stopColor="rgba(244,67,54,0)" />
              </radialGradient>
            </defs>
            {regions.map((r, i) => (
              <g
                key={i}
                onMouseEnter={() => setHover(i)}
                onMouseLeave={() => setHover(null)}
                style={{ cursor: "pointer" }}
              >
                <circle
                  cx={r.x}
                  cy={r.y}
                  r={r.radius}
                  fill="url(#heat)"
                  opacity={Math.max(0.35, r.intensity ?? 0.5)}
                  className="transition-opacity"
                />
                <circle
                  cx={r.x}
                  cy={r.y}
                  r={(r.radius ?? 6) * 0.35}
                  fill="none"
                  stroke="rgba(255,255,255,0.9)"
                  strokeWidth="0.4"
                  strokeDasharray="1 1"
                />
              </g>
            ))}
          </svg>
        )}

        <AnimatePresence>
          {hover !== null && regions[hover] && (
            <motion.div
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="absolute bottom-3 left-3 right-3 rounded-lg bg-foreground/90 backdrop-blur px-3 py-2 text-xs text-background"
            >
              <span className="font-medium">{regions[hover].label}</span>
              <span className="mx-1.5 opacity-40">·</span>
              <span className="opacity-70">
                influence {((regions[hover].intensity ?? 0) * 100).toFixed(0)}%
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {heatmapSrc && (
        <label className="mt-3 flex items-center gap-3 text-xs text-muted-foreground">
          <span className="shrink-0">Heatmap opacity</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="tabular-nums w-8 text-right">
            {Math.round(opacity * 100)}%
          </span>
        </label>
      )}
    </div>
  );
}
