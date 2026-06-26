"use client";

import React from "react";

interface LNIGaugeProps {
  hits: number;
  /** Number of completed reaches that fills the ring (default 30). */
  target?: number;
  label?: string;
}

/**
 * Honest activity gauge for the 9-grid reaching test.
 *
 * The ring fills toward a target number of completed reaches and the centre
 * shows the raw hit count. We intentionally do NOT display an "accuracy %"
 * here: the game only records successful target touches, so there is no miss
 * count to compute a meaningful accuracy from. The tiers below reflect HOW
 * MANY targets were reached, not a hit/miss ratio.
 */
export default function LNIGauge({ hits, target = 30, label = "Target Hits" }: LNIGaugeProps) {
  const safeTarget = Math.max(1, target);
  const progress = Math.min(1, hits / safeTarget);

  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - progress * circumference;

  let strokeColor: string;
  let glowColor: string;
  let rating: string;
  let ratingColor: string;

  if (hits === 0) {
    strokeColor = "#64748b"; // gray
    glowColor = "rgba(100, 116, 139, 0.1)";
    rating = "No Reaches";
    ratingColor = "text-slate-400";
  } else if (progress >= 0.85) {
    strokeColor = "#34d399"; // green
    glowColor = "rgba(52, 211, 153, 0.3)";
    rating = "Very Active";
    ratingColor = "text-emerald-400";
  } else if (progress >= 0.55) {
    strokeColor = "#fbbf24"; // amber
    glowColor = "rgba(251, 191, 36, 0.3)";
    rating = "Active";
    ratingColor = "text-amber-400";
  } else if (progress >= 0.30) {
    strokeColor = "#fb923c"; // orange
    glowColor = "rgba(251, 146, 60, 0.3)";
    rating = "Limited";
    ratingColor = "text-orange-400";
  } else {
    strokeColor = "#fb7185"; // rose
    glowColor = "rgba(251, 113, 133, 0.3)";
    rating = "Low Activity";
    ratingColor = "text-rose-400";
  }

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="lni-ring" style={{ filter: `drop-shadow(0 0 12px ${glowColor})` }}>
        <svg width="120" height="120" viewBox="0 0 120 120">
          {/* Track */}
          <circle
            className="lni-ring-track"
            cx="60"
            cy="60"
            r={radius}
          />
          {/* Value */}
          <circle
            className="lni-ring-value"
            cx="60"
            cy="60"
            r={radius}
            stroke={strokeColor}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold font-mono" style={{ color: strokeColor }}>
            {hits}
          </span>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">
            Hits
          </span>
        </div>
      </div>
      <div className="text-center">
        <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">
          {label}
        </p>
        {hits > 0 && (
          <p className={`text-sm font-semibold ${ratingColor}`}>
            {rating}
          </p>
        )}
      </div>
    </div>
  );
}
