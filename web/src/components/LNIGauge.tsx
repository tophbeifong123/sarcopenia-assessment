"use client";

import React from "react";

interface LNIGaugeProps {
  hits: number;
  misses: number;
  label?: string;
}

export default function LNIGauge({ hits, misses, label = "Total Hits" }: LNIGaugeProps) {
  const total = hits + misses;
  const accuracy = total > 0 ? hits / total : 0;
  
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  // Circular arc matches accuracy ratio (0-1)
  const offset = circumference - accuracy * circumference;

  let strokeColor: string;
  let glowColor: string;
  let rating: string;
  let ratingColor: string;

  if (total === 0) {
    strokeColor = "#64748b"; // gray
    glowColor = "rgba(100, 116, 139, 0.1)";
    rating = "No Attempts";
    ratingColor = "text-slate-400";
  } else if (accuracy >= 0.85) {
    strokeColor = "#34d399"; // green
    glowColor = "rgba(52, 211, 153, 0.3)";
    rating = "Excellent";
    ratingColor = "text-emerald-400";
  } else if (accuracy >= 0.70) {
    strokeColor = "#fbbf24"; // amber
    glowColor = "rgba(251, 191, 36, 0.3)";
    rating = "Good";
    ratingColor = "text-amber-400";
  } else if (accuracy >= 0.50) {
    strokeColor = "#fb923c"; // orange
    glowColor = "rgba(251, 146, 60, 0.3)";
    rating = "Fair";
    ratingColor = "text-orange-400";
  } else {
    strokeColor = "#fb7185"; // rose
    glowColor = "rgba(251, 113, 133, 0.3)";
    rating = "Poor";
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
        {total > 0 && (
          <p className={`text-sm font-semibold ${ratingColor}`}>
            {rating} ({Math.round(accuracy * 100)}% Acc)
          </p>
        )}
      </div>
    </div>
  );
}
