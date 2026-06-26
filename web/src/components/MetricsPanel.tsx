"use client";

import React from "react";
import LNIGauge from "./LNIGauge";

interface ArmStats {
  avgSpeed: number;
  avgReachTime: number;
  maxSpeed: number;
  avgJerk: number;
  romRange: number;
  currentROM: number;
  avgStraightness: number;
}

interface AggregatedStats {
  frameCount: number;
  durationMs: number;
  left: ArmStats;
  right: ArmStats;
  avgLNI: number;
  currentLNI: number;
  dominantSide: string;
}

interface MetricsPanelProps {
  stats: AggregatedStats | null;
  hits: number;
}

function MetricRow({
  label,
  leftVal,
  rightVal,
  unit,
}: {
  label: string;
  leftVal: string;
  rightVal: string;
  unit?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-white/5 last:border-b-0">
      <span className="text-xs text-slate-500 font-medium uppercase tracking-wider w-28">
        {label}
      </span>
      <div className="flex items-center gap-6">
        <span className="text-sm font-mono font-semibold text-cyan-400 w-20 text-right">
          {leftVal}
          {unit && <span className="text-[10px] text-slate-600 ml-0.5">{unit}</span>}
        </span>
        <span className="text-sm font-mono font-semibold text-rose-400 w-20 text-right">
          {rightVal}
          {unit && <span className="text-[10px] text-slate-600 ml-0.5">{unit}</span>}
        </span>
      </div>
    </div>
  );
}

export default function MetricsPanel({ stats, hits }: MetricsPanelProps) {
  if (!stats) {
    return (
      <div className="glass-card p-6 animate-slide-up">
        <div className="flex items-center justify-center h-48">
          <div className="text-center space-y-3">
            <div className="w-10 h-10 mx-auto border-2 border-indigo-500/30 border-t-indigo-400 rounded-full animate-spin" />
            <p className="text-sm text-slate-500">Waiting for pose data...</p>
          </div>
        </div>
      </div>
    );
  }

  const duration = (stats.durationMs / 1000).toFixed(1);

  return (
    <div className="glass-card-glow p-5 animate-slide-up space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
            Arm Kinematics
          </h2>
          <p className="text-xs text-slate-600 font-mono mt-0.5">
            {stats.frameCount} frames · {duration}s
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs font-semibold uppercase tracking-wider">
          <span className="text-cyan-400 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-cyan-400" />
            Left
          </span>
          <span className="text-rose-400 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-400" />
            Right
          </span>
        </div>
      </div>

      {/* Hits Gauge — hero metric */}
      <div className="flex justify-center py-2">
        <LNIGauge hits={hits} label="Target Hits" />
      </div>

      {/* Dominant side */}
      <div className="text-center">
        <span className="metric-badge metric-badge-amber">
          Dominant: {stats.dominantSide === "left" ? "Left Arm" : "Right Arm"}
        </span>
      </div>

      {/* Metrics Grid */}
      <div className="space-y-0">
        <MetricRow
          label="Reach Time Avg"
          leftVal={(stats.left.avgReachTime / 1000).toFixed(2)}
          rightVal={(stats.right.avgReachTime / 1000).toFixed(2)}
          unit="s"
        />
        <MetricRow
          label="Max Speed"
          leftVal={stats.left.maxSpeed.toFixed(1)}
          rightVal={stats.right.maxSpeed.toFixed(1)}
          unit="norm/s"
        />
        <MetricRow
          label="Avg Jerk"
          leftVal={stats.left.avgJerk.toFixed(1)}
          rightVal={stats.right.avgJerk.toFixed(1)}
          unit="norm/s³"
        />
        <MetricRow
          label="ROM Range"
          leftVal={stats.left.romRange.toFixed(0)}
          rightVal={stats.right.romRange.toFixed(0)}
          unit="°"
        />
        <MetricRow
          label="Current ROM"
          leftVal={stats.left.currentROM.toFixed(0)}
          rightVal={stats.right.currentROM.toFixed(0)}
          unit="°"
        />
        <MetricRow
          label="Straightness"
          leftVal={(stats.left.avgStraightness * 100).toFixed(0)}
          rightVal={(stats.right.avgStraightness * 100).toFixed(0)}
          unit="%"
        />
      </div>

      {/* Dominant Side */}
      <div className="pt-2 border-t border-white/5 flex items-center justify-between">
        <span className="text-xs text-slate-500 uppercase tracking-wider font-medium">
          Dominant Side
        </span>
        <span className="text-sm font-mono font-bold text-amber-400">
          {stats.dominantSide.toUpperCase()}
        </span>
      </div>
    </div>
  );
}
