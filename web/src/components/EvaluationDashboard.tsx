"use client";

import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  ScatterChart,
  Scatter,
  Cell,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import LNIGauge from "./LNIGauge";
import type { GameResults } from "@/lib/game-engine";

interface EvaluationDashboardProps {
  results: GameResults;
  report: string | null;
  reportLoading: boolean;
  onGenerateReport: () => void;
  onRestart: () => void;
}

const COLOR_LEFT = "#22d3ee";
const COLOR_RIGHT = "#fb7185";

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="glass-card px-3 py-2 text-xs space-y-1">
      <p className="text-slate-400 font-mono">{label}</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} style={{ color: entry.color || entry.fill }} className="font-semibold">
          {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(2) : entry.value}
        </p>
      ))}
    </div>
  );
}

function StatCard({
  label,
  leftVal,
  rightVal,
  unit,
  higherIsBetter = true,
}: {
  label: string;
  leftVal: number;
  rightVal: number;
  unit: string;
  higherIsBetter?: boolean;
}) {
  const leftBetter = higherIsBetter ? leftVal >= rightVal : leftVal <= rightVal;
  return (
    <div className="glass-card p-4 space-y-2">
      <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </h4>
      <div className="flex items-end justify-between">
        <div className="text-center flex-1">
          <p className={`text-lg font-bold font-mono ${leftBetter ? "text-cyan-400" : "text-cyan-400/60"}`}>
            {leftVal.toFixed(1)}
            <span className="text-[10px] text-slate-600 ml-1">{unit}</span>
          </p>
          <p className="text-[10px] text-slate-600 uppercase">Left</p>
        </div>
        <div className="text-slate-700 text-xs px-2">vs</div>
        <div className="text-center flex-1">
          <p className={`text-lg font-bold font-mono ${!leftBetter ? "text-rose-400" : "text-rose-400/60"}`}>
            {rightVal.toFixed(1)}
            <span className="text-[10px] text-slate-600 ml-1">{unit}</span>
          </p>
          <p className="text-[10px] text-slate-600 uppercase">Right</p>
        </div>
      </div>
    </div>
  );
}

export default function EvaluationDashboard({
  results,
  report,
  reportLoading,
  onGenerateReport,
  onRestart,
}: EvaluationDashboardProps) {
  // ── Derived chart data ──

  const barData = useMemo(() => [
    {
      name: "Reach Time",
      left: results.left.avgReachTime / 1000,
      right: results.right.avgReachTime / 1000,
    },
    {
      name: "Straightness",
      left: results.left.avgStraightness,
      right: results.right.avgStraightness,
    },
    {
      name: "Smoothness",
      left: results.left.avgJerk > 0 ? 1 / (1 + results.left.avgJerk * 0.01) : 1,
      right: results.right.avgJerk > 0 ? 1 / (1 + results.right.avgJerk * 0.01) : 1,
    },
  ], [results]);

  const radarData = useMemo(() => {
    const maxTime = Math.max(results.left.avgReachTime, results.right.avgReachTime, 1);
    return [
      {
        metric: "Speed",
        left: results.left.avgReachTime > 0 ? (1 - results.left.avgReachTime / (maxTime * 1.5)) * 100 : 0,
        right: results.right.avgReachTime > 0 ? (1 - results.right.avgReachTime / (maxTime * 1.5)) * 100 : 0,
      },
      {
        metric: "Accuracy",
        left: results.left.avgStraightness * 100,
        right: results.right.avgStraightness * 100,
      },
      {
        metric: "Smoothness",
        left: results.left.avgJerk > 0 ? Math.max(0, 100 - results.left.avgJerk * 0.5) : 100,
        right: results.right.avgJerk > 0 ? Math.max(0, 100 - results.right.avgJerk * 0.5) : 100,
      },
      {
        metric: "Frequency",
        left: (results.left.reaches / Math.max(results.totalReaches, 1)) * 100,
        right: (results.right.reaches / Math.max(results.totalReaches, 1)) * 100,
      },
    ];
  }, [results]);

  const scatterData = useMemo(() =>
    results.reaches.map((r, i) => ({
      index: i + 1,
      reachTime: r.reachTimeMs / 1000,
      straightness: r.straightness,
      arm: r.arm,
    })),
  [results]);

  const duration = (results.durationMs / 1000).toFixed(0);

  // Determine less-active arm
  const lessActive = results.left.reaches < results.right.reaches ? "Left" : "Right";
  const moreActive = lessActive === "Left" ? "Right" : "Left";

  return (
    <div className="space-y-6 animate-slide-up">
      {/* ══ Header ══ */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold gradient-text">
            Assessment Complete
          </h2>
          <p className="text-sm text-slate-500 mt-1">
            {results.totalReaches} reaches in {duration}s · {results.left.reaches} left · {results.right.reaches} right
          </p>
        </div>
        <button onClick={onRestart} className="btn-secondary text-sm flex items-center gap-2" id="restart-test-btn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="1 4 1 10 7 10" />
            <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
          </svg>
          Restart Test
        </button>
      </div>

      {/* ══ Hero Row: LNI + Stats ══ */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card-glow p-5 flex flex-col items-center justify-center">
          <LNIGauge hits={results.totalReaches} misses={0} label="Target Hits" />
          <p className="text-xs text-slate-500 mt-3 text-center">
            Dominant: <span className="text-indigo-400 font-semibold">{moreActive} Arm</span>
          </p>
        </div>
        <StatCard
          label="Avg Reach Time"
          leftVal={results.left.avgReachTime / 1000}
          rightVal={results.right.avgReachTime / 1000}
          unit="s"
          higherIsBetter={false}
        />
        <StatCard
          label="Path Straightness"
          leftVal={results.left.avgStraightness * 100}
          rightVal={results.right.avgStraightness * 100}
          unit="%"
          higherIsBetter={true}
        />
        <StatCard
          label="Avg Jerk (Smoothness)"
          leftVal={results.left.avgJerk}
          rightVal={results.right.avgJerk}
          unit=""
          higherIsBetter={false}
        />
      </div>

      {/* ══ Charts Row ══ */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Radar Chart */}
        <div className="glass-card p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            Performance Radar
          </h3>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.06)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                <PolarRadiusAxis tick={{ fontSize: 9, fill: "#475569" }} domain={[0, 100]} />
                <Radar name="Left Arm" dataKey="left" stroke={COLOR_LEFT} fill={COLOR_LEFT} fillOpacity={0.15} strokeWidth={2} />
                <Radar name="Right Arm" dataKey="right" stroke={COLOR_RIGHT} fill={COLOR_RIGHT} fillOpacity={0.15} strokeWidth={2} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bar Chart */}
        <div className="glass-card p-4">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
            Left vs Right Comparison
          </h3>
          <div className="h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="left" name="Left Arm" fill={COLOR_LEFT} radius={[4, 4, 0, 0]} />
                <Bar dataKey="right" name="Right Arm" fill={COLOR_RIGHT} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ══ Scatter: Reach Times Over Trial ══ */}
      <div className="glass-card p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
          Individual Reach Times
        </h3>
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 8, bottom: 8, left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" name="Reach #" tick={{ fontSize: 10 }} label={{ value: "Reach #", position: "insideBottom", offset: -2, fontSize: 10, fill: "#64748b" }} />
              <YAxis dataKey="reachTime" name="Time (s)" tick={{ fontSize: 10 }} label={{ value: "Time (s)", angle: -90, position: "insideLeft", fontSize: 10, fill: "#64748b" }} />
              <Tooltip content={<CustomTooltip />} />
              <Scatter data={scatterData} name="Reaches">
                {scatterData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.arm === "left" ? COLOR_LEFT : COLOR_RIGHT} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="flex justify-center gap-6 mt-2 text-xs">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLOR_LEFT }} />
            Left Arm
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLOR_RIGHT }} />
            Right Arm
          </span>
        </div>
      </div>

      {/* ══ AI Report ══ */}
      <div className="glass-card-glow p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
              🤖 Virtual Physio Report
            </h3>
            <p className="text-xs text-slate-600 mt-0.5">AI-powered clinical analysis</p>
          </div>
          {!report && (
            <button
              onClick={onGenerateReport}
              disabled={reportLoading}
              className="btn-primary text-sm flex items-center gap-2"
              id="generate-eval-report-btn"
            >
              {reportLoading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Analyzing...
                </>
              ) : (
                "Generate Clinical Report"
              )}
            </button>
          )}
        </div>

        {report && (
          <div className="report-panel animate-slide-up">
            <div className="prose prose-invert prose-sm max-w-none">
              {report.split("\n").map((line, i) => {
                if (!line.trim()) return <br key={i} />;
                if (line.startsWith("##")) {
                  return <h4 key={i} className="text-sm font-bold text-violet-400 mt-4 mb-1">{line.replace(/^#+\s*/, "")}</h4>;
                }
                if (line.startsWith("- ") || line.startsWith("* ")) {
                  return <p key={i} className="text-sm text-slate-300 pl-4 py-0.5">• {line.slice(2)}</p>;
                }
                if (line.startsWith("**") && line.endsWith("**")) {
                  return <p key={i} className="text-sm font-bold text-slate-200 mt-2">{line.replace(/\*\*/g, "")}</p>;
                }
                return <p key={i} className="text-sm text-slate-400 leading-relaxed">{line}</p>;
              })}
            </div>
          </div>
        )}

        {!report && !reportLoading && (
          <p className="text-sm text-slate-600 text-center py-4">
            Click <span className="text-indigo-400 font-medium">&quot;Generate Clinical Report&quot;</span> for AI analysis
          </p>
        )}
      </div>
    </div>
  );
}
