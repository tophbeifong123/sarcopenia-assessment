"use client";

import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

interface ChartDataPoint {
  t: number;
  leftSpeed: number;
  rightSpeed: number;
  leftJerk: number;
  rightJerk: number;
  lni: number;
  leftROM: number;
  rightROM: number;
}

interface TelemetryChartsProps {
  data: ChartDataPoint[];
}

const COLOR_LEFT = "#22d3ee";
const COLOR_RIGHT = "#fb7185";

const chartMargin = { top: 4, right: 8, bottom: 4, left: -10 };

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="glass-card px-3 py-2 text-xs space-y-1">
      <p className="text-slate-400 font-mono">{label}s</p>
      {payload.map((entry: any, i: number) => (
        <p key={i} style={{ color: entry.color }} className="font-semibold">
          {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(1) : entry.value}
        </p>
      ))}
    </div>
  );
}

function ChartCard({
  title,
  badge,
  children,
}: {
  title: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="glass-card p-4 animate-slide-up">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          {title}
        </h3>
        {badge}
      </div>
      <div className="h-[140px]">{children}</div>
    </div>
  );
}

export default function TelemetryCharts({ data }: TelemetryChartsProps) {
  const hasData = data.length > 2;

  // Memoize latest values for badges
  const latest = useMemo(() => {
    if (data.length === 0) return null;
    return data[data.length - 1];
  }, [data]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {/* Speed Chart */}
      <ChartCard
        title="Arm Speed (px/s)"
        badge={
          latest && (
            <div className="flex items-center gap-3 text-xs font-mono">
              <span style={{ color: COLOR_LEFT }}>L: {latest.leftSpeed.toFixed(0)}</span>
              <span style={{ color: COLOR_RIGHT }}>R: {latest.rightSpeed.toFixed(0)}</span>
            </div>
          )
        }
      >
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={chartMargin}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="leftSpeed"
                stroke={COLOR_LEFT}
                strokeWidth={2}
                dot={false}
                name="Left"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="rightSpeed"
                stroke={COLOR_RIGHT}
                strokeWidth={2}
                dot={false}
                name="Right"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <EmptyChart label="Move your arms to begin..." />
        )}
      </ChartCard>

      {/* Jerk Chart */}
      <ChartCard
        title="Movement Jerk (smoothness)"
        badge={
          latest && (
            <div className="flex items-center gap-3 text-xs font-mono">
              <span style={{ color: COLOR_LEFT }}>L: {latest.leftJerk.toFixed(0)}</span>
              <span style={{ color: COLOR_RIGHT }}>R: {latest.rightJerk.toFixed(0)}</span>
            </div>
          )
        }
      >
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={chartMargin}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="leftJerk"
                stroke={COLOR_LEFT}
                strokeWidth={2}
                dot={false}
                name="Left Jerk"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="rightJerk"
                stroke={COLOR_RIGHT}
                strokeWidth={2}
                dot={false}
                name="Right Jerk"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <EmptyChart label="Jerk data will appear here..." />
        )}
      </ChartCard>

      {/* Shoulder ROM Chart */}
      <ChartCard
        title="Shoulder ROM (degrees)"
        badge={
          latest && (
            <div className="flex items-center gap-3 text-xs font-mono">
              <span style={{ color: COLOR_LEFT }}>L: {latest.leftROM.toFixed(0)}°</span>
              <span style={{ color: COLOR_RIGHT }}>R: {latest.rightROM.toFixed(0)}°</span>
            </div>
          )
        }
      >
        {hasData ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={chartMargin}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} domain={[0, 180]} />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="leftROM"
                stroke={COLOR_LEFT}
                strokeWidth={2}
                dot={false}
                name="Left ROM"
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="rightROM"
                stroke={COLOR_RIGHT}
                strokeWidth={2}
                dot={false}
                name="Right ROM"
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <EmptyChart label="ROM tracking will begin..." />
        )}
      </ChartCard>
    </div>
  );
}

function EmptyChart({ label }: { label: string }) {
  return (
    <div className="h-full flex items-center justify-center">
      <p className="text-sm text-slate-600 animate-pulse">{label}</p>
    </div>
  );
}
