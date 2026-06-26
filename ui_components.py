"""HTML builders for the dashboard cards (kinematics live card + final report)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_max(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _rom_range(rom_min: float, rom_max: float) -> float:
    if rom_max != float("-inf") and rom_min != float("inf"):
        return rom_max - rom_min
    return 0.0


def _metric_row(label: str, left: str, right: str, unit: str,
                value_width: int, font: int, unit_font: int,
                border: str) -> str:
    return f"""<div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: {border};">
<span style="font-size: 10px; font-weight: 600; color: #94A3B8; letter-spacing: 0.05em;">{label}</span>
<div style="display: flex; gap: 16px;">
<span style="color: #22D3EE; font-family: 'JetBrains Mono', monospace; font-size: {font}px; font-weight: 700; width: {value_width}px; text-align: right;">{left}<span style="font-size: {unit_font}px; font-weight: 500; color: #64748B; margin-left: 2px;">{unit}</span></span>
<span style="color: #FB7185; font-family: 'JetBrains Mono', monospace; font-size: {font}px; font-weight: 700; width: {value_width}px; text-align: right;">{right}<span style="font-size: {unit_font}px; font-weight: 500; color: #64748B; margin-left: 2px;">{unit}</span></span>
</div>
</div>"""


def get_kinematics_card_html(
    frame_idx,
    fps,
    total_hits,
    left_hits,
    right_hits,
    left_speeds,
    right_speeds,
    left_jerks,
    right_jerks,
    left_rom_min,
    left_rom_max,
    right_rom_min,
    right_rom_max,
    left_current_rom,
    right_current_rom,
    left_straightness_val,
    right_straightness_val,
    dominant_side,
):
    """Live telemetry card shown beside the video during processing."""
    duration_sec = frame_idx / fps if fps > 0 else 0.0

    left_avg_speed = _safe_mean(left_speeds)
    right_avg_speed = _safe_mean(right_speeds)
    left_max_speed = _safe_max(left_speeds)
    right_max_speed = _safe_max(right_speeds)
    left_avg_jerk = _safe_mean(left_jerks)
    right_avg_jerk = _safe_mean(right_jerks)
    left_rom_range = _rom_range(left_rom_min, left_rom_max)
    right_rom_range = _rom_range(right_rom_min, right_rom_max)

    if total_hits > 0:
        stroke_color, glow_color, offset = "#34d399", "rgba(52, 211, 153, 0.3)", 0.0
    else:
        stroke_color, glow_color, offset = "#64748b", "rgba(100, 116, 139, 0.15)", 301.59

    dominant_side_text = dominant_side.upper()

    rows = (
        _metric_row("TARGET HITS", f"{left_hits}", f"{right_hits}", "", 75, 12, 8, "1px solid rgba(255,255,255,0.05)")
        + _metric_row("AVG SPEED", f"{left_avg_speed:.1f}", f"{right_avg_speed:.1f}", "px/s", 75, 12, 8, "1px solid rgba(255,255,255,0.05)")
        + _metric_row("MAX SPEED", f"{left_max_speed:.1f}", f"{right_max_speed:.1f}", "px/s", 75, 12, 8, "1px solid rgba(255,255,255,0.05)")
        + _metric_row("AVG JERK", f"{left_avg_jerk:.1f}", f"{right_avg_jerk:.1f}", "px/s\u00b3", 75, 12, 8, "1px solid rgba(255,255,255,0.05)")
        + _metric_row("ROM RANGE", f"{left_rom_range:.0f}", f"{right_rom_range:.0f}", "\u00b0", 75, 12, 8, "1px solid rgba(255,255,255,0.05)")
        + _metric_row("CURRENT ROM", f"{left_current_rom:.0f}", f"{right_current_rom:.0f}", "\u00b0", 75, 12, 8, "1px solid rgba(255,255,255,0.05)")
        + _metric_row("STRAIGHTNESS", f"{left_straightness_val:.0f}", f"{right_straightness_val:.0f}", "%", 75, 12, 8, "1px solid rgba(255,255,255,0.05)")
    )

    return f"""<div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 16px; padding: 20px; font-family: 'Inter', sans-serif; box-shadow: 0 4px 20px rgba(0,0,0,0.3); width: 100%; margin: 10px 0;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 10px;">
<div>
<h3 style="margin: 0; font-size: 14px; font-weight: 700; color: #E2E8F0; letter-spacing: 0.05em; text-transform: uppercase;">ARM KINEMATICS</h3>
<p style="margin: 4px 0 0 0; font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">{frame_idx} frames \u00b7 {duration_sec:.1f}s</p>
</div>
<div style="display: flex; gap: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-top: 4px;">
<span style="color: #22D3EE; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #22D3EE;"></span>LEFT</span>
<span style="color: #FB7185; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #FB7185;"></span>RIGHT</span>
</div>
</div>
<div style="display: flex; justify-content: center; margin-bottom: 8px;">
<div style="position: relative; width: 120px; height: 120px; filter: drop-shadow(0 0 12px {glow_color});">
<svg width="120" height="120" viewBox="0 0 120 120" style="transform: rotate(-90deg); width: 120px; height: 120px;">
<circle cx="60" cy="60" r="48" fill="none" stroke="rgba(255, 255, 255, 0.05)" stroke-width="8" />
<circle cx="60" cy="60" r="48" fill="none" stroke="{stroke_color}" stroke-width="8" stroke-linecap="round"
stroke-dasharray="301.59" stroke-dashoffset="{offset}" style="transition: stroke-dashoffset 0.5s ease, stroke 0.5s ease;" />
</svg>
<div style="position: absolute; inset: 0; display: flex; flex-direction: column; justify-content: center; align-items: center;">
<span style="font-size: 26px; font-weight: 800; color: #FFFFFF; line-height: 1;">{total_hits}</span>
<span style="font-size: 9px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; margin-top: 2px;">HITS</span>
</div>
</div>
</div>
<div style="text-align: center; font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 12px;">TARGET HITS</div>
<div style="display: flex; justify-content: center; margin-bottom: 15px;">
<div style="border: 1px solid rgba(245, 158, 11, 0.4); background-color: rgba(245, 158, 11, 0.08); color: #FBBF24; font-size: 10px; font-weight: 700; letter-spacing: 0.05em; padding: 4px 12px; border-radius: 20px; text-transform: uppercase;">
DOMINANT: {dominant_side_text} ARM
</div>
</div>
<div style="display: flex; flex-direction: column;">
{rows}
<div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: none;">
<span style="font-size: 10px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">DOMINANT SIDE</span>
<span style="color: #FBBF24; font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 700; text-transform: uppercase;">{dominant_side_text}</span>
</div>
</div>
</div>"""


def get_report_card_html(
    frame_idx,
    fps,
    total_hits,
    left_hits,
    right_hits,
    dominant_side_en,
    left_avg_speed,
    right_avg_speed,
    left_max_speed,
    right_max_speed,
    left_avg_jerk,
    right_avg_jerk,
    left_rom_range,
    right_rom_range,
    left_current_rom,
    right_current_rom,
    left_straightness_val,
    right_straightness_val,
    lnu_risk,
    lnu_color,
):
    """Final assessment report card shown after processing finishes."""
    duration_sec = frame_idx / fps if fps > 0 else 0.0

    rows = (
        _metric_row("TARGET HITS", f"{left_hits}", f"{right_hits}", "", 85, 13, 9, "1px solid #334155")
        + _metric_row("AVG SPEED", f"{left_avg_speed:.1f}", f"{right_avg_speed:.1f}", "px/s", 85, 13, 9, "1px solid #334155")
        + _metric_row("MAX SPEED", f"{left_max_speed:.1f}", f"{right_max_speed:.1f}", "px/s", 85, 13, 9, "1px solid #334155")
        + _metric_row("AVG JERK", f"{left_avg_jerk:.1f}", f"{right_avg_jerk:.1f}", "px/s\u00b3", 85, 13, 9, "1px solid #334155")
        + _metric_row("ROM RANGE", f"{left_rom_range:.0f}", f"{right_rom_range:.0f}", "\u00b0", 85, 13, 9, "1px solid #334155")
        + _metric_row("CURRENT ROM", f"{left_current_rom:.0f}", f"{right_current_rom:.0f}", "\u00b0", 85, 13, 9, "1px solid #334155")
        + _metric_row("STRAIGHTNESS", f"{left_straightness_val:.0f}", f"{right_straightness_val:.0f}", "%", 85, 13, 9, "1px solid #334155")
    )

    return f"""<div style="background-color: #1E293B; border: 1px solid #334155; border-radius: 16px; padding: 24px; font-family: 'Inter', sans-serif; box-shadow: 0 4px 20px rgba(0,0,0,0.3); max-width: 600px; margin: 20px auto;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 12px;">
<div>
<h3 style="margin: 0; font-size: 15px; font-weight: 700; color: #E2E8F0; letter-spacing: 0.05em; text-transform: uppercase;">ARM KINEMATICS</h3>
<p style="margin: 4px 0 0 0; font-size: 11px; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">{frame_idx} frames \u00b7 {duration_sec:.1f}s</p>
</div>
<div style="display: flex; gap: 12px; font-size: 11px; font-weight: 700; text-transform: uppercase; margin-top: 4px;">
<span style="color: #22D3EE; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #22D3EE;"></span>LEFT</span>
<span style="color: #FB7185; display: flex; align-items: center; gap: 4px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background-color: #FB7185;"></span>RIGHT</span>
</div>
</div>
<div style="display: flex; justify-content: center; margin-bottom: 12px;">
<div style="width: 100px; height: 100px; border-radius: 50%; border: 4px solid #334155; display: flex; flex-direction: column; justify-content: center; align-items: center; background: radial-gradient(circle, rgba(15,23,42,1) 60%, rgba(30,41,59,1) 100%); box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
<span style="font-size: 32px; font-weight: 800; color: #FFFFFF; line-height: 1;">{total_hits}</span>
<span style="font-size: 9px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; margin-top: 4px;">HITS</span>
</div>
</div>
<div style="text-align: center; font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 15px;">TARGET HITS</div>
<div style="display: flex; justify-content: center; margin-bottom: 24px;">
<div style="border: 1px solid rgba(245, 158, 11, 0.4); background-color: rgba(245, 158, 11, 0.08); color: #FBBF24; font-size: 11px; font-weight: 700; letter-spacing: 0.05em; padding: 6px 16px; border-radius: 20px; text-transform: uppercase;">
DOMINANT: {dominant_side_en} ARM
</div>
</div>
<div style="display: flex; flex-direction: column;">
{rows}
<div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #334155;">
<span style="font-size: 11px; font-weight: 700; color: #94A3B8; letter-spacing: 0.05em;">DOMINANT SIDE</span>
<span style="color: #FBBF24; font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 700; text-transform: uppercase;">{dominant_side_en}</span>
</div>
</div>
<div style="margin-top: 20px; padding: 16px; background-color: #0F172A; border-radius: 12px; border: 1px solid #334155;">
<p style="margin: 0; font-size: 14px; color: #FFFFFF;">
\U0001F50D <b>\u0e2a\u0e23\u0e38\u0e1b\u0e01\u0e32\u0e23\u0e04\u0e32\u0e14\u0e01\u0e32\u0e23\u0e13\u0e4c\u0e20\u0e32\u0e27\u0e30 Learned Non-Use:</b>
<span style="color: {lnu_color}; font-weight: bold;">{lnu_risk}</span>
</p>
<p style="margin: 5px 0 0 0; font-size: 11px; color: #94A3B8;">
* \u0e40\u0e01\u0e13\u0e11\u0e4c\u0e04\u0e33\u0e19\u0e27\u0e13\u0e08\u0e32\u0e01\u0e04\u0e27\u0e32\u0e21\u0e2a\u0e21\u0e14\u0e38\u0e25\u0e02\u0e2d\u0e07\u0e01\u0e32\u0e23\u0e2a\u0e25\u0e31\u0e1a\u0e22\u0e37\u0e48\u0e19\u0e21\u0e37\u0e2d, \u0e04\u0e27\u0e32\u0e21\u0e40\u0e23\u0e47\u0e27\u0e2a\u0e31\u0e21\u0e1c\u0e31\u0e2a, \u0e2d\u0e07\u0e28\u0e32\u0e44\u0e2b\u0e25\u0e48 (Shoulder Elevation), \u0e41\u0e25\u0e30\u0e23\u0e30\u0e14\u0e31\u0e1a\u0e04\u0e27\u0e32\u0e21\u0e41\u0e01\u0e27\u0e48\u0e07\u0e44\u0e2b\u0e27 (Jitter) \u0e23\u0e30\u0e2b\u0e27\u0e48\u0e32\u0e07\u0e01\u0e32\u0e23\u0e1b\u0e23\u0e30\u0e40\u0e21\u0e34\u0e19
</p>
</div>
</div>"""
