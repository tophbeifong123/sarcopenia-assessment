import { NextRequest, NextResponse } from "next/server";

/**
 * POST /api/generate-report
 *
 * Accepts aggregated kinematic stats from the frontend sliding window
 * and returns an empathetic, clinical summary.
 *
 * In production, this would call an LLM API (e.g., Gemini, GPT-4).
 * For the hackathon MVP, we generate a detailed deterministic report
 * so the demo works without an API key, with an optional LLM path.
 */

interface ArmStats {
  avgSpeed: number;
  maxSpeed: number;
  avgJerk: number;
  romRange: number;
  currentROM: number;
  currentStraightness: number;
}

interface ReportRequest {
  frameCount: number;
  durationMs: number;
  left: ArmStats;
  right: ArmStats;
  avgLNI: number;
  currentLNI: number;
  dominantSide: string;
}

function generateClinicalReport(data: ReportRequest): string {
  const durationSec = (data.durationMs / 1000).toFixed(1);
  const lniPercent = (data.currentLNI * 100).toFixed(1);
  const avgLniPercent = (data.avgLNI * 100).toFixed(1);

  // Determine severity
  let severity: string;
  let riskLevel: string;
  let recommendation: string;
  let emoji: string;

  if (data.currentLNI < 0.15) {
    severity = "Minimal Asymmetry";
    riskLevel = "LOW";
    emoji = "✅";
    recommendation =
      "The movement pattern appears largely symmetrical. Continue regular physical activity. No immediate clinical concern detected.";
  } else if (data.currentLNI < 0.35) {
    severity = "Mild Asymmetry";
    riskLevel = "MODERATE-LOW";
    emoji = "⚠️";
    recommendation =
      "A mild asymmetry is detected between the left and right arms. This could indicate early-stage compensatory behavior or natural dominance. Consider monitoring over multiple sessions and gentle bilateral exercises.";
  } else if (data.currentLNI < 0.55) {
    severity = "Moderate Asymmetry";
    riskLevel = "MODERATE";
    emoji = "🔶";
    recommendation =
      "A notable difference in arm usage is observed. This pattern is consistent with possible learned non-use behavior, where the less-affected arm is underutilized. Constraint-induced movement therapy (CIMT) principles may be beneficial. A clinical assessment is recommended.";
  } else {
    severity = "Significant Asymmetry";
    riskLevel = "HIGH";
    emoji = "🔴";
    recommendation =
      "A significant disparity in arm movement has been detected. The less-active arm shows reduced speed, range of motion, and movement smoothness — a pattern strongly associated with learned non-use. This individual may be at elevated risk for falls and functional decline. Urgent referral to a physiotherapist or occupational therapist is recommended.";
  }

  // Determine which arm is less active
  const lessActive = data.left.avgSpeed < data.right.avgSpeed ? "Left" : "Right";
  const moreActive = lessActive === "Left" ? "Right" : "Left";
  const lessStats = lessActive === "Left" ? data.left : data.right;
  const moreStats = lessActive === "Left" ? data.right : data.left;

  const speedRatio =
    moreStats.avgSpeed > 0
      ? ((lessStats.avgSpeed / moreStats.avgSpeed) * 100).toFixed(0)
      : "N/A";

  const jerkDiff =
    lessStats.avgJerk > 0 && moreStats.avgJerk > 0
      ? (
          ((lessStats.avgJerk - moreStats.avgJerk) / moreStats.avgJerk) *
          100
        ).toFixed(0)
      : "0";

  return `## ${emoji} Movement Analysis Report — ${severity}

**Assessment Duration:** ${durationSec} seconds (${data.frameCount} frames analyzed)
**Risk Level:** ${riskLevel}
**Learned Non-Use Index (LNI):** ${lniPercent}% (Session Avg: ${avgLniPercent}%)

## Kinematic Summary

**Dominant (More Active) Arm: ${moreActive}**
- Average Speed: ${moreStats.avgSpeed.toFixed(1)} px/s (Peak: ${moreStats.maxSpeed.toFixed(1)} px/s)
- Range of Motion: ${moreStats.romRange.toFixed(0)}° dynamic range
- Movement Jerk: ${moreStats.avgJerk.toFixed(1)} px/s³
- Path Straightness: ${(moreStats.currentStraightness * 100).toFixed(0)}%

**Less Active Arm: ${lessActive}**
- Average Speed: ${lessStats.avgSpeed.toFixed(1)} px/s (Peak: ${lessStats.maxSpeed.toFixed(1)} px/s)
- Range of Motion: ${lessStats.romRange.toFixed(0)}° dynamic range
- Movement Jerk: ${lessStats.avgJerk.toFixed(1)} px/s³
- Path Straightness: ${(lessStats.currentStraightness * 100).toFixed(0)}%

## Key Findings

- The ${lessActive} arm operates at ${speedRatio}% of the ${moreActive} arm's speed.
- Jerk differential: ${jerkDiff}% — ${Number(jerkDiff) > 20 ? "indicating less smooth, more hesitant movement in the less active arm" : "movement smoothness is relatively comparable"}.
- ROM difference: ${Math.abs(data.left.romRange - data.right.romRange).toFixed(0)}° — ${Math.abs(data.left.romRange - data.right.romRange) > 15 ? "suggesting restricted movement in the less active arm" : "within typical range"}.

## Clinical Recommendation

${recommendation}

## Fall Risk Assessment

${
  data.currentLNI > 0.4
    ? "⚠️ **Elevated fall risk detected.** The significant asymmetry in arm movement may indicate compromised balance reactions and reduced protective arm responses during unexpected perturbations. Falls prevention strategies should be implemented."
    : "Fall risk from arm asymmetry appears within acceptable parameters at this time. Continue regular monitoring."
}

---
*This report was generated by the Edge Kinematic Analysis System. It is intended for screening purposes only and does not replace a formal clinical assessment by a qualified healthcare professional.*`;
}

function generateGameClinicalReport(data: any): string {
  const durationSec = (data.durationMs / 1000).toFixed(0);
  const lniPercent = (data.lniScore * 100).toFixed(1);

  // Severity classification
  let severity = "Symmetrical Function";
  let riskLevel = "LOW";
  let recommendation = "";
  let emoji = "✅";

  if (data.lniScore < 0.15) {
    severity = "Minimal Asymmetry";
    riskLevel = "LOW";
    emoji = "✅";
    recommendation = "Excellent symmetrical arm function. The client exhibits balanced coordination, speed, and path efficiency between both sides. No clinical indications of Learned Non-Use. Recommend continuing regular bilateral physical activities to preserve motor resilience.";
  } else if (data.lniScore < 0.35) {
    severity = "Mild Asymmetry / Compensatory Behavior";
    riskLevel = "MODERATE-LOW";
    emoji = "⚠️";
    recommendation = "Mild motor asymmetry detected. The client displays slight hesitation or prolonged reach duration on the less-active side, potentially representing early compensatory movement patterns. Introduce active bilateral target reaching exercises, focusing on using the less-active arm first during daily tasks.";
  } else if (data.lniScore < 0.55) {
    severity = "Moderate Learned Non-Use";
    riskLevel = "MODERATE";
    emoji = "🔶";
    recommendation = "Moderate Learned Non-Use pattern identified. There is a clear preference for the dominant side in reaching speed, trajectory straightness, and choice of arm. Consider targeted Constraint-Induced Movement Therapy (CIMT) protocols: restrict the dominant arm for short intervals during functional tasks (e.g., eating, opening doors) to force engagement of the affected side.";
  } else {
    severity = "Significant Learned Non-Use & High Fall Risk";
    riskLevel = "HIGH";
    emoji = "🔴";
    recommendation = "Severe Learned Non-Use detected. The client relies almost exclusively on the dominant arm, while the less-active arm shows severe speed deficits, wandering path trajectory, and significant tremor/jerk. This level of motor disparity compromises bilateral stabilization and protective reactions, substantially increasing fall risks. Urgent occupational therapy or physical therapy assessment is strongly recommended.";
  }

  const left = data.left;
  const right = data.right;

  // Determine less active arm
  const lessActive = left.reaches < right.reaches ? "Left" : (left.reaches > right.reaches ? "Right" : (left.avgReachTime > right.avgReachTime ? "Left" : "Right"));
  const moreActive = lessActive === "Left" ? "Right" : "Left";
  const lessStats = lessActive === "Left" ? left : right;
  const moreStats = lessActive === "Left" ? right : left;

  const usageRatio = ((lessStats.reaches / Math.max(data.totalReaches, 1)) * 100).toFixed(0);
  const speedDiff = moreStats.avgReachTime > 0 
    ? (((lessStats.avgReachTime - moreStats.avgReachTime) / moreStats.avgReachTime) * 100).toFixed(0)
    : "0";

  return `## ${emoji} 9-Grid Reaching Test Report — ${severity}

**Assessment Duration:** ${durationSec} seconds
**Total Reaches Completed:** ${data.totalReaches} trials
**Risk Level:** ${riskLevel}
**Learned Non-Use Index (LNI):** ${lniPercent}%

## Grid Performance Summary

**Dominant (More Active) Arm: ${moreActive}**
- Reaches Completed: ${moreStats.reaches} touches (${((moreStats.reaches / Math.max(data.totalReaches, 1)) * 100).toFixed(0)}% frequency)
- Avg Reach Duration: ${(moreStats.avgReachTime / 1000).toFixed(2)}s (Min: ${(moreStats.minReachTime / 1000).toFixed(2)}s, Max: ${(moreStats.maxReachTime / 1000).toFixed(2)}s)
- Path Straightness: ${(moreStats.avgStraightness * 100).toFixed(0)}% (high ratio indicates straight-line reach)
- Movement Smoothness (Jerk): ${moreStats.avgJerk.toFixed(1)} (lower indicates steadier trajectory)

**Less Active Arm: ${lessActive}**
- Reaches Completed: ${lessStats.reaches} touches (${usageRatio}% frequency)
- Avg Reach Duration: ${(lessStats.avgReachTime / 1000).toFixed(2)}s (Min: ${(lessStats.minReachTime / 1000).toFixed(2)}s, Max: ${(lessStats.maxReachTime / 1000).toFixed(2)}s)
- Path Straightness: ${(lessStats.avgStraightness * 100).toFixed(0)}%
- Movement Smoothness (Jerk): ${lessStats.avgJerk.toFixed(1)}

## Clinical Kinematic Findings

- **Usage Disparity:** The ${lessActive} arm was selected for only ${usageRatio}% of the reaches, showing a marked preference for the ${moreActive} arm.
- **Reach Delay:** The ${lessActive} arm was ${speedDiff}% slower on average to reach targets compared to the ${moreActive} arm, indicating motor planning delay or weakness.
- **Path Deviation:** The ${lessActive} arm showed a path straightness of ${(lessStats.avgStraightness * 100).toFixed(0)}% compared to ${(moreStats.avgStraightness * 100).toFixed(0)}% for the ${moreActive} arm, reflecting coordination difficulties or tremors.
- **Tremor / Spasticity:** Movement jerk of the ${lessActive} arm was ${lessStats.avgJerk.toFixed(1)} compared to ${moreStats.avgJerk.toFixed(1)} on the other side.

## Rehabilitation Plan & Goals

1. **Short-Term Goal (2-4 weeks):** Promote forced-use of the ${lessActive} arm using structured tasks for 15-20 minutes daily. Achieve an LNI score below 40% on repeat testing.
2. **Bilateral Coordination:** Implement activities requiring stabilizing with the ${moreActive} arm and manipulating with the ${lessActive} arm.
3. **Home Exercise:** Gentle range-of-motion stretching combined with target reaching (touching sticky notes placed in a grid pattern on a wall).

## Fall Risk Assessment

${
  data.lniScore > 0.4
    ? "⚠️ **HIGH RISK:** The asymmetry in arm function strongly correlates with impaired postural adjustments. If the client slips, protective arm extension on the less-active side may be too slow or weak, resulting in high impact fractures. Fall prevention guidelines and grab bar installations are advised."
    : "The client displays adequate reflex reaction and symmetrical control; immediate fall risk from upper limb learned non-use is low."
}

---
*This report is generated dynamically by the NeuroMotion AI Engine using kinematic telemetry. It is intended for clinician review as a screening tool.*`;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Check if it is a 9-Grid Gamified Reaching Test format
    if (body.totalReaches !== undefined && body.lniScore !== undefined) {
      const report = generateGameClinicalReport(body);
      return NextResponse.json({ report });
    }

    // Fallback to old format validation
    if (
      !body.left ||
      !body.right ||
      body.frameCount === undefined ||
      body.currentLNI === undefined
    ) {
      return NextResponse.json(
        { error: "Invalid request body. Missing required kinematic data." },
        { status: 400 }
      );
    }

    const report = generateClinicalReport(body);

    return NextResponse.json({ report });
  } catch (error: any) {
    console.error("Report generation error:", error);
    return NextResponse.json(
      { error: "Failed to generate report" },
      { status: 500 }
    );
  }
}
