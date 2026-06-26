import { NextRequest, NextResponse } from "next/server";

/**
 * POST /api/generate-report
 *
 * Accepts aggregated kinematic stats from the frontend and returns an
 * empathetic, clinical screening summary.
 *
 * Primary path: Google Gemini (set GEMINI_API_KEY in the environment).
 * Fallback path: a deterministic rule-based report, so the demo still
 * works without an API key or if the LLM call fails.
 */

const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";

/**
 * Calls the Gemini REST API to produce a clinical screening report.
 * Throws on any failure so the caller can fall back to the rule-based report.
 */
async function generateGeminiReport(
  prompt: string,
  apiKey: string
): Promise<string> {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${apiKey}`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);

  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.4,
          maxOutputTokens: 2048,
          // Disable model "thinking" so the full token budget goes to the
          // actual report. On thinking models (e.g. gemini-2.5-flash) the
          // reasoning tokens otherwise consume the budget and truncate output.
          thinkingConfig: { thinkingBudget: 0 },
        },
      }),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      throw new Error(`Gemini API error ${res.status}: ${errText}`);
    }

    const json = await res.json();
    const candidate = json?.candidates?.[0];
    const text: string | undefined = candidate?.content?.parts
      ?.map((p: any) => p?.text || "")
      .join("")
      .trim();

    if (!text) {
      throw new Error("Gemini returned an empty response");
    }
    // If the report was cut off mid-way, treat it as a failure so the caller
    // falls back to the complete rule-based report instead of showing a stub.
    if (candidate?.finishReason && candidate.finishReason !== "STOP") {
      throw new Error(
        `Gemini response incomplete (finishReason: ${candidate.finishReason})`
      );
    }
    return text;
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Builds a structured prompt for the 9-Grid reaching test data so Gemini
 * produces a Thai-language clinical screening report in the markdown
 * structure the UI already renders.
 */
function buildGamePrompt(data: any): string {
  const durationSec = (data.durationMs / 1000).toFixed(0);
  const lniPercent = (data.lniScore * 100).toFixed(1);
  const left = data.left || {};
  const right = data.right || {};

  const totalReaches = Math.max(Number(data.totalReaches) || 0, 1);
  const fmt = (s: any) => {
    const reaches = s.reaches ?? 0;
    const usagePct = ((reaches / totalReaches) * 100).toFixed(0);
    return [
      `  - จำนวนครั้งที่ใช้แขนนี้แตะเป้า (usage): ${reaches} ครั้ง (${usagePct}% ของทั้งหมด)`,
      `  - เวลาเอื้อมเฉลี่ย (เร็ว = น้อยกว่า): ${((s.avgReachTime ?? 0) / 1000).toFixed(2)} วินาที`,
      `  - เวลาเอื้อม ต่ำสุด/สูงสุด: ${((s.minReachTime ?? 0) / 1000).toFixed(2)}s / ${((s.maxReachTime ?? 0) / 1000).toFixed(2)}s`,
      `  - ความตรงของเส้นทาง (สูง = ตรง/ควบคุมดี): ${((s.avgStraightness ?? 0) * 100).toFixed(0)}%`,
      `  - ความสั่นของการเคลื่อนไหว jerk (ต่ำ = นุ่มนวลกว่า): ${(s.avgJerk ?? 0).toFixed(1)}`,
    ].join("\n");
  };

  // Describe the one-sided edge case so the model interprets LNI correctly.
  const oneSided =
    (left.reaches ?? 0) === 0 || (right.reaches ?? 0) === 0
      ? "\n- หมายเหตุสำคัญ: ผู้ทดสอบใช้แขนเพียงข้างเดียวในการแตะเป้าตลอดการทดสอบ ซึ่งเป็นสัญญาณพฤติกรรม Learned Non-Use ที่ชัดเจนที่สุด"
      : "";

  return `คุณคือผู้ช่วยนักกายภาพบำบัด (virtual physiotherapist) ที่วิเคราะห์ผลการทดสอบการเอื้อมมือแตะเป้าหมายแบบ 9-Grid ของผู้สูงอายุ เพื่อคัดกรองภาวะ Learned Non-Use, ความเสี่ยงกล้ามเนื้อฝ่อลีบ (Sarcopenia) และความเสี่ยงการหกล้ม

ข้อมูลที่วัดได้จากเซสชันนี้ (telemetry จริงจากกล้องด้วย MediaPipe Pose):
- ระยะเวลาทดสอบ: ${durationSec} วินาที
- จำนวนการเอื้อมแตะทั้งหมด: ${data.totalReaches} ครั้ง
- Learned Non-Use Index (LNI): ${lniPercent}%${oneSided}
- แขนซ้าย (Left):
${fmt(left)}
- แขนขวา (Right):
${fmt(right)}

วิธีคำนวณ LNI (เพื่อให้ตีความตัวเลขถูกต้อง): LNI เป็นค่าความไม่สมมาตรระหว่างแขน 0–100% รวมจาก 4 ด้าน ได้แก่ ความต่างของเวลาเอื้อม (น้ำหนัก 30%), ความต่างของความตรงเส้นทาง (20%), ความต่างของ jerk (20%) และความต่างของจำนวนครั้งที่เลือกใช้แต่ละแขน/usage (30%) — ยิ่ง LNI สูงยิ่งไม่สมมาตรและเสี่ยงมาก โปรดวิเคราะห์โดยอ้างอิงทั้ง 4 ด้านนี้ โดยเฉพาะ usage และเวลาเอื้อม

จงเขียน "รายงานคัดกรองทางคลินิก" เป็นภาษาไทย โดยใช้รูปแบบ Markdown ตามโครงสร้างนี้พอดี (ใช้ ## เป็นหัวข้อ, ใช้ - เป็นบุลเล็ต):

## (อีโมจิระดับความเสี่ยง) รายงานผลทดสอบการเอื้อมมือ 9-Grid — (ระดับความรุนแรง)
(สรุประยะเวลา จำนวนครั้ง ระดับความเสี่ยง และ LNI)

## สรุปผลการเคลื่อนไหว
(เปรียบเทียบแขนข้างที่ใช้งานมากกับข้างที่ใช้งานน้อย ด้วยตัวเลขจริง)

## ข้อค้นพบทางคลินิก
(วิเคราะห์ความไม่สมมาตรของการใช้แขน ความเร็ว ความตรงของเส้นทาง และการสั่น/jerk)

## แผนการฟื้นฟู
(ข้อแนะนำกายภาพบำบัดที่ทำได้จริง เช่นหลักการ CIMT, การออกกำลังที่บ้าน เป็นข้อ ๆ)

## ประเมินความเสี่ยงการหกล้ม
(ประเมินความเสี่ยงการหกล้มจากความไม่สมมาตรของแขน)

ข้อกำหนด:
- อ้างอิงตัวเลขจริงจากข้อมูลข้างต้นเสมอ ห้ามแต่งตัวเลขใหม่
- เลือกอีโมจิตามความรุนแรง: LNI < 15% ใช้ ✅, < 35% ใช้ ⚠️, < 55% ใช้ 🔶, ตั้งแต่ 55% ขึ้นไปใช้ 🔴
- ใช้ภาษากระชับ เข้าใจง่าย เหมาะกับการอ่านโดยผู้ดูแลผู้สูงอายุ
- ปิดท้ายด้วยบรรทัด: "*รายงานนี้สร้างด้วย AI เพื่อการคัดกรองเบื้องต้นเท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์ และไม่ทดแทนการประเมินโดยบุคลากรทางการแพทย์*"
- ตอบเฉพาะตัวรายงาน ไม่ต้องมีคำนำหรือคำอธิบายเพิ่ม`;
}

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
- Average Speed: ${moreStats.avgSpeed.toFixed(1)} norm/s (Peak: ${moreStats.maxSpeed.toFixed(1)} norm/s)
- Range of Motion: ${moreStats.romRange.toFixed(0)}° dynamic range
- Movement Jerk: ${moreStats.avgJerk.toFixed(1)} norm/s³
- Path Straightness: ${(moreStats.currentStraightness * 100).toFixed(0)}%

**Less Active Arm: ${lessActive}**
- Average Speed: ${lessStats.avgSpeed.toFixed(1)} norm/s (Peak: ${lessStats.maxSpeed.toFixed(1)} norm/s)
- Range of Motion: ${lessStats.romRange.toFixed(0)}° dynamic range
- Movement Jerk: ${lessStats.avgJerk.toFixed(1)} norm/s³
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
*This report was generated by an automated rule-based kinematic analysis engine. It is intended for screening purposes only and does not replace a formal clinical assessment by a qualified healthcare professional.*`;
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
*This report is generated by an automated rule-based screening engine from the recorded kinematic telemetry. It is intended for clinician review as a screening tool and does not replace a formal clinical assessment.*`;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const apiKey = process.env.GEMINI_API_KEY;

    // Check if it is a 9-Grid Gamified Reaching Test format
    if (body.totalReaches !== undefined && body.lniScore !== undefined) {
      // Primary path: Gemini. Fall back to rule-based on any failure.
      if (apiKey) {
        try {
          const report = await generateGeminiReport(
            buildGamePrompt(body),
            apiKey
          );
          return NextResponse.json({ report, source: "gemini" });
        } catch (err) {
          console.error("Gemini report failed, using rule-based fallback:", err);
        }
      }
      const report = generateGameClinicalReport(body);
      return NextResponse.json({ report, source: "rule-based" });
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

    return NextResponse.json({ report, source: "rule-based" });
  } catch (error: any) {
    console.error("Report generation error:", error);
    return NextResponse.json(
      { error: "Failed to generate report" },
      { status: 500 }
    );
  }
}
