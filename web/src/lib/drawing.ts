/**
 * AR Drawing Utilities — Renders the MediaPipe skeleton AND
 * the 9-Grid gamified overlay with neon-glow effects.
 */

import { LANDMARKS, type Point3D, type ArmMetrics } from "./kinematics";
import {
  GRID_MARGIN,
  GRID_COLS,
  GRID_ROWS,
  getGridCellBounds,
  type TestState,
} from "./game-engine";

// ──────── Pose Connections ────────

const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [23, 25], [25, 27],
  [24, 26], [26, 28],
];

const LEFT_ARM_INDICES: Set<number> = new Set([
  LANDMARKS.LEFT_SHOULDER,
  LANDMARKS.LEFT_ELBOW,
  LANDMARKS.LEFT_WRIST,
]);

const RIGHT_ARM_INDICES: Set<number> = new Set([
  LANDMARKS.RIGHT_SHOULDER,
  LANDMARKS.RIGHT_ELBOW,
  LANDMARKS.RIGHT_WRIST,
]);

const COLOR_LEFT = "#22d3ee";
const COLOR_RIGHT = "#fb7185";
const COLOR_BODY = "#818cf8";

function isLeftArmConnection(a: number, b: number): boolean {
  return (LEFT_ARM_INDICES.has(a) || LEFT_ARM_INDICES.has(b)) &&
    !RIGHT_ARM_INDICES.has(a) && !RIGHT_ARM_INDICES.has(b);
}

function isRightArmConnection(a: number, b: number): boolean {
  return (RIGHT_ARM_INDICES.has(a) || RIGHT_ARM_INDICES.has(b)) &&
    !LEFT_ARM_INDICES.has(a) && !LEFT_ARM_INDICES.has(b);
}

// ──────── Skeleton Drawing ────────

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: Point3D[],
  width: number,
  height: number,
  leftMetrics?: ArmMetrics,
  rightMetrics?: ArmMetrics,
  totalHits?: number,
  mirrorLabels = true,
) {
  if (!landmarks || landmarks.length < 33) return;

  // Draw connections
  for (const [a, b] of POSE_CONNECTIONS) {
    const pA = landmarks[a];
    const pB = landmarks[b];
    if (!pA || !pB) continue;
    if ((pA.visibility ?? 0) < 0.5 || (pB.visibility ?? 0) < 0.5) continue;

    let color = COLOR_BODY;
    let lineWidth = 2;

    if (isLeftArmConnection(a, b)) {
      color = COLOR_LEFT;
      lineWidth = 3;
    } else if (isRightArmConnection(a, b)) {
      color = COLOR_RIGHT;
      lineWidth = 3;
    }

    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(pA.x * width, pA.y * height);
    ctx.lineTo(pB.x * width, pB.y * height);
    ctx.stroke();
    ctx.restore();
  }

  // Draw landmark dots
  for (let i = 0; i < Math.min(landmarks.length, 33); i++) {
    const p = landmarks[i];
    if (!p || (p.visibility ?? 0) < 0.5) continue;

    let color = COLOR_BODY;
    let radius = 3;

    if (LEFT_ARM_INDICES.has(i)) {
      color = COLOR_LEFT;
      radius = i === LANDMARKS.LEFT_WRIST ? 8 : 4;
    } else if (RIGHT_ARM_INDICES.has(i)) {
      color = COLOR_RIGHT;
      radius = i === LANDMARKS.RIGHT_WRIST ? 8 : 4;
    }

    ctx.save();
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x * width, p.y * height, radius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();
  }

  // Speed indicators on wrists
  if (leftMetrics && leftMetrics.speed > 5) {
    drawSpeedRing(ctx, landmarks[LANDMARKS.LEFT_WRIST], leftMetrics.speed, width, height, COLOR_LEFT);
  }
  if (rightMetrics && rightMetrics.speed > 5) {
    drawSpeedRing(ctx, landmarks[LANDMARKS.RIGHT_WRIST], rightMetrics.speed, width, height, COLOR_RIGHT);
  }

  // ROM labels
  if (leftMetrics) {
    drawROMLabel(ctx, landmarks[LANDMARKS.LEFT_ELBOW], leftMetrics.rom, width, height, COLOR_LEFT, "L");
  }
  if (rightMetrics) {
    drawROMLabel(ctx, landmarks[LANDMARKS.RIGHT_ELBOW], rightMetrics.rom, width, height, COLOR_RIGHT, "R");
  }

  // Hits badge
  if (totalHits !== undefined && totalHits > 0) {
    drawHitsBadge(ctx, totalHits, width);
  }

  // Draw hand highlights and labels
  const leftWrist = landmarks[LANDMARKS.LEFT_WRIST];
  const rightWrist = landmarks[LANDMARKS.RIGHT_WRIST];

  if (leftWrist && (leftWrist.visibility ?? 0) > 0.5) {
    drawHandHighlight(ctx, leftWrist, width, height, COLOR_LEFT, "มือซ้าย", mirrorLabels);
  }

  if (rightWrist && (rightWrist.visibility ?? 0) > 0.5) {
    drawHandHighlight(ctx, rightWrist, width, height, COLOR_RIGHT, "มือขวา", mirrorLabels);
  }
}

function drawHandHighlight(
  ctx: CanvasRenderingContext2D,
  point: Point3D,
  w: number,
  h: number,
  color: string,
  label: string,
  mirror = true,
) {
  const x = point.x * w;
  const y = point.y * h;
  const radius = 28; // Circle radius around hand

  ctx.save();
  
  // 1. Draw highlight circle
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.stroke();

  // 2. Draw text label above the circle (mirrored back so it displays normally under CSS mirror)
  ctx.shadowBlur = 0; // Clear shadow for text readability
  ctx.font = "bold 13px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";

  ctx.translate(x, y - radius - 8);
  if (mirror) {
    ctx.scale(-1, 1); // Flip horizontally because the parent canvas is mirrored via scaleX(-1)
  }

  // Draw outline for high contrast
  ctx.strokeStyle = "rgba(15, 23, 42, 0.9)"; // Dark slate outline
  ctx.lineWidth = 4;
  ctx.lineJoin = "round";
  ctx.strokeText(label, 0, 0);

  // Draw fill
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, 0, 0);

  ctx.restore();
}

function drawSpeedRing(
  ctx: CanvasRenderingContext2D,
  point: Point3D,
  speed: number,
  w: number,
  h: number,
  color: string,
) {
  if (!point) return;
  const x = point.x * w;
  const y = point.y * h;
  const radius = Math.min(25, 5 + speed * 0.05);

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.4;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function drawROMLabel(
  ctx: CanvasRenderingContext2D,
  point: Point3D,
  rom: number,
  w: number,
  h: number,
  color: string,
  side: string,
) {
  if (!point || (point.visibility ?? 0) < 0.5) return;
  const x = point.x * w;
  const y = point.y * h;

  ctx.save();
  ctx.font = "bold 11px Inter, system-ui, sans-serif";
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.85;
  ctx.textAlign = side === "L" ? "right" : "left";
  const offsetX = side === "L" ? -14 : 14;
  ctx.fillText(`${Math.round(rom)}°`, x + offsetX, y - 4);
  ctx.restore();
}

function drawHitsBadge(ctx: CanvasRenderingContext2D, hits: number, width: number) {
  const x = width - 16;
  const y = 30;

  ctx.save();
  const text = `Hits: ${hits}`;
  ctx.font = "bold 13px Inter, system-ui, sans-serif";
  const textWidth = ctx.measureText(text).width;
  const pillW = textWidth + 20;
  const pillH = 26;
  ctx.globalAlpha = 0.8;

  const bgColor = "rgba(52, 211, 153, 0.25)";
  const textColor = "#34d399";

  ctx.fillStyle = bgColor;
  ctx.beginPath();
  const rx = x - pillW;
  const ry = y - pillH / 2;
  const r = pillH / 2;
  ctx.moveTo(rx + r, ry);
  ctx.lineTo(rx + pillW - r, ry);
  ctx.arcTo(rx + pillW, ry, rx + pillW, ry + r, r);
  ctx.arcTo(rx + pillW, ry + pillH, rx + pillW - r, ry + pillH, r);
  ctx.lineTo(rx + r, ry + pillH);
  ctx.arcTo(rx, ry + pillH, rx, ry + pillH - r, r);
  ctx.arcTo(rx, ry, rx + r, ry, r);
  ctx.fill();

  ctx.fillStyle = textColor;
  ctx.globalAlpha = 1;
  ctx.textAlign = "right";
  ctx.fillText(text, x - 10, y + 5);
  ctx.restore();
}

// ──────── 9-Grid Drawing ────────

/**
 * Draw the 3x3 grid overlay on the canvas.
 */
export function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  activeCell: number,
  testState: TestState,
  hitFlashCell: number | null,
  frameTime: number,
) {
  const gridX = GRID_MARGIN * width;
  const gridY = GRID_MARGIN * height;
  const gridW = (1 - 2 * GRID_MARGIN) * width;
  const gridH = (1 - 2 * GRID_MARGIN) * height;
  const cellW = gridW / GRID_COLS;
  const cellH = gridH / GRID_ROWS;
  // Draw each cell
  for (let i = 0; i < GRID_COLS * GRID_ROWS; i++) {
    const row = Math.floor(i / GRID_COLS);
    const col = i % GRID_COLS;
    const cx = gridX + col * cellW;
    const cy = gridY + row * cellH;

    ctx.save();

    if ((testState === "STARTING_POSTURE" || testState === "COUNTDOWN") && (i === 0 || i === 2)) {
      // ── Starting posture guide boxes (top-left & top-right) ──
      const pulse = 0.5 + 0.5 * Math.sin(frameTime * 0.005);
      const alpha = 0.15 + pulse * 0.15;
      
      ctx.shadowColor = "#eab308";
      ctx.shadowBlur = 15 + pulse * 10;
      ctx.fillStyle = `rgba(234, 179, 8, ${alpha * 0.4})`;
      ctx.fillRect(cx + 2, cy + 2, cellW - 4, cellH - 4);

      ctx.strokeStyle = `rgba(234, 179, 8, ${0.6 + pulse * 0.4})`;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(cx + 2, cy + 2, cellW - 4, cellH - 4);

      const centerX = cx + cellW / 2;
      const centerY = cy + cellH / 2;
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.scale(-1, 1);
      ctx.font = "bold 11px Inter, system-ui, sans-serif";
      ctx.fillStyle = "#eab308";
      ctx.textAlign = "center";
      // i === 0 is visually on the right under scaleX(-1), so user's right hand goes there (R)
      // i === 2 is visually on the left under scaleX(-1), so user's left hand goes there (L)
      ctx.fillText(i === 0 ? "START POSTURE R" : "START POSTURE L", 0, 0);
      ctx.restore();

    } else if (testState === "PLAYING" && i === activeCell) {
      // ── Active cell: neon glow ──
      const pulse = 0.5 + 0.5 * Math.sin(frameTime * 0.005);
      const alpha = 0.15 + pulse * 0.15;

      // Glow fill
      ctx.shadowColor = "#22d3ee";
      ctx.shadowBlur = 20 + pulse * 15;
      ctx.fillStyle = `rgba(34, 211, 238, ${alpha})`;
      ctx.fillRect(cx + 2, cy + 2, cellW - 4, cellH - 4);

      // Border
      ctx.strokeStyle = `rgba(34, 211, 238, ${0.6 + pulse * 0.4})`;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(cx + 2, cy + 2, cellW - 4, cellH - 4);

      // Target crosshair
      const centerX = cx + cellW / 2;
      const centerY = cy + cellH / 2;
      const crossSize = Math.min(cellW, cellH) * 0.15;

      ctx.strokeStyle = `rgba(34, 211, 238, ${0.4 + pulse * 0.3})`;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(centerX - crossSize, centerY);
      ctx.lineTo(centerX + crossSize, centerY);
      ctx.moveTo(centerX, centerY - crossSize);
      ctx.lineTo(centerX, centerY + crossSize);
      ctx.stroke();

      // Target circle
      ctx.beginPath();
      ctx.arc(centerX, centerY, crossSize * 0.7, 0, Math.PI * 2);
      ctx.stroke();

    } else if (hitFlashCell === i) {
      // ── Hit flash effect ──
      ctx.fillStyle = "rgba(52, 211, 153, 0.4)";
      ctx.fillRect(cx + 2, cy + 2, cellW - 4, cellH - 4);
      ctx.strokeStyle = "rgba(52, 211, 153, 0.8)";
      ctx.lineWidth = 2;
      ctx.strokeRect(cx + 2, cy + 2, cellW - 4, cellH - 4);

    } else {
      // ── Inactive cell ──
      // No fill to make it 100% transparent and allow perfect visibility
      ctx.strokeStyle = testState === "IDLE"
        ? "rgba(255, 255, 255, 0.08)"
        : "rgba(255, 255, 255, 0.04)";
      ctx.lineWidth = 1;
      if (testState === "IDLE") {
        ctx.setLineDash([4, 4]);
      }
      ctx.strokeRect(cx + 2, cy + 2, cellW - 4, cellH - 4);
      ctx.setLineDash([]);
    }

    // Cell label (small, top-left corner from user's perspective)
    if (testState !== "EVALUATING") {
      ctx.save();
      ctx.font = "10px Inter, system-ui, sans-serif";
      ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
      
      const visualCol = 2 - col;
      const visualRow = row;
      const lane = visualRow * 3 + visualCol + 1;

      // Translate to cell top-left visually, scale(-1, 1), and write
      ctx.translate(cx + cellW - 8, cy + 16);
      ctx.scale(-1, 1);
      ctx.textAlign = "left";
      ctx.fillText(`${lane}`, 0, 0);
      ctx.restore();
    }

    ctx.restore();
  }
}

/**
 * Draw a hit sparkle animation at a point.
 */
export function drawHitSparkle(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  progress: number, // 0–1
) {
  const px = x * width;
  const py = y * height;
  const maxRadius = 30;
  const radius = maxRadius * progress;
  const alpha = 1 - progress;

  ctx.save();
  ctx.strokeStyle = `rgba(52, 211, 153, ${alpha})`;
  ctx.lineWidth = 2;
  ctx.shadowColor = "#34d399";
  ctx.shadowBlur = 15;
  ctx.beginPath();
  ctx.arc(px, py, radius, 0, Math.PI * 2);
  ctx.stroke();

  // Inner ring
  ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.5})`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(px, py, radius * 0.5, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}
