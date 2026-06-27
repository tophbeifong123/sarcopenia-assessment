/**
 * Game Engine — Types, grid math, collision detection,
 * reach metrics computation, and results aggregation
 * for the 9-Grid Gamified Reaching Clinical Test.
 */

// ──────── Types ────────

export type TestState = "IDLE" | "STARTING_POSTURE" | "COUNTDOWN" | "PLAYING" | "EVALUATING";

export interface PathPoint {
  x: number;  // normalized 0–1
  y: number;  // normalized 0–1
  t: number;  // performance.now() ms
}

export interface CellHitData {
  arm: "left" | "right";
  cellIndex: number;
  timestamp: number;
  pathPoints: PathPoint[];
  reachStartTime: number;
}

export interface ReachRecord {
  index: number;
  targetCell: number;
  arm: "left" | "right";
  reachTimeMs: number;
  straightness: number;  // 0–1, higher = straighter path
  jerk: number;          // lower = smoother movement
}

export interface ArmStats {
  reaches: number;
  avgReachTime: number;
  avgStraightness: number;
  avgJerk: number;
  maxReachTime: number;
  minReachTime: number;
}

export interface GameResults {
  totalReaches: number;
  durationMs: number;
  left: ArmStats;
  right: ArmStats;
  lniScore: number;
  reaches: ReachRecord[];
}

// ──────── Grid Constants ────────

export const GRID_MARGIN = 0.00;  // 0% margin to fill the entire screen
export const GRID_COLS = 3;
export const GRID_ROWS = 3;
export const GRID_TOTAL = GRID_COLS * GRID_ROWS;

export interface CellBounds {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  row: number;
  col: number;
}

/**
 * Get the bounding box of a grid cell in normalized (0–1) coordinates.
 */
export function getGridCellBounds(cellIndex: number): CellBounds {
  const row = Math.floor(cellIndex / GRID_COLS);
  const col = cellIndex % GRID_COLS;
  const gridW = 1 - 2 * GRID_MARGIN;
  const gridH = 1 - 2 * GRID_MARGIN;
  const cellW = gridW / GRID_COLS;
  const cellH = gridH / GRID_ROWS;

  const x1 = GRID_MARGIN + col * cellW;
  const y1 = GRID_MARGIN + row * cellH;

  return {
    x1,
    y1,
    x2: x1 + cellW,
    y2: y1 + cellH,
    cx: x1 + cellW / 2,
    cy: y1 + cellH / 2,
    row,
    col,
  };
}

/**
 * Check if a normalized (x, y) point is inside a grid cell.
 */
export function isPointInCell(x: number, y: number, cellIndex: number): boolean {
  const b = getGridCellBounds(cellIndex);
  return x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2;
}

/**
 * Check if a circle centered at normalized (x, y) with a pixel radius overlaps with a grid cell.
 */
export function isCircleCollidingWithCell(
  x: number,
  y: number,
  cellIndex: number,
  width: number,
  height: number,
  radius: number = 28
): boolean {
  const b = getGridCellBounds(cellIndex);

  // Find the closest point on the cell bounding box to the circle center
  const closestX = Math.max(b.x1, Math.min(x, b.x2));
  const closestY = Math.max(b.y1, Math.min(y, b.y2));

  // Calculate distance in pixel space
  const dx = (x - closestX) * width;
  const dy = (y - closestY) * height;

  return (dx * dx + dy * dy) <= (radius * radius);
}

export function getRandomCell(excludeIndex: number): number {
  let next: number;
  do {
    next = Math.floor(Math.random() * GRID_TOTAL);
  } while (next === excludeIndex);
  return next;
}

/**
 * Pick a random cell index that is NOT in the exclude list.
 */
export function getRandomCellWithExclusions(excludeIndices: number[]): number {
  const possibleCells: number[] = [];
  for (let i = 0; i < GRID_TOTAL; i++) {
    if (!excludeIndices.includes(i)) {
      possibleCells.push(i);
    }
  }
  if (possibleCells.length === 0) {
    // Fallback in case everything is excluded
    return Math.floor(Math.random() * GRID_TOTAL);
  }
  const randIdx = Math.floor(Math.random() * possibleCells.length);
  return possibleCells[randIdx];
}

/**
 * Pick a random cell index that is NOT in the exclude list,
 * while balancing the selections between Left, Center, and Right columns.
 * 
 * Column mapping:
 * - Left (Col 0): cells 0, 3, 6
 * - Center (Col 1): cells 1, 4, 7
 * - Right (Col 2): cells 2, 5, 8
 */
export function getBalancedRandomCell(
  excludeIndices: number[],
  cellHistory: number[]
): number {
  const possibleCells: number[] = [];
  for (let i = 0; i < GRID_TOTAL; i++) {
    if (!excludeIndices.includes(i)) {
      possibleCells.push(i);
    }
  }

  if (possibleCells.length === 0) {
    // Fallback: pick any random cell
    return Math.floor(Math.random() * GRID_TOTAL);
  }

  // 1. Calculate column index for each cell in history
  const getCol = (cell: number) => cell % GRID_COLS;

  // 2. Count occurrences of each column in history
  const colCounts = [0, 0, 0];
  for (const cell of cellHistory) {
    const col = getCol(cell);
    if (col >= 0 && col < 3) {
      colCounts[col]++;
    }
  }

  // 3. Among the possible (non-excluded) cells, group them by column
  const cellsByCol: Record<number, number[]> = { 0: [], 1: [], 2: [] };
  for (const cell of possibleCells) {
    const col = getCol(cell);
    cellsByCol[col].push(cell);
  }

  // 4. Find the minimum column selection count among the columns that have at least one possible cell
  let minCount = Infinity;
  for (let col = 0; col < 3; col++) {
    if (cellsByCol[col].length > 0) {
      if (colCounts[col] < minCount) {
        minCount = colCounts[col];
      }
    }
  }

  // 5. Find all columns that share this minimum count (and have possible cells)
  const candidateCols: number[] = [];
  for (let col = 0; col < 3; col++) {
    if (cellsByCol[col].length > 0 && colCounts[col] === minCount) {
      candidateCols.push(col);
    }
  }

  // 6. Gather all possible cells belonging to these candidate columns
  const candidateCells: number[] = [];
  for (const col of candidateCols) {
    candidateCells.push(...cellsByCol[col]);
  }

  if (candidateCells.length === 0) {
    // Fallback: pick any possible cell
    const randIdx = Math.floor(Math.random() * possibleCells.length);
    return possibleCells[randIdx];
  }

  // 7. Pick a random cell from candidate cells
  const randIdx = Math.floor(Math.random() * candidateCells.length);
  return candidateCells[randIdx];
}


// ──────── Reach Metrics ────────

/**
 * Path straightness — ratio of start-to-end displacement to total path length.
 * 1.0 = perfectly straight, 0.0 = extremely wandering.
 */
export function calculateStraightness(path: PathPoint[]): number {
  if (path.length < 2) return 1;

  const start = path[0];
  const end = path[path.length - 1];

  const displacement = Math.sqrt(
    (end.x - start.x) ** 2 + (end.y - start.y) ** 2
  );

  let pathLength = 0;
  for (let i = 1; i < path.length; i++) {
    pathLength += Math.sqrt(
      (path[i].x - path[i - 1].x) ** 2 +
      (path[i].y - path[i - 1].y) ** 2
    );
  }

  if (pathLength < 0.001) return 1;
  return Math.min(1, displacement / pathLength);
}

/**
 * Average jerk magnitude — measures movement smoothness.
 * Jerk = derivative of acceleration. Lower = smoother.
 */
export function calculateJerk(path: PathPoint[]): number {
  if (path.length < 4) return 0;

  // Velocities
  const vel: { vx: number; vy: number; t: number }[] = [];
  for (let i = 1; i < path.length; i++) {
    const dt = (path[i].t - path[i - 1].t) / 1000;
    if (dt <= 0) continue;
    vel.push({
      vx: (path[i].x - path[i - 1].x) / dt,
      vy: (path[i].y - path[i - 1].y) / dt,
      t: path[i].t,
    });
  }

  // Accelerations
  const acc: { ax: number; ay: number; t: number }[] = [];
  for (let i = 1; i < vel.length; i++) {
    const dt = (vel[i].t - vel[i - 1].t) / 1000;
    if (dt <= 0) continue;
    acc.push({
      ax: (vel[i].vx - vel[i - 1].vx) / dt,
      ay: (vel[i].vy - vel[i - 1].vy) / dt,
      t: vel[i].t,
    });
  }

  // Jerk
  let totalJerk = 0;
  let count = 0;
  for (let i = 1; i < acc.length; i++) {
    const dt = (acc[i].t - acc[i - 1].t) / 1000;
    if (dt <= 0) continue;
    const jx = (acc[i].ax - acc[i - 1].ax) / dt;
    const jy = (acc[i].ay - acc[i - 1].ay) / dt;
    totalJerk += Math.sqrt(jx ** 2 + jy ** 2);
    count++;
  }

  return count > 0 ? totalJerk / count : 0;
}

/**
 * Process a raw hit event into a ReachRecord with computed metrics.
 */
export function processHit(hitData: CellHitData, reachIndex: number): ReachRecord {
  const reachTimeMs = hitData.timestamp - hitData.reachStartTime;
  const straightness = calculateStraightness(hitData.pathPoints);
  const jerk = calculateJerk(hitData.pathPoints);

  return {
    index: reachIndex,
    targetCell: hitData.cellIndex,
    arm: hitData.arm,
    reachTimeMs,
    straightness,
    jerk,
  };
}

// ──────── Results Aggregation ────────

function calcArmStats(records: ReachRecord[]): ArmStats {
  if (records.length === 0) {
    return {
      reaches: 0,
      avgReachTime: 0,
      avgStraightness: 0,
      avgJerk: 0,
      maxReachTime: 0,
      minReachTime: 0,
    };
  }

  const times = records.map((r) => r.reachTimeMs);
  const straights = records.map((r) => r.straightness);
  const jerks = records.map((r) => r.jerk);

  return {
    reaches: records.length,
    avgReachTime: times.reduce((a, b) => a + b, 0) / times.length,
    avgStraightness: straights.reduce((a, b) => a + b, 0) / straights.length,
    avgJerk: jerks.reduce((a, b) => a + b, 0) / jerks.length,
    maxReachTime: Math.max(...times),
    minReachTime: Math.min(...times),
  };
}

/**
 * Aggregate all reach records into final game results.
 */
export function aggregateResults(
  reaches: ReachRecord[],
  durationMs: number,
): GameResults {
  const left = reaches.filter((r) => r.arm === "left");
  const right = reaches.filter((r) => r.arm === "right");

  const leftStats = calcArmStats(left);
  const rightStats = calcArmStats(right);

  // ── LNI Calculation (authoritative end-of-test score) ──
  //
  // This is the 4-axis score: it blends asymmetry in reach SPEED (time),
  // path STRAIGHTNESS, movement JERK, and — crucially — USAGE FREQUENCY
  // (which arm the user actually chose to reach with). The live preview in
  // kinematics.ts is a 3-axis motor-only estimate; this one is the clinical
  // result because choosing not to use an arm is the core sign of non-use.
  // Returns 0–1 where 0 = perfectly symmetric, 1 = maximum asymmetry.
  let lniScore = 0;
  if (leftStats.reaches > 0 && rightStats.reaches > 0) {
    // Speed asymmetry: relative difference in average reach time (0–1)
    const maxAvgTime = Math.max(leftStats.avgReachTime, rightStats.avgReachTime);
    const speedAsym = maxAvgTime > 0
      ? Math.abs(leftStats.avgReachTime - rightStats.avgReachTime) / maxAvgTime
      : 0;

    // Straightness asymmetry — both values are already in 0–1, so their
    // absolute difference is also bounded to 0–1.
    const straightAsym = Math.abs(
      leftStats.avgStraightness - rightStats.avgStraightness
    );

    // Jerk asymmetry (0–1)
    const totalJerk = leftStats.avgJerk + rightStats.avgJerk;
    const jerkAsym = totalJerk > 0
      ? Math.abs(leftStats.avgJerk - rightStats.avgJerk) / totalJerk
      : 0;

    // Usage asymmetry — which arm is used more (0–1)
    const totalReaches = leftStats.reaches + rightStats.reaches;
    const usageAsym = totalReaches > 0
      ? Math.abs(leftStats.reaches - rightStats.reaches) / totalReaches
      : 0;

    lniScore =
      speedAsym * 0.30 +
      straightAsym * 0.20 +
      jerkAsym * 0.20 +
      usageAsym * 0.30;

    lniScore = Math.min(1, Math.max(0, lniScore));
  } else if (leftStats.reaches > 0 || rightStats.reaches > 0) {
    // Only one arm was ever used to reach a target. This is total one-sided
    // usage — the strongest behavioural sign of learned non-use — so usage
    // asymmetry is maximal (1.0). We weight it by the usage axis and add a
    // baseline for the unmeasurable motor axes of the unused arm.
    lniScore = Math.min(1, 0.30 * 1.0 + 0.45);
  }

  return {
    totalReaches: reaches.length,
    durationMs,
    left: leftStats,
    right: rightStats,
    lniScore,
    reaches,
  };
}
