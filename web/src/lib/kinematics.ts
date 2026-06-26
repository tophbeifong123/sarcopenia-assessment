/**
 * Edge Kinematic Engine — runs entirely in the browser.
 * Calculates Speed, Path Straightness, Jerk, and Learned Non-Use Index (LNI)
 * from MediaPipe Pose landmarks.
 */

// MediaPipe Pose landmark indices
export const LANDMARKS = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
} as const;

export interface Point3D {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface ArmMetrics {
  speed: number;         // normalized units/s — instantaneous wrist speed
  acceleration: number;  // normalized units/s² — instantaneous acceleration
  jerk: number;          // normalized units/s³ — rate of change of acceleration (smoothness)
  straightness: number;  // 0-1 — ratio of displacement to path length
  rom: number;           // shoulder elevation angle in degrees
}

export interface FrameData {
  timestamp: number;     // performance.now() ms
  leftWrist: Point3D;
  rightWrist: Point3D;
  leftElbow: Point3D;
  rightElbow: Point3D;
  leftShoulder: Point3D;
  rightShoulder: Point3D;
  leftHip: Point3D;
  rightHip: Point3D;
  leftMetrics: ArmMetrics;
  rightMetrics: ArmMetrics;
  lniScore: number;      // Learned Non-Use Index: 0 = equal, 1 = max asymmetry
}

// ──────── Math utilities ────────

function dist(a: Point3D, b: Point3D): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2);
}

/**
 * Shoulder elevation/abduction angle — the angle between the upper arm
 * (shoulder→elbow) and the torso (shoulder→hip).
 * This matches the Streamlit analyzer's `calculate_shoulder_angle`, so both
 * systems measure the SAME joint and the values are directly comparable.
 */
function shoulderAngle(shoulder: Point3D, elbow: Point3D, hip: Point3D): number {
  const vArm = { x: elbow.x - shoulder.x, y: elbow.y - shoulder.y, z: elbow.z - shoulder.z };
  const vTorso = { x: hip.x - shoulder.x, y: hip.y - shoulder.y, z: hip.z - shoulder.z };
  const dot = vArm.x * vTorso.x + vArm.y * vTorso.y + vArm.z * vTorso.z;
  const magArm = Math.sqrt(vArm.x ** 2 + vArm.y ** 2 + vArm.z ** 2);
  const magTorso = Math.sqrt(vTorso.x ** 2 + vTorso.y ** 2 + vTorso.z ** 2);
  if (magArm === 0 || magTorso === 0) return 0;
  const cosAngle = Math.max(-1, Math.min(1, dot / (magArm * magTorso)));
  return (Math.acos(cosAngle) * 180) / Math.PI;
}

// ──────── Sliding Window Buffer ────────

export class KinematicsBuffer {
  private buffer: FrameData[] = [];
  private readonly maxSize: number;

  // Accumulators for path length (for straightness)
  private leftPathLength = 0;
  private rightPathLength = 0;

  // Track min/max ROM over the window
  private leftMinROM = Infinity;
  private leftMaxROM = -Infinity;
  private rightMinROM = Infinity;
  private rightMaxROM = -Infinity;

  constructor(maxSize = 120) {
    this.maxSize = maxSize;
  }

  get frames(): readonly FrameData[] {
    return this.buffer;
  }

  get length(): number {
    return this.buffer.length;
  }

  get latestFrame(): FrameData | undefined {
    return this.buffer[this.buffer.length - 1];
  }

  /**
   * Push a new set of landmarks and compute all metrics.
   */
  push(landmarks: Point3D[], timestamp: number): FrameData {
    const leftWrist = landmarks[LANDMARKS.LEFT_WRIST];
    const rightWrist = landmarks[LANDMARKS.RIGHT_WRIST];
    const leftElbow = landmarks[LANDMARKS.LEFT_ELBOW];
    const rightElbow = landmarks[LANDMARKS.RIGHT_ELBOW];
    const leftShoulder = landmarks[LANDMARKS.LEFT_SHOULDER];
    const rightShoulder = landmarks[LANDMARKS.RIGHT_SHOULDER];
    const leftHip = landmarks[LANDMARKS.LEFT_HIP];
    const rightHip = landmarks[LANDMARKS.RIGHT_HIP];

    const prev = this.buffer[this.buffer.length - 1];
    const prevPrev = this.buffer[this.buffer.length - 2];

    // ── dt in seconds ──
    const dt = prev ? (timestamp - prev.timestamp) / 1000 : 0;
    const dtSafe = dt > 0 ? dt : 1 / 30; // fallback 30fps

    // ── Speed (px/s) ──
    const leftSpeed = prev ? dist(leftWrist, prev.leftWrist) / dtSafe : 0;
    const rightSpeed = prev ? dist(rightWrist, prev.rightWrist) / dtSafe : 0;

    // ── Acceleration (px/s²) ──
    const leftAccel = prev ? Math.abs(leftSpeed - prev.leftMetrics.speed) / dtSafe : 0;
    const rightAccel = prev ? Math.abs(rightSpeed - prev.rightMetrics.speed) / dtSafe : 0;

    // ── Jerk (px/s³) — rate of change of acceleration ──
    const leftJerk = prevPrev ? Math.abs(leftAccel - prev.leftMetrics.acceleration) / dtSafe : 0;
    const rightJerk = prevPrev ? Math.abs(rightAccel - prev.rightMetrics.acceleration) / dtSafe : 0;

    // ── Path length accumulation ──
    if (prev) {
      this.leftPathLength += dist(leftWrist, prev.leftWrist);
      this.rightPathLength += dist(rightWrist, prev.rightWrist);
    }

    // ── Straightness — displacement / path length ──
    const leftStartWrist = this.buffer.length > 0 ? this.buffer[0].leftWrist : leftWrist;
    const rightStartWrist = this.buffer.length > 0 ? this.buffer[0].rightWrist : rightWrist;
    const leftDisplacement = dist(leftWrist, leftStartWrist);
    const rightDisplacement = dist(rightWrist, rightStartWrist);
    const leftStraightness = this.leftPathLength > 0.001
      ? Math.min(1, leftDisplacement / this.leftPathLength)
      : 1;
    const rightStraightness = this.rightPathLength > 0.001
      ? Math.min(1, rightDisplacement / this.rightPathLength)
      : 1;

    // ── ROM — shoulder elevation angle (matches the Streamlit analyzer) ──
    const leftROM = shoulderAngle(leftShoulder, leftElbow, leftHip);
    const rightROM = shoulderAngle(rightShoulder, rightElbow, rightHip);
    this.leftMinROM = Math.min(this.leftMinROM, leftROM);
    this.leftMaxROM = Math.max(this.leftMaxROM, leftROM);
    this.rightMinROM = Math.min(this.rightMinROM, rightROM);
    this.rightMaxROM = Math.max(this.rightMaxROM, rightROM);

    const leftMetrics: ArmMetrics = {
      speed: leftSpeed,
      acceleration: leftAccel,
      jerk: leftJerk,
      straightness: leftStraightness,
      rom: leftROM,
    };

    const rightMetrics: ArmMetrics = {
      speed: rightSpeed,
      acceleration: rightAccel,
      jerk: rightJerk,
      straightness: rightStraightness,
      rom: rightROM,
    };

    // ── Learned Non-Use Index (LNI) ──
    const lniScore = this.computeLNI(leftMetrics, rightMetrics);

    const frame: FrameData = {
      timestamp,
      leftWrist,
      rightWrist,
      leftElbow,
      rightElbow,
      leftShoulder,
      rightShoulder,
      leftHip,
      rightHip,
      leftMetrics,
      rightMetrics,
      lniScore,
    };

    this.buffer.push(frame);

    // Evict oldest frame if over capacity
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift();
    }

    return frame;
  }

  /**
   * Compute the LIVE (real-time) Learned Non-Use Index preview.
   *
   * This is a 3-axis preview shown while the user is moving: it blends
   * asymmetry in average wrist SPEED, shoulder ROM range, and movement JERK.
   * It does NOT include usage/choice frequency because no reach targets have
   * been completed yet during free movement.
   *
   * The authoritative end-of-test score is computed separately in
   * `aggregateResults` (game-engine.ts), which adds a 4th axis (which arm the
   * user actually CHOSE to reach with). The two intentionally differ in scope:
   * this one previews motor asymmetry, the final one scores behavioural non-use.
   *
   * Returns 0–1 where 0 = perfectly symmetric, 1 = maximum asymmetry.
   */
  private computeLNI(left: ArmMetrics, right: ArmMetrics): number {
    if (this.buffer.length < 5) return 0;

    // Average metrics over the window
    const n = this.buffer.length;
    let avgLeftSpeed = 0, avgRightSpeed = 0;
    let avgLeftJerk = 0, avgRightJerk = 0;
    const windowSize = Math.min(n, 60);

    for (let i = n - windowSize; i < n; i++) {
      const f = this.buffer[i];
      avgLeftSpeed += f.leftMetrics.speed;
      avgRightSpeed += f.rightMetrics.speed;
      avgLeftJerk += f.leftMetrics.jerk;
      avgRightJerk += f.rightMetrics.jerk;
    }
    avgLeftSpeed /= windowSize;
    avgRightSpeed /= windowSize;
    avgLeftJerk /= windowSize;
    avgRightJerk /= windowSize;

    // Speed asymmetry (0–1)
    const totalSpeed = avgLeftSpeed + avgRightSpeed;
    const speedAsymmetry = totalSpeed > 0.1
      ? Math.abs(avgLeftSpeed - avgRightSpeed) / totalSpeed
      : 0;

    // ROM asymmetry
    const leftROMRange = this.leftMaxROM - this.leftMinROM;
    const rightROMRange = this.rightMaxROM - this.rightMinROM;
    const totalROM = leftROMRange + rightROMRange;
    const romAsymmetry = totalROM > 1
      ? Math.abs(leftROMRange - rightROMRange) / totalROM
      : 0;

    // Jerk asymmetry — higher jerk means less smooth movement
    const totalJerk = avgLeftJerk + avgRightJerk;
    const jerkAsymmetry = totalJerk > 0.1
      ? Math.abs(avgLeftJerk - avgRightJerk) / totalJerk
      : 0;

    // Weighted composite
    const lni = (
      speedAsymmetry * 0.40 +
      romAsymmetry * 0.30 +
      jerkAsymmetry * 0.30
    );

    return Math.min(1, Math.max(0, lni));
  }

  /**
   * Get aggregated stats for the report.
   */
  getAggregatedStats() {
    if (this.buffer.length < 2) {
      return null;
    }

    const n = this.buffer.length;
    let leftSpeedSum = 0, rightSpeedSum = 0;
    let leftJerkSum = 0, rightJerkSum = 0;
    let leftSpeedMax = 0, rightSpeedMax = 0;
    let lniSum = 0;

    for (const f of this.buffer) {
      leftSpeedSum += f.leftMetrics.speed;
      rightSpeedSum += f.rightMetrics.speed;
      leftJerkSum += f.leftMetrics.jerk;
      rightJerkSum += f.rightMetrics.jerk;
      leftSpeedMax = Math.max(leftSpeedMax, f.leftMetrics.speed);
      rightSpeedMax = Math.max(rightSpeedMax, f.rightMetrics.speed);
      lniSum += f.lniScore;
    }

    return {
      frameCount: n,
      durationMs: this.buffer[n - 1].timestamp - this.buffer[0].timestamp,
      left: {
        avgSpeed: leftSpeedSum / n,
        maxSpeed: leftSpeedMax,
        avgJerk: leftJerkSum / n,
        romRange: this.leftMaxROM - this.leftMinROM,
        currentROM: this.buffer[n - 1].leftMetrics.rom,
        currentStraightness: this.buffer[n - 1].leftMetrics.straightness,
      },
      right: {
        avgSpeed: rightSpeedSum / n,
        maxSpeed: rightSpeedMax,
        avgJerk: rightJerkSum / n,
        romRange: this.rightMaxROM - this.rightMinROM,
        currentROM: this.buffer[n - 1].rightMetrics.rom,
        currentStraightness: this.buffer[n - 1].rightMetrics.straightness,
      },
      avgLNI: lniSum / n,
      currentLNI: this.buffer[n - 1].lniScore,
      dominantSide: (leftSpeedSum > rightSpeedSum) ? "left" : "right",
    };
  }

  /**
   * Get the recent chart data for Recharts (last N points, downsampled).
   */
  getChartData(maxPoints = 60): Array<{
    t: number;
    leftSpeed: number;
    rightSpeed: number;
    leftJerk: number;
    rightJerk: number;
    lni: number;
    leftROM: number;
    rightROM: number;
  }> {
    const step = Math.max(1, Math.floor(this.buffer.length / maxPoints));
    const data = [];
    for (let i = 0; i < this.buffer.length; i += step) {
      const f = this.buffer[i];
      data.push({
        t: Math.round((f.timestamp - this.buffer[0].timestamp) / 100) / 10,
        leftSpeed: Math.round(f.leftMetrics.speed * 10) / 10,
        rightSpeed: Math.round(f.rightMetrics.speed * 10) / 10,
        leftJerk: Math.round(f.leftMetrics.jerk),
        rightJerk: Math.round(f.rightMetrics.jerk),
        lni: Math.round(f.lniScore * 100) / 100,
        leftROM: Math.round(f.leftMetrics.rom),
        rightROM: Math.round(f.rightMetrics.rom),
      });
    }
    return data;
  }

  reset() {
    this.buffer = [];
    this.leftPathLength = 0;
    this.rightPathLength = 0;
    this.leftMinROM = Infinity;
    this.leftMaxROM = -Infinity;
    this.rightMinROM = Infinity;
    this.rightMaxROM = -Infinity;
  }
}
