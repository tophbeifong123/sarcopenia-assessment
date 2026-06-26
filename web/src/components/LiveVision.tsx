"use client";

import React, {
  useRef,
  useEffect,
  useCallback,
  useState,
} from "react";
import Webcam from "react-webcam";
import { KinematicsBuffer, type FrameData, type Point3D } from "@/lib/kinematics";
import { drawSkeleton, drawGrid, drawHitSparkle, getCoverFit } from "@/lib/drawing";
import { TestState, isCircleCollidingWithCell, type PathPoint } from "@/lib/game-engine";

// MediaPipe Pose types (loaded via CDN script, available on window)
interface MediaPipePose {
  setOptions(options: Record<string, unknown>): void;
  onResults(callback: (results: MediaPipeResults) => void): void;
  send(inputs: { image: HTMLVideoElement }): Promise<void>;
  close(): Promise<void>;
  initialize(): Promise<void>;
}

interface MediaPipeResults {
  poseLandmarks?: Point3D[];
  poseWorldLandmarks?: Point3D[];
  image?: HTMLCanvasElement | HTMLImageElement | ImageBitmap;
}

interface MediaPipePoseConstructor {
  new (config: { locateFile: (file: string) => string }): MediaPipePose;
}

declare global {
  interface Window {
    Pose?: MediaPipePoseConstructor;
  }
}

interface LiveVisionProps {
  onFrame?: (frame: FrameData) => void;
  kinematicsBuffer: React.MutableRefObject<KinematicsBuffer>;
  isRunning: boolean;
  testState: TestState;
  activeCell: number;
  hitFlashCell: number | null;
  onHit: (
    arm: "left" | "right",
    cellIndex: number,
    path: PathPoint[],
    leftHandCell: number,
    rightHandCell: number
  ) => void;
  onPostureAchieved: (achieved: boolean) => void;
  totalHits?: number;
  mirrorView?: boolean;
}

/**
 * Load a script from CDN and return a promise that resolves when loaded.
 */
function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    // Check if already loaded
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.crossOrigin = "anonymous";
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}

/**
 * Wait for MediaPipe Pose to be available on the global scope.
 * The IIFE in pose.js registers via G() which attaches to `this` (globalThis).
 */
function waitForPose(timeoutMs = 10000): Promise<MediaPipePoseConstructor> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    function check() {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const g = globalThis as any;
      const PoseClass = g.Pose || window.Pose;
      if (PoseClass) {
        resolve(PoseClass);
        return;
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error("Timeout waiting for MediaPipe Pose to load"));
        return;
      }
      setTimeout(check, 100);
    }
    check();
  });
}

/**
 * Determine which cell index (0-8) the hand's outer circle (radius 28) overlaps. Returns -1 if outside.
 */
function getHandCellIndex(point: Point3D | undefined, w: number, h: number): number {
  if (!point || (point.visibility ?? 0) < 0.5) return -1;
  for (let i = 0; i < 9; i++) {
    if (isCircleCollidingWithCell(point.x, point.y, i, w, h, 28)) {
      return i;
    }
  }
  return -1;
}

export default function LiveVision({
  onFrame,
  kinematicsBuffer,
  isRunning,
  testState,
  activeCell,
  hitFlashCell,
  onHit,
  onPostureAchieved,
  totalHits,
  mirrorView = true,
}: LiveVisionProps) {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef<MediaPipePose | null>(null);
  const animFrameRef = useRef<number>(0);
  const [cameraReady, setCameraReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingStatus, setLoadingStatus] = useState("Loading AI Model...");
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  // Native video frame size that MediaPipe normalized coords are relative to.
  const videoSizeRef = useRef({ width: 640, height: 480 });

  // Refs for game mechanics to prevent stale closures in pose.onResults
  const testStateRef = useRef(testState);
  const activeCellRef = useRef(activeCell);
  const hitFlashCellRef = useRef(hitFlashCell);
  const onHitRef = useRef(onHit);
  const onPostureAchievedRef = useRef(onPostureAchieved);
  const postureAchievedRef = useRef(false);

  const leftWristPathRef = useRef<PathPoint[]>([]);
  const rightWristPathRef = useRef<PathPoint[]>([]);
  const sparkleRef = useRef<{ x: number; y: number; startTime: number } | null>(null);
  const hitTriggeredRef = useRef<boolean>(false);
  const lastVideoTimeRef = useRef<number>(-1);

  useEffect(() => {
    testStateRef.current = testState;
    activeCellRef.current = activeCell;
    hitFlashCellRef.current = hitFlashCell;
    onHitRef.current = onHit;
    onPostureAchievedRef.current = onPostureAchieved;

    if (testState === "STARTING_POSTURE") {
      postureAchievedRef.current = false;
    }
  }, [testState, activeCell, hitFlashCell, onHit, onPostureAchieved]);

  useEffect(() => {
    leftWristPathRef.current = [];
    rightWristPathRef.current = [];
    hitTriggeredRef.current = false;
  }, [activeCell]);

  // Initialize MediaPipe Pose via CDN script
  useEffect(() => {
    let cancelled = false;

    async function initPose() {
      try {
        setLoadingStatus("Downloading MediaPipe Pose...");
        await loadScript(
          "https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/pose.js"
        );

        if (cancelled) return;

        setLoadingStatus("Waiting for AI model to register...");
        const PoseClass = await waitForPose(15000);

        if (cancelled) return;

        setLoadingStatus("Initializing AI model...");

        const pose = new PoseClass({
          locateFile: (file: string) =>
            `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`,
        });

        pose.setOptions({
          modelComplexity: 0, // 0 is Lite model - much faster inference with minimal CPU/GPU usage
          smoothLandmarks: true, // Enable temporal smoothing to eliminate joint jitter
          enableSegmentation: false,
          smoothSegmentation: false,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        pose.onResults((results: MediaPipeResults) => {
          if (!canvasRef.current) return;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;

          const landmarks = results.poseLandmarks;

          // Reset any prior transform and clear the full bitmap.
          ctx.setTransform(1, 0, 0, 1, 0, 0);
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (landmarks && landmarks.length >= 33) {
            const timestamp = performance.now();

            // Correct the camera mirroring by using anatomically correct landmarks
            const correctedLandmarks = landmarks;

            const frame = kinematicsBuffer.current.push(correctedLandmarks, timestamp);

            // Map normalized landmark space onto the SAME object-fit:cover crop
            // the browser applied to the <video>. We read the video element's
            // live intrinsic size and rendered box so the skeleton/grid line up
            // exactly, regardless of any box vs. camera aspect-ratio mismatch.
            const videoEl = webcamRef.current?.video;
            const vW = videoEl?.videoWidth || videoSizeRef.current.width;
            const vH = videoEl?.videoHeight || videoSizeRef.current.height;
            const fit = getCoverFit(canvas.width, canvas.height, vW, vH);
            ctx.setTransform(fit.scale, 0, 0, fit.scale, fit.offsetX, fit.offsetY);

            // Starting posture verification
            if (testStateRef.current === "STARTING_POSTURE" || testStateRef.current === "COUNTDOWN") {
              const leftWrist = correctedLandmarks[15];
              const rightWrist = correctedLandmarks[16];

              const leftHandCell = getHandCellIndex(leftWrist, vW, vH);
              const rightHandCell = getHandCellIndex(rightWrist, vW, vH);

              // We need one hand in Cell 1 (index 0) and one hand in Cell 3 (index 2)
              const isPostureCorrect =
                (leftHandCell === 0 && rightHandCell === 2) ||
                (leftHandCell === 2 && rightHandCell === 0);

              if (isPostureCorrect !== postureAchievedRef.current) {
                postureAchievedRef.current = isPostureCorrect;
                onPostureAchievedRef.current(isPostureCorrect);
              }
            }

            // 1. Draw base skeleton
            drawSkeleton(
              ctx,
              correctedLandmarks,
              vW,
              vH,
              frame.leftMetrics,
              frame.rightMetrics,
              totalHits,
              mirrorView
            );

            // 2. Draw 3x3 Grid if not evaluating
            if (testStateRef.current !== "EVALUATING") {
              drawGrid(
                ctx,
                vW,
                vH,
                activeCellRef.current,
                testStateRef.current,
                hitFlashCellRef.current,
                timestamp
              );
            }

            // 3. Draw sparkle if active
            if (sparkleRef.current) {
              const elapsed = timestamp - sparkleRef.current.startTime;
              const duration = 300; // 300ms animation
              if (elapsed < duration) {
                drawHitSparkle(
                  ctx,
                  sparkleRef.current.x,
                  sparkleRef.current.y,
                  vW,
                  vH,
                  elapsed / duration
                );
              } else {
                sparkleRef.current = null;
              }
            }

            // 4. Collision detection (only if PLAYING, active cell is valid, and not yet triggered)
            if (testStateRef.current === "PLAYING" && activeCellRef.current !== -1 && !hitTriggeredRef.current) {
              const leftWrist = correctedLandmarks[15]; // LEFT_WRIST
              const rightWrist = correctedLandmarks[16]; // RIGHT_WRIST

              // Track left wrist path
              if (leftWrist && (leftWrist.visibility ?? 0) > 0.5) {
                leftWristPathRef.current.push({
                  x: leftWrist.x,
                  y: leftWrist.y,
                  t: timestamp,
                });

                if (isCircleCollidingWithCell(leftWrist.x, leftWrist.y, activeCellRef.current, vW, vH, 28)) {
                  hitTriggeredRef.current = true;
                  sparkleRef.current = { x: leftWrist.x, y: leftWrist.y, startTime: timestamp };
                  
                  const leftHandCell = getHandCellIndex(leftWrist, vW, vH);
                  const rightHandCell = getHandCellIndex(rightWrist, vW, vH);
                  
                  onHitRef.current("left", activeCellRef.current, [...leftWristPathRef.current], leftHandCell, rightHandCell);
                }
              }

              // Track right wrist path
              if (!hitTriggeredRef.current && rightWrist && (rightWrist.visibility ?? 0) > 0.5) {
                rightWristPathRef.current.push({
                  x: rightWrist.x,
                  y: rightWrist.y,
                  t: timestamp,
                });

                if (isCircleCollidingWithCell(rightWrist.x, rightWrist.y, activeCellRef.current, vW, vH, 28)) {
                  hitTriggeredRef.current = true;
                  sparkleRef.current = { x: rightWrist.x, y: rightWrist.y, startTime: timestamp };
                  
                  const leftHandCell = getHandCellIndex(leftWrist, vW, vH);
                  const rightHandCell = getHandCellIndex(rightWrist, vW, vH);

                  onHitRef.current("right", activeCellRef.current, [...rightWristPathRef.current], leftHandCell, rightHandCell);
                }
              }
            }

            if (onFrame) {
              onFrame(frame);
            }
          } else {
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
          }
        });

        // Initialize the model (downloads WASM + model files)
        await pose.initialize();

        if (cancelled) return;

        poseRef.current = pose;
        setLoading(false);
        setLoadingStatus("");
      } catch (err) {
        console.error("MediaPipe initialization error:", err);
        setLoadingStatus("Failed to load AI model. Please refresh.");
      }
    }

    initPose();

    return () => {
      cancelled = true;
      if (poseRef.current) {
        poseRef.current.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start/stop detection loop
  useEffect(() => {
    let active = true;

    async function runDetection() {
      if (!active) return;

      if (
        !webcamRef.current ||
        !webcamRef.current.video ||
        !poseRef.current ||
        !isRunning
      ) {
        if (active) {
          animFrameRef.current = requestAnimationFrame(runDetection);
        }
        return;
      }

      const video = webcamRef.current.video;
      if (video.readyState !== 4) {
        if (active) {
          animFrameRef.current = requestAnimationFrame(runDetection);
        }
        return;
      }

      // Only run inference if a new frame has actually been captured by the webcam!
      // This prevents double-inference at 60Hz on a 30Hz video feed, saving 50% CPU/GPU load.
      if (video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;
        try {
          await poseRef.current.send({ image: video });
        } catch {
          // MediaPipe can throw if busy — just skip frame
        }
      }

      if (active) {
        animFrameRef.current = requestAnimationFrame(runDetection);
      }
    }

    if (cameraReady && !loading) {
      animFrameRef.current = requestAnimationFrame(runDetection);
    }

    return () => {
      active = false;
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [cameraReady, loading, isRunning]);

  // Handle camera ready
  const handleUserMedia = useCallback(() => {
    setCameraReady(true);
    if (webcamRef.current?.video) {
      const video = webcamRef.current.video;
      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      setDimensions({ width: w, height: h });
      videoSizeRef.current = { width: w, height: h };
    }
  }, []);


  return (
    <div className="webcam-container" style={{ aspectRatio: `${dimensions.width}/${dimensions.height}` }}>
      {/* Loading overlay */}
      {(loading || !cameraReady) && (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-black/80 gap-4">
          <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-400 rounded-full animate-spin" />
          <p className="text-sm text-slate-400 font-medium tracking-wide">
            {loadingStatus || "Waiting for camera..."}
          </p>
        </div>
      )}

      {/* Webcam */}
      <Webcam
        ref={webcamRef}
        audio={false}
        mirrored={mirrorView}
        onUserMedia={handleUserMedia}
        onLoadedMetadata={handleUserMedia}
        videoConstraints={{
          width: 1280,
          height: 720,
          facingMode: "user",
        }}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          display: "block",
        }}
      />

      {/* AR Canvas Overlay */}
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: mirrorView ? "scaleX(-1)" : "none", // Mirror to match webcam
          pointerEvents: "none",
        }}
      />

      {/* Top-left status */}
      {cameraReady && isRunning && !loading && (
        <div className="absolute top-4 left-4 z-10">
          <span className="status-live">Live Analysis</span>
        </div>
      )}

      {/* Top-right frame counter */}
      {cameraReady && !loading && (
        <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
          <span className="text-xs text-slate-500 font-mono">
            {kinematicsBuffer.current.length} frames
          </span>
        </div>
      )}
    </div>
  );
}
