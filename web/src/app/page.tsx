"use client";

import React, { useRef, useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import { KinematicsBuffer, type FrameData } from "@/lib/kinematics";
import MetricsPanel from "@/components/MetricsPanel";
import { TestState, GameResults, ReachRecord, processHit, aggregateResults, getRandomCellWithExclusions, type PathPoint } from "@/lib/game-engine";
import { playDing, playStartChime, playCompleteChime, playCountdownBeep } from "@/lib/audio";

// Dynamic imports — no SSR for webcam/chart components
const LiveVision = dynamic(() => import("@/components/LiveVision"), {
  ssr: false,
  loading: () => (
    <div className="webcam-container aspect-video flex items-center justify-center bg-black/40">
      <div className="text-center space-y-3">
        <div className="w-12 h-12 mx-auto border-4 border-indigo-500/30 border-t-indigo-400 rounded-full animate-spin" />
        <p className="text-sm text-slate-500 font-medium">Loading Vision Engine...</p>
      </div>
    </div>
  ),
});

const TelemetryCharts = dynamic(() => import("@/components/TelemetryCharts"), {
  ssr: false,
  loading: () => (
    <div className="glass-card p-6 h-48 flex items-center justify-center">
      <p className="text-sm text-slate-500 font-medium">Loading Charts...</p>
    </div>
  ),
});

const EvaluationDashboard = dynamic(() => import("@/components/EvaluationDashboard"), {
  ssr: false,
  loading: () => (
    <div className="glass-card p-8 text-center space-y-4">
      <div className="w-10 h-10 mx-auto border-4 border-indigo-500/30 border-t-indigo-400 rounded-full animate-spin" />
      <p className="text-sm text-slate-500 font-medium">Aggregating reach telemetry...</p>
    </div>
  ),
});

export default function HomePage() {
  const kinematicsBuffer = useRef(new KinematicsBuffer(150));
  const [isRunning, setIsRunning] = useState(true);
  const [chartData, setChartData] = useState<any[]>([]);
  const [stats, setStats] = useState<any>(null);
  const frameCountRef = useRef(0);

  // 9-Grid Game States
  const [testState, setTestState] = useState<TestState>("IDLE");
  const [countdown, setCountdown] = useState(90);
  const [activeCell, setActiveCell] = useState(0);
  const [hitFlashCell, setHitFlashCell] = useState<number | null>(null);
  const [results, setResults] = useState<GameResults | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [postureAchieved, setPostureAchieved] = useState(false);
  const [countdownProgress, setCountdownProgress] = useState(0);
  const [mirrorView, setMirrorView] = useState(true);

  const reachesRef = useRef<ReachRecord[]>([]);
  const reachStartTimeRef = useRef<number>(0);
  const testTimerRef = useRef<NodeJS.Timeout | null>(null);
  const testStartTimeRef = useRef<number>(0);

  // Throttled update — update charts every 4 frames to keep 30+ FPS
  const handleFrame = useCallback((_frame: FrameData) => {
    frameCountRef.current += 1;
    if (frameCountRef.current % 4 === 0) {
      const newChartData = kinematicsBuffer.current.getChartData(80);
      const newStats = kinematicsBuffer.current.getAggregatedStats();

      if (newStats) {
        // Straightness must reflect PER-REACH movement quality, updated only
        // for the arm that actually completed each reach. We therefore derive
        // it from the completed reach records (not the continuous pose buffer),
        // so a right-arm reach never changes the left arm's straightness, and
        // vice versa. The displayed value is the average over that arm's reaches.
        const reaches = reachesRef.current;
        let leftSum = 0, leftCount = 0;
        let rightSum = 0, rightCount = 0;
        let leftTimeSum = 0, rightTimeSum = 0;
        for (const r of reaches) {
          if (r.arm === "left") { leftSum += r.straightness; leftTimeSum += r.reachTimeMs; leftCount++; }
          else { rightSum += r.straightness; rightTimeSum += r.reachTimeMs; rightCount++; }
        }
        newStats.left.avgStraightness = leftCount > 0 ? leftSum / leftCount : 0;
        newStats.right.avgStraightness = rightCount > 0 ? rightSum / rightCount : 0;
        newStats.left.avgReachTime = leftCount > 0 ? leftTimeSum / leftCount : 0;
        newStats.right.avgReachTime = rightCount > 0 ? rightTimeSum / rightCount : 0;
      }

      setChartData(newChartData);
      setStats(newStats);
    }
  }, []);

  const handleReset = useCallback(() => {
    kinematicsBuffer.current.reset();
    setChartData([]);
    setStats(null);
    frameCountRef.current = 0;
    
    // Reset test state
    setTestState("IDLE");
    setCountdown(90);
    setResults(null);
    setReport(null);
    setHitFlashCell(null);
    setPostureAchieved(false);
    reachesRef.current = [];
  }, []);

  // End standard 90s trial
  const handleEndTest = useCallback(() => {
    playCompleteChime();
    setTestState("EVALUATING");
    const duration = performance.now() - testStartTimeRef.current;
    const finalResults = aggregateResults(reachesRef.current, duration);
    setResults(finalResults);
  }, []);

  // Handle countdown interval (the actual 90s trial)
  useEffect(() => {
    if (testState === "PLAYING") {
      testTimerRef.current = setInterval(() => {
        setCountdown((prev) => {
          if (prev <= 1) {
            clearInterval(testTimerRef.current!);
            handleEndTest();
            return 0;
          }
          // Play warning beeps at the final 3 seconds (3, 2, 1) before the end of the test
          if (prev <= 4 && prev > 1) {
            playCountdownBeep(false);
          }
          return prev - 1;
        });
      }, 1000);
    } else {
      if (testTimerRef.current) {
        clearInterval(testTimerRef.current);
      }
    }

    return () => {
      if (testTimerRef.current) {
        clearInterval(testTimerRef.current);
      }
    };
  }, [testState, handleEndTest]);

  // Start the 90s clinical reaching test
  const handleStartTest = useCallback(() => {
    // Reset kinematics first
    kinematicsBuffer.current.reset();
    setChartData([]);
    setStats(null);
    frameCountRef.current = 0;

    // Set game stats
    reachesRef.current = [];
    setCountdown(90);
    const startCell = Math.floor(Math.random() * 9);
    setActiveCell(startCell);
    setResults(null);
    setReport(null);
    setHitFlashCell(null);

    // Play chime and transition state
    playStartChime();
    setTestState("PLAYING");
    reachStartTimeRef.current = performance.now();
    testStartTimeRef.current = performance.now();
  }, []);

  // Handle 3-2-1 countdown sequence before beginning
  useEffect(() => {
    let timer: NodeJS.Timeout;
    let progressTimer: NodeJS.Timeout;

    if (testState === "COUNTDOWN") {
      let count = 3;
      setCountdown(3);
      setCountdownProgress(0);
      playCountdownBeep(false); // count 3

      timer = setInterval(() => {
        count -= 1;
        if (count === 0) {
          clearInterval(timer);
          clearInterval(progressTimer);
          playCountdownBeep(true); // GO!
          handleStartTest();
        } else {
          setCountdown(count);
          playCountdownBeep(false); // count 2, 1
        }
      }, 1000);

      // Smooth progress calculation over 3 seconds (3000ms)
      const startTime = performance.now();
      progressTimer = setInterval(() => {
        const elapsed = performance.now() - startTime;
        const pct = Math.min(100, (elapsed / 3000) * 100);
        setCountdownProgress(pct);
      }, 50);
    }
    return () => {
      if (timer) clearInterval(timer);
      if (progressTimer) clearInterval(progressTimer);
    };
  }, [testState, handleStartTest]);

  // Starting posture callback from LiveVision
  const handlePostureAchieved = useCallback((achieved: boolean) => {
    setPostureAchieved(achieved);
    setTestState((current) => {
      if (achieved && current === "STARTING_POSTURE") {
        return "COUNTDOWN";
      } else if (!achieved && current === "COUNTDOWN") {
        return "STARTING_POSTURE";
      }
      return current;
    });
  }, []);

  // Collision hit callback
  const handleHit = useCallback((
    arm: "left" | "right",
    cellIndex: number,
    path: PathPoint[],
    leftHandCell: number,
    rightHandCell: number
  ) => {
    playDing();

    const now = performance.now();
    const hitData = {
      arm,
      cellIndex,
      timestamp: now,
      pathPoints: path,
      reachStartTime: reachStartTimeRef.current,
    };

    const reachIndex = reachesRef.current.length + 1;
    const record = processHit(hitData, reachIndex);
    reachesRef.current.push(record);

    // Neon hit flash on the target cell
    setHitFlashCell(cellIndex);
    
    // Hide active cell so the user sees a clear transition gap
    setActiveCell(-1);

    setTimeout(() => {
      setHitFlashCell(null);

      // Determine cell exclusions: the cell that was just hit, plus the cells where hands are currently located
      const exclusions = [cellIndex];
      if (leftHandCell !== -1) exclusions.push(leftHandCell);
      if (rightHandCell !== -1) exclusions.push(rightHandCell);

      // Choose next cell with exclusions
      const nextCell = getRandomCellWithExclusions(exclusions);
      setActiveCell(nextCell);
      reachStartTimeRef.current = performance.now();
    }, 400); // 400ms delay for clear visual gap
  }, []);

  // Generate the AI virtual physio report
  const handleGenerateReport = async () => {
    if (!results) return;
    setReportLoading(true);
    try {
      const res = await fetch("/api/generate-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(results),
      });
      if (!res.ok) throw new Error("Failed to generate report");
      const data = await res.json();
      setReport(data.report);
    } catch (err) {
      console.error(err);
      setReport("Error generating clinical report. Please try again.");
    } finally {
      setReportLoading(false);
    }
  };

  const handleRestart = useCallback(() => {
    handleReset();
  }, [handleReset]);

  return (
    <div className="min-h-screen bg-grid">
      {/* ══ Header ══ */}
      <header className="border-b border-white/5 bg-black/20 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Logo */}
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight">
                <span className="gradient-text">KONKAE.COM</span>
              </h1>
              <p className="text-[10px] text-slate-600 font-medium tracking-widest uppercase">
                By Toto and King
              </p>
            </div>
          </div>

          {/* Connected Streamlit App Navigation Button */}
          <div className="flex items-center gap-3">
            <a
              href="http://localhost:8501"
              target="_self"
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl bg-indigo-950/40 text-indigo-300 border border-indigo-800/80 hover:bg-indigo-900/50 hover:text-white transition-all duration-200"
              title="สลับไประบบอัปโหลดวิดีโอประเมินผล (Streamlit)"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <span className="hidden md:inline">ระบบอัปโหลดวิดีโอ (Streamlit)</span>
              <span className="inline md:hidden">อัปโหลด (Streamlit)</span>
            </a>
          </div>

          <div className="flex items-center gap-3">
            {testState === "PLAYING" && (
              <div className={`flex items-center gap-2 mr-4 px-3 py-1.5 rounded-xl border transition-all duration-300 ${
                countdown <= 3 
                  ? "bg-rose-500/20 border-rose-500/30 text-rose-400 animate-pulse scale-105" 
                  : "bg-indigo-500/10 border-indigo-500/20 text-cyan-400"
              }`}>
                <span className={`w-2 h-2 rounded-full ${countdown <= 3 ? "bg-rose-500 animate-ping" : "bg-cyan-400 animate-pulse"}`} />
                <span className="text-xs font-semibold font-mono">
                  Timer: {countdown}s
                </span>
                <span className="text-xs text-slate-400">|</span>
                <span className={`text-xs font-mono font-medium ${countdown <= 3 ? "text-rose-400/80" : "text-slate-400"}`}>
                  Score: {reachesRef.current.length}
                </span>
              </div>
            )}

            {/* Mirror View Toggle */}
            <button
              onClick={() => setMirrorView(!mirrorView)}
              className={`p-2 rounded-xl border transition-all duration-200 cursor-pointer ${
                mirrorView 
                  ? 'bg-cyan-950/40 text-cyan-400 border-cyan-800/80 hover:bg-cyan-900/50' 
                  : 'bg-zinc-900 border-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-800'
              }`}
              title="สลับโหมดกระจกสะท้อน (Mirror Camera)"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="shrink-0">
                <polygon points="12 2 2 7 12 12 22 7 12 2" />
                <polyline points="2 17 12 22 22 17" />
                <polyline points="2 12 12 17 22 12" />
              </svg>
            </button>

            {/* Reset */}
            <button
              id="reset-session-btn"
              onClick={handleReset}
              className="btn-secondary text-xs flex items-center gap-1.5"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="1 4 1 10 7 10" />
                <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10" />
              </svg>
              Reset
            </button>

            {/* Start/Pause */}
            <button
              id="toggle-analysis-btn"
              onClick={() => setIsRunning(!isRunning)}
              className={`text-xs font-semibold px-4 py-2 rounded-xl transition-all flex items-center gap-1.5 ${
                isRunning
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/15"
                  : "bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/15"
              }`}
            >
              {isRunning ? (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse-dot" />
                  Analyzing
                </>
              ) : (
                <>
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  Paused
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      {/* ══ Main Content ══ */}
      <main className="max-w-[1600px] mx-auto px-4 sm:px-6 py-6 space-y-6">
        {testState === "EVALUATING" && results ? (
          <EvaluationDashboard
            results={results}
            report={report}
            reportLoading={reportLoading}
            onGenerateReport={handleGenerateReport}
            onRestart={handleRestart}
          />
        ) : (
          <>
            {/* Top Row: Camera + Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Webcam — takes 2/3 width */}
              <div className="lg:col-span-2 space-y-4">
                 {/* 9-Grid Control Panel Overlay */}
                {testState === "IDLE" && (
                  <div className="glass-card-glow p-6 text-center space-y-4 animate-slide-up">
                    <h3 className="text-base font-bold text-slate-200">
                      🎮 9-Grid Gamified Reaching Test
                    </h3>
                    <p className="text-xs text-slate-400 max-w-xl mx-auto leading-relaxed">
                      This clinical test measures upper extremity coordination, reach duration, movement smoothness, and path straightness. 
                      Move both arms in front of the camera, then click below to prepare. Place your hands in the highlighted top corners to start!
                    </p>
                    <button
                      id="start-9-grid-btn"
                      onClick={() => setTestState("STARTING_POSTURE")}
                      className="btn-primary text-xs font-semibold px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 to-violet-600 hover:from-indigo-600 hover:to-violet-700 shadow-md shadow-indigo-500/20"
                    >
                      Start Clinical Test
                    </button>
                  </div>
                )}

                {testState === "STARTING_POSTURE" && (
                  <div className="glass-card p-4 text-center space-y-2 border border-yellow-500/30 bg-yellow-500/5 animate-slide-up">
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse" />
                      <span className="text-xs font-bold text-yellow-400 uppercase tracking-wide">
                        Starting Posture Required
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-300">
                      Place **both hands** inside the top corners (**START POSTURE L** and **START POSTURE R**, highlighted in orange) to trigger the countdown.
                    </p>
                  </div>
                )}

                {testState === "COUNTDOWN" && (
                  <div className="glass-card p-4 text-center space-y-2 border border-emerald-500/30 bg-emerald-500/5 animate-slide-up">
                    <div className="flex items-center justify-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
                      <span className="text-xs font-bold text-emerald-400 uppercase tracking-wide">
                        Hold starting posture!
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-300">
                      Keep your hands in the top corners. Starting in {countdown} seconds...
                    </p>
                  </div>
                )}

                {testState === "PLAYING" && (
                  <div className="glass-card p-3 flex items-center justify-between animate-slide-up">
                    <div className="flex items-center gap-3">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                      </span>
                      <p className="text-xs text-slate-300 font-medium">
                        Reach and touch the glowing cell <span className="text-cyan-400 font-bold font-mono">#{activeCell + 1}</span>
                      </p>
                    </div>
                    <div className="w-1/3 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div 
                        className="bg-indigo-500 h-1.5 transition-all duration-1000" 
                        style={{ width: `${(countdown / 90) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                <div className="relative overflow-hidden rounded-2xl">
                  <LiveVision
                    onFrame={handleFrame}
                    kinematicsBuffer={kinematicsBuffer}
                    isRunning={isRunning}
                    testState={testState}
                    activeCell={activeCell}
                    hitFlashCell={hitFlashCell}
                    onHit={handleHit}
                    onPostureAchieved={handlePostureAchieved}
                    totalHits={reachesRef.current.length}
                    mirrorView={mirrorView}
                  />

                  {/* Absolute Countdown Overlay */}
                  {testState === "COUNTDOWN" && (
                    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-black/75 backdrop-blur-[3px] animate-fade-in">
                      {/* SVG Circular Progress Ring */}
                      <div className="relative w-28 h-28 flex items-center justify-center">
                        <svg className="w-full h-full transform -rotate-90">
                          <circle
                            cx="56"
                            cy="56"
                            r="48"
                            stroke="rgba(255, 255, 255, 0.05)"
                            strokeWidth="6"
                            fill="transparent"
                          />
                          <circle
                            cx="56"
                            cy="56"
                            r="48"
                            stroke="#6366f1"
                            strokeWidth="6"
                            fill="transparent"
                            strokeDasharray={2 * Math.PI * 48}
                            strokeDashoffset={2 * Math.PI * 48 * (1 - countdownProgress / 100)}
                            strokeLinecap="round"
                            className="transition-all duration-75 ease-out"
                            style={{
                              filter: "drop-shadow(0 0 8px rgba(99, 102, 241, 0.5))"
                            }}
                          />
                        </svg>
                        <span className="absolute text-5xl font-black text-white font-mono">
                          {countdown}
                        </span>
                      </div>

                      <div className="mt-6 text-center space-y-1">
                        <p className="text-xs text-indigo-300 font-bold tracking-wider uppercase animate-pulse">
                          Calibrating Posture...
                        </p>
                        <p className="text-[10px] text-slate-400 font-medium">
                          Keep both hands steady ({countdownProgress.toFixed(0)}%)
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Absolute Ending Countdown Overlay */}
                  {testState === "PLAYING" && countdown <= 3 && (
                    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-rose-950/20 backdrop-blur-[0.5px] pointer-events-none animate-fade-in">
                      <div className="w-24 h-24 rounded-full border-4 border-rose-500 flex items-center justify-center bg-rose-500/10 shadow-lg shadow-rose-500/30 animate-scale-up">
                        <span className="text-5xl font-black text-rose-500 font-mono animate-ping">
                          {countdown}
                        </span>
                      </div>
                      <p className="text-xs text-rose-400 font-bold tracking-wider uppercase mt-4 animate-pulse">
                        Ending soon...
                      </p>
                    </div>
                  )}
                </div>

                {/* Legend bar below webcam */}
                <div className="flex items-center justify-center gap-6 text-xs text-slate-500">
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 rounded bg-cyan-400" />
                    Left Arm
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 rounded bg-rose-400" />
                    Right Arm
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-0.5 rounded bg-indigo-400" />
                    Body
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full border border-violet-400/50" />
                    Hits Overlay
                  </span>
                </div>
              </div>

              {/* Metrics Panel — takes 1/3 */}
              <div>
                <MetricsPanel stats={stats} hits={reachesRef.current.length} />
              </div>
            </div>

            {/* Telemetry Charts */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
                  Live Telemetry
                </h2>
                <span className="text-[10px] text-slate-600 font-mono">
                  {chartData.length > 0
                    ? `${chartData.length} data points`
                    : "waiting..."}
                </span>
              </div>
              <TelemetryCharts data={chartData} />
            </section>
          </>
        )}

        {/* Footer */}
        <footer className="border-t border-white/5 pt-6 pb-8 text-center">
          <p className="text-xs text-slate-600">
            <span className="gradient-text font-semibold">KONKAE.COM</span>{" "}
            — By Toto and King
          </p>
          <p className="text-[10px] text-slate-700 mt-1">
            Digital Aiding 4 Aging Hackathon · All inference runs on-device via MediaPipe WebAssembly
          </p>
        </footer>
      </main>
    </div>
  );
}
