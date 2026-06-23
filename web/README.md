# 🦾 KONKAE (คนแก่) - Interactive Webcam Reaching Game (Next.js App)

This directory contains the **Next.js Web Application** for the **KONKAE (คนแก่)** Dexterity & Sarcopenia Analyzer project, developed for the **Digital Aiding 4 Aging Hackathon** by Toto and King.

It provides an interactive, client-side, on-device webcam clinical reaching test (9-Grid layout) utilizing MediaPipe Pose WebAssembly and renders real-time telemetry graphs along with clinical reports.

For the full project details, architecture, and documentation, please refer to the main [Root README.md](file:///d:/vibe-hack-real/sarcopenia-assessment/README.md).

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v18 or higher recommended)
- **NPM** or another package manager (Yarn, Pnpm, Bun)

### Installation

Install the project dependencies:

```bash
npm install
```

### Run the Development Server

Start the Next.js server locally:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the application.

---

## 📂 Key Components in this Folder

* **[src/app/page.tsx](file:///d:/vibe-hack-real/sarcopenia-assessment/web/src/app/page.tsx)**: Main game page containing the 9-Grid trial layout, game loops, and UI controls.
* **[src/components/LiveVision.tsx](file:///d:/vibe-hack-real/sarcopenia-assessment/web/src/components/LiveVision.tsx)**: Manages webcam feeds and interacts with MediaPipe Pose via CDN integration.
* **[src/components/EvaluationDashboard.tsx](file:///d:/vibe-hack-real/sarcopenia-assessment/web/src/components/EvaluationDashboard.tsx)**: Interactive assessment results and generated clinical rehabilitation plan.
* **[src/app/api/generate-report/route.ts](file:///d:/vibe-hack-real/sarcopenia-assessment/web/src/app/api/generate-report/route.ts)**: API handler for evaluating the Learned Non-Use (LNI) metrics and fall risk warnings.
