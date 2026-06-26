"""Core computer-vision pipeline for the dexterity assessment.

``VideoProcessor`` runs the real MediaPipe Pose + OpenCV colour-target pipeline
over a video file and yields a :class:`FrameResult` per frame. It contains no
Streamlit code so it can be tested headlessly and reused by other frontends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import cv2
import numpy as np

from kinematics import (
    TemporalSmoother,
    calculate_shoulder_angle,
    dist_3d_norm,
    displacement_and_path_length,
    calculate_straightness,
    calculate_jerk,
)
from pose_backend import POSE_LANDMARK, PoseEstimator, draw_pose

# HSV colour ranges for the target boxes.
_COLOR_RANGES = {
    "Green": [((35, 40, 40), (85, 255, 255))],
    "Blue": [((100, 40, 40), (140, 255, 255))],
    "Red": [((0, 40, 40), (10, 255, 255)), ((170, 40, 40), (180, 255, 255))],
}


@dataclass
class ProcessorConfig:
    target_color: str = "Green"
    ref_point_mode: str = "Wrist"
    min_frames_in_box: int = 3
    skip_seconds: float = 0.0
    mirror_view: bool = False
    left_margin: float = 0.0
    right_margin: float = 0.0
    top_margin: float = 0.0
    bottom_margin: float = 0.0
    smoothing_window: int = 5


@dataclass
class FrameResult:
    """Per-frame snapshot consumed by the UI layer."""

    frame_idx: int
    timestamp_sec: float
    is_recording: bool
    annotated_frame_rgb: np.ndarray
    active_cells: list
    new_log_lines: list = field(default_factory=list)

    # Live kinematics snapshot
    total_hits: int = 0
    left_hits: int = 0
    right_hits: int = 0
    left_speeds: list = field(default_factory=list)
    right_speeds: list = field(default_factory=list)
    left_jerks: list = field(default_factory=list)
    right_jerks: list = field(default_factory=list)
    left_rom_min: float = float("inf")
    left_rom_max: float = float("-inf")
    right_rom_min: float = float("inf")
    right_rom_max: float = float("-inf")
    left_current_rom: float = 0.0
    right_current_rom: float = 0.0
    left_straightness: float = 0.0
    right_straightness: float = 0.0
    dominant_side: str = "WAITING"

    # CSV row for the frame inspector
    history_row: dict = field(default_factory=dict)


class VideoProcessor:
    def __init__(self, video_path: str, config: ProcessorConfig):
        self.config = config
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._pose = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Final-summary stats populated as processing runs.
        self.target_instance_counter = 0
        self.total_hits = 0
        self.total_misses = 0
        self.left_hits_count = 0
        self.right_hits_count = 0
        self.max_left_angle = 0.0
        self.max_right_angle = 0.0
        self.reaction_times: list = []
        self.left_reaction_times: list = []
        self.right_reaction_times: list = []
        self.left_jitter_sum = 0.0
        self.right_jitter_sum = 0.0
        self.left_jitter_frames = 0
        self.right_jitter_frames = 0
        self.left_speeds: list = []
        self.right_speeds: list = []
        self.left_jerks: list = []
        self.right_jerks: list = []
        self.left_rom_min = float("inf")
        self.left_rom_max = float("-inf")
        self.right_rom_min = float("inf")
        self.right_rom_max = float("-inf")
        self.left_current_rom = 0.0
        self.right_current_rom = 0.0
        self.left_wrist_norm_history: list = []
        self.right_wrist_norm_history: list = []
        self.reaches: list[dict] = []

    # -- public helpers ---------------------------------------------------
    def release(self) -> None:
        try:
            self.cap.release()
        finally:
            self._pose.close()

    def color_mask(self, hsv: np.ndarray) -> np.ndarray:
        ranges = _COLOR_RANGES.get(self.config.target_color, _COLOR_RANGES["Green"])
        mask = None
        for low, high in ranges:
            m = cv2.inRange(hsv, np.array(low), np.array(high))
            mask = m if mask is None else (mask | m)
        return mask

    @property
    def dominant_side_en(self) -> str:
        return "LEFT" if sum(self.left_speeds) > sum(self.right_speeds) else "RIGHT"

    # -- main loop --------------------------------------------------------
    def process(self) -> Iterator[FrameResult]:
        cfg = self.config
        width, height, fps = self.width, self.height, self.fps

        active_targets_map: dict = {}
        hits_log: dict = {}
        any_hit_logged: dict = {}
        trial_start_times: dict = {}
        consecutive_frames_counter = {i: {"Left": 0, "Right": 0} for i in range(1, 10)}
        target_paths: dict = {}

        smoothers = {
            name: TemporalSmoother(cfg.smoothing_window)
            for name in (
                "ls", "rs", "le", "re", "lw", "rw", "li", "ri", "lh", "rh",
            )
        }

        prev_left_wrist = prev_right_wrist = None
        prev_left_speed = prev_right_speed = 0.0
        prev_left_accel = prev_right_accel = 0.0
        left_arm_raised = right_arm_raised = False
        left_raise_count = right_raise_count = 0
        left_wrist_history: list = []
        right_wrist_history: list = []
        left_path_length = right_path_length = 0.0

        frame_idx = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time_sec = frame_idx / fps
            is_recording = current_time_sec >= cfg.skip_seconds
            new_logs: list = []

            if cfg.mirror_view:
                frame = cv2.flip(frame, 1)
            annotated = frame.copy()

            # --- grid geometry ---
            x_start = int(width * (cfg.left_margin / 100.0))
            x_end = int(width * (1.0 - cfg.right_margin / 100.0))
            y_start = int(height * (cfg.top_margin / 100.0))
            y_end = int(height * (1.0 - cfg.bottom_margin / 100.0))
            grid_w = x_end - x_start
            grid_h = y_end - y_start
            cell_w = grid_w // 3
            cell_h = grid_h // 3
            cell_area = max(1, cell_w * cell_h)

            cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), (120, 120, 120), 1)
            for i in range(1, 3):
                cv2.line(annotated, (x_start + i * cell_w, y_start), (x_start + i * cell_w, y_end), (120, 120, 120), 1)
                cv2.line(annotated, (x_start, y_start + i * cell_h), (x_end, y_start + i * cell_h), (120, 120, 120), 1)
            for r in range(3):
                for c in range(3):
                    cell_num = r * 3 + c + 1
                    cv2.putText(annotated, str(cell_num), (x_start + c * cell_w + 10, y_start + r * cell_h + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

            # --- colour target detection ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            color_mask = self.color_mask(hsv)
            current_active_cells: list = []
            for r in range(3):
                for c in range(3):
                    cell_num = r * 3 + c + 1
                    x1 = x_start + c * cell_w
                    y1 = y_start + r * cell_h
                    x2 = x_start + (c + 1) * cell_w
                    y2 = y_start + (r + 1) * cell_h
                    cell_mask = color_mask[y1:y2, x1:x2]
                    ratio = (np.sum(cell_mask > 0) / cell_area) * 100
                    if ratio > 10.0:
                        current_active_cells.append(cell_num)

            # --- demo-phase overlay ---
            if not is_recording:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (width, height), (48, 27, 6), -1)
                annotated = cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(annotated, "DEMO / TEACHING PHASE", (width // 2 - 220, height // 2 - 40),
                            font, 0.6, (244, 162, 89), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"System will start analysis in {max(0.0, cfg.skip_seconds - current_time_sec):.1f}s",
                            (width // 2 - 170, height // 2 - 10), font, 0.5, (243, 244, 246), 1, cv2.LINE_AA)
                bar_w, bar_h = 360, 8
                bar_x = (width - bar_w) // 2
                bar_y = height // 2 + 15
                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
                fill_w = int(bar_w * (current_time_sec / max(0.1, cfg.skip_seconds)))
                cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (244, 162, 89), -1)

            # --- pose tracking ---
            left_speed = right_speed = left_jerk = right_jerk = 0.0
            left_angle = right_angle = 0.0
            left_wrist_pt = right_wrist_pt = None
            timestamp_ms = int(frame_idx * (1000.0 / fps)) if fps > 0 else frame_idx * 33
            pose_landmarks = self._pose.process(frame, timestamp_ms)
            has_pose = pose_landmarks is not None
            smoothed_landmarks = {}

            if has_pose:
                lm = pose_landmarks.landmark

                def d(landmark):
                    return {"x": landmark.x, "y": landmark.y, "z": landmark.z}

                l_sh = smoothers["ls"].smooth(d(lm[POSE_LANDMARK.LEFT_SHOULDER]))
                r_sh = smoothers["rs"].smooth(d(lm[POSE_LANDMARK.RIGHT_SHOULDER]))
                l_el = smoothers["le"].smooth(d(lm[POSE_LANDMARK.LEFT_ELBOW]))
                r_el = smoothers["re"].smooth(d(lm[POSE_LANDMARK.RIGHT_ELBOW]))
                l_wr = smoothers["lw"].smooth(d(lm[POSE_LANDMARK.LEFT_WRIST]))
                r_wr = smoothers["rw"].smooth(d(lm[POSE_LANDMARK.RIGHT_WRIST]))
                l_ix = smoothers["li"].smooth(d(lm[POSE_LANDMARK.LEFT_INDEX]))
                r_ix = smoothers["ri"].smooth(d(lm[POSE_LANDMARK.RIGHT_INDEX]))
                l_hip = smoothers["lh"].smooth(d(lm[POSE_LANDMARK.LEFT_HIP]))
                r_hip = smoothers["rh"].smooth(d(lm[POSE_LANDMARK.RIGHT_HIP]))

                smoothed_landmarks = {
                    11: l_sh,
                    12: r_sh,
                    13: l_el,
                    14: r_el,
                    15: l_wr,
                    16: r_wr,
                    19: l_ix,
                    20: r_ix,
                    23: l_hip,
                    24: r_hip,
                }

                left_angle = calculate_shoulder_angle((l_sh["x"], l_sh["y"]), (l_el["x"], l_el["y"]), (l_hip["x"], l_hip["y"]))
                right_angle = calculate_shoulder_angle((r_sh["x"], r_sh["y"]), (r_el["x"], r_el["y"]), (r_hip["x"], r_hip["y"]))

                if is_recording:
                    self.max_left_angle = max(self.max_left_angle, left_angle)
                    self.max_right_angle = max(self.max_right_angle, right_angle)
                    self.left_rom_min = min(self.left_rom_min, left_angle)
                    self.left_rom_max = max(self.left_rom_max, left_angle)
                    self.right_rom_min = min(self.right_rom_min, right_angle)
                    self.right_rom_max = max(self.right_rom_max, right_angle)
                    self.left_current_rom = left_angle
                    self.right_current_rom = right_angle

                    self.left_wrist_norm_history.append(l_wr)
                    self.right_wrist_norm_history.append(r_wr)
                    if len(self.left_wrist_norm_history) >= 2:
                        left_path_length += dist_3d_norm(self.left_wrist_norm_history[-1], self.left_wrist_norm_history[-2])
                    if len(self.right_wrist_norm_history) >= 2:
                        right_path_length += dist_3d_norm(self.right_wrist_norm_history[-1], self.right_wrist_norm_history[-2])

                    if left_angle > 60.0:
                        if not left_arm_raised:
                            left_arm_raised = True
                            left_raise_count += 1
                            new_logs.append(f"[Frame {frame_idx}] Left Arm Raise #{left_raise_count} (Angle: {left_angle:.1f}°)")
                    elif left_angle < 30.0:
                        left_arm_raised = False
                    if right_angle > 60.0:
                        if not right_arm_raised:
                            right_arm_raised = True
                            right_raise_count += 1
                            new_logs.append(f"[Frame {frame_idx}] Right Arm Raise #{right_raise_count} (Angle: {right_angle:.1f}°)")
                    elif right_angle < 30.0:
                        right_arm_raised = False

                dt_safe = 1.0 / fps if fps > 0 else 1.0 / 30.0
                if prev_left_wrist is not None:
                    left_speed = dist_3d_norm(l_wr, prev_left_wrist) / dt_safe
                    left_accel = abs(left_speed - prev_left_speed) / dt_safe
                    left_jerk = abs(left_accel - prev_left_accel) / dt_safe
                    prev_left_accel = left_accel
                else:
                    prev_left_accel = 0.0
                if prev_right_wrist is not None:
                    right_speed = dist_3d_norm(r_wr, prev_right_wrist) / dt_safe
                    right_accel = abs(right_speed - prev_right_speed) / dt_safe
                    right_jerk = abs(right_accel - prev_right_accel) / dt_safe
                    prev_right_accel = right_accel
                else:
                    prev_right_accel = 0.0

                prev_left_wrist, prev_right_wrist = l_wr, r_wr
                prev_left_speed, prev_right_speed = left_speed, right_speed

                if is_recording:
                    self.left_speeds.append(left_speed)
                    self.right_speeds.append(right_speed)
                    self.left_jerks.append(left_jerk)
                    self.right_jerks.append(right_jerk)

                l_wrist_px = (int(l_wr["x"] * width), int(l_wr["y"] * height))
                r_wrist_px = (int(r_wr["x"] * width), int(r_wr["y"] * height))

                if is_recording:
                    left_wrist_history.append(l_wrist_px)
                    right_wrist_history.append(r_wrist_px)
                    if len(left_wrist_history) >= 3:
                        v1 = np.array(left_wrist_history[-1]) - np.array(left_wrist_history[-2])
                        v2 = np.array(left_wrist_history[-2]) - np.array(left_wrist_history[-3])
                        self.left_jitter_sum += np.linalg.norm(v1 - v2)
                        self.left_jitter_frames += 1
                    if len(right_wrist_history) >= 3:
                        v1 = np.array(right_wrist_history[-1]) - np.array(right_wrist_history[-2])
                        v2 = np.array(right_wrist_history[-2]) - np.array(right_wrist_history[-3])
                        self.right_jitter_sum += np.linalg.norm(v1 - v2)
                        self.right_jitter_frames += 1

                if cfg.ref_point_mode == "Index Finger Tip":
                    left_wrist_pt = (int(l_ix["x"] * width), int(l_ix["y"] * height))
                    right_wrist_pt = (int(r_ix["x"] * width), int(r_ix["y"] * height))
                else:
                    left_wrist_pt = l_wrist_px
                    right_wrist_pt = r_wrist_px

                draw_pose(annotated, smoothed_landmarks)

            # --- target appearance state machine ---
            if is_recording:
                for cell_num in current_active_cells:
                    if cell_num not in active_targets_map:
                        self.target_instance_counter += 1
                        active_targets_map[cell_num] = self.target_instance_counter
                        hits_log[self.target_instance_counter] = {"Left": False, "Right": False}
                        any_hit_logged[self.target_instance_counter] = False
                        # Reset frames counter for this cell when a new target appears
                        consecutive_frames_counter[cell_num] = {"Left": 0, "Right": 0}
                        col = (cell_num - 1) % 3
                        side = "LEFT" if col == 0 else "RIGHT" if col == 2 else "CENTER"
                        new_logs.append(f"[Frame {frame_idx}] Target {self.target_instance_counter} appeared on {side}")
                        trial_start_times[self.target_instance_counter] = frame_idx / fps
            else:
                active_targets_map = {}

            # Record path points for active targets if pose exists
            if has_pose and is_recording:
                ref_left = l_ix if cfg.ref_point_mode == "Index Finger Tip" else l_wr
                ref_right = r_ix if cfg.ref_point_mode == "Index Finger Tip" else r_wr
                for cell_num, inst_id in active_targets_map.items():
                    if inst_id not in target_paths:
                        target_paths[inst_id] = {"Left": [], "Right": []}
                    target_paths[inst_id]["Left"].append({"x": ref_left["x"], "y": ref_left["y"], "t": timestamp_ms})
                    target_paths[inst_id]["Right"].append({"x": ref_right["x"], "y": ref_right["y"], "t": timestamp_ms})

            # --- collision detection ---
            hands_pts = {"Left": left_wrist_pt, "Right": right_wrist_pt}
            if is_recording:
                for cell_num, inst_id in list(active_targets_map.items()):
                    r = (cell_num - 1) // 3
                    c = (cell_num - 1) % 3
                    x1 = x_start + c * cell_w
                    y1 = y_start + r * cell_h
                    x2 = x_start + (c + 1) * cell_w
                    y2 = y_start + (r + 1) * cell_h
                    
                    # 10% tolerance padding to make collision detection smoother
                    pad_w = int(cell_w * 0.10)
                    pad_h = int(cell_h * 0.10)
                    x1_pad, x2_pad = x1 - pad_w, x2 + pad_w
                    y1_pad, y2_pad = y1 - pad_h, y2 + pad_h
                    
                    for hand_side, pt in hands_pts.items():
                        if pt is None:
                            continue
                        px, py = pt
                        if x1_pad <= px <= x2_pad and y1_pad <= py <= y2_pad:
                            # Use cumulative frame count instead of strictly consecutive frames to mitigate jitter
                            consecutive_frames_counter[cell_num][hand_side] += 1
                            if consecutive_frames_counter[cell_num][hand_side] >= cfg.min_frames_in_box:
                                if not hits_log[inst_id][hand_side]:
                                    hits_log[inst_id][hand_side] = True
                                    any_hit_logged[inst_id] = True
                                    trial_start = trial_start_times.get(inst_id, frame_idx / fps)
                                    reaction = (frame_idx / fps) - trial_start
                                    
                                    # Calculate reach metrics
                                    path_points = target_paths.get(inst_id, {}).get(hand_side, [])
                                    st_val = calculate_straightness(path_points)
                                    jk_val = calculate_jerk(path_points)
                                    
                                    self.reaches.append({
                                        "index": len(self.reaches) + 1,
                                        "targetCell": cell_num,
                                        "arm": hand_side.lower(),
                                        "reachTimeMs": reaction * 1000.0,
                                        "straightness": st_val,
                                        "jerk": jk_val
                                    })
                                    
                                    new_logs.append(f"[Frame {frame_idx}] Target {inst_id} HIT by {hand_side.upper()} Hand")
                                    self.total_hits += 1
                                    self.reaction_times.append(reaction)
                                    if hand_side == "Left":
                                        self.left_hits_count += 1
                                        self.left_reaction_times.append(reaction)
                                    else:
                                        self.right_hits_count += 1
                                        self.right_reaction_times.append(reaction)
                                    cv2.putText(annotated, "HIT!", (x1 + 10, y1 + cell_h - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    consecutive_frames_counter[cell_num][hand_side] = 0

            # --- target disappearance state machine ---
            if is_recording:
                for cell_num in list(active_targets_map.keys()):
                    if cell_num not in current_active_cells:
                        inst_id = active_targets_map[cell_num]
                        if not any_hit_logged.get(inst_id, False):
                            # Proximity fallback check (20% padding) when target disappears:
                            # If the hand is near the cell when it disappears, it means the hand touched it
                            # and covered the target (causing it to disappear), which counts as a HIT.
                            r = (cell_num - 1) // 3
                            c = (cell_num - 1) % 3
                            x1 = x_start + c * cell_w
                            y1 = y_start + r * cell_h
                            x2 = x_start + (c + 1) * cell_w
                            y2 = y_start + (r + 1) * cell_h
                            
                            pad_w = int(cell_w * 0.20)
                            pad_h = int(cell_h * 0.20)
                            x1_pad, x2_pad = x1 - pad_w, x2 + pad_w
                            y1_pad, y2_pad = y1 - pad_h, y2 + pad_h
                            
                            was_near = False
                            near_hand = None
                            for hand_side, pt in hands_pts.items():
                                if pt is not None:
                                    px, py = pt
                                    if x1_pad <= px <= x2_pad and y1_pad <= py <= y2_pad:
                                        was_near = True
                                        near_hand = hand_side
                                        break
                                        
                            if was_near:
                                hits_log[inst_id][near_hand] = True
                                any_hit_logged[inst_id] = True
                                trial_start = trial_start_times.get(inst_id, frame_idx / fps)
                                reaction = (frame_idx / fps) - trial_start
                                
                                # Calculate reach metrics
                                path_points = target_paths.get(inst_id, {}).get(near_hand, [])
                                st_val = calculate_straightness(path_points)
                                jk_val = calculate_jerk(path_points)
                                
                                self.reaches.append({
                                    "index": len(self.reaches) + 1,
                                    "targetCell": cell_num,
                                    "arm": near_hand.lower(),
                                    "reachTimeMs": reaction * 1000.0,
                                    "straightness": st_val,
                                    "jerk": jk_val
                                })
                                
                                new_logs.append(f"[Frame {frame_idx}] Target {inst_id} HIT by {near_hand.upper()} Hand")
                                self.total_hits += 1
                                self.reaction_times.append(reaction)
                                if near_hand == "Left":
                                    self.left_hits_count += 1
                                    self.left_reaction_times.append(reaction)
                                else:
                                    self.right_hits_count += 1
                                    self.right_reaction_times.append(reaction)
                                cv2.putText(annotated, "HIT!", (x1 + 10, y1 + cell_h - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            else:
                                new_logs.append(f"[Frame {frame_idx}] Target {inst_id} disappeared (MISSED)")
                                self.total_misses += 1
                        del active_targets_map[cell_num]

            # Draw target active/hit boundaries
            for cell_num, inst_id in active_targets_map.items():
                r = (cell_num - 1) // 3
                c = (cell_num - 1) % 3
                glow = (0, 255, 0) if any_hit_logged.get(inst_id, False) else (0, 0, 255)
                cv2.rectangle(annotated, (x_start + c * cell_w + 3, y_start + r * cell_h + 3),
                              (x_start + (c + 1) * cell_w - 3, y_start + (r + 1) * cell_h - 3), glow, 3)

            # --- tracking cursors ---
            for hand_side, pt in hands_pts.items():
                if pt is None:
                    continue
                px, py = pt
                color = (200, 209, 31) if hand_side == "Left" else (89, 162, 244)
                cv2.circle(annotated, (px, py), 10, color, -1)
                cv2.circle(annotated, (px, py), 15, (255, 255, 255), 2)
                text = "มือซ้าย" if hand_side == "Left" else "มือขวา"
                cv2.putText(annotated, text, (px - 25, py - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # --- live derived metrics ---
            left_speed_sum = sum(self.left_speeds)
            right_speed_sum = sum(self.right_speeds)
            if left_speed_sum == 0.0 and right_speed_sum == 0.0:
                dominant_side = "WAITING"
            else:
                dominant_side = "left" if left_speed_sum > right_speed_sum else "right"

            left_done = [r for r in self.reaches if r["arm"] == "left"]
            right_done = [r for r in self.reaches if r["arm"] == "right"]
            left_straightness = float(np.mean([r["straightness"] for r in left_done])) * 100.0 if left_done else 0.0
            right_straightness = float(np.mean([r["straightness"] for r in right_done])) * 100.0 if right_done else 0.0

            # --- CSV history row ---
            left_hand_x = (left_wrist_pt[0] - x_start) / grid_w * 100 if left_wrist_pt is not None else None
            left_hand_y = (left_wrist_pt[1] - y_start) / grid_h * 100 if left_wrist_pt is not None else None
            right_hand_x = (right_wrist_pt[0] - x_start) / grid_w * 100 if right_wrist_pt is not None else None
            right_hand_y = (right_wrist_pt[1] - y_start) / grid_h * 100 if right_wrist_pt is not None else None

            active_cells_list = list(active_targets_map.keys())
            primary_active_cell = active_cells_list[0] if active_cells_list else None
            t_x1 = t_y1 = t_x2 = t_y2 = 0.0
            if primary_active_cell is not None:
                tr = (primary_active_cell - 1) // 3
                tc = (primary_active_cell - 1) % 3
                t_x1, t_y1 = tc * 33.33, tr * 33.33
                t_x2, t_y2 = (tc + 1) * 33.33, (tr + 1) * 33.33

            left_hit = any(hits_log.get(i, {}).get("Left", False) for i in active_targets_map.values())
            right_hit = any(hits_log.get(i, {}).get("Right", False) for i in active_targets_map.values())

            history_row = {
                "Frame Index": frame_idx,
                "Timestamp (sec)": current_time_sec,
                "Active Target Cell": ", ".join(map(str, active_cells_list)) if active_cells_list else "None",
                "Target X1 (%)": t_x1,
                "Target Y1 (%)": t_y1,
                "Target X2 (%)": t_x2,
                "Target Y2 (%)": t_y2,
                "Left Hand X (%)": left_hand_x if left_hand_x is not None else "N/A",
                "Left Hand Y (%)": left_hand_y if left_hand_y is not None else "N/A",
                "Right Hand X (%)": right_hand_x if right_hand_x is not None else "N/A",
                "Right Hand Y (%)": right_hand_y if right_hand_y is not None else "N/A",
                "Left Hand Hit": "Yes" if left_hit else "No",
                "Right Hand Hit": "Yes" if right_hit else "No",
                "Left Arm Speed (px/s)": round(left_speed, 1) if (has_pose and prev_left_wrist is not None) else 0.0,
                "Right Arm Speed (px/s)": round(right_speed, 1) if (has_pose and prev_right_wrist is not None) else 0.0,
                "Left Movement Jerk (px/s3)": round(left_jerk, 1) if (has_pose and prev_left_wrist is not None) else 0.0,
                "Right Movement Jerk (px/s3)": round(right_jerk, 1) if (has_pose and prev_right_wrist is not None) else 0.0,
                "Left Shoulder Angle (deg)": round(left_angle, 1) if has_pose else 0.0,
                "Right Shoulder Angle (deg)": round(right_angle, 1) if has_pose else 0.0,
            }

            yield FrameResult(
                frame_idx=frame_idx,
                timestamp_sec=current_time_sec,
                is_recording=is_recording,
                annotated_frame_rgb=cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                active_cells=current_active_cells,
                new_log_lines=new_logs,
                total_hits=self.total_hits,
                left_hits=self.left_hits_count,
                right_hits=self.right_hits_count,
                left_speeds=self.left_speeds,
                right_speeds=self.right_speeds,
                left_jerks=self.left_jerks,
                right_jerks=self.right_jerks,
                left_rom_min=self.left_rom_min,
                left_rom_max=self.left_rom_max,
                right_rom_min=self.right_rom_min,
                right_rom_max=self.right_rom_max,
                left_current_rom=self.left_current_rom,
                right_current_rom=self.right_current_rom,
                left_straightness=left_straightness,
                right_straightness=right_straightness,
                dominant_side=dominant_side,
                history_row=history_row,
            )

    # -- final summary ----------------------------------------------------
    def summarize(self) -> dict:
        left_reaches = sum(1 for r in self.reaches if r["arm"] == "left")
        right_reaches = sum(1 for r in self.reaches if r["arm"] == "right")
        total_hits = len(self.reaches)

        left_avg_reach_time = float(np.mean([r["reachTimeMs"] for r in self.reaches if r["arm"] == "left"])) if left_reaches > 0 else 0.0
        right_avg_reach_time = float(np.mean([r["reachTimeMs"] for r in self.reaches if r["arm"] == "right"])) if right_reaches > 0 else 0.0
        left_min_reach_time = float(np.min([r["reachTimeMs"] for r in self.reaches if r["arm"] == "left"])) if left_reaches > 0 else 0.0
        right_min_reach_time = float(np.min([r["reachTimeMs"] for r in self.reaches if r["arm"] == "right"])) if right_reaches > 0 else 0.0
        left_max_reach_time = float(np.max([r["reachTimeMs"] for r in self.reaches if r["arm"] == "left"])) if left_reaches > 0 else 0.0
        right_max_reach_time = float(np.max([r["reachTimeMs"] for r in self.reaches if r["arm"] == "right"])) if right_reaches > 0 else 0.0

        left_avg_straightness = float(np.mean([r["straightness"] for r in self.reaches if r["arm"] == "left"])) if left_reaches > 0 else 1.0
        right_avg_straightness = float(np.mean([r["straightness"] for r in self.reaches if r["arm"] == "right"])) if right_reaches > 0 else 1.0

        left_avg_jerk = float(np.mean([r["jerk"] for r in self.reaches if r["arm"] == "left"])) if left_reaches > 0 else 0.0
        right_avg_jerk = float(np.mean([r["jerk"] for r in self.reaches if r["arm"] == "right"])) if right_reaches > 0 else 0.0

        # Dominant side based on usage frequency, fallback to avg speed
        if left_reaches != right_reaches:
            dominant_side_en = "LEFT" if left_reaches > right_reaches else "RIGHT"
        else:
            left_speed_sum = sum(self.left_speeds)
            right_speed_sum = sum(self.right_speeds)
            dominant_side_en = "LEFT" if left_speed_sum > right_speed_sum else "RIGHT"

        left_rom_range = self.left_rom_max - self.left_rom_min if self.left_rom_max != float("-inf") else 0.0
        right_rom_range = self.right_rom_max - self.right_rom_min if self.right_rom_max != float("-inf") else 0.0

        # LNI Score calculation (4-axis: reach duration, path straightness, jerk, usage frequency)
        lni_score = 0.0
        if left_reaches > 0 and right_reaches > 0:
            max_avg_time = max(left_avg_reach_time, right_avg_reach_time)
            speed_asym = abs(left_avg_reach_time - right_avg_reach_time) / max_avg_time if max_avg_time > 0 else 0.0
            straight_asym = abs(left_avg_straightness - right_avg_straightness)
            total_jerk = left_avg_jerk + right_avg_jerk
            jerk_asym = abs(left_avg_jerk - right_avg_jerk) / total_jerk if total_jerk > 0 else 0.0
            usage_asym = abs(left_reaches - right_reaches) / total_hits if total_hits > 0 else 0.0

            lni_score = speed_asym * 0.30 + straight_asym * 0.20 + jerk_asym * 0.20 + usage_asym * 0.30
            lni_score = min(1.0, max(0.0, lni_score))
        elif left_reaches > 0 or right_reaches > 0:
            # One sided reaches is max usage asymmetry
            lni_score = min(1.0, 0.30 * 1.0 + 0.45)

        # Risk levels matching Next.js
        if lni_score < 0.15:
            lnu_risk = "Low Risk (ความเสี่ยงต่ำ - ทั้งสองข้างใช้งานสมมาตรดี)"
            lnu_color = "#34D399"
        elif lni_score < 0.35:
            lnu_risk = "Mild Asymmetry (ความไม่สมมาตรเล็กน้อย - เริ่มมีความแตกต่างระหว่างสองฝั่ง)"
            lnu_color = "#FBBF24"
        elif lni_score < 0.55:
            lnu_risk = "Moderate Risk (ความเสี่ยงปานกลาง - มีแนวโน้มชดเชยการใช้กำลังสองฝั่งไม่สมดุล)"
            lnu_color = "#F59E0B"
        else:
            lnu_risk = "High Risk (ความเสี่ยงสูง - ตรวจพบภาวะฝืนไม่ใช้งานแขนข้างที่อ่อนแรงชัดเจน)"
            lnu_color = "#EF4444"

        # Duration
        duration_sec = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps if self.fps > 0 else 0.0
        if duration_sec <= 0:
            duration_sec = (self.total_frames / self.fps) if self.fps > 0 else 0.0

        return {
            "total_hits": total_hits,
            "left_reaches": left_reaches,
            "right_reaches": right_reaches,
            "left_hits": left_reaches,
            "right_hits": right_reaches,
            "duration_sec": duration_sec,
            "duration_ms": duration_sec * 1000.0,
            "dominant_side_en": dominant_side_en,
            
            "left_avg_reach_time": left_avg_reach_time,
            "right_avg_reach_time": right_avg_reach_time,
            "left_min_reach_time": left_min_reach_time,
            "right_min_reach_time": right_min_reach_time,
            "left_max_reach_time": left_max_reach_time,
            "right_max_reach_time": right_max_reach_time,
            
            "left_avg_speed": float(np.mean(self.left_speeds)) if self.left_speeds else 0.0,
            "right_avg_speed": float(np.mean(self.right_speeds)) if self.right_speeds else 0.0,
            "left_max_speed": float(np.max(self.left_speeds)) if self.left_speeds else 0.0,
            "right_max_speed": float(np.max(self.right_speeds)) if self.right_speeds else 0.0,
            
            "left_avg_jerk": left_avg_jerk,
            "right_avg_jerk": right_avg_jerk,
            "left_rom_range": left_rom_range,
            "right_rom_range": right_rom_range,
            "left_current_rom": self.left_current_rom,
            "right_current_rom": self.right_current_rom,
            
            "left_avg_straightness": left_avg_straightness,
            "right_avg_straightness": right_avg_straightness,
            "left_straightness": left_avg_straightness * 100.0,
            "right_straightness": right_avg_straightness * 100.0,
            
            "lni_score": lni_score,
            "lnu_risk": lnu_risk,
            "lnu_color": lnu_color,
            "reaches": self.reaches,
        }
