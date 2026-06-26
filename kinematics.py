"""Pure kinematics math: angles, smoothing, distances and path metrics.

These helpers are framework-agnostic (no Streamlit / OpenCV) so they can be
unit-tested and reused by the live-webcam frontend as well.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

Point3D = dict  # {'x': float, 'y': float, 'z': float}


def calculate_shoulder_angle(shoulder, elbow, hip) -> float:
    """Shoulder abduction/elevation angle (deg) relative to the vertical torso.

    Each argument is an ``(x, y)`` pair in normalized image coordinates.
    """
    v_arm = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    v_torso = np.array([hip[0] - shoulder[0], hip[1] - shoulder[1]])

    norm_arm = np.linalg.norm(v_arm)
    norm_torso = np.linalg.norm(v_torso)
    if norm_arm == 0 or norm_torso == 0:
        return 0.0

    cos_theta = float(np.dot(v_arm, v_torso) / (norm_arm * norm_torso))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


class TemporalSmoother:
    """Moving-average smoother for a 3D landmark dictionary."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history: list[Point3D] = []

    def smooth(self, pt: Point3D) -> Point3D:
        self.history.append(pt)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        n = len(self.history)
        return {
            "x": sum(p["x"] for p in self.history) / n,
            "y": sum(p["y"] for p in self.history) / n,
            "z": sum(p["z"] for p in self.history) / n,
        }


def dist_3d_norm(p1: Point3D, p2: Point3D) -> float:
    """3D Euclidean distance between two landmark dictionaries."""
    dx = p1["x"] - p2["x"]
    dy = p1["y"] - p2["y"]
    dz = p1["z"] - p2["z"]
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def displacement_and_path_length(history: Sequence[Point3D]) -> tuple[float, float]:
    """Net displacement (start->end) and total path length of a trajectory."""
    if len(history) < 2:
        return 0.0, 0.0
    displacement = dist_3d_norm(history[0], history[-1])
    path_length = 0.0
    for i in range(1, len(history)):
        path_length += dist_3d_norm(history[i], history[i - 1])
    return displacement, path_length


def straightness_pct(history: Sequence[Point3D]) -> float:
    """Movement straightness as a percentage (displacement / path length)."""
    if not history:
        return 0.0
    displacement, path_length = displacement_and_path_length(history)
    if path_length <= 0.001:
        return 0.0
    return displacement / path_length * 100.0


def calculate_straightness(path: list[dict]) -> float:
    """Path straightness: start-to-end displacement divided by total path length."""
    if len(path) < 2:
        return 1.0

    start = path[0]
    end = path[-1]
    
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    displacement = (dx * dx + dy * dy) ** 0.5

    path_length = 0.0
    for i in range(1, len(path)):
        p1 = path[i - 1]
        p2 = path[i]
        p_dx = p2["x"] - p1["x"]
        p_dy = p2["y"] - p1["y"]
        path_length += (p_dx * p_dx + p_dy * p_dy) ** 0.5

    if path_length < 0.001:
        return 1.0
        
    return min(1.0, displacement / path_length)


def calculate_jerk(path: list[dict]) -> float:
    """Average jerk magnitude over a path of points with 'x', 'y' and 't' (ms) keys.
    
    Matches the JS implementation in game-engine.ts.
    """
    if len(path) < 4:
        return 0.0

    # Velocities
    vel = []
    for i in range(1, len(path)):
        dt = (path[i]["t"] - path[i - 1]["t"]) / 1000.0
        if dt <= 0:
            continue
        vel.append({
            "vx": (path[i]["x"] - path[i - 1]["x"]) / dt,
            "vy": (path[i]["y"] - path[i - 1]["y"]) / dt,
            "t": path[i]["t"]
        })

    # Accelerations
    acc = []
    for i in range(1, len(vel)):
        dt = (vel[i]["t"] - vel[i - 1]["t"]) / 1000.0
        if dt <= 0:
            continue
        acc.append({
            "ax": (vel[i]["vx"] - vel[i - 1]["vx"]) / dt,
            "ay": (vel[i]["vy"] - vel[i - 1]["vy"]) / dt,
            "t": vel[i]["t"]
        })

    # Jerk
    total_jerk = 0.0
    count = 0
    for i in range(1, len(acc)):
        dt = (acc[i]["t"] - acc[i - 1]["t"]) / 1000.0
        if dt <= 0:
            continue
        jx = (acc[i]["ax"] - acc[i - 1]["ax"]) / dt
        jy = (acc[i]["ay"] - acc[i - 1]["ay"]) / dt
        total_jerk += (jx * jx + jy * jy) ** 0.5
        count += 1

    return total_jerk / count if count > 0 else 0.0
