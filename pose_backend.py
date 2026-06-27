"""Pose-estimation backend built on the MediaPipe **Tasks** API.

The legacy ``mediapipe.solutions`` module is not shipped in some MediaPipe
builds (notably the Python 3.14 wheels), which is why the original
``mp.solutions.pose`` import fails at runtime. This module wraps the modern
``mediapipe.tasks.python.vision.PoseLandmarker`` instead and exposes a small,
stable surface that the rest of the app relies on:

* :class:`PoseEstimator` - load the model (auto-downloading it once) and run
  per-frame detection on BGR frames.
* :data:`POSE_LANDMARK` - landmark-index constants (same numbering as the old
  ``PoseLandmark`` enum), so existing index lookups keep working.
* :func:`draw_pose` - draw the skeleton with OpenCV (the Tasks build has no
  ``drawing_utils`` helper).
"""

from __future__ import annotations

import os
import urllib.request
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Landmark indices match the classic MediaPipe Pose ordering (33 landmarks).
POSE_LANDMARK = SimpleNamespace(
    LEFT_SHOULDER=11,
    RIGHT_SHOULDER=12,
    LEFT_ELBOW=13,
    RIGHT_ELBOW=14,
    LEFT_WRIST=15,
    RIGHT_WRIST=16,
    LEFT_INDEX=19,
    RIGHT_INDEX=20,
    LEFT_HIP=23,
    RIGHT_HIP=24,
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "pose_landmarker_full.task")


def ensure_model(path: str = _MODEL_PATH, url: str = _MODEL_URL) -> str:
    """Return the local model path, downloading it once if necessary."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


class PoseEstimator:
    """Thin wrapper around ``PoseLandmarker`` running in VIDEO mode."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        model_path = model_path or ensure_model()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int):
        """Detect a single pose and return ``pose_landmarks`` or ``None``.

        The returned object mimics the legacy result: it has a
        ``landmark`` list indexable by :data:`POSE_LANDMARK` constants, where
        each entry exposes ``.x``, ``.y`` and ``.z`` in normalized coordinates.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        if not result.pose_landmarks:
            return None
        return SimpleNamespace(landmark=result.pose_landmarks[0])

    def close(self) -> None:
        self._landmarker.close()


# Connections used to draw the skeleton (resolved lazily for portability).
def _connections():
    try:
        return [
            (c.start, c.end)
            for c in vision.PoseLandmarksConnections.POSE_LANDMARKS
        ]
    except Exception:
        return []


_POSE_CONNECTIONS = _connections()


def draw_pose(frame_bgr: np.ndarray, landmarks) -> None:
    """Draw the pose skeleton + joints onto ``frame_bgr`` in place."""
    if landmarks is None:
        return
    h, w = frame_bgr.shape[:2]
    
    if isinstance(landmarks, dict):
        pts = {idx: (int(lm["x"] * w), int(lm["y"] * h)) for idx, lm in landmarks.items()}
    else:
        # Fallback for raw MediaPipe landmark object
        pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks.landmark)}

    # Connections for stable upper body skeleton
    connections = [
        (11, 12),  # Shoulder to Shoulder
        (11, 13),  # L Shoulder to L Elbow
        (13, 15),  # L Elbow to L Wrist
        (15, 19),  # L Wrist to L Index
        (12, 14),  # R Shoulder to R Elbow
        (14, 16),  # R Elbow to R Wrist
        (16, 20),  # R Wrist to R Index
        (11, 23),  # L Shoulder to L Hip
        (12, 24),  # R Shoulder to R Hip
        (23, 24),  # L Hip to R Hip
    ]

    # Draw lines
    for start, end in connections:
        if start in pts and end in pts:
            cv2.line(frame_bgr, pts[start], pts[end], (88, 205, 175), 3, cv2.LINE_AA)
            
    # Draw active joint circles
    active_indices = {11, 12, 13, 14, 15, 16, 19, 20, 23, 24}
    for idx in active_indices:
        if idx in pts:
            cv2.circle(frame_bgr, pts[idx], 5, (245, 158, 11), -1, cv2.LINE_AA)
