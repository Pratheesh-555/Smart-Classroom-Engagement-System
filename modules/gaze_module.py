"""
Eye Gaze Analysis Module
=========================
Uses MediaPipe FaceLandmarker (Tasks API — compatible with mediapipe ≥0.10.21
and Python 3.13) to estimate iris position and classify gaze direction.

The module detects face landmarks including iris (indices 468-477) and
computes horizontal/vertical iris-position ratios to determine if a
student is looking at the screen, away, or down.

Returns a dict with:
    attention_score   : float (0-1, class average)
    looking_away_rate : float (0-1, fraction looking away)
    faces             : list[dict]
"""

import os
import cv2
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_DIR, "face_landmarker.task")


class GazeAnalyzer:
    def __init__(self):
        self._available = False

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )

            if not os.path.exists(_MODEL_PATH):
                print(f"[EduLens] Gaze model not found at {_MODEL_PATH}")
                return

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_faces=30,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            self._available = True
            print("[EduLens] Gaze module ready (FaceLandmarker Tasks API)")

        except (ImportError, AttributeError, RuntimeError) as e:
            print(f"[EduLens] Gaze module disabled: {e}")
            self._landmarker = None

        # ── Iris / eye landmark indices (same as before) ──
        self.LEFT_IRIS_CENTER = 468
        self.RIGHT_IRIS_CENTER = 473
        self.LEFT_EYE_INNER = 133
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_OUTER = 263
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374

        # ── Gaze classification thresholds ──
        self.H_AWAY_LOW = 0.30
        self.H_AWAY_HIGH = 0.70
        self.V_DOWN_THRESHOLD = 0.65

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_point(self, landmark, h, w):
        """Convert a NormalizedLandmark to pixel coordinates."""
        return np.array([landmark.x * w, landmark.y * h])

    def _compute_iris_ratios(self, landmarks, h, w):
        """
        Compute horizontal and vertical iris-position ratios for both eyes.
        """
        pt = lambda idx: self._get_point(landmarks[idx], h, w)

        # Left eye
        left_iris = pt(self.LEFT_IRIS_CENTER)
        left_outer = pt(self.LEFT_EYE_OUTER)
        left_inner = pt(self.LEFT_EYE_INNER)
        left_top = pt(self.LEFT_EYE_TOP)
        left_bottom = pt(self.LEFT_EYE_BOTTOM)

        left_w = max(np.linalg.norm(left_inner - left_outer), 1.0)
        left_h = max(np.linalg.norm(left_top - left_bottom), 1.0)
        left_h_ratio = np.linalg.norm(left_iris - left_outer) / left_w
        left_v_ratio = np.linalg.norm(left_iris - left_top) / left_h

        # Right eye
        right_iris = pt(self.RIGHT_IRIS_CENTER)
        right_outer = pt(self.RIGHT_EYE_OUTER)
        right_inner = pt(self.RIGHT_EYE_INNER)
        right_top = pt(self.RIGHT_EYE_TOP)
        right_bottom = pt(self.RIGHT_EYE_BOTTOM)

        right_w = max(np.linalg.norm(right_inner - right_outer), 1.0)
        right_h = max(np.linalg.norm(right_top - right_bottom), 1.0)
        right_h_ratio = np.linalg.norm(right_iris - right_outer) / right_w
        right_v_ratio = np.linalg.norm(right_iris - right_top) / right_h

        # Average both eyes
        h_ratio = (left_h_ratio + right_h_ratio) / 2.0
        v_ratio = (left_v_ratio + right_v_ratio) / 2.0

        gaze_x = round((h_ratio - 0.5) * 2, 3)
        gaze_y = round((v_ratio - 0.5) * 2, 3)

        return h_ratio, v_ratio, (gaze_x, gaze_y)

    def _classify_gaze(self, h_ratio, v_ratio):
        """Classify gaze into screen / away / down."""
        if v_ratio > self.V_DOWN_THRESHOLD:
            return "down"
        if h_ratio < self.H_AWAY_LOW or h_ratio > self.H_AWAY_HIGH:
            return "away"
        return "screen"

    def _compute_attention_score(self, looking_at, h_ratio, v_ratio):
        """Derive a 0-1 attention score from the gaze classification."""
        if looking_at == "screen":
            h_dev = abs(h_ratio - 0.5)
            v_dev = abs(v_ratio - 0.5)
            centred = 1.0 - (h_dev + v_dev)
            return round(max(0.6, min(1.0, 0.7 + centred * 0.3)), 2)
        elif looking_at == "down":
            return 0.35
        else:
            return 0.15

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze_frame(self, frame):
        """
        Analyse a single BGR video frame for eye-gaze patterns.
        Falls back to neutral scores if mediapipe is unavailable.
        """
        _neutral = {
            "attention_score": 0.5,
            "looking_away_rate": 0.0,
            "faces": [],
        }

        if not self._available or self._landmarker is None:
            return _neutral

        try:
            import mediapipe as mp

            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = self._landmarker.detect(mp_image)

            if not result.face_landmarks:
                return _neutral

            faces = []
            for face_lm in result.face_landmarks:
                # FaceLandmarker returns 478 landmarks including iris
                if len(face_lm) < 478:
                    continue

                h_ratio, v_ratio, gaze_dir = self._compute_iris_ratios(
                    face_lm, h, w
                )
                looking_at = self._classify_gaze(h_ratio, v_ratio)
                attention = self._compute_attention_score(
                    looking_at, h_ratio, v_ratio
                )

                faces.append({
                    "looking_at": looking_at,
                    "gaze_direction": gaze_dir,
                    "attention_score": attention,
                })

            if faces:
                avg_attention = round(
                    sum(f["attention_score"] for f in faces) / len(faces), 2
                )
                looking_away_count = sum(
                    1 for f in faces if f["looking_at"] != "screen"
                )
                looking_away_rate = round(looking_away_count / len(faces), 2)
            else:
                avg_attention = 0.5
                looking_away_rate = 0.0

            return {
                "attention_score": avg_attention,
                "looking_away_rate": looking_away_rate,
                "faces": faces,
            }

        except Exception as e:
            # Never crash — return neutral on any error
            return _neutral
