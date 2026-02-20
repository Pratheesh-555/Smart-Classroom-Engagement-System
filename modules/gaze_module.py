"""
Eye Gaze Analysis Module — STUB
================================
Friends: Replace the placeholder logic inside `analyze_frame()`
with a real gaze-tracking implementation (e.g. MediaPipe Face Mesh).

Expected return format per face:
    {
        "looking_at"      : "screen" | "away" | "down",
        "gaze_direction"  : (x, y),          # normalised gaze vector
        "attention_score"  : float (0-1),
    }

The module-level `analyze_frame` returns a dict with:
    attention_score   : float (0-1, class average)
    looking_away_rate : float (0-1, fraction looking away)
    faces             : list[dict]
"""

import random


class GazeAnalyzer:
    def analyze_frame(self, frame):
        """
        Analyse a video frame for eye-gaze patterns.

        TODO ── Replace random stubs with real inference:
            1. Use MediaPipe Face Mesh to get 468 face landmarks.
            2. Extract iris landmarks (468-477) to compute gaze vector.
            3. Classify as screen / away / down.
        """
        return {
            "attention_score": round(random.uniform(0.45, 0.85), 2),
            "looking_away_rate": round(random.uniform(0.10, 0.40), 2),
            "faces": [],
        }
