"""
Multimodal Fusion Engine
========================
Combines facial (Fi), posture (Pi) and gaze/audio (Ai) scores
into a single per-classroom engagement metric.

    Ei = alpha * Fi + beta * Pi + gamma * Ai       (PDF Page 7)
    E_class = (1/N) * sum(Ei)

Uses a 10-second sliding window and a 40 % drop-detection threshold.
"""

from collections import deque
from config import ALPHA, BETA, GAMMA, ENGAGEMENT_DROP_THRESHOLD, SLIDING_WINDOW_SECONDS


class FusionEngine:
    def __init__(self):
        self.history = deque(maxlen=500)

    def compute(self, facial_summary, gaze_results, posture_results):
        """
        Fuse the three modality scores into dashboard-ready metrics.

        Parameters
        ----------
        facial_summary : dict from FacialAnalyzer.get_class_summary()
        gaze_results   : dict from GazeAnalyzer.analyze_frame()
        posture_results: dict from PostureAnalyzer.analyze_frame()

        Returns
        -------
        dict with keys used directly by the Streamlit dashboard.
        """
        fi = facial_summary.get("avg_engagement", 0.5)
        pi = posture_results.get("engagement_score", 0.5)
        ai = gaze_results.get("attention_score", 0.5)

        overall = ALPHA * fi + BETA * pi + GAMMA * ai

        result = {
            "overall_engagement": round(overall, 3),
            "facial_score": round(fi, 3),
            "posture_score": round(pi, 3),
            "gaze_score": round(ai, 3),
            "confusion_rate": facial_summary.get("confusion_rate", 0.0),
            "looking_away_rate": gaze_results.get("looking_away_rate", 0.0),
            "slouching_rate": posture_results.get("slouching_rate", 0.0),
            "num_students": facial_summary.get("num_students", 0),
            "emotion_distribution": facial_summary.get("emotion_distribution", {}),
            "drop_detected": overall < ENGAGEMENT_DROP_THRESHOLD,
        }

        self.history.append(result)
        return result

    def get_sliding_window_avg(self, fps, window_sec=SLIDING_WINDOW_SECONDS):
        """Return average engagement over the last `window_sec` seconds."""
        n = int(fps * window_sec) if fps > 0 else len(self.history)
        recent = list(self.history)[-n:]
        if not recent:
            return 0.0
        return sum(r["overall_engagement"] for r in recent) / len(recent)
