"""
Body Posture Analysis Module
=============================
Uses YOLOv8-pose (from modules/pose/yolov8n-pose.pt) for real-time pose
estimation and classifies student concentration based on body gestures.

Gesture classification adapted from:
  modules/pose/gestures.py
  (https://github.com/RAJADURAI3/Real-Time-Gesture-Recognition-PyTorch-YOLOv8-OpenCV-)

Returns
-------
{
    "engagement_score"   : float (0-1),
    "slouching_rate"     : float (0-1),
    "concentration_rate" : float (0-1),
    "posture_states"     : {"upright": int, "slouching": int,
                            "head_down": int, "shifting": int},
    "per_person"         : [{"person_id", "gestures", "dynamic_actions",
                             "concentrated"}, ...],
    "gesture_log"        : [{"Frame", "PersonID", "Gestures",
                             "DynamicActions", "Concentrated"}, ...],
}

Extra keys are ignored by the existing fusion.py / app.py (they use .get()),
so backwards compatibility is preserved.
"""

import os
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Resolve the local .pt model shipped inside modules/pose/
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_DIR, "pose", "yolov8n-pose.pt")


# ═══════════════════════════════════════════════════════════════════════════
# Keypoint helpers  (adapted from pose/gestures.py — converted to numpy)
# ═══════════════════════════════════════════════════════════════════════════
def _above(p1, p2):
    return p1[1] < p2[1]


def _below(p1, p2):
    return p1[1] > p2[1]


def _aligned(p1, p2, tol=20):
    return abs(p1[1] - p2[1]) < tol


def _near(p1, p2, tol=40):
    return float(np.linalg.norm(p1 - p2)) < tol


def _valid(pt):
    """True when a keypoint was actually detected (not all-zero)."""
    return not np.allclose(pt, 0)


# ═══════════════════════════════════════════════════════════════════════════
# Static gesture classifier  (full port of pose/gestures.py)
# ═══════════════════════════════════════════════════════════════════════════
def _classify_gestures(kp):
    """
    Classify static gestures from 17 COCO keypoints (numpy float32 array,
    shape (17, 2)).  Returns list of (gesture_name, confidence).
    """
    if kp.shape[0] < 17:
        return [("Incomplete Pose", 1.0)]

    nose            = kp[0]
    left_eye        = kp[1];  right_eye       = kp[2]
    left_shoulder   = kp[5];  right_shoulder  = kp[6]
    left_elbow      = kp[7];  right_elbow     = kp[8]
    left_wrist      = kp[9];  right_wrist     = kp[10]
    left_hip        = kp[11]; right_hip       = kp[12]
    left_knee       = kp[13]; right_knee      = kp[14]
    left_ankle      = kp[15]; right_ankle     = kp[16]

    if not (_valid(left_shoulder) and _valid(right_shoulder)):
        return [("Incomplete Pose", 1.0)]

    gestures = []

    # ── Posture ──────────────────────────────────────────────────────────
    hips_ok  = _valid(left_hip)  and _valid(right_hip)
    knees_ok = _valid(left_knee) and _valid(right_knee)

    if knees_ok and hips_ok:
        if _below(left_knee, left_hip) and _below(right_knee, right_hip):
            gestures.append(("Standing", 0.9))
        elif _above(left_knee, left_hip) and _above(right_knee, right_hip):
            gestures.append(("Sitting", 0.9))
        if (abs(left_hip[1] - left_knee[1]) < 30
                and abs(right_hip[1] - right_knee[1]) < 30):
            gestures.append(("Squat", 0.85))

    if hips_ok:
        if (abs(left_shoulder[1] - left_hip[1]) < 20
                and abs(right_shoulder[1] - right_hip[1]) < 20):
            gestures.append(("Lying Down", 0.9))

    # ── Slouching (torso-ratio heuristic for seated students) ────────────
    if hips_ok:
        mid_sh = (left_shoulder + right_shoulder) / 2.0
        mid_hp = (left_hip + right_hip) / 2.0
        torso_len = abs(mid_sh[1] - mid_hp[1])
        sh_width  = float(np.linalg.norm(left_shoulder - right_shoulder))
        if sh_width > 0 and (torso_len / sh_width) < 1.0:
            gestures.append(("Slouching", 0.85))

    # ── Head ─────────────────────────────────────────────────────────────
    if _valid(nose):
        if nose[0] < left_shoulder[0] and nose[0] < right_shoulder[0]:
            gestures.append(("Head Turn Left", 0.85))
        elif nose[0] > left_shoulder[0] and nose[0] > right_shoulder[0]:
            gestures.append(("Head Turn Right", 0.85))
        if _above(nose, left_shoulder) and _above(nose, right_shoulder):
            gestures.append(("Head Up", 0.8))
        elif _below(nose, left_shoulder) and _below(nose, right_shoulder):
            gestures.append(("Head Down", 0.8))

    if _valid(left_eye) and _valid(right_eye):
        if abs(left_eye[0] - right_eye[0]) > 40:
            gestures.append(("Head Rotated", 0.7))

    # ── Arms ─────────────────────────────────────────────────────────────
    lw_ok = _valid(left_wrist);  rw_ok = _valid(right_wrist)
    if lw_ok and rw_ok and _valid(nose):
        if _above(left_wrist, nose) and _above(right_wrist, nose):
            gestures.append(("Raise Hands", 0.9))
        elif _above(left_wrist, nose):
            gestures.append(("Left Hand Raised", 0.85))
        elif _above(right_wrist, nose):
            gestures.append(("Right Hand Raised", 0.85))

    if lw_ok and rw_ok:
        if (_aligned(left_wrist, left_shoulder)
                and _aligned(right_wrist, right_shoulder)):
            gestures.append(("T-Pose", 0.8))
        if hips_ok:
            if _near(left_wrist, left_hip) and _near(right_wrist, right_hip):
                gestures.append(("Hands on Hips", 0.8))
        if _near(left_wrist, right_shoulder) and _near(right_wrist, left_shoulder):
            gestures.append(("Crossed Arms", 0.7))

    le_ok = _valid(left_elbow); re_ok = _valid(right_elbow)
    if lw_ok and le_ok and left_wrist[0] < left_elbow[0] and _aligned(left_wrist, left_shoulder):
        gestures.append(("Point Left", 0.75))
    if rw_ok and re_ok and right_wrist[0] > right_elbow[0] and _aligned(right_wrist, right_shoulder):
        gestures.append(("Point Right", 0.75))

    # ── Legs ─────────────────────────────────────────────────────────────
    la_ok = _valid(left_ankle); ra_ok = _valid(right_ankle)
    if la_ok and ra_ok and knees_ok:
        if left_ankle[1] < left_knee[1] and right_ankle[1] < right_knee[1]:
            gestures.append(("Jumping", 0.9))
        if left_ankle[0] - right_ankle[0] > 50:
            gestures.append(("Step Right", 0.8))
        elif right_ankle[0] - left_ankle[0] > 50:
            gestures.append(("Step Left", 0.8))

    if not gestures:
        gestures.append(("Neutral", 0.6))

    return gestures


# ═══════════════════════════════════════════════════════════════════════════
# Concentration mapping
# ═══════════════════════════════════════════════════════════════════════════
_CONCENTRATED = {
    "Standing", "Sitting", "Head Up", "Raise Hands",
    "Left Hand Raised", "Right Hand Raised",
    "Neutral", "Hands on Hips", "Crossed Arms",
}
_NOT_CONCENTRATED = {
    "Head Down", "Lying Down", "Slouching",
    "Head Turn Left", "Head Turn Right", "Head Rotated",
    "Squat", "Jumping", "T-Pose",
}


def _is_concentrated(gesture_names):
    """Return True when the balance of evidence says the student is focused."""
    names = set(gesture_names)
    if names & _NOT_CONCENTRATED:
        return False
    if names & _CONCENTRATED:
        return True
    return True  # neutral → tentatively concentrated


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic action tracker  (direct port of pose/gestures.py GestureTracker)
# ═══════════════════════════════════════════════════════════════════════════
class _DynamicTracker:
    def __init__(self, history=30):
        self._histories = {}
        self._maxlen = history

    def update(self, person_id, gestures):
        if person_id not in self._histories:
            self._histories[person_id] = deque(maxlen=self._maxlen)
        self._histories[person_id].append(gestures)
        return self._detect(person_id)

    def _detect(self, pid):
        names = [g[0] for frame in self._histories[pid] for g in frame]
        actions = []
        if "Head Up" in names and "Head Down" in names:
            actions.append(("Nodding", 0.9))
        if "Left Hand Raised" in names and "Point Left" in names:
            actions.append(("Waving Left", 0.85))
        if "Right Hand Raised" in names and "Point Right" in names:
            actions.append(("Waving Right", 0.85))
        if "Step Left" in names and "Step Right" in names:
            actions.append(("Walking", 0.9))
        if names.count("Jumping") > 5:
            actions.append(("Repeated Jumping", 0.95))
        return actions


# ═══════════════════════════════════════════════════════════════════════════
# Shift (fidget) detector — hip-centre deltas across frames
# ═══════════════════════════════════════════════════════════════════════════
class _ShiftTracker:
    def __init__(self, history_len=15, threshold=25.0):
        self._history = {}
        self._maxlen = history_len
        self._threshold = threshold

    def update(self, person_idx, kp):
        lh, rh = kp[11], kp[12]
        if not (_valid(lh) and _valid(rh)):
            return False
        mid = (lh + rh) / 2.0
        if person_idx not in self._history:
            self._history[person_idx] = deque(maxlen=self._maxlen)
        self._history[person_idx].append(mid)
        if len(self._history[person_idx]) < 4:
            return False
        pts = np.array(self._history[person_idx])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) > self._threshold


# ═══════════════════════════════════════════════════════════════════════════
# Main analyser
# ═══════════════════════════════════════════════════════════════════════════
class PostureAnalyzer:
    """
    Drop-in replacement for the stub PostureAnalyzer.

    * Loads the YOLOv8-pose model from ``modules/pose/yolov8n-pose.pt``.
    * Classifies every detected person's gestures.
    * Determines per-student **concentration** status.
    * Tracks dynamic actions (nodding, waving …) over time.
    * Accumulates gesture counts per frame for heatmap visualisation.
    """

    _ENGAGEMENT = {
        "upright":  1.0,
        "slouching": 0.3,
        "head_down": 0.2,
        "shifting":  0.4,
    }

    def __init__(self, model_path: str = _MODEL_PATH, conf: float = 0.5):
        self._model = YOLO(model_path)
        self._conf = conf
        self._shift_tracker   = _ShiftTracker()
        self._dynamic_tracker = _DynamicTracker()
        self._frame_count = 0
        self._heatmap_log = []          # [{frame, gestures:{name:count}}]

    # -----------------------------------------------------------------
    def analyze_frame(self, frame):
        """
        Analyse a BGR video frame for body posture and concentration.

        Returns a dict consumed by FusionEngine.compute() and the dashboard.
        """
        self._frame_count += 1

        empty = self._empty_result()
        if frame is None:
            return empty

        # ── YOLOv8-pose inference ────────────────────────────────────
        results = self._model(frame, conf=self._conf, verbose=False)
        if not results or results[0].keypoints is None:
            return empty

        kps = results[0].keypoints
        num_people = len(kps)
        if num_people == 0:
            return empty

        # ── Per-person analysis ──────────────────────────────────────
        posture_counts = {"upright": 0, "slouching": 0,
                          "head_down": 0, "shifting": 0}
        per_person     = []
        concentrated_n = 0
        frame_gestures = defaultdict(int)

        for idx in range(num_people):
            kp = kps[idx].xy[0].cpu().numpy()          # (17, 2)

            # Full static gesture list
            gestures = _classify_gestures(kp)
            gesture_names = [g[0] for g in gestures]

            # Dynamic actions over time
            dynamic_actions = self._dynamic_tracker.update(idx, gestures)

            # Shifting / fidget detection
            is_shifting = self._shift_tracker.update(idx, kp)

            # ── Map gestures → posture bucket ────────────────────────
            if "Head Down" in gesture_names:
                bucket = "head_down"
            elif ("Slouching" in gesture_names
                  or "Lying Down" in gesture_names
                  or "Squat" in gesture_names):
                bucket = "slouching"
            elif is_shifting:
                bucket = "shifting"
            else:
                bucket = "upright"

            posture_counts[bucket] += 1

            # ── Concentration ────────────────────────────────────────
            concentrated = _is_concentrated(gesture_names)
            if is_shifting:
                concentrated = False
            # Nodding → engaged (override to concentrated)
            if ("Nodding", 0.9) in dynamic_actions:
                concentrated = True
            if concentrated:
                concentrated_n += 1

            # ── Accumulate gesture counts for heatmap ────────────────
            for g in gesture_names:
                frame_gestures[g] += 1

            per_person.append({
                "person_id":       idx,
                "gestures":        gesture_names,
                "dynamic_actions": [a[0] for a in dynamic_actions],
                "concentrated":    concentrated,
            })

        # ── Heatmap log entry ────────────────────────────────────────
        self._heatmap_log.append({
            "frame":    self._frame_count,
            "gestures": dict(frame_gestures),
        })

        # ── Aggregate scores ─────────────────────────────────────────
        eng_vals = [self._ENGAGEMENT.get(
            "upright" if p["concentrated"] else
            ("head_down" if "Head Down" in p["gestures"] else "slouching"),
            0.5) for p in per_person]
        engagement_score  = round(float(np.mean(eng_vals)), 2)

        slouch_count      = posture_counts["slouching"] + posture_counts["head_down"]
        slouching_rate    = round(slouch_count / num_people, 2)
        concentration_rate = round(concentrated_n / num_people, 2)

        # ── Build gesture_log (same schema as pose/main.py JSON) ─────
        gesture_log = [{
            "Frame":          self._frame_count,
            "PersonID":       p["person_id"],
            "Gestures":       p["gestures"],
            "DynamicActions": p["dynamic_actions"],
            "Concentrated":   p["concentrated"],
        } for p in per_person]

        return {
            # --- keys used by fusion.py / app.py (backward-compat) ---
            "engagement_score": engagement_score,
            "slouching_rate":   slouching_rate,
            "posture_states":   posture_counts,
            # --- new keys for concentration & visualisation ----------
            "concentration_rate": concentration_rate,
            "per_person":         per_person,
            "gesture_log":        gesture_log,
        }

    # -----------------------------------------------------------------
    def get_heatmap_data(self):
        """
        Return accumulated gesture counts suitable for a Plotly / Seaborn
        heatmap (frames × gesture types).

        Returns
        -------
        dict with keys *frames* (list[int]), *actions* (list[str]),
        *matrix* (list[list[int]]).
        """
        if not self._heatmap_log:
            return {"frames": [], "actions": [], "matrix": []}

        all_gestures = sorted(
            {g for entry in self._heatmap_log for g in entry["gestures"]})
        frames = [e["frame"] for e in self._heatmap_log]
        matrix = [[e["gestures"].get(g, 0) for g in all_gestures]
                   for e in self._heatmap_log]
        return {"frames": frames, "actions": all_gestures, "matrix": matrix}

    # -----------------------------------------------------------------
    @staticmethod
    def _empty_result():
        return {
            "engagement_score":   0.0,
            "slouching_rate":     0.0,
            "concentration_rate": 0.0,
            "posture_states": {"upright": 0, "slouching": 0,
                               "head_down": 0, "shifting": 0},
            "per_person":    [],
            "gesture_log":   [],
        }
