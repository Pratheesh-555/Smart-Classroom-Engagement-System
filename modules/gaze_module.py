"""
Eye Gaze Analysis Module
=========================
Adapted from the Pupil Invisible / YOLOv8 tutorial notebook (modules/gaze/).

The tutorial uses pre-recorded gaze coordinates from eye-tracking glasses and
YOLOv8 object segmentation to determine whether gaze falls on detected objects.
Here we replicate this approach for a regular webcam by:
  1. Using MediaPipe Face Mesh (refine_landmarks=True) to obtain iris
     landmark positions (indices 468-477) — replacing the hardware gaze data.
  2. Computing a normalised iris-position ratio within the eye socket to
     derive horizontal / vertical gaze direction — analogous to the
     tutorial's gaze coordinate overlay.
  3. Classifying gaze as "screen" / "away" / "down" by checking whether
     the estimated gaze falls within the forward-facing zone — mirroring
     the tutorial's gaze-to-object-mask intersection logic.

Landmark reference (refine_landmarks=True adds 10 iris landmarks):
    Left  iris center : 468      Right iris center : 473
    Left  eye corners : 33 (outer), 133 (inner)
    Right eye corners : 362 (inner), 263 (outer)
    Left  eye vertical: 159 (top), 145 (bottom)
    Right eye vertical: 386 (top), 374 (bottom)

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

import cv2
import numpy as np


class GazeAnalyzer:
    def __init__(self):
        import mediapipe as mp

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=30,
            refine_landmarks=True,   # enables iris landmarks 468-477
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ── Iris landmark indices ──
        self.LEFT_IRIS_CENTER = 468
        self.RIGHT_IRIS_CENTER = 473

        # ── Eye corner indices (for horizontal ratio) ──
        self.LEFT_EYE_INNER = 133
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_INNER = 362
        self.RIGHT_EYE_OUTER = 263

        # ── Eye vertical indices (for vertical ratio) ──
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374

        # ── Gaze classification thresholds ──
        self.H_AWAY_LOW = 0.30       # looking too far to one side
        self.H_AWAY_HIGH = 0.70
        self.V_DOWN_THRESHOLD = 0.65  # looking down

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_landmark_point(self, landmarks, idx, h, w):
        """Convert a normalised landmark to pixel coordinates."""
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    def _compute_iris_ratios(self, landmarks, h, w):
        """
        Compute horizontal and vertical iris-position ratios for both eyes.

        Horizontal ratio: 0 → iris at outer corner, 1 → iris at inner corner
            ~0.5 means the person is looking straight ahead.
        Vertical ratio : 0 → iris at top lid, 1 → iris at bottom lid
            ~0.5 means looking level; >0.65 means looking down.

        This mirrors the tutorial's technique of mapping gaze coordinates
        onto a frame and checking intersection with target regions.
        """
        pt = lambda idx: self._get_landmark_point(landmarks, idx, h, w)

        # ── Left eye ──
        left_iris = pt(self.LEFT_IRIS_CENTER)
        left_outer = pt(self.LEFT_EYE_OUTER)
        left_inner = pt(self.LEFT_EYE_INNER)
        left_top = pt(self.LEFT_EYE_TOP)
        left_bottom = pt(self.LEFT_EYE_BOTTOM)

        left_w = max(np.linalg.norm(left_inner - left_outer), 1.0)
        left_h = max(np.linalg.norm(left_top - left_bottom), 1.0)
        left_h_ratio = np.linalg.norm(left_iris - left_outer) / left_w
        left_v_ratio = np.linalg.norm(left_iris - left_top) / left_h

        # ── Right eye ──
        right_iris = pt(self.RIGHT_IRIS_CENTER)
        right_outer = pt(self.RIGHT_EYE_OUTER)
        right_inner = pt(self.RIGHT_EYE_INNER)
        right_top = pt(self.RIGHT_EYE_TOP)
        right_bottom = pt(self.RIGHT_EYE_BOTTOM)

        right_w = max(np.linalg.norm(right_inner - right_outer), 1.0)
        right_h = max(np.linalg.norm(right_top - right_bottom), 1.0)
        right_h_ratio = np.linalg.norm(right_iris - right_outer) / right_w
        right_v_ratio = np.linalg.norm(right_iris - right_top) / right_h

        # ── Average both eyes for stability ──
        h_ratio = (left_h_ratio + right_h_ratio) / 2.0
        v_ratio = (left_v_ratio + right_v_ratio) / 2.0

        # Normalised gaze direction vector (deviation from centre)
        gaze_x = round((h_ratio - 0.5) * 2, 3)   # -1 → right, +1 → left
        gaze_y = round((v_ratio - 0.5) * 2, 3)    # -1 → up,    +1 → down

        return h_ratio, v_ratio, (gaze_x, gaze_y)

    def _classify_gaze(self, h_ratio, v_ratio):
        """
        Classify gaze into screen / away / down.

        Analogous to the tutorial checking whether gaze circles overlap
        a detected-object mask — here the "object" is the forward screen zone.
        """
        if v_ratio > self.V_DOWN_THRESHOLD:
            return "down"
        if h_ratio < self.H_AWAY_LOW or h_ratio > self.H_AWAY_HIGH:
            return "away"
        return "screen"

    def _compute_attention_score(self, looking_at, h_ratio, v_ratio):
        """
        Derive a 0-1 attention score from the gaze classification and
        how centred the iris position is (closer to centre = higher score).
        """
        if looking_at == "screen":
            h_dev = abs(h_ratio - 0.5)
            v_dev = abs(v_ratio - 0.5)
            centred = 1.0 - (h_dev + v_dev)
            return round(max(0.6, min(1.0, 0.7 + centred * 0.3)), 2)
        elif looking_at == "down":
            return 0.35
        else:   # away
            return 0.15

    # ------------------------------------------------------------------
    # Public API — matches the interface consumed by FusionEngine
    # ------------------------------------------------------------------
    def analyze_frame(self, frame):
        """
        Analyse a single BGR video frame for eye-gaze patterns.

        Uses MediaPipe Face Mesh with iris landmarks (468-477) to compute
        gaze direction, adapted from the YOLOv8 gaze tutorial's approach
        of determining where a person is looking within the scene.
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        faces = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                h_ratio, v_ratio, gaze_dir = self._compute_iris_ratios(
                    landmarks, h, w
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

        # ── Aggregate across all detected faces ──
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
