import torch
from collections import deque

# -------------------------
# Helper functions
# -------------------------
def above(p1, p2): return p1[1] < p2[1]
def below(p1, p2): return p1[1] > p2[1]
def aligned(p1, p2, tol=20): return abs(p1[1] - p2[1]) < tol
def near(p1, p2, tol=40): return torch.norm(p1 - p2) < tol


# -------------------------
# Static gesture classifier
# -------------------------
def classify_body_gesture(keypoints, threshold=0.8):
    """
    Classify static gestures from YOLO keypoints.
    Returns a list of (gesture_name, confidence_score).
    """
    kp = keypoints.xy[0]
    if kp.shape[0] < 17:
        return [("Incomplete Pose", 1.0)]

    # Keypoint mapping (COCO order)
    nose = kp[0]
    left_eye, right_eye = kp[1], kp[2]
    left_shoulder, right_shoulder = kp[5], kp[6]
    left_elbow, right_elbow = kp[7], kp[8]
    left_wrist, right_wrist = kp[9], kp[10]
    left_hip, right_hip = kp[11], kp[12]
    left_knee, right_knee = kp[13], kp[14]
    left_ankle, right_ankle = kp[15], kp[16]

    gestures = []

    # --- Posture ---
    if below(left_knee, left_hip) and below(right_knee, right_hip):
        gestures.append(("Standing", 0.9))
    elif above(left_knee, left_hip) and above(right_knee, right_hip):
        gestures.append(("Sitting", 0.9))
    if abs(left_hip[1] - left_knee[1]) < 30 and abs(right_hip[1] - right_knee[1]) < 30:
        gestures.append(("Squat", 0.85))
    if abs(left_shoulder[1] - left_hip[1]) < 20 and abs(right_shoulder[1] - right_hip[1]) < 20:
        gestures.append(("Lying Down", 0.9))

    # --- Head ---
    if nose[0] < left_shoulder[0] and nose[0] < right_shoulder[0]:
        gestures.append(("Head Turn Left", 0.85))
    elif nose[0] > left_shoulder[0] and nose[0] > right_shoulder[0]:
        gestures.append(("Head Turn Right", 0.85))
    if above(nose, left_shoulder) and above(nose, right_shoulder):
        gestures.append(("Head Up", 0.8))
    elif below(nose, left_shoulder) and below(nose, right_shoulder):
        gestures.append(("Head Down", 0.8))
    if abs(left_eye[0] - right_eye[0]) > 40:
        gestures.append(("Head Rotated", 0.7))

    # --- Arms ---
    if above(left_wrist, nose) and above(right_wrist, nose):
        gestures.append(("Raise Hands", 0.9))
    elif above(left_wrist, nose):
        gestures.append(("Left Hand Raised", 0.85))
    elif above(right_wrist, nose):
        gestures.append(("Right Hand Raised", 0.85))
    if aligned(left_wrist, left_shoulder) and aligned(right_wrist, right_shoulder):
        gestures.append(("T-Pose", 0.8))
    if near(left_wrist, left_hip) and near(right_wrist, right_hip):
        gestures.append(("Hands on Hips", 0.8))
    if left_wrist[0] < left_elbow[0] and aligned(left_wrist, left_shoulder):
        gestures.append(("Point Left", 0.75))
    if right_wrist[0] > right_elbow[0] and aligned(right_wrist, right_shoulder):
        gestures.append(("Point Right", 0.75))
    if near(left_wrist, right_shoulder) and near(right_wrist, left_shoulder):
        gestures.append(("Crossed Arms", 0.7))

    # --- Legs ---
    if left_ankle[1] < left_knee[1] and right_ankle[1] < right_knee[1]:
        gestures.append(("Jumping", 0.9))
    if left_ankle[0] - right_ankle[0] > 50:
        gestures.append(("Step Right", 0.8))
    elif right_ankle[0] - left_ankle[0] > 50:
        gestures.append(("Step Left", 0.8))

    if not gestures:
        gestures.append(("Neutral", 0.6))

    # Filter by confidence threshold
    filtered = [(name, score) for name, score in gestures if score >= threshold]
    if not filtered:
        filtered = [("Neutral", 0.6)]

    return filtered


# -------------------------
# Dynamic action tracker
# -------------------------
class GestureTracker:
    """
    Tracks gestures over time to detect dynamic actions like nodding, waving, walking, jumping.
    """
    def __init__(self, history=30):
        self.history = history
        self.person_histories = {}

    def update(self, person_id, keypoints):
        gestures = classify_body_gesture(keypoints)
        if person_id not in self.person_histories:
            self.person_histories[person_id] = deque(maxlen=self.history)
        self.person_histories[person_id].append(gestures)
        return self.detect_dynamic_actions(person_id)

    def detect_dynamic_actions(self, person_id):
        history = list(self.person_histories[person_id])
        actions = []

        # Flatten history into gesture names
        gesture_names = [g[0] for frame in history for g in frame]

        # Dynamic actions
        if "Head Up" in gesture_names and "Head Down" in gesture_names:
            actions.append(("Nodding", 0.9))
        if "Left Hand Raised" in gesture_names and "Point Left" in gesture_names:
            actions.append(("Waving Left", 0.85))
        if "Right Hand Raised" in gesture_names and "Point Right" in gesture_names:
            actions.append(("Waving Right", 0.85))
        if "Step Left" in gesture_names and "Step Right" in gesture_names:
            actions.append(("Walking", 0.9))
        if gesture_names.count("Jumping") > 5:
            actions.append(("Repeated Jumping", 0.95))

        return actions
