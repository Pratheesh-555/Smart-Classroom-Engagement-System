"""Configuration for Smart Classroom Engagement Detection System"""

# --- Model Settings ---
YOLO_FACE_MODEL = "yolov8n-face-keypoints.pt"   # Face-specific YOLOv8 from notebook
YOLO_CONFIDENCE = 0.6                            # Confidence threshold (notebook uses 0.6)

# --- Emotion Buffer (from notebook's voting mechanism) ---
EMOTION_BUFFER_SIZE = 5                          # Buffer size for per-face emotion voting

# --- Engagement Scoring Weights (PDF Page 7: Ei = alpha*Fi + beta*Pi + gamma*Ai) ---
ALPHA = 0.5   # Facial expression weight
BETA = 0.3    # Posture weight
GAMMA = 0.2   # Audio/gaze participation weight

# --- Facial Scoring Weights (PDF Page 4: Fi = w1*attention + w2*positive - w3*confusion - w4*fatigue) ---
W1_ATTENTION = 0.3
W2_POSITIVE = 0.3
W3_CONFUSION = 0.2
W4_FATIGUE = 0.2

# --- Thresholds (PDF Page 7) ---
ENGAGEMENT_DROP_THRESHOLD = 0.40
SLIDING_WINDOW_SECONDS = 10

# --- Emotion-to-Engagement Mapping ---
EMOTION_MAP = {
    "happy":    {"attention": 0.8, "positive": 1.0, "confusion": 0.0, "fatigue": 0.0},
    "neutral":  {"attention": 0.5, "positive": 0.3, "confusion": 0.1, "fatigue": 0.3},
    "sad":      {"attention": 0.3, "positive": 0.0, "confusion": 0.2, "fatigue": 0.8},
    "angry":    {"attention": 0.6, "positive": 0.0, "confusion": 0.3, "fatigue": 0.2},
    "surprise": {"attention": 0.8, "positive": 0.5, "confusion": 0.5, "fatigue": 0.0},
    "fear":     {"attention": 0.4, "positive": 0.0, "confusion": 0.8, "fatigue": 0.1},
    "disgust":  {"attention": 0.3, "positive": 0.0, "confusion": 0.4, "fatigue": 0.5},
}

# --- Video Processing ---
FRAME_SKIP = 8
MAX_FACES = 30

# --- Default Video Path ---
DEFAULT_VIDEO = "dataset/face_video.mp4"
