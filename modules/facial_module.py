"""
Facial Expression Analysis Module
==================================
Face detection  : YOLOv8 with yolov8n-face-keypoints.pt  (from notebook)
Face tracking   : ultralytics .track(persist=True)        (from notebook)
Emotion classify: DeepFace on YOLO-cropped faces
Emotion buffer  : Per-face voting buffer with deque       (from notebook)

Scoring formula (PDF Page 4):
    Fi = w1(attention) + w2(positive) - w3(confusion) - w4(fatigue)
"""

import cv2
import numpy as np
import torch
from collections import deque, Counter

from config import (
    YOLO_FACE_MODEL, YOLO_CONFIDENCE, EMOTION_MAP,
    W1_ATTENTION, W2_POSITIVE, W3_CONFUSION, W4_FATIGUE,
    EMOTION_BUFFER_SIZE,
)


class FacialAnalyzer:
    def __init__(self):
        # ── YOLOv8 face detector (from notebook cell 9) ──
        from ultralytics import YOLO
        from deepface import DeepFace

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.face_detector = YOLO(YOLO_FACE_MODEL)
        self.face_detector.to(self.device)

        # ── Preload DeepFace emotion model so first-frame isn't stuck ──
        self.DeepFace = DeepFace
        dummy = np.zeros((48, 48, 3), dtype=np.uint8)
        try:
            DeepFace.analyze(
                img_path=dummy,
                actions=["emotion"],
                detector_backend="skip",
                enforce_detection=False,
                silent=True,
            )
        except Exception:
            pass

        # ── Per-face emotion buffer (from notebook cell 21) ──
        #    {face_id: deque(maxlen=EMOTION_BUFFER_SIZE)}
        self.person_buffers = {}

    # ------------------------------------------------------------------
    # Core analysis — one frame at a time
    # ------------------------------------------------------------------
    def analyze_frame(self, frame):
        """
        Detect faces (YOLOv8) → crop → classify emotion (DeepFace) → buffer vote.

        Returns list[dict] with one entry per detected face.
        """
        DeepFace = self.DeepFace
        faces = []

        # ── YOLOv8 face detection with tracking (notebook cells 19/21) ──
        results = self.face_detector.track(
            frame, persist=True, device=self.device, verbose=False,
        )

        for r in results[0]:
            if r.boxes.conf.item() < YOLO_CONFIDENCE:
                continue

            # ── Bounding-box extraction (exact notebook code) ──
            x_c, y_c, w_b, h_b = r.boxes.xywh.cpu().numpy()[0]
            x_min = int(x_c - w_b / 2)
            x_max = int(x_c + w_b / 2)
            y_min = int(y_c - h_b / 2)
            y_max = int(y_c + h_b / 2)

            # Clamp to frame bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            # ── Crop face (notebook: frame[y_min:y_max, x_min:x_max]) ──
            cropped_face = frame[y_min:y_max, x_min:x_max]
            if cropped_face.size == 0:
                continue

            # ── Tracking ID (notebook cell 21: r.boxes.id) ──
            face_id = None
            if r.boxes.id is not None:
                face_id = int(r.boxes.id.item())

            # ── Keypoints (available for future gaze work) ──
            keypoints = None
            if r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy()[0]

            # ── Emotion classification on the cropped face ──
            # Resize to 48x48 — the FER model's native input size — for speed
            dominant_emotion = "neutral"
            emotion_scores = {}
            try:
                small_face = cv2.resize(cropped_face, (48, 48))
                emo_result = DeepFace.analyze(
                    img_path=small_face,
                    actions=["emotion"],
                    detector_backend="skip",  # face already cropped by YOLO
                    enforce_detection=False,
                    silent=True,
                )
                if not isinstance(emo_result, list):
                    emo_result = [emo_result]
                emotion_scores = emo_result[0].get("emotion", {})
                dominant_emotion = emo_result[0].get("dominant_emotion", "neutral")
            except Exception:
                pass

            # ── Buffer voting (adapted from notebook cell 21) ──
            if face_id is not None:
                if face_id not in self.person_buffers:
                    self.person_buffers[face_id] = deque(maxlen=EMOTION_BUFFER_SIZE)
                self.person_buffers[face_id].append(dominant_emotion)

                if len(self.person_buffers[face_id]) >= 3:
                    dominant_emotion = Counter(
                        self.person_buffers[face_id]
                    ).most_common(1)[0][0]

            # ── Engagement sub-scores (PDF page 4 formula) ──
            mapping = EMOTION_MAP.get(dominant_emotion, EMOTION_MAP["neutral"])
            fi = (
                W1_ATTENTION * mapping["attention"]
                + W2_POSITIVE * mapping["positive"]
                - W3_CONFUSION * mapping["confusion"]
                - W4_FATIGUE * mapping["fatigue"]
            )

            faces.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "face_id": face_id,
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
                "engagement_score": max(0.0, min(1.0, fi)),
                "attention": mapping["attention"],
                "confusion": mapping["confusion"],
                "fatigue": mapping["fatigue"],
                "confidence": r.boxes.conf.item(),
                "keypoints": keypoints,
            })

        return faces

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def get_class_summary(self, faces):
        """Aggregate per-face results into classroom-level metrics."""
        if not faces:
            return {
                "num_students": 0,
                "avg_engagement": 0.0,
                "confusion_rate": 0.0,
                "attention_rate": 0.0,
                "fatigue_rate": 0.0,
                "emotion_distribution": {},
            }

        n = len(faces)
        emotion_dist = {}
        for f in faces:
            e = f["dominant_emotion"]
            emotion_dist[e] = emotion_dist.get(e, 0) + 1
        for k in emotion_dist:
            emotion_dist[k] /= n

        return {
            "num_students": n,
            "avg_engagement": float(np.mean([f["engagement_score"] for f in faces])),
            "confusion_rate": float(np.mean([f["confusion"] for f in faces])),
            "attention_rate": float(np.mean([f["attention"] for f in faces])),
            "fatigue_rate": float(np.mean([f["fatigue"] for f in faces])),
            "emotion_distribution": emotion_dist,
        }

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def annotate_frame(self, frame, faces):
        """Draw bounding boxes, emotion labels, and face IDs on a frame copy."""
        out = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for f in faces:
            x1, y1, x2, y2 = f["bbox"]
            score = f["engagement_score"]
            if score > 0.5:
                color = (0, 200, 0)
            elif score > 0.3:
                color = (0, 180, 255)
            else:
                color = (0, 0, 255)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"{f['dominant_emotion']} {score:.0%}"
            cv2.putText(out, label, (x1, y1 - 8), font, 0.55, color, 2)

            # Show face tracking ID (from notebook)
            if f.get("face_id") is not None:
                id_label = f"id:{f['face_id']}"
                cv2.putText(out, id_label, (x2 + 4, y1 + 14), font, 0.45, (212, 122, 66), 1)

        return out
