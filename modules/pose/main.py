import cv2
import time
import json
from ultralytics import YOLO
from gestures import classify_body_gesture, GestureTracker

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Initialize gesture tracker
tracker = GestureTracker(history=30)

# Skeleton connections (COCO format)
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
]

# Open webcam (Windows fix with DirectShow)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Webcam could not be opened. Check device index or permissions.")

# Video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

# FPS tracker
last_t = time.time()
fps = 0.0

# Fullscreen window
cv2.namedWindow("Gesture AI Demo", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gesture AI Demo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# JSON logging setup
json_data = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_count += 1

    # Run YOLOv8 with tracking
    results = model.track(frame, persist=True)

    # Compute FPS
    now = time.time()
    dt = now - last_t
    last_t = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    # Annotated frame from YOLO
    annotated_frame = results[0].plot() if results and len(results) > 0 else frame

    if results and len(results[0].keypoints) > 0:
        kps = results[0].keypoints
        boxes = results[0].boxes

        for i in range(len(kps)):
            keypoints = kps[i]
            box_xyxy = boxes[i].xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box_xyxy

            # Tracking ID
            track_id = i
            if boxes[i].id is not None:
                try:
                    track_id = int(boxes[i].id.item())
                except Exception:
                    track_id = i

            # Static gestures
            gestures = classify_body_gesture(keypoints, threshold=0.8)
            gesture_labels = [f"{name} ({score:.2f})" for name, score in gestures]

            # Dynamic actions
            dynamic_actions = tracker.update(track_id, keypoints)
            dyn_labels = [f"{name} ({score:.2f})" for name, score in dynamic_actions]

            # Combine labels
            all_labels = gesture_labels + dyn_labels
            text = f"ID {track_id}: " + ", ".join(all_labels)

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Put text above box
            cv2.putText(annotated_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw skeleton
            kp = keypoints.xy[0].cpu().numpy()
            for (x, y) in kp:
                cv2.circle(annotated_frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            for (p1, p2) in SKELETON:
                if p1 < len(kp) and p2 < len(kp):
                    cv2.line(annotated_frame, tuple(kp[p1].astype(int)),
                             tuple(kp[p2].astype(int)), (255, 0, 0), 2)

            # --- JSON logging ---
            entry = {
                "Frame": frame_count,
                "PersonID": track_id,
                "Gestures": [name for name, _ in gestures],
                "DynamicActions": [name for name, _ in dynamic_actions]
            }
            json_data.append(entry)

    # HUD: FPS
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

    # Show frame
    cv2.imshow("Gesture AI Demo", annotated_frame)

    # Save frame
    if out.isOpened():
        out.write(annotated_frame)

    # Exit with q or ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
if out.isOpened():
    out.release()
cv2.destroyAllWindows()

# Save JSON at the end
with open("gesture_log.json", "w") as jf:
    json.dump(json_data, jf, indent=4)
