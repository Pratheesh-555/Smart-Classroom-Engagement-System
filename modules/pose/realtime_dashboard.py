import json
import time

def load_json():
    with open("gesture_log.json", "r") as f:
        return json.load(f)

print("Realtime Gesture Dashboard (press Ctrl+C to stop)")
last_len = 0

try:
    while True:
        data = load_json()
        if len(data) > last_len:
            new_entries = data[last_len:]
            for entry in new_entries:
                print(f"Frame {entry['Frame']} | Person {entry['PersonID']} | "
                      f"Gestures: {', '.join(entry['Gestures'])} | "
                      f"Actions: {', '.join(entry['DynamicActions'])}")
            last_len = len(data)
        time.sleep(1)  # refresh every second
except KeyboardInterrupt:
    print("Dashboard stopped.")
