import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load JSON log
with open("gesture_log.json", "r") as f:
    data = json.load(f)

# Count gestures per person
gesture_counts = defaultdict(lambda: defaultdict(int))
for entry in data:
    pid = entry["PersonID"]
    for g in entry["Gestures"]:
        gesture_counts[pid][g] += 1

# Plot per person
for pid, counts in gesture_counts.items():
    plt.figure(figsize=(8, 4))
    plt.bar(counts.keys(), counts.values())
    plt.title(f"Gesture Frequency for Person {pid}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

