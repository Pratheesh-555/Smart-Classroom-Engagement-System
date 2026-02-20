import json
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

# Load JSON log
with open("gesture_log.json", "r") as f:
    data = json.load(f)

# Build frame vs action matrix
action_matrix = defaultdict(lambda: defaultdict(int))
for entry in data:
    frame = entry["Frame"]
    for action in entry["DynamicActions"]:
        action_matrix[frame][action] += 1

# Convert to heatmap-friendly format
frames = sorted(action_matrix.keys())
actions = sorted({a for frame in frames for a in action_matrix[frame]})
matrix = [[action_matrix[frame].get(a, 0) for a in actions] for frame in frames]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(matrix, xticklabels=actions, yticklabels=frames, cmap="YlGnBu")
plt.title("Dynamic Actions Heatmap (Frame vs Action)")
plt.xlabel("Actions")
plt.ylabel("Frame")
plt.show()

