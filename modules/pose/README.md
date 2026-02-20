 🧠 Gesture AI Demo

**Real-time human pose estimation and gesture control using PyTorch, YOLOv8, and OpenCV.**

## 📖 Overview
This project demonstrates a modular and responsive gesture recognition system powered by **YOLOv8-Pose (Ultralytics)** and **PyTorch**.  
It captures human poses via webcam, classifies static gestures, tracks dynamic actions, and logs results into a structured JSON file.  
The system is designed for **AR/VR, robotics, accessibility applications, and interactive AI demos**.

 🚀 Features

- Built on **PyTorch** with YOLOv8-Pose for real-time human pose estimation.
- **Static gesture classification** (standing, sitting, raise hands, T-pose, etc.).
- **Dynamic action recognition** (nodding, waving, walking, jumping).
- **Webcam-based input** for live demos.
- **JSON-only logging** for clean, structured analytics.
- Lightweight and fast — runs on CPU or GPU.
- Visualization scripts for offline analysis and realtime console dashboards.

## 📂 Project Structure
pose_gesture_control/ 
│── main.py # Runs YOLOv8, logs gestures/actions to JSON 
│── gestures.py # Static + dynamic gesture classifier 
│── gesture_log.json # Output log file (generated at runtime) 
│── visualize_multi_person.py # Bar chart visualization 
│── visualize_actions_heatmap.py # Heatmap visualization 
│── realtime_dashboard.py # Console-based live dashboard 
│── .gitignore # Excludes venv, cache, logs, model weights

 📦 Installation

```bash
# Clone the repository
git clone https://github.com/RAJADURAI3/gesture-ai-demo.git
cd gesture-ai-demo

# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate   # On Windows
# or
source venv/bin/activate       # On Linux/Mac

# Install dependencies
pip install torch torchvision torchaudio
pip install opencv-python ultralytics matplotlib seaborn

```

🎮 Usage

1. Run Gesture Demo

```bash

python main.py

```
Opens webcam in fullscreen.

Detects gestures/actions.

Logs results into gesture_log.json.

2. Visualize Offline

```bash

python visualize_multi_person.py
python visualize_actions_heatmap.py

```

3. Realtime Console Dashboard

```bash

python realtime_dashboard.py

```
Prints new gestures/actions to console as they are logged.

📊 Output Format

Each entry in gesture_log.json looks like:

```Json

{
  "Frame": 42,
  "PersonID": 1,
  "Gestures": ["Standing", "Raise Hands"],
  "DynamicActions": ["Waving Right"]
}


Future work

Integrate a web dashboard (Flask + Plotly) for interactive visualization.


👨‍💻 Author
Rajadurai MSc in Data Science | Computer Vision 
Enthusiast 
Focused on real-time AI demos, AR/VR, and accessibility applications.
