"""
Smart Classroom Engagement Detection System — Streamlit Dashboard
=================================================================
Matches the Teacher Dashboard Interface from the project PDF (Page 8):
  - 4 donut gauges: Overall Engagement, Confused Students, Looking Away, Slouching
  - Alert banner on engagement drop
  - Smart intervention suggestions (Page 9)
  - Engagement timeline chart
  - Annotated video playback

All visualisations update LIVE while the video is being processed.
"""

import sys, os
import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import FRAME_SKIP, DEFAULT_VIDEO, ENGAGEMENT_DROP_THRESHOLD
from modules.facial_module import FacialAnalyzer
from modules.gaze_module import GazeAnalyzer
from modules.posture_module import PostureAnalyzer
from modules.fusion import FusionEngine

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Classroom Engagement System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 0.8rem; }
    .gauge-title {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 600;
        color: #e0e0e0;
        margin-top: -10px;
    }
    .gauge-subtitle {
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        margin-top: -6px;
        margin-bottom: 12px;
    }
    .alert-box {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        font-weight: 600;
        margin: 8px 0;
    }
    .suggestion-box {
        background-color: #1a1a2e;
        border-left: 4px solid #ff6b6b;
        color: #ccc;
        padding: 14px 20px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .metric-card {
        background-color: #16213e;
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-card h3 { margin: 0; color: #ff6b6b; font-size: 1.8rem; }
    .metric-card p  { margin: 0; color: #888; font-size: 0.85rem; }
    div[data-testid="stSidebar"] { background-color: #111827; }
    /* tighter spacing between sections */
    .stSubheader { margin-top: 0.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Unique key generator
# ──────────────────────────────────────────────
_key_counter = [0]
def _key(prefix):
    _key_counter[0] += 1
    return f"{prefix}_{_key_counter[0]}"


# ──────────────────────────────────────────────
# Cached model loading
# ──────────────────────────────────────────────
@st.cache_resource
def load_analyzers():
    return FacialAnalyzer(), GazeAnalyzer(), PostureAnalyzer(), FusionEngine()


# ──────────────────────────────────────────────
# Gauge helper — donut chart matching PDF page 8
# ──────────────────────────────────────────────
def create_donut_gauge(value, color="#ff6b6b"):
    fig = go.Figure(go.Pie(
        values=[value, 100 - value],
        hole=0.75,
        marker=dict(colors=[color, "rgba(60,60,80,0.25)"]),
        textinfo="none",
        hoverinfo="none",
        showlegend=False,
        direction="clockwise",
        sort=False,
        rotation=90,
    ))
    fig.add_annotation(
        text=f"<b>{value}%</b>",
        font=dict(size=32, color="white"),
        showarrow=False, x=0.5, y=0.5,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=5, r=5, t=5, b=5),
    )
    return fig


def create_timeline_chart(results):
    df = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["overall_engagement"] * 100,
        name="Overall Engagement", line=dict(color="#ff6b6b", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,107,107,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["confusion_rate"] * 100,
        name="Confusion", line=dict(color="#e74c3c", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["looking_away_rate"] * 100,
        name="Looking Away", line=dict(color="#3498db", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=ENGAGEMENT_DROP_THRESHOLD * 100,
                  line_dash="dash", line_color="yellow",
                  annotation_text="Drop Threshold (40%)")
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
        height=300,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ──────────────────────────────────────────────
# Smart Intervention Engine (PDF Page 9)
# ──────────────────────────────────────────────
INTERVENTIONS = [
    (lambda m: m["confusion_rate"] > 0.40,
     "High Confusion Cluster",
     "Re-explain concept with worked numerical example"),
    (lambda m: m["fatigue_rate"] > 0.50,
     "Fatigue Signals",
     "2-minute stretch break to refresh students"),
    (lambda m: m["looking_away_rate"] > 0.35,
     "Attention Drift",
     "Ask directed questions to specific students"),
    (lambda m: m["slouching_rate"] > 0.40,
     "Widespread Slouching",
     "Quick energizer activity or change of pace"),
    (lambda m: m["avg_engagement"] < ENGAGEMENT_DROP_THRESHOLD,
     "Engagement Drop",
     "Pause and re-explain with visual example"),
]

def get_suggestion(metrics):
    for check, title, suggestion in INTERVENTIONS:
        if check(metrics):
            return title, suggestion
    return "All Clear", "Class engagement is within normal range"


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
st.sidebar.title("Configuration")

video_source = st.sidebar.radio("Video Source", ["Upload Video", "Default Video", "Webcam"])
video_path = None

if video_source == "Upload Video":
    uploaded = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_path = tfile.name
elif video_source == "Default Video":
    default = Path(DEFAULT_VIDEO)
    if default.exists():
        video_path = str(default)
        st.sidebar.success(f"Loaded: {DEFAULT_VIDEO}")
    else:
        st.sidebar.error(f"{DEFAULT_VIDEO} not found")
else:
    video_path = 0

frame_skip = st.sidebar.slider("Process every N-th frame", 1, 15, FRAME_SKIP)

st.sidebar.markdown("---")
st.sidebar.subheader("Fusion Weights")
alpha = st.sidebar.slider("Facial (alpha)", 0.0, 1.0, 0.5, 0.05)
beta  = st.sidebar.slider("Posture (beta)", 0.0, 1.0, 0.3, 0.05)
gamma = st.sidebar.slider("Gaze (gamma)",   0.0, 1.0, 0.2, 0.05)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("Smart Classroom Engagement Detection System")
st.caption("AI-powered monitoring and intervention system")

# Top-level status bar (visible without scrolling)
ph_top_status = st.empty()

# ──────────────────────────────────────────────
# Live dashboard layout
# ──────────────────────────────────────────────
st.subheader("Live Engagement Overview")

# 4 gauge columns
g1, g2, g3, g4 = st.columns(4)
with g1:
    ph_g1 = st.empty()
    st.markdown('<p class="gauge-title">Overall Engagement</p>', unsafe_allow_html=True)
    ph_s1 = st.empty()
with g2:
    ph_g2 = st.empty()
    st.markdown('<p class="gauge-title">Confused Students</p>', unsafe_allow_html=True)
    ph_s2 = st.empty()
with g3:
    ph_g3 = st.empty()
    st.markdown('<p class="gauge-title">Looking Away</p>', unsafe_allow_html=True)
    ph_s3 = st.empty()
with g4:
    ph_g4 = st.empty()
    st.markdown('<p class="gauge-title">Slouching</p>', unsafe_allow_html=True)
    ph_s4 = st.empty()

# Alert + suggestion
ph_alert = st.empty()
ph_suggestion = st.empty()

# Two-column layout: video feed | stats
col_vid, col_stats = st.columns([3, 1])
with col_vid:
    ph_video = st.empty()
with col_stats:
    ph_students = st.empty()
    ph_dominant = st.empty()
    ph_frames_done = st.empty()

# Timeline + Emotion side by side
col_tl, col_emo = st.columns([2, 1])
with col_tl:
    st.subheader("Engagement Timeline")
    ph_timeline = st.empty()
with col_emo:
    st.subheader("Emotion Distribution")
    ph_emotions = st.empty()

# Post-analysis frame slider
ph_slider_header = st.empty()
ph_slider = st.empty()
ph_slider_img = st.empty()


# ──────────────────────────────────────────────
# Helper: push current metrics to all placeholders
# ──────────────────────────────────────────────
def update_dashboard(results, latest_frame=None, processed=0, total=0):
    if not results:
        return

    avg_eng  = np.mean([r["overall_engagement"] for r in results]) * 100
    avg_conf = np.mean([r["confusion_rate"] for r in results]) * 100
    avg_look = np.mean([r["looking_away_rate"] for r in results]) * 100
    avg_slou = np.mean([r["slouching_rate"] for r in results]) * 100
    last     = results[-1]

    # Gauges
    ph_g1.plotly_chart(create_donut_gauge(round(avg_eng), "#ff6b6b"),
                       use_container_width=True, key=_key("eng"))
    status = "Good" if avg_eng >= 60 else "Warning" if avg_eng >= 40 else "Critical"
    ph_s1.markdown(f'<p class="gauge-subtitle">{status} status</p>', unsafe_allow_html=True)

    ph_g2.plotly_chart(create_donut_gauge(round(avg_conf), "#e74c3c"),
                       use_container_width=True, key=_key("conf"))
    ph_s2.markdown('<p class="gauge-subtitle">Need explanation</p>', unsafe_allow_html=True)

    ph_g3.plotly_chart(create_donut_gauge(round(avg_look), "#3498db"),
                       use_container_width=True, key=_key("look"))
    ph_s3.markdown('<p class="gauge-subtitle">Attention drift</p>', unsafe_allow_html=True)

    ph_g4.plotly_chart(create_donut_gauge(round(avg_slou), "#9b59b6"),
                       use_container_width=True, key=_key("slou"))
    ph_s4.markdown('<p class="gauge-subtitle">Low motivation</p>', unsafe_allow_html=True)

    # Alert
    if avg_eng < (ENGAGEMENT_DROP_THRESHOLD * 100):
        ph_alert.markdown(
            '<div class="alert-box">Alert: Engagement Drop Detected</div>',
            unsafe_allow_html=True,
        )
    else:
        ph_alert.empty()

    # Suggestion
    summary_for_suggestion = {
        "avg_engagement": avg_eng / 100,
        "confusion_rate": avg_conf / 100,
        "looking_away_rate": avg_look / 100,
        "slouching_rate": avg_slou / 100,
        "fatigue_rate": np.mean([r.get("confusion_rate", 0) for r in results]),
    }
    title, suggestion = get_suggestion(summary_for_suggestion)
    ph_suggestion.markdown(
        f'<div class="suggestion-box"><strong>{title}:</strong> {suggestion}</div>',
        unsafe_allow_html=True,
    )

    # Live video frame
    if latest_frame is not None:
        ph_video.image(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB),
                       caption="Live Analysis", use_container_width=True)

    # Side stats
    n_students = last.get("num_students", 0)
    ph_students.metric("Students Detected", n_students)

    # Most common emotion across all results
    all_emos_flat = {}
    for r in results:
        for emo, val in r.get("emotion_distribution", {}).items():
            all_emos_flat[emo] = all_emos_flat.get(emo, 0) + val
    if all_emos_flat:
        dom = max(all_emos_flat, key=all_emos_flat.get)
        ph_dominant.metric("Dominant Emotion", dom.capitalize())

    if total > 0:
        ph_frames_done.metric("Frames Analyzed", f"{processed}/{total}")

    # Timeline
    if len(results) >= 2:
        ph_timeline.plotly_chart(create_timeline_chart(results),
                                use_container_width=True, key=_key("tl"))

    # Emotion distribution
    if all_emos_flat:
        total_emo = sum(all_emos_flat.values()) or 1
        emo_fig = go.Figure(go.Bar(
            x=list(all_emos_flat.keys()),
            y=[round(v / total_emo * 100, 1) for v in all_emos_flat.values()],
            marker_color="#ff6b6b",
        ))
        emo_fig.update_layout(
            yaxis_title="%",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=40, r=20, t=20, b=40),
        )
        ph_emotions.plotly_chart(emo_fig, use_container_width=True, key=_key("emo"))


# ──────────────────────────────────────────────
# Show initial empty gauges OR restore from session
# ──────────────────────────────────────────────
if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    frames = st.session_state.get("frames", [])
    update_dashboard(
        results,
        latest_frame=frames[-1] if frames else None,
        processed=len(results),
        total=len(results),
    )
else:
    ph_g1.plotly_chart(create_donut_gauge(0, "#555"), use_container_width=True, key=_key("i1"))
    ph_s1.markdown('<p class="gauge-subtitle">Waiting...</p>', unsafe_allow_html=True)
    ph_g2.plotly_chart(create_donut_gauge(0, "#555"), use_container_width=True, key=_key("i2"))
    ph_s2.markdown('<p class="gauge-subtitle">Waiting...</p>', unsafe_allow_html=True)
    ph_g3.plotly_chart(create_donut_gauge(0, "#555"), use_container_width=True, key=_key("i3"))
    ph_s3.markdown('<p class="gauge-subtitle">Waiting...</p>', unsafe_allow_html=True)
    ph_g4.plotly_chart(create_donut_gauge(0, "#555"), use_container_width=True, key=_key("i4"))
    ph_s4.markdown('<p class="gauge-subtitle">Waiting...</p>', unsafe_allow_html=True)
    ph_suggestion.info("Select a video source and click **Analyze Video** to start.")


# ──────────────────────────────────────────────
# Process video — LIVE dashboard updates
# ──────────────────────────────────────────────
if st.sidebar.button("Analyze Video", type="primary"):
    if video_path is None:
        st.error("Please select a video source first.")
    else:
        # Toast notification — visible from anywhere on the page
        st.toast("Starting analysis...", icon="^")

        # Top-of-page status (visible without scrolling)
        with ph_top_status.status("Analyzing video...", expanded=True) as top_status:
            top_status.write("Loading models (first run may download ~100 MB)...")
            facial, gaze, posture, fusion = load_analyzers()
            top_status.write("Models loaded. Processing frames...")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Could not open video.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frames_to_process = max(total_frames // frame_skip, 1)

                progress = st.sidebar.progress(0, text="Starting...")
                all_results = []
                annotated_frames = []
                frame_idx = 0
                processed = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip == 0:
                        processed += 1
                        pct = min(processed / frames_to_process, 1.0)
                        progress.progress(pct, text=f"Frame {processed}/{frames_to_process}")
                        top_status.write(f"Frame {processed}/{frames_to_process}")

                        # --- Run all three modules ---
                        face_results = facial.analyze_frame(frame)
                        gaze_results = gaze.analyze_frame(frame)
                        posture_results = posture.analyze_frame(frame)

                        # --- Fuse ---
                        summary = facial.get_class_summary(face_results)
                        fused = fusion.compute(summary, gaze_results, posture_results)
                        fused["timestamp"] = round(frame_idx / fps, 2)
                        fused["frame_idx"] = frame_idx
                        all_results.append(fused)

                        # --- Annotate frame ---
                        annotated = facial.annotate_frame(frame, face_results)
                        annotated_frames.append(annotated)

                        # --- LIVE UPDATE dashboard ---
                        update_dashboard(all_results, latest_frame=annotated,
                                         processed=processed, total=frames_to_process)

                    frame_idx += 1

                cap.release()
                progress.progress(1.0, text="Complete!")
                top_status.update(label="Analysis complete!", state="complete", expanded=False)

                # Store final results
                st.session_state["results"] = all_results
                st.session_state["frames"] = annotated_frames
                st.session_state["fps"] = fps

                st.toast("Analysis complete!", icon="^")

# ──────────────────────────────────────────────
# Post-analysis: frame slider
# ──────────────────────────────────────────────
if "results" in st.session_state and st.session_state.get("frames"):
    frames = st.session_state["frames"]
    ph_slider_header.subheader("Frame-by-Frame Playback")
    idx = ph_slider.slider("Browse frames", 0, len(frames) - 1, len(frames) - 1)
    ph_slider_img.image(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB),
                        use_container_width=True)
