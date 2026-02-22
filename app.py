"""
EduLens — Smart Classroom Engagement System
============================================
Enterprise-grade dashboard: facial expression, gaze tracking,
posture detection with real-time fusion scoring.

Design: Refined dark neutral — surgical, professional, university-ready.
"""

import sys, os, io, csv, time
import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import FRAME_SKIP, DEFAULT_VIDEO, ENGAGEMENT_DROP_THRESHOLD
from modules.facial_module import FacialAnalyzer
from modules.gaze_module import GazeAnalyzer
from modules.posture_module import PostureAnalyzer
from modules.fusion import FusionEngine

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduLens · Engagement System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📡",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System — CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');

:root {
  --n-950: #09090B; --n-900: #0F1014; --n-850: #141519;
  --n-800: #18191E; --n-750: #1E1F26; --n-700: #25262E;
  --n-600: #34353F; --n-500: #52535F; --n-400: #72737F;
  --n-300: #9899A6; --n-200: #C4C5CF; --n-100: #E8E9EF;
  --n-50:  #F5F5F8;

  --accent: #2563EB; --accent-hover: #1D4FD8;
  --accent-muted: rgba(37,99,235,0.10);
  --accent-border: rgba(37,99,235,0.22);

  --green: #16A34A; --green-bg: rgba(22,163,74,0.08);
  --green-border: rgba(22,163,74,0.20);
  --amber: #D97706; --amber-bg: rgba(217,119,6,0.08);
  --amber-border: rgba(217,119,6,0.20);
  --red: #DC2626; --red-bg: rgba(220,38,38,0.07);
  --red-border: rgba(220,38,38,0.18);

  --font: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --mono: 'DM Mono', 'SF Mono', monospace;
  --border: rgba(255,255,255,0.06); --border-hi: rgba(255,255,255,0.10);
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.4);
  --shadow: 0 4px 12px rgba(0,0,0,0.5);
  --shadow-lg: 0 12px 32px rgba(0,0,0,0.6);
  --r-xs: 6px; --r-sm: 8px; --r: 10px; --r-md: 12px; --r-lg: 16px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
section[data-testid="stMain"] {
  font-family: var(--font) !important;
  background: var(--n-950) !important;
  color: var(--n-100) !important;
}

.block-container {
  padding: 0 2rem 3rem !important;
  max-width: 1440px !important;
}

footer, #MainMenu { visibility: hidden !important; display: none !important; }

header[data-testid="stHeader"] {
  background: var(--n-950) !important;
  border-bottom: none !important;
}
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

button[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* Force sidebar always open */
section[data-testid="stSidebar"] {
  width: 310px !important;
  min-width: 310px !important;
  max-width: 310px !important;
  transform: none !important;
  position: relative !important;
  visibility: visible !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
  display: block !important;
  width: 310px !important;
  min-width: 310px !important;
  transform: none !important;
  margin-left: 0 !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--n-700); border-radius: 2px; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-dot {
  0%,100% { opacity:1; } 50% { opacity:0.3; }
}

/* ── GLOW BORDER ANIMATION ── */
@keyframes glowPulseBlue {
  0%,100% { box-shadow: inset 0 0 18px 2px rgba(37,99,235,0.15), 0 0 30px 2px rgba(37,99,235,0.08); }
  50%     { box-shadow: inset 0 0 32px 4px rgba(37,99,235,0.30), 0 0 50px 6px rgba(37,99,235,0.16); }
}
@keyframes glowPulseGreen {
  0%   { box-shadow: inset 0 0 30px 4px rgba(22,163,74,0.30), 0 0 40px 4px rgba(22,163,74,0.15); }
  100% { box-shadow: inset 0 0 0 0 transparent, 0 0 0 0 transparent; }
}
@keyframes glowPulseRed {
  0%,100% { box-shadow: inset 0 0 18px 2px rgba(220,38,38,0.15), 0 0 30px 2px rgba(220,38,38,0.08); }
  50%     { box-shadow: inset 0 0 32px 4px rgba(220,38,38,0.30), 0 0 50px 6px rgba(220,38,38,0.16); }
}

.glow-analyzing [data-testid="stAppViewContainer"] {
  animation: glowPulseBlue 2.5s ease-in-out infinite !important;
  border-radius: 0 !important;
}
.glow-complete [data-testid="stAppViewContainer"] {
  animation: glowPulseGreen 2s ease-out forwards !important;
}
.glow-alert [data-testid="stAppViewContainer"] {
  animation: glowPulseRed 1.8s ease-in-out infinite !important;
}

/* ── SIDEBAR ── */
div[data-testid="stSidebar"] {
  background: var(--n-900) !important;
  border-right: 1px solid var(--border) !important;
}
div[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

.sb-brand { padding: 22px 20px 18px; border-bottom: 1px solid var(--border); }
.sb-name {
  font-family: var(--font); font-size: 0.875rem; font-weight: 700;
  color: var(--n-50); letter-spacing: -0.02em;
  display: flex; align-items: center; gap: 8px;
}
.sb-name-dot { width: 7px; height: 7px; background: var(--accent); border-radius: 50%; flex-shrink: 0; }
.sb-name-sub {
  font-family: var(--mono); font-size: 0.58rem; color: var(--n-500);
  letter-spacing: 1px; text-transform: uppercase; margin-top: 5px; padding-left: 15px;
}
.sb-lbl {
  font-family: var(--mono); font-size: 0.57rem; font-weight: 500; color: var(--n-500);
  text-transform: uppercase; letter-spacing: 1.5px; padding: 18px 20px 8px;
}
.sb-hr { height:1px; background: var(--border); margin: 10px 20px; }
.sb-foot { padding: 14px 20px; border-top: 1px solid var(--border); }
.sb-foot p { font-family: var(--mono); font-size: 0.57rem; color: var(--n-500); line-height: 1.9; }

div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
  font-family: var(--font) !important; font-size: 0.73rem !important;
  font-weight: 500 !important; color: var(--n-300) !important;
}
div[data-testid="stSidebar"] .stFileUploader,
div[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: var(--n-850) !important; border: 1px dashed var(--border-hi) !important;
  border-radius: var(--r) !important;
}

.norm-badge {
  display: inline-block; padding: 3px 8px; border-radius: 4px;
  font-family: var(--mono); font-size: 0.56rem;
  background: var(--accent-muted); color: var(--accent);
  border: 1px solid var(--accent-border); margin-top: 4px;
}

/* ── TOPBAR ── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 0 18px; border-bottom: 1px solid var(--border);
  margin-bottom: 24px; animation: fadeIn 0.35s ease both;
}
.topbar h1 {
  font-family: var(--font); font-size: 1.1rem; font-weight: 700;
  color: var(--n-50); letter-spacing: -0.025em; margin-bottom: 4px; line-height: 1;
}
.topbar p { font-family: var(--font); font-size: 0.74rem; color: var(--n-400); line-height: 1; }

.status-badge {
  display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
  border-radius: 100px; font-family: var(--mono); font-size: 0.59rem;
  letter-spacing: 0.5px; border: 1px solid;
}
.b-ready     { color:var(--green);  background:var(--green-bg);  border-color:var(--green-border); }
.b-analyzing { color:var(--amber);  background:var(--amber-bg);  border-color:var(--amber-border); }
.b-complete  { color:var(--accent); background:var(--accent-muted); border-color:var(--accent-border); }
.b-dot { width:5px; height:5px; border-radius:50%; background:currentColor; animation: pulse-dot 2s ease infinite; }

/* ── SECTION LABEL ── */
.sec {
  font-family: var(--mono); font-size: 0.58rem; font-weight: 500; color: var(--n-500);
  text-transform: uppercase; letter-spacing: 1.8px; margin: 24px 0 12px;
}

/* ── KPI CARD ── */
.kpi {
  background: var(--n-850); border: 1px solid var(--border); border-radius: var(--r-md);
  padding: 16px 12px 12px; text-align: center;
  box-shadow: var(--shadow-sm); transition: border-color .2s, box-shadow .2s, transform .2s;
  animation: fadeIn .4s ease both;
}
.kpi:hover { border-color: var(--border-hi); box-shadow: var(--shadow); transform: translateY(-1px); }
.kpi-label {
  font-family: var(--font); font-size: 0.68rem; font-weight: 600;
  color: var(--n-300); margin-top: 2px; margin-bottom: 2px;
}
.kpi-sub {
  font-family: var(--mono); font-size: 0.54rem; color: var(--n-500);
  text-transform: uppercase; letter-spacing: 0.9px;
}

/* ── ALERT ── */
.alert-bar {
  display: flex; align-items: center; gap: 10px;
  background: var(--red-bg); border: 1px solid var(--red-border); border-radius: var(--r);
  padding: 11px 16px; font-family: var(--font); font-size: 0.77rem;
  font-weight: 500; color: #FCA5A5; animation: fadeIn .25s ease; margin-top: 10px;
}

/* ── INSIGHT ── */
.insight {
  background: var(--n-850); border: 1px solid var(--border);
  border-left: 2px solid var(--accent); border-radius: var(--r);
  padding: 11px 16px; margin-top: 8px; animation: fadeIn .25s ease;
}
.insight-tag {
  font-family: var(--mono); font-size: 0.56rem; color: var(--accent);
  text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;
}
.insight-body { font-family: var(--font); font-size: 0.76rem; color: var(--n-300); line-height: 1.5; }

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: var(--n-850) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; padding: 14px !important;
  box-shadow: var(--shadow-sm) !important; transition: border-color .2s !important;
}
[data-testid="stMetric"]:hover { border-color: var(--border-hi) !important; }
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important; font-weight: 500 !important;
  font-size: 1.25rem !important; color: var(--n-50) !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--font) !important; font-size: 0.6rem !important;
  font-weight: 500 !important; color: var(--n-400) !important;
  text-transform: uppercase !important; letter-spacing: 1px !important;
}

/* ── BUTTONS ── */
.stButton > button {
  font-family: var(--font) !important; font-weight: 600 !important;
  font-size: 0.78rem !important; border-radius: var(--r-sm) !important;
  transition: all .15s ease !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--accent) !important; border: none !important; color: #fff !important;
  padding: 10px 20px !important; box-shadow: 0 1px 4px rgba(37,99,235,.35) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--accent-hover) !important; transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(37,99,235,.4) !important;
}

/* ── SLIDER ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
  font-family: var(--mono) !important; font-size: 0.63rem !important;
  background: var(--n-800) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-xs) !important; color: var(--n-100) !important;
}

/* ── PROGRESS ── */
.stProgress > div > div > div > div { background: var(--accent) !important; border-radius: 100px !important; }
.stProgress > div > div { background: var(--n-700) !important; border-radius: 100px !important; }

/* ── STATUS WIDGET ── */
[data-testid="stStatusWidget"] {
  background: var(--n-850) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r) !important; font-family: var(--font) !important;
  font-size: 0.78rem !important;
}

/* ── TOAST ── */
[data-testid="stToast"] {
  background: var(--n-800) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; font-family: var(--font) !important;
  font-size: 0.78rem !important; box-shadow: var(--shadow-lg) !important;
}

/* ── IMAGE ── */
[data-testid="stImage"] img { border-radius: var(--r-sm) !important; }
.modebar { display: none !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
  background: var(--n-850) !important; border: 1px dashed var(--border-hi) !important;
  border-radius: var(--r) !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] small {
  font-family: var(--font) !important; font-size: 0.72rem !important; color: var(--n-400) !important;
}

/* ── EMPTY VIDEO SLOT ── */
.video-empty {
  background: var(--n-900); border: 1px solid var(--border); border-radius: var(--r-md);
  min-height: 280px; display: flex; align-items: center; justify-content: center;
  font-family: var(--font); font-size: 0.75rem; color: var(--n-500);
}

/* ── DIALOG ── */
[data-testid="stDialog"] > div {
  background: var(--n-900) !important; border: 1px solid var(--border-hi) !important;
  border-radius: var(--r-lg) !important; box-shadow: var(--shadow-lg) !important;
  max-width: 460px !important;
}
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p {
  font-family: var(--font) !important; color: var(--n-300) !important;
  font-size: 0.78rem !important; line-height: 1.6 !important;
}

/* ── REPORT SECTION ── */
.report-card {
  background: var(--n-850); border: 1px solid var(--border); border-radius: var(--r-md);
  padding: 20px; animation: fadeIn .4s ease both;
}
.report-card h3 {
  font-family: var(--font); font-size: 0.82rem; font-weight: 700;
  color: var(--n-50); margin-bottom: 12px;
}
.report-stat {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 0; border-bottom: 1px solid var(--border);
  font-family: var(--font); font-size: 0.74rem;
}
.report-stat:last-child { border-bottom: none; }
.report-stat-label { color: var(--n-400); }
.report-stat-value { font-family: var(--mono); font-weight: 600; color: var(--n-100); }

/* ── DOWNLOAD BTN ── */
.stDownloadButton > button {
  background: var(--n-800) !important; border: 1px solid var(--border-hi) !important;
  color: var(--n-100) !important; font-family: var(--font) !important;
  font-weight: 600 !important; font-size: 0.74rem !important;
  border-radius: var(--r-sm) !important;
}
.stDownloadButton > button:hover {
  background: var(--n-750) !important; border-color: var(--accent-border) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] {
  font-family: var(--font) !important; font-size: 0.74rem !important;
  font-weight: 600 !important; color: var(--n-400) !important;
  padding: 8px 16px !important; border-radius: var(--r-sm) var(--r-sm) 0 0 !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
_kc = [0]
def _key(p):
    _kc[0] += 1
    return f"{p}_{_kc[0]}"


@st.cache_resource
def load_analyzers():
    return FacialAnalyzer(), GazeAnalyzer(), PostureAnalyzer(), FusionEngine()


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans"),
    showlegend=False,
)


def create_gauge(value: int, color: str = "#2563EB") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[100], hole=0.80,
        marker=dict(colors=["rgba(255,255,255,0.04)"]),
        textinfo="none", hoverinfo="none", showlegend=False, sort=False,
    ))
    fig.add_trace(go.Pie(
        values=[max(value, 0), 100 - max(value, 0)],
        hole=0.74,
        marker=dict(colors=[color, "rgba(0,0,0,0)"], line=dict(width=0)),
        textinfo="none", hoverinfo="none", showlegend=False, sort=False,
        direction="clockwise", rotation=90,
        domain=dict(x=[0.07, 0.93], y=[0.07, 0.93]),
    ))
    fig.add_annotation(
        text=f"<b>{value}%</b>",
        font=dict(size=19, color="rgba(255,255,255,0.88)", family="DM Mono"),
        showarrow=False, x=0.5, y=0.5,
    )
    fig.update_layout(**_BASE, height=148, margin=dict(l=0, r=0, t=4, b=4))
    return fig


def create_timeline(results: list) -> go.Figure:
    df = pd.DataFrame(results)
    ax = dict(
        gridcolor="rgba(255,255,255,0.04)",
        tickfont=dict(family="DM Mono", size=9, color="rgba(255,255,255,0.22)"),
        showgrid=True, zeroline=False,
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["overall_engagement"] * 100,
        name="Engagement",
        line=dict(color="#2563EB", width=2, shape="spline"),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["confusion_rate"] * 100,
        name="Confusion",
        line=dict(color="#DC2626", width=1.5, dash="dot", shape="spline"),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["looking_away_rate"] * 100,
        name="Looking Away",
        line=dict(color="#D97706", width=1.5, dash="dot", shape="spline"),
        mode="lines",
    ))
    fig.add_hline(
        y=ENGAGEMENT_DROP_THRESHOLD * 100,
        line_dash="dash", line_color="rgba(255,255,255,0.10)", line_width=1,
        annotation_text=f"Threshold {int(ENGAGEMENT_DROP_THRESHOLD*100)}%",
        annotation_font=dict(color="rgba(255,255,255,0.22)", size=9, family="DM Mono"),
        annotation_position="bottom right",
    )
    layout = {**_BASE, "showlegend": True}
    fig.update_layout(
        **layout,
        height=240,
        margin=dict(l=42, r=16, t=8, b=36),
        xaxis=dict(**ax, title=None),
        yaxis=dict(**ax, title=None, range=[0, 105]),
        legend=dict(
            orientation="h", x=0, y=1.10,
            font=dict(family="DM Sans", size=10, color="rgba(255,255,255,0.38)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
    )
    return fig


def create_emotion_chart(all_emos: dict) -> go.Figure:
    total = sum(all_emos.values()) or 1
    pairs = sorted(
        [(round(v / total * 100, 1), k) for k, v in all_emos.items()],
        reverse=True
    )
    vals, labels = zip(*pairs) if pairs else ([], [])
    colors = ["#2563EB"] + ["rgba(255,255,255,0.07)"] * (len(labels) - 1)
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(vals),
        marker=dict(color=colors, line=dict(width=0), cornerradius=4),
        text=[f"{v}%" for v in vals],
        textposition="outside",
        textfont=dict(size=9, family="DM Mono", color="rgba(255,255,255,0.28)"),
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_BASE, height=185,
        margin=dict(l=8, r=8, t=20, b=30),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            tickfont=dict(family="DM Mono", size=8, color="rgba(255,255,255,0.18)"),
            zeroline=False,
        ),
        xaxis=dict(tickfont=dict(family="DM Sans", size=9, color="rgba(255,255,255,0.32)")),
        bargap=0.42,
    )
    return fig


def create_posture_heatmap(posture_analyzer) -> go.Figure:
    """Create a heatmap of gestures over time from the PostureAnalyzer."""
    hdata = posture_analyzer.get_heatmap_data()
    if not hdata["frames"] or not hdata["actions"]:
        return None
    fig = go.Figure(go.Heatmap(
        z=list(zip(*hdata["matrix"])) if hdata["matrix"] else [],
        x=hdata["frames"],
        y=hdata["actions"],
        colorscale=[[0, "rgba(9,9,11,1)"], [0.5, "rgba(37,99,235,0.4)"], [1, "#2563EB"]],
        showscale=False,
        hovertemplate="Frame %{x}<br>%{y}: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **_BASE, height=220,
        margin=dict(l=100, r=16, t=8, b=36),
        xaxis=dict(
            title=None,
            tickfont=dict(family="DM Mono", size=8, color="rgba(255,255,255,0.22)"),
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(family="DM Sans", size=9, color="rgba(255,255,255,0.32)"),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Intervention Engine
# ─────────────────────────────────────────────────────────────────────────────
INTERVENTIONS = [
    (lambda m: m["confusion_rate"] > 0.40,
     "High Confusion Detected",
     "Re-explain the current concept using a worked example or visual aid."),
    (lambda m: m["fatigue_rate"] > 0.50,
     "Fatigue Signals Present",
     "A short break is recommended to restore student focus."),
    (lambda m: m["looking_away_rate"] > 0.35,
     "Attention Drift",
     "Direct questions to specific students to re-engage the class."),
    (lambda m: m["slouching_rate"] > 0.40,
     "Low Posture Engagement",
     "Consider an energizer or short movement activity."),
    (lambda m: m["avg_engagement"] < ENGAGEMENT_DROP_THRESHOLD,
     "Engagement Below Threshold",
     "Pause the lesson and re-engage using interactive questioning."),
]

def get_suggestion(m: dict):
    for check, title, body in INTERVENTIONS:
        try:
            if check(m):
                return title, body
        except Exception:
            continue
    return "Engagement On Track", "Class engagement is within expected levels — continue."


# ─────────────────────────────────────────────────────────────────────────────
# CSV Report Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_csv_report(results):
    """Generate an in-memory CSV from analysis results."""
    if not results:
        return None
    output = io.StringIO()
    fields = ["timestamp", "frame_idx", "overall_engagement", "facial_score",
              "posture_score", "gaze_score", "confusion_rate",
              "looking_away_rate", "slouching_rate", "num_students"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-name"><span class="sb-name-dot"></span>EduLens</div>
        <div class="sb-name-sub">Engagement Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">Video Source</div>', unsafe_allow_html=True)
    video_source = st.radio(
        "Source", ["Upload Video", "Default Video", "Webcam"],
        label_visibility="collapsed",
    )

    video_path = None
    if video_source == "Upload Video":
        uploaded = st.file_uploader(
            "Drop video", type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            video_path = tfile.name
    elif video_source == "Default Video":
        default = Path(DEFAULT_VIDEO)
        if default.exists():
            video_path = str(default)
            st.success(f"Loaded · {DEFAULT_VIDEO}")
        else:
            st.error(f"Not found: {DEFAULT_VIDEO}")
    else:
        video_path = 0

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">Processing</div>', unsafe_allow_html=True)
    frame_skip = st.slider("Sample every N-th frame", 1, 15, FRAME_SKIP)

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-lbl">Fusion Weights</div>', unsafe_allow_html=True)
    alpha = st.slider("Facial", 0.0, 1.0, 0.5, 0.05)
    beta  = st.slider("Posture", 0.0, 1.0, 0.3, 0.05)
    gamma = st.slider("Gaze", 0.0, 1.0, 0.2, 0.05)

    # Show normalized weights
    w_total = alpha + beta + gamma
    if w_total > 0:
        na, nb, ng = alpha/w_total, beta/w_total, gamma/w_total
    else:
        na, nb, ng = 1/3, 1/3, 1/3
    st.markdown(
        f'<div class="norm-badge">Normalized: {na:.0%} · {nb:.0%} · {ng:.0%}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)
    analyze_clicked = st.button("▶ Run Analysis", type="primary", use_container_width=True)

    st.markdown("""
    <div class="sb-foot">
        <p>Facial · Gaze · Posture<br>Multimodal Fusion Engine</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header + Glow State
# ─────────────────────────────────────────────────────────────────────────────
has_results  = "results" in st.session_state and st.session_state["results"]
is_analyzing = st.session_state.get("analyzing", False)

# Apply glow border class
if is_analyzing:
    st.markdown('<script>document.body.classList.add("glow-analyzing");document.body.classList.remove("glow-complete","glow-alert");</script>', unsafe_allow_html=True)
    badge = '<span class="status-badge b-analyzing"><span class="b-dot"></span>Analyzing</span>'
elif has_results:
    avg_eng = float(np.mean([r["overall_engagement"] for r in st.session_state["results"]])) * 100
    if avg_eng < ENGAGEMENT_DROP_THRESHOLD * 100:
        st.markdown('<script>document.body.classList.add("glow-alert");document.body.classList.remove("glow-analyzing","glow-complete");</script>', unsafe_allow_html=True)
    else:
        st.markdown('<script>document.body.classList.add("glow-complete");document.body.classList.remove("glow-analyzing","glow-alert");</script>', unsafe_allow_html=True)
    badge = '<span class="status-badge b-complete"><span class="b-dot"></span>Complete</span>'
else:
    st.markdown('<script>document.body.classList.remove("glow-analyzing","glow-complete","glow-alert");</script>', unsafe_allow_html=True)
    badge = '<span class="status-badge b-ready"><span class="b-dot"></span>Ready</span>'

st.markdown(f"""
<div class="topbar">
    <div>
        <h1>Classroom Engagement Monitor</h1>
        <p>Multimodal real-time analysis · facial expression · gaze · posture</p>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">{badge}</div>
</div>
""", unsafe_allow_html=True)

# ── Getting Started dialog ──
@st.dialog("Getting Started")
def show_onboarding():
    st.markdown("""
    Select a video source, configure analysis parameters, then run the
    engagement analysis.

    **Steps:**
    1. **Select source** — Choose Upload, Default, or Webcam from the sidebar
    2. **Set weights** — Adjust the facial, posture, and gaze fusion weights
    3. **Run analysis** — Click the Run Analysis button to start processing
    """)
    if st.button("Got it", type="primary", use_container_width=True):
        st.rerun()

if "onboard_shown" not in st.session_state:
    st.session_state["onboard_shown"] = False

if not st.session_state["onboard_shown"]:
    st.session_state["onboard_shown"] = True
    show_onboarding()

info_col_l, info_col_r = st.columns([10, 1])
with info_col_r:
    if st.button("ℹ", key="info_toggle", help="Show getting started guide"):
        show_onboarding()

ph_top_status = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Gauges
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Engagement Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    ph_g1 = st.empty()
    st.markdown('<p class="kpi-label">Engagement</p>', unsafe_allow_html=True)
    ph_s1 = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    ph_g2 = st.empty()
    st.markdown('<p class="kpi-label">Confusion</p>', unsafe_allow_html=True)
    ph_s2 = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    ph_g3 = st.empty()
    st.markdown('<p class="kpi-label">Looking Away</p>', unsafe_allow_html=True)
    ph_s3 = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="kpi">', unsafe_allow_html=True)
    ph_g4 = st.empty()
    st.markdown('<p class="kpi-label">Slouching</p>', unsafe_allow_html=True)
    ph_s4 = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

ph_alert      = st.empty()
ph_suggestion = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Live Feed + Stats
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Live Feed · Student Metrics</div>', unsafe_allow_html=True)
col_feed, col_stats = st.columns([3, 2], gap="medium")

with col_feed:
    ph_video = st.empty()

with col_stats:
    m1, m2, m3 = st.columns(3, gap="small")
    with m1: ph_students    = st.empty()
    with m2: ph_dominant    = st.empty()
    with m3: ph_frames_done = st.empty()

    st.markdown('<div class="sec" style="margin-top:16px">Emotion Distribution</div>', unsafe_allow_html=True)
    ph_emotions = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Timeline
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec">Engagement Timeline</div>', unsafe_allow_html=True)
ph_timeline = st.empty()

ph_slider_header = st.empty()
ph_slider        = st.empty()
ph_slider_img    = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Report Section (placeholder)
# ─────────────────────────────────────────────────────────────────────────────
ph_report = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Update
# ─────────────────────────────────────────────────────────────────────────────
def _gc(v):
    if v >= 65: return "#16A34A"
    if v >= 40: return "#D97706"
    return "#DC2626"

def _gl(v):
    if v >= 65: return "Good"
    if v >= 40: return "Moderate"
    return "Critical"


def update_dashboard(results, latest_frame=None, processed=0, total=0):
    if not results: return

    avg_eng  = float(np.mean([r["overall_engagement"] for r in results])) * 100
    avg_conf = float(np.mean([r["confusion_rate"]      for r in results])) * 100
    avg_look = float(np.mean([r["looking_away_rate"]   for r in results])) * 100
    avg_slou = float(np.mean([r["slouching_rate"]      for r in results])) * 100

    ev = round(avg_eng)
    ph_g1.plotly_chart(create_gauge(ev, _gc(ev)),           use_container_width=True, key=_key("g1"))
    ph_s1.markdown(f'<p class="kpi-sub">{_gl(ev)}</p>',    unsafe_allow_html=True)
    ph_g2.plotly_chart(create_gauge(round(avg_conf), "#DC2626"), use_container_width=True, key=_key("g2"))
    ph_s2.markdown('<p class="kpi-sub">Needs attention</p>', unsafe_allow_html=True)
    ph_g3.plotly_chart(create_gauge(round(avg_look), "#D97706"), use_container_width=True, key=_key("g3"))
    ph_s3.markdown('<p class="kpi-sub">Attention drift</p>',  unsafe_allow_html=True)
    ph_g4.plotly_chart(create_gauge(round(avg_slou), "#7C3AED"), use_container_width=True, key=_key("g4"))
    ph_s4.markdown('<p class="kpi-sub">Posture quality</p>',  unsafe_allow_html=True)

    if avg_eng < ENGAGEMENT_DROP_THRESHOLD * 100:
        ph_alert.markdown(
            '<div class="alert-bar">⚠&nbsp; Engagement below threshold — instructor intervention recommended.</div>',
            unsafe_allow_html=True,
        )
    else:
        ph_alert.empty()

    m = {
        "avg_engagement":    avg_eng  / 100,
        "confusion_rate":    avg_conf / 100,
        "looking_away_rate": avg_look / 100,
        "slouching_rate":    avg_slou / 100,
        "fatigue_rate":      float(np.mean([r.get("confusion_rate", 0) for r in results])),
    }
    title, body = get_suggestion(m)
    ph_suggestion.markdown(
        f'<div class="insight"><div class="insight-tag">{title}</div>'
        f'<div class="insight-body">{body}</div></div>',
        unsafe_allow_html=True,
    )

    if latest_frame is not None:
        try:
            ph_video.image(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB),
                           use_container_width=True)
        except Exception:
            pass

    last = results[-1]
    ph_students.metric("Students", last.get("num_students", 0))

    all_emos: dict = {}
    for r in results:
        for emo, val in r.get("emotion_distribution", {}).items():
            all_emos[emo] = all_emos.get(emo, 0) + val

    if all_emos:
        ph_dominant.metric("Dominant", max(all_emos, key=all_emos.get).capitalize())
    if total > 0:
        ph_frames_done.metric("Frames", f"{processed}/{total}")

    if len(results) >= 2:
        ph_timeline.plotly_chart(create_timeline(results),
                                 use_container_width=True, key=_key("tl"))
    if all_emos:
        ph_emotions.plotly_chart(create_emotion_chart(all_emos),
                                 use_container_width=True, key=_key("em"))


# ─────────────────────────────────────────────────────────────────────────────
# Render Report
# ─────────────────────────────────────────────────────────────────────────────
def render_report(results, posture_obj=None):
    """Render post-analysis report section."""
    if not results:
        return

    avg_eng  = float(np.mean([r["overall_engagement"] for r in results])) * 100
    avg_conf = float(np.mean([r["confusion_rate"]      for r in results])) * 100
    avg_look = float(np.mean([r["looking_away_rate"]   for r in results])) * 100
    avg_slou = float(np.mean([r["slouching_rate"]      for r in results])) * 100
    avg_face = float(np.mean([r.get("facial_score", 0) for r in results])) * 100
    avg_gaze = float(np.mean([r.get("gaze_score", 0)   for r in results])) * 100
    avg_post = float(np.mean([r.get("posture_score", 0) for r in results])) * 100
    total_students = max(r.get("num_students", 0) for r in results)
    duration = results[-1].get("timestamp", 0) if results else 0
    drops = sum(1 for r in results if r.get("drop_detected", False))

    with ph_report.container():
        st.markdown('<div class="sec">Analysis Report</div>', unsafe_allow_html=True)

        tab_summary, tab_details, tab_heatmap = st.tabs(["📊 Summary", "📋 Details", "🗺 Posture Heatmap"])

        with tab_summary:
            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            sc1.metric("Overall Engagement", f"{avg_eng:.1f}%")
            sc2.metric("Total Students", total_students)
            sc3.metric("Duration", f"{duration:.1f}s")
            sc4.metric("Drop Events", drops)

            st.markdown('<div class="sec" style="margin-top:16px">Score Breakdown</div>', unsafe_allow_html=True)
            sb1, sb2, sb3 = st.columns(3, gap="small")
            sb1.metric("Facial Score", f"{avg_face:.1f}%")
            sb2.metric("Posture Score", f"{avg_post:.1f}%")
            sb3.metric("Gaze Score", f"{avg_gaze:.1f}%")

            # Intervention summary
            m = {
                "avg_engagement": avg_eng / 100,
                "confusion_rate": avg_conf / 100,
                "looking_away_rate": avg_look / 100,
                "slouching_rate": avg_slou / 100,
                "fatigue_rate": avg_conf / 100,
            }
            title, body = get_suggestion(m)
            st.markdown(
                f'<div class="insight" style="margin-top:16px">'
                f'<div class="insight-tag">Recommendation</div>'
                f'<div class="insight-body"><strong>{title}</strong> — {body}</div></div>',
                unsafe_allow_html=True,
            )

            # CSV download
            csv_data = build_csv_report(results)
            if csv_data:
                st.download_button(
                    "⬇ Download CSV Report",
                    data=csv_data,
                    file_name=f"edulens_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with tab_details:
            # Per-frame data table
            df = pd.DataFrame([{
                "Time (s)": r.get("timestamp", 0),
                "Engagement": f"{r['overall_engagement']*100:.1f}%",
                "Confusion": f"{r['confusion_rate']*100:.1f}%",
                "Looking Away": f"{r['looking_away_rate']*100:.1f}%",
                "Slouching": f"{r['slouching_rate']*100:.1f}%",
                "Students": r.get("num_students", 0),
            } for r in results])
            st.dataframe(df, use_container_width=True, height=320)

        with tab_heatmap:
            if posture_obj is not None:
                heatmap_fig = create_posture_heatmap(posture_obj)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True, key=_key("hm"))
                else:
                    st.info("No posture data available for heatmap.")
            else:
                st.info("Posture heatmap is available after running an analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# Initial Render
# ─────────────────────────────────────────────────────────────────────────────
if has_results:
    r  = st.session_state["results"]
    fr = st.session_state.get("frames", [])
    update_dashboard(r, latest_frame=fr[-1] if fr else None,
                     processed=len(r), total=len(r))
    render_report(r, posture_obj=st.session_state.get("posture_obj"))
else:
    for ph, k in [(ph_g1,"e1"),(ph_g2,"e2"),(ph_g3,"e3"),(ph_g4,"e4")]:
        ph.plotly_chart(create_gauge(0, "rgba(255,255,255,0.05)"),
                        use_container_width=True, key=_key(k))
    for ph in [ph_s1, ph_s2, ph_s3, ph_s4]:
        ph.markdown('<p class="kpi-sub">—</p>', unsafe_allow_html=True)

    ph_video.markdown(
        '<div class="video-empty">No video loaded — select a source from the sidebar</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run Analysis
# ─────────────────────────────────────────────────────────────────────────────
if analyze_clicked:
    if video_path is None:
        st.error("No video source selected. Choose one from the sidebar.")
    else:
        st.session_state["analyzing"] = True
        # Activate glow border
        st.markdown('<script>document.body.classList.add("glow-analyzing");document.body.classList.remove("glow-complete","glow-alert");</script>', unsafe_allow_html=True)
        st.toast("Starting analysis…")

        cap = None
        try:
            with ph_top_status.status("Analyzing video…", expanded=False) as sts:
                sts.write("Loading models…")
                # Clear cache if first run after restart (picks up fixed modules)
                if "_analyzers_loaded" not in st.session_state:
                    load_analyzers.clear()
                    st.session_state["_analyzers_loaded"] = True
                facial, gaze, posture, fusion = load_analyzers()
                sts.write("Models ready — processing frames…")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Could not open video. Please try a different file.")
                else:
                    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    fps           = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frames_to_proc = max(total_frames // frame_skip, 1)

                    progress    = st.sidebar.progress(0, text="Initialising…")
                    all_results = []
                    ann_frames  = []
                    frame_idx   = 0
                    processed   = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        if frame_idx % frame_skip == 0:
                            processed += 1
                            pct = min(processed / frames_to_proc, 1.0)
                            progress.progress(pct, text=f"Frame {processed} / {frames_to_proc}")
                            sts.write(f"Frame {processed} / {frames_to_proc}")

                            # Each module wrapped individually — one failure
                            # never prevents the others from contributing.
                            try:
                                face_res = facial.analyze_frame(frame)
                            except Exception:
                                face_res = []

                            try:
                                gaze_res = gaze.analyze_frame(frame)
                            except Exception:
                                gaze_res = {"attention_score": 0.5, "looking_away_rate": 0.0, "faces": []}

                            try:
                                posture_res = posture.analyze_frame(frame)
                            except Exception:
                                posture_res = {"engagement_score": 0.5, "slouching_rate": 0.0}

                            summary = facial.get_class_summary(face_res)
                            fused   = fusion.compute(
                                summary, gaze_res, posture_res,
                                alpha=alpha, beta=beta, gamma=gamma,
                            )
                            fused["timestamp"] = round(frame_idx / fps, 2)
                            fused["frame_idx"] = frame_idx
                            all_results.append(fused)

                            annotated = facial.annotate_frame(frame, face_res)
                            ann_frames.append(annotated)

                            update_dashboard(all_results, latest_frame=annotated,
                                             processed=processed, total=frames_to_proc)
                        frame_idx += 1

                    progress.progress(1.0, text="Done")
                    sts.update(label="Analysis complete", state="complete", expanded=False)

                    st.session_state.update({
                        "results":      all_results,
                        "frames":       ann_frames,
                        "fps":          fps,
                        "analyzing":    False,
                        "posture_obj":  posture,
                    })

                    # Switch glow to complete
                    if all_results:
                        avg_e = float(np.mean([r["overall_engagement"] for r in all_results])) * 100
                        if avg_e < ENGAGEMENT_DROP_THRESHOLD * 100:
                            st.markdown('<script>document.body.classList.add("glow-alert");document.body.classList.remove("glow-analyzing","glow-complete");</script>', unsafe_allow_html=True)
                        else:
                            st.markdown('<script>document.body.classList.add("glow-complete");document.body.classList.remove("glow-analyzing","glow-alert");</script>', unsafe_allow_html=True)

                    st.toast("Analysis complete ✓")

                    # Render report
                    render_report(all_results, posture_obj=posture)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.session_state["analyzing"] = False
        finally:
            if cap is not None:
                cap.release()
            st.session_state["analyzing"] = False


# ─────────────────────────────────────────────────────────────────────────────
# Frame Playback
# ─────────────────────────────────────────────────────────────────────────────
if "results" in st.session_state and st.session_state.get("frames"):
    frames = st.session_state["frames"]
    ph_slider_header.markdown(
        '<div class="sec">Frame-by-Frame Playback</div>',
        unsafe_allow_html=True,
    )
    idx = ph_slider.slider(
        "Scrub", 0, len(frames) - 1, len(frames) - 1,
        label_visibility="collapsed",
    )
    pc, _ = ph_slider_img.columns([3, 2])
    try:
        pc.image(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB), use_container_width=True)
    except Exception:
        pc.warning("Could not display this frame.")