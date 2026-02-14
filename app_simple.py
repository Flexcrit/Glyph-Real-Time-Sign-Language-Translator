"""
app_simple.py

ASL Sign Language Detector — Real-time letter prediction
with minimalist, professional interface.
"""

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import time
import os
import json
import pickle

def normalize_hand_landmarks(hand_landmarks) -> np.ndarray:
    """Normalize hand landmarks (same as collect_letters.py)."""
    coords = np.array([
        [lm.x, lm.y, lm.z]
        for lm in hand_landmarks.landmark
    ])
    wrist = coords[0].copy()
    coords = coords - wrist
    palm_size = np.linalg.norm(coords[9])
    if palm_size > 1e-6:
        coords = coords / palm_size
    return coords.flatten().astype(np.float32)


def load_letter_model():
    """Load trained letter model and labels."""
    model_path = "models/letter_model.pkl"
    labels_path = "letter_labels.json"
    
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(labels_path, "r") as f:
        labels = json.load(f)
    
    labels = {int(k): v for k, v in labels.items()}
    return model, labels


st.set_page_config(
    page_title="Glyph",
    layout="wide"
)

st.markdown("""
<style>
    /* ── Import Professional Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #111118;
        --bg-card: #16161f;
        --bg-card-hover: #1e1e2a;
        --border-color: #2a2a3a;
        --border-subtle: #1e1e2a;
        --text-primary: #f0f0f5;
        --text-secondary: #a0a0b0;
        --text-muted: #606070;
        --accent: #6366f1;
        --accent-dim: rgba(255,255,255,0.08);
        --green: #22c55e;
        --green-dim: rgba(34,197,94,0.12);
        --amber: #f59e0b;
        --amber-dim: rgba(245,158,11,0.12);
        --red: #ef4444;
        --red-dim: rgba(239,68,68,0.10);
        --radius: 8px;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Animated Background — Aurora + Grid + Blobs ── */

    @keyframes auroraShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes blobDrift1 {
        0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); }
        20% { transform: translate(80px, -60px) rotate(45deg) scale(1.15); }
        40% { transform: translate(-40px, 80px) rotate(90deg) scale(0.9); }
        60% { transform: translate(60px, 40px) rotate(180deg) scale(1.1); }
        80% { transform: translate(-80px, -30px) rotate(270deg) scale(0.95); }
    }

    @keyframes blobDrift2 {
        0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); }
        25% { transform: translate(-70px, 50px) rotate(-60deg) scale(1.2); }
        50% { transform: translate(50px, -80px) rotate(-120deg) scale(0.85); }
        75% { transform: translate(90px, 30px) rotate(-200deg) scale(1.1); }
    }

    @keyframes blobDrift3 {
        0%, 100% { transform: translate(0, 0) scale(1); }
        33% { transform: translate(-60px, -70px) scale(1.25); }
        66% { transform: translate(70px, 50px) scale(0.8); }
    }

    @keyframes gridScroll {
        0%   { transform: translate(0, 0); }
        100% { transform: translate(30px, 30px); }
    }

    @keyframes shimmer {
        0%   { opacity: 0.04; }
        50%  { opacity: 0.08; }
        100% { opacity: 0.04; }
    }

    /* ── Global Reset ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stMainBlockContainer"] {
        position: relative;
        z-index: 2;
    }

    /* Layer 1: Aurora gradient sweep */
    [data-testid="stAppViewContainer"] {
        position: relative;
        overflow: hidden;
    }

    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: fixed;
        inset: 0;
        background: linear-gradient(
            125deg,
            rgba(99, 102, 241, 0.18) 0%,
            rgba(139, 92, 246, 0.12) 15%,
            transparent 30%,
            rgba(34, 211, 238, 0.14) 45%,
            transparent 55%,
            rgba(168, 85, 247, 0.16) 70%,
            transparent 80%,
            rgba(59, 130, 246, 0.12) 100%
        );
        background-size: 400% 400%;
        animation: auroraShift 15s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    /* Layer 2: Dot grid overlay */
    [data-testid="stAppViewContainer"]::after {
        content: '';
        position: fixed;
        inset: -30px;
        background-image: radial-gradient(circle, rgba(255,255,255,0.06) 1px, transparent 1px);
        background-size: 30px 30px;
        animation: gridScroll 8s linear infinite, shimmer 6s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Layer 3: Bright gradient blobs via stApp pseudo-elements */
    [data-testid="stApp"]::before {
        content: '';
        position: fixed;
        width: 600px;
        height: 600px;
        top: -15%;
        left: -10%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.5) 0%, rgba(99, 102, 241, 0.15) 40%, transparent 70%);
        filter: blur(80px);
        opacity: 0.35;
        pointer-events: none;
        z-index: 0;
        animation: blobDrift1 25s ease-in-out infinite;
    }

    [data-testid="stApp"]::after {
        content: '';
        position: fixed;
        width: 550px;
        height: 550px;
        bottom: -10%;
        right: -8%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(34, 211, 238, 0.45) 0%, rgba(34, 211, 238, 0.1) 40%, transparent 70%);
        filter: blur(80px);
        opacity: 0.3;
        pointer-events: none;
        z-index: 0;
        animation: blobDrift2 22s ease-in-out infinite;
    }

    /* Layer 4: Third blob injected via stMain */
    [data-testid="stMain"]::before {
        content: '';
        position: fixed;
        width: 500px;
        height: 500px;
        top: 35%;
        left: 25%;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.4) 0%, rgba(168, 85, 247, 0.08) 45%, transparent 70%);
        filter: blur(90px);
        opacity: 0.25;
        pointer-events: none;
        z-index: 0;
        animation: blobDrift3 18s ease-in-out infinite;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
    }

    [data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary) !important;
        font-size: 0.85rem !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: var(--border-color) !important;
        margin: 1.2rem 0 !important;
    }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em !important;
    }

    h1 { font-size: 2rem !important; }

    /* ── Horizontal Rules ── */
    hr {
        border-color: var(--border-color) !important;
        opacity: 0.5 !important;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.01em !important;
        border-radius: var(--radius) !important;
        padding: 0.65rem 1.6rem !important;
        transition: var(--transition) !important;
        border: 1px solid var(--border-color) !important;
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
    }

    .stButton > button:hover {
        background-color: var(--bg-card-hover) !important;
        border-color: #3a3a4a !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3) !important;
    }

    /* Primary button override */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #f0f0f5 !important;
        color: #0a0a0f !important;
        border: 1px solid #f0f0f5 !important;
        font-weight: 600 !important;
    }

    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background-color: #ffffff !important;
        border-color: #ffffff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(255,255,255,0.15) !important;
    }

    /* ── Alert / Info / Warning / Success boxes ── */
    [data-testid="stAlert"] {
        font-family: 'Inter', sans-serif !important;
        border-radius: var(--radius) !important;
        font-size: 0.85rem !important;
        border: 1px solid var(--border-color) !important;
    }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] label {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    /* ── Custom Component Styles ── */

    /* Hero Section */
    .hero-section {
        padding: 0 0 0.25rem 0;
        animation: fadeInDown 0.8s ease-out;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-16px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .hero-label {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        border: 1px solid var(--border-color);
        border-radius: 100px;
        padding: 0.35rem 1rem;
        margin-bottom: 1.2rem;
        animation: fadeIn 1s ease-out 0.2s both;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        line-height: 1.15;
        margin: 0 0 1rem 0;
        animation: fadeIn 0.8s ease-out 0.1s both;
    }

    .hero-description {
        font-size: 0.95rem;
        line-height: 1.75;
        color: var(--text-secondary);
        max-width: 720px;
        margin-bottom: 0.8rem;
        animation: fadeIn 0.8s ease-out 0.3s both;
    }

    .hero-divider {
        width: 48px;
        height: 1px;
        background: var(--border-color);
        margin: 2rem 0;
        animation: expandWidth 0.6s ease-out 0.5s both;
    }

    @keyframes expandWidth {
        from { width: 0; }
        to { width: 48px; }
    }

    /* Status Indicator */
    .status-row {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        padding: 0.8rem 0;
        animation: fadeIn 0.8s ease-out 0.6s both;
    }

    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }

    .status-dot.active {
        background-color: var(--green);
        box-shadow: 0 0 8px rgba(34,197,94,0.4);
        animation: pulse 2s infinite;
    }

    .status-dot.inactive {
        background-color: var(--text-muted);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Camera Feed Container */
    .feed-container {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        animation: fadeIn 0.6s ease-out;
    }

    .feed-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    /* Detection Panel */
    .detection-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        animation: fadeIn 0.6s ease-out 0.15s both;
    }

    .det-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    .det-letter {
        font-size: 3.5rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1;
        margin: 0.5rem 0 0.3rem 0;
    }

    .det-letter.high { color: var(--green); }
    .det-letter.medium { color: var(--amber); }

    .det-confidence {
        font-size: 0.85rem;
        font-weight: 500;
        color: var(--text-secondary);
    }

    .det-waiting {
        font-size: 0.9rem;
        color: var(--text-muted);
        padding: 1.5rem 0;
    }

    /* Stats Panel */
    .stats-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 0.75rem;
        animation: fadeIn 0.6s ease-out 0.25s both;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid var(--border-subtle);
    }

    .stat-row:last-child { border-bottom: none; }

    .stat-label {
        font-size: 0.78rem;
        color: var(--text-muted);
        font-weight: 400;
    }

    .stat-value {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: var(--text-muted);
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        animation: fadeIn 1s ease-out;
    }

    .footer strong {
        color: var(--text-secondary);
        font-weight: 600;
    }

    .footer .separator {
        display: inline-block;
        width: 3px;
        height: 3px;
        border-radius: 50%;
        background: var(--text-muted);
        margin: 0 0.6rem;
        vertical-align: middle;
    }

    /* Camera active banner */
    .camera-active-banner {
        background: var(--green-dim);
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: var(--radius);
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        color: var(--green);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.4s ease-out;
    }

    .camera-error-banner {
        background: var(--red-dim);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: var(--radius);
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        color: var(--red);
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* Sidebar heading style */
    .sidebar-heading {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
        padding-top: 0.25rem;
    }

    .sidebar-model-status {
        font-size: 0.82rem;
        padding: 0.65rem 0.9rem;
        border-radius: var(--radius);
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    .sidebar-model-status.loaded {
        background: var(--green-dim);
        border: 1px solid rgba(34,197,94,0.15);
        color: var(--green);
    }

    .sidebar-model-status.missing {
        background: var(--amber-dim);
        border: 1px solid rgba(245,158,11,0.15);
        color: var(--amber);
    }

    .sidebar-letters {
        font-size: 0.8rem;
        color: var(--text-muted);
        line-height: 1.6;
        padding: 0.4rem 0;
    }

    .sidebar-legend {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 0.4rem 0;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    .legend-dot.red { background: #ef4444; }
    .legend-dot.blue { background: #3b82f6; }

    /* ── App Title ── */
    .app-title {
        text-align: left;
        font-size: 5.5rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin: 0 0 0.25rem 0;
        color: #f0f0f5;
    }

    .app-title span {
        display: inline-block;
        animation: letterWave 6s ease-in-out infinite;
    }

    .app-title span:nth-child(1) { animation-delay: 0s; }
    .app-title span:nth-child(2) { animation-delay: 0.2s; }
    .app-title span:nth-child(3) { animation-delay: 0.4s; }
    .app-title span:nth-child(4) { animation-delay: 0.6s; }
    .app-title span:nth-child(5) { animation-delay: 0.8s; }

    @keyframes letterWave {
        0% { 
            opacity: 1; 
            transform: translateY(0) scale(1);
        }
        12.5% { 
            opacity: 0; 
            transform: translateY(-20px) scale(0.8);
        }
        25% { 
            opacity: 0; 
            transform: translateY(20px) scale(0.8);
        }
        37.5%, 100% { 
            opacity: 1; 
            transform: translateY(0) scale(1);
        }
    }

    /* ── Sentence Builder Panel ── */
    .sentence-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 0.75rem;
        animation: fadeIn 0.6s ease-out 0.3s both;
    }

    .sentence-label {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    .sentence-text {
        font-size: 1.3rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: var(--text-primary);
        min-height: 2.8rem;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius);
        word-wrap: break-word;
        line-height: 1.6;
        font-family: 'Inter', monospace;
    }

    .sentence-text .cursor-blink {
        display: inline-block;
        width: 2px;
        height: 1.2em;
        background: var(--green);
        margin-left: 2px;
        vertical-align: text-bottom;
        animation: blink 1s step-end infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }

    .sentence-hint {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.6rem;
    }

    .sentence-controls {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }

    /* ── Compact text-editing buttons (Space/Delete/Clear) ── */
    [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"] .stButton > button {
        font-size: 0.65rem !important;
        padding: 0.32rem 0.6rem !important;
        letter-spacing: 0.05em !important;
        min-height: unset !important;
        height: auto !important;
        line-height: 1.3 !important;
        white-space: nowrap !important;
    }

    [data-testid="stColumn"]:last-child [data-testid="stHorizontalBlock"] {
        margin-top: 1.5rem !important;
        margin-bottom: 0.05rem !important;
    }

    /* ── Floating 3D Alphabet Letters ── */
    .floating-letters {
        position: fixed;
        inset: 0;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
        perspective: 800px;
    }

    .floating-letter {
        position: absolute;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.12);
        text-shadow: 0 0 50px rgba(139, 92, 246, 0.15);
        user-select: none;
        animation-timing-function: ease-in-out;
        animation-iteration-count: infinite;
        animation-fill-mode: both;
    }

    @keyframes letterFloat1 {
        0%   { transform: translateY(110vh) rotateX(0deg) rotateY(0deg) rotateZ(0deg); opacity: 0; }
        10%  { opacity: 0.15; }
        50%  { opacity: 0.22; }
        90%  { opacity: 0.1; }
        100% { transform: translateY(-20vh) rotateX(360deg) rotateY(180deg) rotateZ(45deg); opacity: 0; }
    }

    @keyframes letterFloat2 {
        0%   { transform: translateY(110vh) rotateX(0deg) rotateY(0deg) rotateZ(0deg); opacity: 0; }
        10%  { opacity: 0.18; }
        40%  { opacity: 0.25; }
        85%  { opacity: 0.08; }
        100% { transform: translateY(-15vh) rotateX(-180deg) rotateY(360deg) rotateZ(-30deg); opacity: 0; }
    }

    @keyframes letterFloat3 {
        0%   { transform: translateY(110vh) rotateX(0deg) rotateY(0deg) rotateZ(0deg) scale(0.8); opacity: 0; }
        15%  { opacity: 0.14; }
        50%  { opacity: 0.2; transform: translateY(50vh) rotateX(180deg) rotateY(90deg) rotateZ(20deg) scale(1.1); }
        90%  { opacity: 0.08; }
        100% { transform: translateY(-25vh) rotateX(360deg) rotateY(270deg) rotateZ(-45deg) scale(0.9); opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

import random
random.seed(42)

letters_config = []
for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    left = (i * 3.7 + random.uniform(0, 5)) % 95
    size = random.choice([60, 75, 90, 110, 130, 150])
    delay = round(random.uniform(0, 20), 1)
    duration = round(random.uniform(18, 35), 1)
    anim = random.choice(['letterFloat1', 'letterFloat2', 'letterFloat3'])
    letters_config.append((ch, left, size, delay, duration, anim))

letters_html = '<div class="floating-letters">\n'
for ch, left, size, delay, duration, anim in letters_config:
    letters_html += (
        f'  <span class="floating-letter" style="'
        f'left:{left}%;'
        f'font-size:{size}px;'
        f'animation-name:{anim};'
        f'animation-duration:{duration}s;'
        f'animation-delay:{delay}s;'
        f'">{ch}</span>\n'
    )
letters_html += '</div>'

st.markdown(letters_html, unsafe_allow_html=True)


if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""
if 'last_stable_letter' not in st.session_state:
    st.session_state.last_stable_letter = None
if 'stable_count' not in st.session_state:
    st.session_state.stable_count = 0
if 'letter_committed' not in st.session_state:
    st.session_state.letter_committed = False

letter_model, letter_labels = load_letter_model()
model_loaded = letter_model is not None


st.sidebar.markdown('<div class="sidebar-heading">Settings</div>', unsafe_allow_html=True)

camera_index = st.sidebar.selectbox(
    "Camera Input",
    options=[0, 1, 2],
    index=0,
    help="Select capture device index"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">Model Status</div>', unsafe_allow_html=True)

if model_loaded:
    st.sidebar.markdown(
        f'<div class="sidebar-model-status loaded">Model active — {len(letter_labels)} letters loaded</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f'<div class="sidebar-letters">{", ".join(sorted(letter_labels.values()))}</div>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<div class="sidebar-model-status missing">No trained model detected</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("""
    To enable detection, run:
    1. `python collect_letters.py`
    2. `python train_letters.py`
    3. Restart this application
    """)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-heading">Landmark Legend</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div class="sidebar-legend">
    <div class="legend-item"><span class="legend-dot red"></span> Left Hand</div>
    <div class="legend-item"><span class="legend-dot blue"></span> Right Hand</div>
</div>
""", unsafe_allow_html=True)


st.markdown('<div class="app-title"><span>G</span><span>l</span><span>y</span><span>p</span><span>h</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <div class="hero-title">ASL Translation System</div>
    <p class="hero-description">
        Glyph translates American Sign Language into written text in real time. 
        It identifies hand gestures for letters A through Z to facilitate clear communication between the deaf community and hearing individuals. 
        The system uses computer vision to track hand movements and map them to the English alphabet.
    </p>
    <p class="hero-description">
        It provides a direct bridge for users to spell out words and sentences effortlessly.
        It focuses on accuracy and speed to ensure conversations flow naturally. This tool supports
        accessibility by making digital and physical spaces more inclusive for everyone.
    </p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

status_class = "active" if model_loaded else "inactive"
status_text = "Ready" if model_loaded else "Model required"
st.markdown(f"""
<div class="status-row">
    <div class="status-indicator">
        <span class="status-dot {status_class}"></span>
        System: {status_text}
    </div>
</div>
""", unsafe_allow_html=True)


col_a, col_b, _ = st.columns([1, 1, 3])

with col_a:
    if st.button("Start Camera", type="primary"):
        st.session_state.camera_active = True

with col_b:
    if st.button("Stop Camera"):
        st.session_state.camera_active = False

st.markdown("")


col1, col2 = st.columns([2.2, 1])

with col1:
    st.markdown('<div class="feed-label">Live Feed</div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

with col2:
    detection_placeholder = st.empty()

    if st.session_state.camera_active:
        _, sc1, sc2, sc3, _ = st.columns([0.2, 0.85, 1, 0.6, 0.1])
        with sc1:
            if st.button("Space"):
                st.session_state.sentence += " "
                st.session_state.letter_committed = False
                st.session_state.last_stable_letter = None
                st.session_state.stable_count = 0
        with sc2:
            if st.button("Backspace"):
                st.session_state.sentence = st.session_state.sentence[:-1]
                st.session_state.letter_committed = False
                st.session_state.last_stable_letter = None
                st.session_state.stable_count = 0
        with sc3:
            if st.button("Clear"):
                st.session_state.sentence = ""
                st.session_state.letter_committed = False
                st.session_state.last_stable_letter = None
                st.session_state.stable_count = 0

    sentence_placeholder = st.empty()
    stats_placeholder = st.empty()


if st.session_state.camera_active:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.markdown(
            '<div class="camera-error-banner">Unable to access camera. Check device permissions or try a different input index.</div>',
            unsafe_allow_html=True
        )
        st.session_state.camera_active = False
    else:
        st.markdown(
            '<div class="camera-active-banner"><span class="status-dot active"></span> Camera active — present a hand sign to begin detection</div>',
            unsafe_allow_html=True
        )

        frame_count = 0
        start_time = time.time()

        while st.session_state.camera_active:
            ret, frame = cap.read()

            if not ret:
                st.markdown(
                    '<div class="camera-error-banner">Failed to read from camera.</div>',
                    unsafe_allow_html=True
                )
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            current_prediction = None
            current_confidence = 0.0

            if results.multi_hand_landmarks:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    hand_label = "Right" if results.multi_handedness[idx].classification[0].label == "Left" else "Left"

                    hand_color = (0, 0, 255) if hand_label == "Left" else (255, 0, 0)
                    mp_drawing.draw_landmarks(
                        frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=hand_color, thickness=2)
                    )

                    if model_loaded:
                        landmarks = normalize_hand_landmarks(hand_lms)
                        proba = letter_model.predict_proba([landmarks])[0]
                        pred_idx = np.argmax(proba)
                        confidence = proba[pred_idx]
                        pred_letter = letter_labels.get(pred_idx, "?")

                        if confidence > 0.5:
                            current_prediction = pred_letter
                            current_confidence = confidence

                            wrist = hand_lms.landmark[0]
                            middle_tip = hand_lms.landmark[12]

                            text_x = int(middle_tip.x * w) - 30
                            text_y = int(min(wrist.y, middle_tip.y) * h) - 40

                            text_x = max(10, min(text_x, w - 100))
                            text_y = max(50, min(text_y, h - 10))

                            label_text = f"{pred_letter} {confidence:.0%}"
                            (tw, th_text), _ = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                            )

                            cv2.rectangle(
                                frame,
                                (text_x - 5, text_y - th_text - 10),
                                (text_x + tw + 5, text_y + 5),
                                (0, 0, 0), -1
                            )
                            border_color = (0, 255, 0) if confidence >= 0.8 else (0, 255, 255)
                            cv2.rectangle(
                                frame,
                                (text_x - 5, text_y - th_text - 10),
                                (text_x + tw + 5, text_y + 5),
                                border_color, 2
                            )

                            text_color = (0, 255, 0) if confidence >= 0.8 else (0, 255, 255)
                            cv2.putText(
                                frame, label_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3
                            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            if current_prediction and current_confidence >= 0.6:
                if current_prediction == st.session_state.last_stable_letter:
                    if not st.session_state.letter_committed:
                        st.session_state.stable_count += 1
                        if st.session_state.stable_count >= 15:
                            st.session_state.sentence += current_prediction
                            st.session_state.letter_committed = True
                else:
                    st.session_state.last_stable_letter = current_prediction
                    st.session_state.stable_count = 1
                    st.session_state.letter_committed = False

                conf_class = "high" if current_confidence >= 0.8 else "medium"
                hold_pct = min(st.session_state.stable_count / 15 * 100, 100)
                hold_text = "Confirmed" if st.session_state.letter_committed else f"Hold steady... {hold_pct:.0f}%"
                detection_placeholder.markdown(f"""
                <div class="detection-panel">
                    <div class="det-label">Detected Letter</div>
                    <div class="det-letter {conf_class}">{current_prediction}</div>
                    <div class="det-confidence">Confidence: {current_confidence:.1%}</div>
                    <div class="det-confidence" style="margin-top:0.3rem;font-size:0.75rem;">{hold_text}</div>
                </div>
                """, unsafe_allow_html=True)
            elif current_prediction:
                conf_class = "medium"
                st.session_state.stable_count = 0
                st.session_state.letter_committed = False
                detection_placeholder.markdown(f"""
                <div class="detection-panel">
                    <div class="det-label">Detected Letter</div>
                    <div class="det-letter {conf_class}">{current_prediction}</div>
                    <div class="det-confidence">Confidence: {current_confidence:.1%}</div>
                    <div class="det-confidence" style="margin-top:0.3rem;font-size:0.75rem;">Low confidence — hold steady</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.session_state.stable_count = 0
                st.session_state.letter_committed = False
                detection_placeholder.markdown("""
                <div class="detection-panel">
                    <div class="det-label">Detection</div>
                    <div class="det-waiting">Present a hand sign to begin</div>
                </div>
                """, unsafe_allow_html=True)

            display_sentence = st.session_state.sentence if st.session_state.sentence else ""
            sentence_placeholder.markdown(f"""
            <div class="sentence-panel">
                <div class="sentence-label">Sentence Builder</div>
                <div class="sentence-text">{display_sentence}<span class="cursor-blink"></span></div>
                <div class="sentence-hint">Hold a sign steady to add a letter. Use the buttons above to add spaces or delete.</div>
            </div>
            """, unsafe_allow_html=True)

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            model_status = "Active" if model_loaded else "Not loaded"

            stats_placeholder.markdown(f"""
            <div class="stats-panel">
                <div class="det-label">Performance</div>
                <div class="stat-row">
                    <span class="stat-label">Frame Rate</span>
                    <span class="stat-value">{fps:.1f} fps</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Frames Processed</span>
                    <span class="stat-value">{frame_count}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Model</span>
                    <span class="stat-value">{model_status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.03)

        cap.release()
        hands.close()


st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>ASL Translation System</strong>
    <span class="separator"></span>
    A Project by Flexcrit
</div>
""", unsafe_allow_html=True)
