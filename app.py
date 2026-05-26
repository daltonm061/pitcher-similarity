import warnings
import math
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="MLB Pitcher Similarity Finder",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── GLOBAL RESET ── */
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
.stApp {
    background: #080c14;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, #0d1f3510 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 100%, #0a1a2d08 0%, transparent 60%);
}
@keyframes slideUpFade {
    0% { opacity: 0; transform: translateY(15px); }
    100% { opacity: 1; transform: translateY(0); }
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 2.5rem 2rem 2.5rem !important; max-width: 100% !important; }

/* ── APP BAR ── */
.app-bar {
    background: linear-gradient(90deg, #0a0e18 0%, #0f1a2a 50%, #0a0e18 100%);
    border-bottom: 1px solid #1a2a40;
    padding: 18px 40px; display: flex; align-items: center; gap: 16px;
    position: relative;
}
.app-bar::after {
    content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #d4a84820, #d4a84840, #d4a84820, transparent);
}
.app-bar-title {
    font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 800;
    color: #e8dcc8; letter-spacing: 4px; text-transform: uppercase; margin: 0; line-height: 1;
    text-shadow: 0 0 12px rgba(232, 220, 200, 0.4);
}
.app-bar-sub {
    font-size: 10px; color: #8ab0c8; letter-spacing: 2px; margin-top: 4px;
    font-family: 'JetBrains Mono', monospace; font-weight: 400;
}

/* Status bar */
.status-bar {
    background: #0a0e16; border-bottom: 1px solid #141e2e;
    padding: 6px 40px; font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: #7aaac0; display: flex; gap: 18px; flex-wrap: wrap; letter-spacing: 0.3px;
}

/* ── SECTION LABELS ── */
.sec-label {
    font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 700;
    color: #d4a848; letter-spacing: 3px; text-transform: uppercase;
    border-bottom: 1px solid #141e2e; padding-bottom: 8px; margin-bottom: 14px;
    position: relative;
    text-shadow: 0 0 8px rgba(212, 168, 72, 0.4);
}

/* ── PITCH CARDS ── */
.pitch-card {
    background: rgba(12, 20, 32, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid #162236;
    border-radius: 16px; padding: 18px 20px; margin-bottom: 8px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideUpFade 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}
.pitch-card:hover {
    border-color: #1e3250;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transform: translateY(-2px) scale(1.01);
}
.pitch-card-title {
    font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 12px;
}
.field-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #7aaac0;
    text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 3px; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
/* Prevent vertical letter-wrap on radio button choices in cramped columns */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stRadio"] [data-baseweb="radio"] div {
    white-space: nowrap !important;
}

/* ── NUMBER INPUTS ── */
.stNumberInput { margin-bottom: 4px !important; }
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
.stNumberInput label,
.stNumberInput > label,
[data-testid="stNumberInputContainer"] label,
[data-testid="stNumberInput"] label,
.stSelectbox label,
[data-testid="stSelectbox"] > label,
[data-baseweb="select"] ~ label,
[data-testid="stWidgetLabel"] p { 
    display: none !important; 
    height: 0 !important; 
    overflow: hidden !important; 
    margin: 0 !important; 
    padding: 0 !important;
    visibility: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}
/* ── MULTISELECT — bright tag X buttons ── */
span[data-baseweb="tag"] {
    background: #1a3050 !important;
    border: 1px solid #2a4a70 !important;
}
span[data-baseweb="tag"] span[role="presentation"] {
    color: #d8cbb4 !important;
    font-weight: 600 !important;
}
span[data-baseweb="tag"] [data-testid="stMarkdownContainer"] {
    color: #d8cbb4 !important;
}
/* The X button on each tag */
span[data-baseweb="tag"] button,
span[data-baseweb="tag"] [aria-label="Remove"] {
    color: #e8c060 !important;
    opacity: 1 !important;
    font-size: 14px !important;
}
span[data-baseweb="tag"] button:hover,
span[data-baseweb="tag"] [aria-label="Remove"]:hover {
    color: #ff8080 !important;
    background: transparent !important;
}

/* ── SELECTBOX — fix text contrast ── */
.stSelectbox > div > div,
[data-baseweb="select"] > div {
    background: #0c1220 !important;
    color: #d8cbb4 !important;
    border-color: #1a2a40 !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] [class*="singleValue"],
[data-baseweb="select"] [class*="placeholder"] {
    color: #d8cbb4 !important;
}
[data-baseweb="menu"] {
    background: #0e1828 !important;
}
[data-baseweb="menu"] li {
    color: #d8cbb4 !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [aria-selected="true"] {
    background: #1a2a40 !important;
    color: #e8dcc8 !important;
}
/* Text inputs — pitch metrics and leaderboard filters */
.stTextInput > div > div > input::-webkit-search-cancel-button,
.stTextInput > div > div > input::-webkit-clear-button,
.stTextInput > div > div > input::-ms-clear {
    display: none !important;
}
.stTextInput > div > div > input,
.stTextInput input,
[data-testid="stTextInput"] input {
    background: #0c1220 !important; color: #d8cbb4 !important;
    border: 1px solid #1a2a40 !important; border-radius: 8px !important;
    font-size: 14px !important; font-family: 'JetBrains Mono', monospace !important;
    padding: 9px 12px !important; font-weight: 500 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextInput input:focus {
    border-color: #d4a848 !important;
    box-shadow: 0 0 0 2px #d4a84818, 0 0 16px #d4a84810 !important;
    outline: none !important;
}
[data-testid="stTextInput"] > div,
.stTextInput > div > div {
    background: #0c1220 !important;
    border-color: #1a2a40 !important;
}
/* Native Streamlit clear button — styled to look like pill × */
[data-testid="stTextInputClearButton"] {
    color: #6a90a8 !important;
    background: transparent !important;
    border: none !important;
}
[data-testid="stTextInputClearButton"]:hover {
    color: #e8dcc8 !important;
    background: transparent !important;
}

/* Number inputs — broad selectors to catch all Streamlit versions */
.stNumberInput > div > div > input,
.stNumberInput input,
[data-testid="stNumberInput"] input,
[data-testid="stNumberInputContainer"] input,
[data-baseweb="input"] input,
[data-baseweb="base-input"] input {
    background: #0c1220 !important; color: #d8cbb4 !important;
    border: 1px solid #1a2a40 !important; border-radius: 8px !important;
    font-size: 14px !important; font-family: 'JetBrains Mono', monospace !important;
    padding: 9px 12px !important; font-weight: 500 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stNumberInput > div > div > input:focus,
.stNumberInput input:focus,
[data-testid="stNumberInput"] input:focus,
[data-baseweb="input"] input:focus,
[data-baseweb="base-input"] input:focus {
    border-color: #d4a848 !important;
    box-shadow: 0 0 0 2px #d4a84818, 0 0 16px #d4a84810 !important;
    outline: none !important;
}
/* Input wrapper background (the container div that shows white) */
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-testid="stNumberInputContainer"] > div,
.stNumberInput > div > div {
    background: #0c1220 !important;
    border-color: #1a2a40 !important;
}
[data-testid="InputInstructions"] { display: none !important; }
[data-baseweb="tooltip"] { display: none !important; }
[role="tooltip"] { display: none !important; }
.stNumberInput button {
    display: none !important;
}

/* ── RADIO ── */
.stRadio > label { display: none !important; }
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #b8c8d8 !important; font-size: 14px !important; font-weight: 500 !important;
}

/* ── SLIDER ── */
.stSlider > label { color: #7aaac0 !important; font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 500; }

/* ── SECONDARY BUTTONS (title cards, leaderboard sort, add-pitch dropdown) ── */
/* All non-run, non-back buttons default to dark theme */
.stButton > div > button,
[data-testid="stButton"] > button,
[data-testid="baseButton-secondary"],
button[kind="secondary"] {
    background: linear-gradient(165deg, #0c1420, #0a1220) !important;
    border: 1px solid #1a2a40 !important;
    color: #d8cbb4 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stButton > div > button:hover,
[data-testid="stButton"] > button:hover,
button[kind="secondary"]:hover {
    border-color: #d4a84840 !important;
    color: #e8dcc8 !important;
    background: linear-gradient(165deg, #0e1828, #0c1420) !important;
}
/* Active/selected sort button */
[data-testid="baseButton-primary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a5a, #0e2a42) !important;
    border: 1px solid #d4a84850 !important;
    color: #d4a848 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
button[kind="primary"]:hover {
    border-color: #d4a848 !important;
    background: linear-gradient(135deg, #1e4060, #122a3a) !important;
}

/* ── RUN BUTTON ── */
.run-btn-wrap > div > button {
    background: linear-gradient(135deg, #d4a848 0%, #e8c05a 50%, #d4a848 100%) !important;
    color: #080c14 !important; font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important; font-size: 15px !important;
    letter-spacing: 3px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 10px !important;
    padding: 14px 40px !important; width: 100% !important;
    white-space: nowrap !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 2px 12px #d4a84830 !important;
}
.run-btn-wrap > div > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 28px #d4a84850, 0 0 0 1px #d4a84830 !important;
}
.run-btn-wrap > div > button:active {
    transform: translateY(0) !important;
}

/* ── LEADERBOARD FILTER CLEAR (×) BUTTONS ── */
[data-testid="stButton"] button[kind="secondary"][title^="Clear"] {
    background: transparent !important;
    border: 1px solid #2a3a50 !important;
    color: #4a6880 !important;
    border-radius: 4px !important;
    font-size: 10px !important;
    padding: 2px 4px !important;
    min-height: 0 !important;
    line-height: 1 !important;
}
[data-testid="stButton"] button[kind="secondary"][title^="Clear"]:hover {
    border-color: #e0606060 !important;
    color: #e06060 !important;
    background: #e0606010 !important;
}

/* ── REMOVE PITCH BUTTON (✕) ── */
[data-testid="stButton"] button[kind="secondary"][data-testid*="_remove_"],
div:has(> button[title^="Remove"]) button,
button[title^="Remove"] {
    background: transparent !important;
    border: 1px solid #2a3a50 !important;
    color: #6a90a8 !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    padding: 4px 8px !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
button[title^="Remove"]:hover {
    border-color: #e0606040 !important;
    color: #e06060 !important;
    background: #e0606008 !important;
}

/* ── BACK BUTTON ── */
.back-btn-wrap > div > button {
    background: #0c142010 !important; color: #d4a848 !important;
    border: 1px solid #d4a84830 !important; font-size: 12px !important;
    padding: 6px 18px !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; letter-spacing: 1.5px !important;
    font-weight: 600 !important;
    width: auto !important; white-space: nowrap !important;
    transition: all 0.2s !important;
}
.back-btn-wrap > div > button:hover {
    background: #d4a84810 !important; border-color: #d4a84850 !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: rgba(12, 20, 32, 0.65);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid #162236; border-radius: 16px; padding: 16px 20px;
    transition: all 0.3s;
    animation: slideUpFade 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}
[data-testid="metric-container"]:hover { border-color: #1e3250; transform: translateY(-1px) scale(1.01); }
[data-testid="metric-container"] label {
    color: #3d5a78 !important; font-size: 9px !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #d4a848 !important; font-family: 'Inter', sans-serif !important;
    font-size: 22px !important; font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 11px !important; font-family: 'JetBrains Mono', monospace !important;
}

hr { border-color: #141e2e !important; margin: 24px 0 !important; }

/* ── EXPANDERS — Streamlit 1.55 uses data-testid selectors ── */
/* Target every possible selector variant across Streamlit versions */
.streamlit-expanderHeader,
[data-testid="stExpander"] > div:first-child,
[data-testid="stExpanderToggleIcon"] ~ div,
details > summary,
details summary {
    background: linear-gradient(165deg, #0c1420 0%, #0a1220 100%) !important;
    color: #d8cbb4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.5px; font-size: 11px !important; font-weight: 600 !important;
    border: 1px solid #162236 !important; border-radius: 10px !important;
    transition: border-color 0.2s !important;
    list-style: none !important;
}
.streamlit-expanderHeader:hover,
[data-testid="stExpander"] > div:first-child:hover,
details > summary:hover {
    border-color: #2a4060 !important;
    color: #e8dcc8 !important;
}
/* The actual text inside the expander header summary */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] > details > summary * {
    color: #d8cbb4 !important;
    font-weight: 600 !important;
}
/* Expander content area */
.streamlit-expanderContent,
[data-testid="stExpander"] > details {
    background: #0a0e16 !important;
}
[data-testid="stExpanderDetails"] {
    background: #0a0e16 !important;
    border: 1px solid #141e2e !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}
/* Expander arrow/chevron icon */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary svg path,
details summary svg,
details summary svg path {
    fill: #d8cbb4 !important;
    stroke: #d8cbb4 !important;
}
/* Status bar text */
.status-bar { color: #7aaac0 !important; }
.status-bar span { color: #7aaac0 !important; }

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid #162236 !important; border-radius: 10px !important; }

/* ── TRACKMAN CARD ── */
.tm-card {
    background: rgba(12, 20, 32, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid #162236;
    border-top: 2px solid #d4a84830; border-radius: 16px; padding: 22px 24px;
    margin-bottom: 12px;
    animation: slideUpFade 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

/* ── SIMILARITY BARS ── */
.sim-bar-bg { background: #141e2e; border-radius: 4px; height: 6px; width: 100%; margin-top: 4px; }
.sim-bar-fill { border-radius: 4px; height: 6px; transition: width 0.4s cubic-bezier(0.4,0,0.2,1); }

/* ── METRIC COMPARE ROW ── */
.metric-row {
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid #141e2e; padding: 8px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
.metric-label { color: #7aaac0; width: 80px; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
.metric-mlb   { width: 70px; text-align: right; font-weight: 600; }
.metric-you   { width: 60px; text-align: right; color: #90b8d0; }
.metric-bar-wrap { flex: 1; position: relative; height: 16px; }
.metric-bar-center { position: absolute; left: 50%; top: 50%; width: 1px; height: 12px;
    background: #1e3250; transform: translateY(-50%); }
.metric-bar-fill { position: absolute; top: 50%; height: 6px; border-radius: 3px;
    transform: translateY(-50%); }

/* ── FILE UPLOADER ── */
.stFileUploader > label { color: #7aaac0 !important; font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important; text-transform: uppercase;
    letter-spacing: 1px; font-weight: 500; }

/* Drop zone container */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzoneInstructions"],
.stFileUploader section {
    background: linear-gradient(165deg, #0c1420, #0a1220) !important;
    border: 1px solid #1a2a40 !important;
    border-radius: 10px !important;
    color: #a0c0d4 !important;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzoneInstructions"]:hover {
    border-color: #d4a84840 !important;
    background: linear-gradient(165deg, #0e1828, #0c1420) !important;
}
/* "Browse files" button inside uploader */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: linear-gradient(165deg, #0e1828, #0c1420) !important;
    border: 1px solid #1a2a40 !important;
    color: #d4a848 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    border-color: #d4a84850 !important;
    background: #d4a84810 !important;
}
/* Instruction text */
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #6a90a8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
}
/* Uploaded file pill */
[data-testid="stFileUploaderFile"],
[data-testid="uploadedFile"] {
    background: #0e1828 !important;
    border: 1px solid #1a2a40 !important;
    border-radius: 8px !important;
    color: #a0c0d4 !important;
}
[data-testid="stFileUploaderFile"] span,
[data-testid="uploadedFile"] span {
    color: #a0c0d4 !important;
}
/* Delete X button on uploaded file */
[data-testid="stFileUploaderDeleteBtn"] button {
    color: #e06060 !important;
    background: transparent !important;
    border: none !important;
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    background: linear-gradient(165deg, #0c1420, #0a1220) !important;
    border: 1px solid #162236 !important; border-radius: 10px !important;
    color: #b8c8d8 !important; font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    border-color: #d4a84840 !important; color: #d4a848 !important;
    box-shadow: 0 2px 12px #d4a84815 !important;
}

/* ── SELECTBOX ── */
[data-baseweb="select"] > div {
    background: #0c1220 !important; border-color: #1a2a40 !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] > div:focus-within {
    border-color: #d4a848 !important;
    box-shadow: 0 0 0 2px #d4a84818 !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0e16; }
::-webkit-scrollbar-thumb { background: #1a2a40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2a3a50; }

/* ── SMOOTH ANIMATIONS ── */
* { transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1); }

/* ── MOBILE RESPONSIVE ── */
@media (max-width: 640px) {
    .block-container { padding: 0.8rem 0.8rem 1.5rem 0.8rem !important; }
    .app-bar { padding: 12px 16px !important; gap: 10px; }
    .app-bar-title { font-size: 14px !important; letter-spacing: 2px !important; }
    .pitch-card { padding: 10px 12px !important; }
    .pitch-card-title { font-size: 11px !important; letter-spacing: 1.5px !important; }
    .stNumberInput > div > div > input { font-size: 13px !important; padding: 7px 10px !important; }
    .run-btn-wrap > div > button { font-size: 13px !important; letter-spacing: 2px !important; padding: 12px 20px !important; }
}

/* ── Reusable section header (was inlined ~14 times) ── */
.section-h {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #c49148;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a2a40;
}
.section-h-sub {
    color: #3a5a78;
    font-size: 9px;
    font-weight: 400;
    letter-spacing: 1px;
    text-transform: none;
}

/* ── Reusable small monospace footnote (was inlined ~33 times) ── */
.note-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #5a7a90;
    line-height: 1.7;
}
.note-mono-tight {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: #3a5a78;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PITCH_GROUPS = {
    "4-Seam":        ["FF"],
    "2-Seam/Sinker": ["FT", "SI"],
    "Cutter":        ["FC"],
    "Slider":        ["SL"],
    "Sweeper":       ["ST"],
    "Curveball":     ["CU", "CS", "KC"],
    "Splitter":      ["FS"],
    "Changeup":      ["CH"],
    "Knuckleball":   ["KN"],
}
# Pitch type aliases for cross-group similarity scoring
# If a pitcher lacks the primary group but has an alias, use the alias metrics
PITCH_ALIASES = {
    "Cutter":  ["Slider"],
    "Slider":  ["Cutter", "Sweeper"],   # Slider can match Sweeper profiles too
    "Sweeper": ["Slider", "Cutter"],    # Sweeper can match Slider and Cutter
}

PITCH_COLORS = {
    "4-Seam":        "#e63946",
    "2-Seam/Sinker": "#f4a261",
    "Cutter":        "#2a9d8f",
    "Slider":        "#457b9d",
    "Sweeper":       "#a855f7",
    "Curveball":     "#00b4d8",
    "Splitter":      "#e9c46a",
    "Changeup":      "#06d6a0",
    "Knuckleball":   "#cccccc",
}

# ── Brand & data constants (audit fix #1, #4, #6, #17) ───────────────────────
# Single source of truth for year range, brand gold, default placeholder values,
# and MLB per-pitch medians.
_DATA_YEAR_RANGE = "2017–2025"
_BRAND_GOLD      = "#d4a848"     # primary accent (was mixed with #c49148)

# Default placeholder examples (consistent across input + calculator)
_PLACEHOLDER_REL_HEIGHT = "e.g. 5.80"
_PLACEHOLDER_REL_SIDE   = "e.g. 1.90"
_PLACEHOLDER_EXTENSION  = "e.g. 6.40"

# ── Movement-data source adjustments ────────────────────────────────
# Our model is trained on Statcast pitch-by-pitch data, which has used
# Hawk-Eye optical tracking since the 2020 season. TrackMan (Doppler
# radar) and Rapsodo (radar + camera hybrid) report systematically
# different break numbers for the same pitch.
#
# Physics of the bias (best public understanding):
#   - TrackMan radar measures the ball's trajectory continuously over
#     the full ~55 ft from release to plate.
#   - Hawk-Eye's cameras lose the ball reliably around y=40 ft (last
#     ~15 ft of flight is occluded by batter/catcher), so Hawk-Eye
#     EXTRAPOLATES the final segment using the trajectory it saw.
#   - That extrapolation slightly underestimates ongoing Magnus
#     acceleration, so Hawk-Eye reports less total break than TrackMan
#     for the same pitch.
#   - The bias scales WITH MAGNITUDE — a pitch with 2× the Magnus
#     acceleration has 2× the absolute gap. A multiplicative
#     correction handles this correctly; a flat offset does not.
#
# Conversion model:
#     hawkeye_value  =  entered_value  /  scale
#
# Where `scale` > 1.0 for sources that read higher than Hawk-Eye.
# Per-pitch-type overrides let us tune for cases where the bias
# magnitude differs (fastballs tend to show larger gaps than gyro-y
# breaking balls). Public calibration data is sparse — values below
# reflect reported industry comparisons (Driveline, PitcherList) and
# aim to capture direction + rough magnitude.
_DATA_SOURCES = ["Hawk-Eye / Statcast", "TrackMan", "Rapsodo"]
_DATA_SOURCE_SCALE = {
    "Hawk-Eye / Statcast": {
        "default": {"ivb": 1.00, "hb": 1.00},
    },
    "TrackMan": {
        # TrackMan reads ~10% higher IVB than Hawk-Eye on average;
        # slightly more on Magnus-heavy pitches (4-Seam, Cutter),
        # slightly less on gyro-dominant ones (Slider, Sweeper).
        "default":         {"ivb": 1.10, "hb": 1.05},
        "4-Seam":          {"ivb": 1.12, "hb": 1.05},
        "2-Seam/Sinker":   {"ivb": 1.10, "hb": 1.06},
        "Cutter":          {"ivb": 1.12, "hb": 1.05},
        "Slider":          {"ivb": 1.08, "hb": 1.06},
        "Sweeper":         {"ivb": 1.08, "hb": 1.08},
        "Curveball":       {"ivb": 1.10, "hb": 1.05},
        "Splitter":        {"ivb": 1.10, "hb": 1.05},
        "Changeup":        {"ivb": 1.10, "hb": 1.05},
        "Knuckleball":     {"ivb": 1.05, "hb": 1.05},
    },
    "Rapsodo": {
        # Rapsodo's hybrid sensor is closer to Hawk-Eye than TrackMan
        # but still reads marginally higher.
        "default":         {"ivb": 1.05, "hb": 1.03},
    },
}

def _apply_data_source_adjustment(pitch_group: str, ivb, hb, source: str):
    """Convert entered IVB/HB from `source` to a Hawk-Eye/Statcast
    equivalent via multiplicative scaling.

    Returns (adj_ivb, adj_hb). Both inputs and outputs are arm-side-
    positive HB convention. NaN/None inputs pass through unchanged.

    The multiplicative form means the absolute adjustment scales with
    movement magnitude: a 20" TrackMan IVB drops more than a 5" one.
    """
    if source not in _DATA_SOURCE_SCALE or source == "Hawk-Eye / Statcast":
        return ivb, hb
    table = _DATA_SOURCE_SCALE[source]
    s = table.get(pitch_group) or table.get("default") or {"ivb": 1.0, "hb": 1.0}
    s_ivb = max(float(s.get("ivb", 1.0)), 1e-6)
    s_hb  = max(float(s.get("hb",  1.0)), 1e-6)
    adj_ivb = (None if ivb is None else float(ivb) / s_ivb)
    adj_hb  = (None if hb  is None else float(hb)  / s_hb)
    return adj_ivb, adj_hb


# MLB per-pitch league medians, arm-side positive HB convention (matches
# calculator input). Used by movement plot, improvement suggestions, and
# nearest-shape fallback. Previously redefined inline in the calculator screen.
_MLB_PITCH_MEDIANS = {
    "4-Seam":        {"velo": 93.8, "ivb": 16.0, "hb":   7.7, "spin_rate": 2274},
    "2-Seam/Sinker": {"velo": 93.2, "ivb":  9.3, "hb":  15.2, "spin_rate": 2160},
    "Cutter":        {"velo": 88.7, "ivb":  7.7, "hb":  -2.3, "spin_rate": 2365},
    "Slider":        {"velo": 85.3, "ivb":  1.8, "hb":  -4.4, "spin_rate": 2391},
    "Sweeper":       {"velo": 82.1, "ivb":  1.0, "hb": -14.0, "spin_rate": 2571},
    "Curveball":     {"velo": 79.4, "ivb": -9.9, "hb":  -8.6, "spin_rate": 2503},
    "Splitter":      {"velo": 86.1, "ivb":  3.9, "hb":  10.8, "spin_rate": 1370},
    "Changeup":      {"velo": 85.5, "ivb":  6.6, "hb":  14.3, "spin_rate": 1740},
}

# Clock-tilt → expected active-spin fraction lookup. Crude but useful — a 12:00
# tilt 4-seam has ~95% backspin (high active spin), a 9:00 tilt slider has
# heavy gyro (low active spin). Used for the optional spin-axis input.
# Active fraction is what fraction of total spin contributes to Magnus break;
# the rest is gyro. We translate clock → fraction per pitch type from training
# data percentiles.
_CLOCK_ACTIVE_SPIN = {
    # tilt hour : default fraction (0-1). User-entered tilt overrides imputed.
    # Approximations from Statcast spin-axis vs spin-efficiency joint distribution.
    1:  0.85, 2:  0.78, 3:  0.55, 4:  0.30, 5:  0.20, 6:  0.25,
    7:  0.30, 8:  0.55, 9:  0.65, 10: 0.78, 11: 0.90, 12: 0.95,
}

def _ssw_fraction_from_clock(tilt_hour: float, pitch_group: str) -> float:
    """Estimate SSW (seam-shifted wake) magnitude fraction from clock tilt.

    SSW% ≈ 1 - active_spin_fraction (the part of break not from Magnus).
    Returns a value in [0, 0.5]. For pitches where tilt doesn't predict SSW
    strongly (4-Seams, Cutters), returns small values regardless of tilt.
    """
    if tilt_hour is None:
        return 0.0
    h = round(float(tilt_hour)) % 12
    if h == 0:
        h = 12
    active = _CLOCK_ACTIVE_SPIN.get(h, 0.6)
    # Pitches where SSW is the dominant signal: 2-Seam, Splitter, Curveball, Slider
    # Pitches where Magnus dominates: 4-Seam, Cutter, Changeup
    _SSW_MAX = {
        "2-Seam/Sinker": 0.35, "Splitter": 0.40, "Curveball": 0.30,
        "Slider": 0.25, "Sweeper": 0.20, "Changeup": 0.15,
        "4-Seam": 0.10, "Cutter": 0.10, "Knuckleball": 0.50,
    }
    return (1.0 - active) * _SSW_MAX.get(pitch_group, 0.20)

def _pitch_explainer(group: str, shape_row: dict, sp_val: float,
                       ood_warnings: list, imputed: list) -> str:
    """Generate a 1-sentence human explanation for why a pitch graded the way
    it did. Returns plain text (no HTML). Uses observed features, OOD flags,
    and per-pitch-type medians to find the dominant driver."""
    if shape_row is None:
        return ""
    velo  = shape_row.get("start_speed")
    ivb   = shape_row.get("ivb_in")
    hb    = shape_row.get("hb_arm_in")
    if hb is not None:
        hb = -hb  # convert glove-side to arm-side for narrative
    spin  = shape_row.get("spin_rate")
    meds  = _MLB_PITCH_MEDIANS.get(group, {})
    parts = []
    # Grade-anchored tone
    if sp_val >= 115:
        tone = "Elite"
    elif sp_val >= 105:
        tone = "Above-average"
    elif sp_val >= 95:
        tone = "Average"
    else:
        tone = "Below-average"
    # Dominant driver: largest abs Z-score from per-type median
    drivers = []
    if velo is not None and meds.get("velo") is not None:
        dz = velo - meds["velo"]
        if abs(dz) >= 1.5:
            drivers.append((abs(dz)/2.0, f"velo ({velo:.1f} mph {'above' if dz>0 else 'below'} type average)"))
    if ivb is not None and meds.get("ivb") is not None:
        di = ivb - meds["ivb"]
        if abs(di) >= 2.5:
            drivers.append((abs(di)/3.5, f"IVB ({ivb:+.1f}″, {abs(di):.1f}″ {'above' if di>0 else 'below'} type average)"))
    if hb is not None and meds.get("hb") is not None:
        dh = hb - meds["hb"]
        if abs(dh) >= 3.0:
            drivers.append((abs(dh)/4.0, f"HB ({hb:+.1f}″, {abs(dh):.1f}″ {'arm-side' if dh>0 else 'glove-side'} of average)"))
    drivers.sort(reverse=True)
    if drivers:
        parts.append(f"{tone} — driven by {drivers[0][1]}")
        if len(drivers) > 1 and drivers[1][0] > 0.5:
            parts[-1] += f" and {drivers[1][1]}"
    else:
        parts.append(f"{tone} — shape sits near league average for this pitch type")
    parts[-1] += "."
    # OOD flag note
    if ood_warnings:
        extreme = [w for w in ood_warnings if w.get("severity") == "extreme"]
        if extreme:
            parts.append(f"⚠ Prediction less reliable — {extreme[0]['feat']} is well outside training range.")
    # Imputed input note
    imp_input = [f for f in imputed if f in ("ivb","hb","spin_rate","rel_height","rel_side","extension")]
    if imp_input:
        parts.append(f"Some inputs missing ({', '.join(imp_input)}) — score uses league fillers.")
    return " ".join(parts)

# ── Weights — #4 rel_height cut 25% (50→38), #5 velo base weight (scaled dynamically)
# ── Gaussian similarity model ─────────────────────────────────────────────────
# Similarity decays exponentially with distance: sim = exp(-0.5 * (d/σ)²)
# σ = "ideal tolerance" — at exactly σ away, similarity = 0.607 (still good)
# At 2σ → 0.135, at 3σ → 0.011 (near zero). Fast falloff beyond tolerance.
#
# Tolerances (σ) chosen to match desired match tightness:
#   rel_height : ±0.20 ft   (very tight — slot matters a lot)
#   rel_side   : ±0.30 ft   (tight)
#   velo       : ±1.5  mph  (tight)
#   ivb        : ±2.5  in   (moderate)
#   hb         : ±2.5  in   (moderate)
#   extension  : ±0.50 ft   (loose — least important)
#
# Weights control contribution of each dimension when all are filled in.
# Hand mismatch → hard zero (multiplier, not additive penalty).

# ── Similarity model: Gaussian decay + weighted geometric mean ───────────────
# σ = "ideal tolerance" — at d=σ the dimension scores 0.607 (solid match)
# At d=2σ → 0.135, at d=3σ → 0.011 (exponential falloff)
# Geometric mean makes misses compound: two bad metrics hurt much more than one.
# Tolerances match your specified targets exactly.
SIGMA = dict(
    rel_height = 0.20,   # ±0.20 ft release height
    rel_side   = 0.30,   # ±0.30 ft release side
    velo       = 1.2,    # ±1.2 mph velocity (tighter — velo is critical)
    ivb        = 2.5,    # ±2.5" induced vertical break
    hb         = 2.5,    # ±2.5" horizontal break
    extension  = 0.50,   # ±0.50 ft extension (least important)
)

# Weights control how much each dimension pulls in the geometric mean exponent.
# Higher weight = that dimension dominates more when it's an outlier.
WEIGHTS = dict(
    rel_height = 3.0,   # slot height — critical for arm-slot matching
    rel_side   = 2.5,   # slot side — important
    velo       = 5.0,   # velocity — most important, boosted
    ivb        = 3.0,   # vertical break — critical
    hb         = 3.0,   # horizontal break — critical
    extension  = 0.5,   # extension — least important, intentionally low
)

# Velo boost: harder throwers need tighter velo matching
# Scales σ_velo DOWN (tighter) for 95+ mph pitchers
VELO_BOOST_THRESHOLD = 95.0
VELO_BOOST_MIN_SIGMA = 0.8   # at 102+ mph, σ tightens to 0.8 mph

# TrackMan column name mappings (common variations)
TM_COL_MAP = {
    "pitch_type":     ["autopitchtype","pitchtype","pitch type","auto pitch type","taggedpitchtype"],
    "velo":           ["relspeed","velocity","pitch speed","pitchspeed","releasespeed","speed"],
    "ivb":            ["inducedvertbreak","ivb","induced vert break","inducedverticalbreak","vertbreak","verticalbreak"],
    "hb":             ["horizbreak","horzbreak","hb","horizontal break","horizbreakcatcher","horzbreakcatcher"],
    "extension":      ["extension","releaseextension"],
    "rel_height":     ["relheight","releaseheight","relz"],
    "rel_side":       ["relside","releaseside","relx"],
    "vaa":            ["vertapprangle","vaa","verticalapproachangle","vapproachangle"],
    "haa":            ["horizapprangle","haa","horizontalapproachangle","happroachangle"],
}

TM_PITCH_MAP = {
    # 4-Seam — listed first so "fastball" matches here before sinker/2-seam
    "4-seam fastball": "4-Seam", "4seam fastball": "4-Seam",
    "4-seam": "4-Seam", "four-seam": "4-Seam", "four seam": "4-Seam",
    "fastball": "4-Seam",   # generic "fastball" = 4-seam unless sinker/cutter specified
    # 2-Seam / Sinker
    "2-seam fastball": "2-Seam/Sinker", "two-seam fastball": "2-Seam/Sinker",
    "sinker": "2-Seam/Sinker", "two-seam": "2-Seam/Sinker", "two seam": "2-Seam/Sinker",
    "2-seam": "2-Seam/Sinker",
    # Cutter
    "cutter": "Cutter", "cut fastball": "Cutter",
    # Slider / Sweeper
    "slider": "Slider",
    "sweeper": "Sweeper",
    # Curveball
    "curveball": "Curveball", "curve": "Curveball",
    "knuckle curve": "Curveball", "knucklecurve": "Curveball",
    # Splitter
    "splitter": "Splitter", "split-finger": "Splitter", "splitfinger": "Splitter",
    "split finger": "Splitter",
    # Changeup
    "changeup": "Changeup", "change-up": "Changeup", "change up": "Changeup",
    # Knuckleball
    "knuckleball": "Knuckleball",
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("screen","title"), ("results",None), ("user_snapshot",{}), ("mode","arsenal"), ("lb_sort","velo"), ("lb_asc",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── APP BAR — render immediately so health check passes ──────────────────────
st.markdown("""
<div class="app-bar">
  <span style="font-size:30px;line-height:1;opacity:0.9">⚾</span>
  <div>
    <div class="app-bar-title">Pitcher Similarity Engine</div>
    <div class="app-bar-sub">STATCAST 2017–2025 · ARM-SIDE NORMALIZED · GAUSSIAN SCORING</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DMStuff+ model ─────────────────────────────────────────────────────────────
import joblib as _joblib, os as _os

_MODEL_PATH = _os.path.join(_os.path.dirname(__file__), "model", "lgbm_model_2020_2023.joblib")

# ── DM Stuff+ v5 (current production model) ───────────────────────────────────
# Loads dm_stuff_plus_v5.joblib + optional dm_stuff_plus_v5_norms.json override.
# Used by the new "DM Stuff+ Calculator" screen.
_V5_BUNDLE_PATH = _os.path.join(_os.path.dirname(__file__), "models", "dm_stuff_plus_v5.joblib")
_V5_NORMS_PATH  = _os.path.join(_os.path.dirname(__file__), "models", "dm_stuff_plus_v5_norms.json")
_V6_BUNDLE_PATH = _os.path.join(_os.path.dirname(__file__), "models", "dm_stuff_plus_v6.joblib")
_V7_BUNDLE_PATH = _os.path.join(_os.path.dirname(__file__), "models", "dm_stuff_plus_v7.joblib")

_V8_BUNDLE_PATH  = _os.path.join(_os.path.dirname(__file__), "models", "dm_stuff_plus_v8.joblib")
_V8B_BUNDLE_PATH = _os.path.join(_os.path.dirname(__file__), "models", "dm_stuff_plus_v8b.joblib")
_V8C_BUNDLE_PATH = _os.path.join(_os.path.dirname(__file__), "models", "dm_stuff_plus_v8c.joblib")


def _is_bundle_valid(bundle):
    """Reject bundles with placeholder norms (mean=0, sd=1 — the snapshot
    saved BEFORE production-aligned standardization). Such bundles produce
    wild Stuff+ values like 1186 because sp = 100 + (raw - 0)/1 * 10.
    """
    if not bundle:
        return False, "empty bundle"
    norms = bundle.get("norms", {})
    overall = norms.get("overall", {})
    mean = overall.get("mean")
    sd = overall.get("sd")
    if mean is None or sd is None:
        return False, "missing norms.overall.mean/sd"
    # Placeholder marker: snapshot defaults are exactly (0.0, 1.0)
    if abs(mean) < 1e-9 and abs(sd - 1.0) < 1e-9:
        return False, f"placeholder norms (mean={mean}, sd={sd}) — partial snapshot"
    # Sanity: real norms have sd between 0.05 and 5.0
    if sd < 0.01 or sd > 10.0:
        return False, f"sd={sd:.4f} outside healthy range"
    return True, None


@st.cache_resource
def load_dm_v5():
    """Returns (bundle_dict, norms_dict) or (None, None) if not present.
    Prefers highest available bundle (v8 → v7 → v6 → v5), but VALIDATES
    each one's norms. Skips any bundle with placeholder norms (mean=0, sd=1).
    """
    candidates = [
        ("v8c", _V8C_BUNDLE_PATH),
        ("v8b", _V8B_BUNDLE_PATH),
        ("v8",  _V8_BUNDLE_PATH),
        ("v7",  _V7_BUNDLE_PATH),
        ("v6",  _V6_BUNDLE_PATH),
        ("v5",  _V5_BUNDLE_PATH),
    ]
    for tag, path in candidates:
        if not _os.path.exists(path):
            continue
        try:
            bundle = _joblib.load(path)
        except Exception as _e:
            print(f"  ! {tag} at {path} failed to load: {_e}")
            continue
        ok, why = _is_bundle_valid(bundle)
        if not ok:
            print(f"  ! {tag} at {path} REJECTED — {why}")
            continue
        norms = bundle.get("norms", {})
        # Only apply norms override file if it matches the loaded bundle version.
        # Different versions have wildly different norm scales (e.g., v5 sd~0.004
        # vs v8c sd~0.38), so mixing them produces wrong Stuff+ values.
        version = bundle.get("version", "")
        if "v5" in version and _os.path.exists(_V5_NORMS_PATH):
            try:
                import json as _json
                with open(_V5_NORMS_PATH) as _f:
                    norms = _json.load(_f)
                print(f"  ✓ Applied v5 norms override")
            except Exception:
                pass
        print(f"  ✓ Loaded Stuff+ bundle: {path} (version={bundle.get('version','?')})")
        return bundle, norms
    print("  ! No valid Stuff+ bundle found")
    return None, None

_v5_bundle, _v5_norms = load_dm_v5()
_V5_AVAILABLE = _v5_bundle is not None


# ── Zone-Stuff+ model loader (optional second model) ──────────────────────
# Auto-discovers any zone_stuff_*.joblib in models/, preferring the highest
# version number. Validates required keys before accepting.
import glob as _glob

def load_zone_stuff():
    """Load zone-Stuff+ bundle, scanning models/ for any zone_stuff_*.joblib.
    Returns None if no valid bundle found.
    """
    pattern = "models/zone_stuff_*.joblib"
    candidates = sorted(_glob.glob(pattern), reverse=True)  # newest version first
    if not candidates:
        # Also try without models/ prefix in case of working-dir weirdness
        candidates = sorted(_glob.glob("zone_stuff_*.joblib"), reverse=True)
    for path in candidates:
        try:
            b = _joblib.load(path)
        except Exception as _e:
            print(f"  ! failed to load {path}: {_e}")
            continue
        required = ["model", "features", "cat_features", "group_to_int",
                     "norms", "per_type_scalers", "fallback_scaler"]
        missing = [k for k in required if k not in b]
        if missing:
            print(f"  ! {path} is missing keys: {missing}")
            continue
        print(f"  ✓ Loaded zone-Stuff+ bundle: {path} (version={b.get('version', 'unknown')})")
        return b
    return None

_zone_bundle = load_zone_stuff()
_ZONE_AVAILABLE = _zone_bundle is not None


# ── Usage-Outcome model v3b loader ──────────────────────────────────────
# Trained by train_usage_model_v3b.py. Predicts arsenal-level outcome
# (composite of CSW, Whiff, xwOBA, delta_run_exp) from full arsenal
# context: shape × usage × batter quality, on monthly grain.
# Bundle contains:
#   model_vs_RHB, model_vs_LHB  — per-platoon LightGBM regressors
#   feature_names               — exact order required for predict()
#   tunnel_lambda_vs_{R,L}HB    — fitted pairwise tunneling slopes
#   tunnel_sigmas               — Gaussian kernel bandwidths
#   league_batter               — {xwoba, k, bb} league means for fillers
#   fe_kappa, fe_table          — pitcher fixed-effects (we don't use FE
#                                  at calculator inference time — we
#                                  predict on residualized scale)
_USAGE_V3B_PATH = _os.path.join(_os.path.dirname(__file__),
                                  "models", "usage_outcome_v3b.joblib")

@st.cache_resource
def load_usage_v3b():
    """Return v3b bundle dict or None if file not present."""
    if not _os.path.exists(_USAGE_V3B_PATH):
        return None
    try:
        b = _joblib.load(_USAGE_V3B_PATH)
        print(f"  ✓ Loaded usage-outcome v3b "
              f"(features={len(b.get('feature_names', [])) if b else 0})")
        return b
    except Exception as _e:
        print(f"  ! Failed to load usage v3b: {_e}")
        return None

_usage_v3b = load_usage_v3b()
_USAGE_V3B_AVAILABLE = _usage_v3b is not None


# ── v3b: arsenal-outcome prediction + usage suggestion helpers ──────────
# These are the inference-time entry points. Context features that the
# calculator doesn't capture (count state, batter quality, in-PA
# sequence) are set to neutral / league-typical values so the predicted
# per-pitch f() can be interpreted as "expected per-pitch outcome
# against a league-typical batter pool across a typical count
# distribution." This is the right semantic for cross-pitcher arsenal
# ranking. It is NOT correct for predicting any specific game.

# v3b's per-pitch feature schema (must match train_usage_model_v3b.py
# build_per_pitch_features exactly — keep in sync if either file changes)
_V3B_NEUTRAL_FEATURES = {
    "balls":              1,       # mid count
    "strikes":            1,
    "is_same_hand":       0,       # default to opposite-hand matchup
    "times_faced":        2,       # midpoint of 1/2/3+
    "pitch_num_in_app":   40,
    "pitch_num_log":      np.log1p(40),
    "pitch_in_pa":        3,
    "prior_velo":         np.nan,  # LightGBM handles NaN
    "prior_ivb":          np.nan,
    "prior_hb":           np.nan,
    "prior_pt_int":       np.nan,
    "velo_gap_prior":     np.nan,
    "ivb_gap_prior":      np.nan,
    "hb_gap_prior":       np.nan,
}

# Pitch group → integer (matches GROUP_TO_INT in train_usage_model_v3b.py)
_V3B_GROUP_TO_INT = {g: i + 1 for i, g in enumerate([
    "4-Seam", "2-Seam/Sinker", "Cutter",
    "Slider", "Sweeper", "Curveball",
    "Splitter", "Changeup", "Knuckleball",
])}

def _ordinal_word(n: int) -> str:
    """1 → '#1', 2 → '#2', etc. Short, monospace-friendly."""
    try:
        return f"#{int(n)}"
    except (TypeError, ValueError):
        return str(n)


# Module-level availability flag so the UI can hide the PDF section
# when the dependency is missing (deployed envs may not have reportlab).
try:
    import reportlab as _rl_check  # noqa: F401
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False


def build_calculator_pdf(cache: dict, arsenal_summary: dict = None) -> bytes:
    """Render the calculator's current results into a single-page PDF.

    Inputs (all optional; resilient to missing pieces):
      cache:            the _dm_cache dict (must contain `scores`,
                        `pitches_dict`, `dm_added`, `hand_code`)
      arsenal_summary:  optional {"arsenal_sp": float, "grade": str,
                        "vs_rhb": float, "vs_lhb": float} dict for the
                        headline numbers; if absent, we compute basics.

    Returns the PDF as bytes (for st.download_button). On any
    exception, returns a small error PDF so the download still
    succeeds and the failure is visible.
    """
    import io as _io_pdf
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas as _rl_canvas

    buf = _io_pdf.BytesIO()
    try:
        scores  = cache.get("scores", {})
        pdict   = cache.get("pitches_dict", {})
        added   = cache.get("dm_added", [])
        hand    = cache.get("hand_code", "R")
        src     = cache.get("data_source", "Hawk-Eye / Statcast")

        c = _rl_canvas.Canvas(buf, pagesize=letter)
        W, H = letter
        
        def _draw_bg():
            c.setFillColorRGB(0.05, 0.08, 0.13)
            c.rect(0, 0, W, H, fill=1, stroke=0)
            
        _draw_bg()
        # ── Header ─────────────────────────────────────────────────
        c.setFillColorRGB(0.83, 0.57, 0.28)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(0.6 * inch, H - 0.7 * inch, "DM STUFF+  ·  ARSENAL REPORT")
        c.setFillColorRGB(0.4, 0.55, 0.65)
        c.setFont("Helvetica", 9)
        import time as _t_pdf
        c.drawString(0.6 * inch, H - 0.95 * inch,
                      f"Generated {_t_pdf.strftime('%Y-%m-%d %H:%M')}  ·  "
                      f"{('RHP' if hand == 'R' else 'LHP')}  ·  "
                      f"source: {src}")

        # ── Arsenal grade banner ───────────────────────────────────
        if arsenal_summary:
            asp = arsenal_summary.get("arsenal_sp")
            grade = arsenal_summary.get("grade", "")
            vs_r  = arsenal_summary.get("vs_rhb")
            vs_l  = arsenal_summary.get("vs_lhb")
            c.setFillColorRGB(0.06, 0.10, 0.16)
            c.rect(0.6 * inch, H - 1.85 * inch, W - 1.2 * inch, 0.75 * inch, fill=1, stroke=0)
            c.setFillColorRGB(0.83, 0.57, 0.28)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(0.75 * inch, H - 1.25 * inch, "ARSENAL STUFF+")
            c.setFillColorRGB(0.83, 0.65, 0.28)
            c.setFont("Helvetica-Bold", 28)
            if asp is not None:
                c.drawRightString(W - 0.75 * inch, H - 1.55 * inch,
                                   f"{asp:.1f}   {grade}")
            if vs_r is not None and vs_l is not None:
                c.setFillColorRGB(0.5, 0.65, 0.75)
                c.setFont("Helvetica", 9)
                c.drawString(0.75 * inch, H - 1.65 * inch,
                              f"vs RHB: {vs_r:.1f}    vs LHB: {vs_l:.1f}")

        # ── Per-pitch table ────────────────────────────────────────
        y = H - 2.15 * inch
        c.setFillColorRGB(0.83, 0.57, 0.28)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(0.6 * inch, y, "PER-PITCH SCORES")
        y -= 0.20 * inch
        c.setFont("Helvetica-Bold", 8)
        c.setFillColorRGB(0.5, 0.65, 0.75)
        # Column headers
        cols = [(0.60, "Pitch"), (1.85, "Velo"), (2.40, "iVB"),
                (2.95, "HB"),    (3.50, "Spin"), (4.10, "Use%"),
                (4.75, "Stuff+"),(5.45, "vs RHB"), (6.15, "vs LHB"),
                (6.95, "Comp")]
        for x_in, label in cols:
            c.drawString(x_in * inch, y, label)
        y -= 0.04 * inch
        c.setStrokeColorRGB(0.10, 0.16, 0.25)
        c.line(0.6 * inch, y, W - 0.6 * inch, y)
        y -= 0.18 * inch
        c.setFont("Helvetica", 9)
        for grp in added:
            if grp not in scores: continue
            row = scores[grp]
            sr  = row.get("shape_row", {})
            sp_v = row.get("stuff_plus_overall", row.get("stuff_plus"))
            sp_r = row.get("stuff_plus_vs_rhb")
            sp_l = row.get("stuff_plus_vs_lhb")
            nn = row.get("nearest_pitcher", {}) or {}
            nn_label = (f"{nn['name'].split(',')[0].strip() if ',' in str(nn.get('name', '')) else (str(nn.get('name','')).split()[-1] if nn.get('name') else '')}"
                        f" '{int(nn['year']) % 100:02d}" if nn.get("year") else "")
            _pdg = pdict.get(grp, {})
            _ivb_disp = _pdg.get("ivb_entered") if _pdg.get("ivb_entered") is not None else sr.get("ivb_in")
            _hb_disp = _pdg.get("hb_entered") if _pdg.get("hb_entered") is not None else sr.get("hb_arm_in")
            cells = [
                ("", grp),
                (1.85 * inch, f"{sr.get('start_speed', '—'):.1f}" if sr.get('start_speed') is not None else "—"),
                (2.40 * inch, f"{_ivb_disp:+.1f}" if _ivb_disp is not None else "—"),
                (2.95 * inch, f"{-_hb_disp:+.1f}" if _hb_disp is not None else "—"),
                (3.50 * inch, f"{int(sr.get('spin_rate', 0))}"      if sr.get('spin_rate')   else "—"),
                (4.10 * inch, f"{int(pdict.get(grp, {}).get('usage_pct', 0))}%"
                              if pdict.get(grp, {}).get('usage_pct') else "—"),
                (4.75 * inch, f"{sp_v:.1f}" if sp_v is not None else "—"),
                (5.45 * inch, f"{sp_r:.1f}" if sp_r is not None else "—"),
                (6.15 * inch, f"{sp_l:.1f}" if sp_l is not None else "—"),
                (6.95 * inch, nn_label or "—"),
            ]
            c.setFillColorRGB(0.78, 0.85, 0.90)
            c.drawString(0.60 * inch, y, str(cells[0][1]))
            for x_pos, val in cells[1:]:
                c.drawString(x_pos, y, str(val))
            y -= 0.20 * inch
            if y < 1.2 * inch:
                break

        # ── Movement Plot ──────────────────────────────────────────
        plot_bytes = cache.get("plot_img_bytes")
        if plot_bytes:
            from reportlab.lib.utils import ImageReader
            img = ImageReader(_io_pdf.BytesIO(plot_bytes))
            plot_h = 3.5 * inch
            plot_w = 4.0 * inch
            y -= plot_h + 0.2 * inch
            if y < 0.8 * inch:
                c.showPage()
                _draw_bg()
                y = H - plot_h - 1 * inch
            c.drawImage(img, W / 2 - plot_w / 2, y, width=plot_w, height=plot_h, preserveAspectRatio=True, mask='auto')


        # ── Footer ─────────────────────────────────────────────────
        c.setFillColorRGB(0.30, 0.45, 0.55)
        c.setFont("Helvetica-Oblique", 7)
        c.drawCentredString(W / 2, 0.5 * inch,
                             f"DM Stuff+ Calculator  ·  Model trained on "
                             f"Statcast {_DATA_YEAR_RANGE}  ·  HB in arm-side-positive convention")
        c.save()
        return buf.getvalue()
    except Exception as _pdf_err:
        # Failure mode: minimal "report failed" PDF
        buf2 = _io_pdf.BytesIO()
        try:
            c = _rl_canvas.Canvas(buf2, pagesize=letter)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, letter[1] - 72, "DM Stuff+ PDF generation failed")
            c.setFont("Helvetica", 10)
            c.drawString(72, letter[1] - 100, f"Reason: {_pdf_err}")
            c.save()
            return buf2.getvalue()
        except Exception:
            return b""


def _v3b_per_pitch_feature_row(pitch_group, velo, ivb, hb_arm_positive,
                                 spin_rate, rel_height, rel_side_arm, extension,
                                 hand, is_same_hand):
    """Build one row for v3b's per-pitch model in the exact feature order
    the bundle expects.

    Inputs use CALCULATOR conventions:
      hb_arm_positive: arm-side-positive (matches calculator input)
      hand: 'R' or 'L'

    The model was trained with HB in `hb_arm_in` glove-side-positive
    convention. We negate the input to match.
    """
    if _usage_v3b is None:
        return None
    feat_names = _usage_v3b.get("feature_names") or []
    if not feat_names:
        return None
    lb = _usage_v3b.get("league_batter") or {}
    bat_xwoba = float(lb.get("xwoba", 0.310))
    bat_k     = float(lb.get("k",     0.225))
    bat_bb    = float(lb.get("bb",    0.085))
    is_lefty  = 1 if str(hand).upper().startswith("L") else 0
    # Calculator HB is arm-side-positive; v3b internal hb_arm is glove-side-positive
    hb_arm = -float(hb_arm_positive) if hb_arm_positive is not None else 0.0
    pt_int = _V3B_GROUP_TO_INT.get(pitch_group, 0)
    velo_f  = float(velo) if velo is not None else 90.0
    ivb_f   = float(ivb)  if ivb  is not None else 12.0
    spin_f  = float(spin_rate) if spin_rate is not None else 2200.0
    ext_f   = float(extension) if extension is not None else 6.4
    rh_f    = float(rel_height) if rel_height is not None else 5.8
    rs_f    = float(rel_side_arm) if rel_side_arm is not None else 1.5
    rs_arm  = abs(rs_f)   # arm-side positive
    bauer   = spin_f / max(velo_f, 50.0)
    total_break = (ivb_f**2 + hb_arm**2) ** 0.5
    ivb_per_spin = ivb_f / max(spin_f, 500.0) * 1000
    hb_per_spin  = hb_arm / max(spin_f, 500.0) * 1000
    feat_dict = {
        "velo":              velo_f,
        "ivb":               ivb_f,
        "hb_arm":            hb_arm,
        "spin_rate":         spin_f,
        "extension":         ext_f,
        "rel_height":        rh_f,
        "rel_side_arm":      rs_arm,
        "pitch_type_int":    pt_int,
        "is_lefty":          is_lefty,
        "balls":             _V3B_NEUTRAL_FEATURES["balls"],
        "strikes":           _V3B_NEUTRAL_FEATURES["strikes"],
        "is_same_hand":      int(is_same_hand),
        "total_break":       total_break,
        "bauer":             bauer,
        "ivb_per_spin":      ivb_per_spin,
        "hb_per_spin":       hb_per_spin,
        "batter_xwoba":      bat_xwoba,
        "batter_k_pct":      bat_k,
        "batter_bb_pct":     bat_bb,
        "batter_has_prior":  1,
        "times_faced":       _V3B_NEUTRAL_FEATURES["times_faced"],
        "pitch_num_in_app":  _V3B_NEUTRAL_FEATURES["pitch_num_in_app"],
        "pitch_num_log":     _V3B_NEUTRAL_FEATURES["pitch_num_log"],
        "pitch_in_pa":       _V3B_NEUTRAL_FEATURES["pitch_in_pa"],
        "prior_velo":        _V3B_NEUTRAL_FEATURES["prior_velo"],
        "prior_ivb":         _V3B_NEUTRAL_FEATURES["prior_ivb"],
        "prior_hb":          _V3B_NEUTRAL_FEATURES["prior_hb"],
        "prior_pt_int":      _V3B_NEUTRAL_FEATURES["prior_pt_int"],
        "velo_gap_prior":    _V3B_NEUTRAL_FEATURES["velo_gap_prior"],
        "ivb_gap_prior":     _V3B_NEUTRAL_FEATURES["ivb_gap_prior"],
        "hb_gap_prior":      _V3B_NEUTRAL_FEATURES["hb_gap_prior"],
    }
    # Return in the exact column order the bundle expects
    return [feat_dict.get(n, np.nan) for n in feat_names]


def _v3b_predict_arsenal(pitches_dict, rel_height, rel_side_arm, extension,
                           hand, usages):
    """Predict v3b arsenal-level outcome on the residualized scale.

    pitches_dict: {group: {velo, ivb, hb (arm-side+), spin_rate}, ...}
    usages: {group: fraction in [0,1]} — must sum to ~1
    Returns: (pred_avg, per_group_pred dict, tunnel_correction)
              Both stands (R, L) are predicted and averaged.
    """
    if _usage_v3b is None or not pitches_dict:
        return None, {}, 0.0
    feat_names = _usage_v3b["feature_names"]
    out_per_stand = {}
    per_group_by_stand = {"R": {}, "L": {}}
    for stand_code in ("R", "L"):
        model = _usage_v3b.get(f"model_vs_{stand_code}HB")
        if model is None:
            continue
        is_same_hand_flag = 1 if str(hand).upper().startswith(stand_code) else 0
        per_group = {}
        for grp, pd_g in pitches_dict.items():
            row = _v3b_per_pitch_feature_row(
                pitch_group=grp,
                velo=pd_g.get("velo"),
                ivb=pd_g.get("ivb"),
                hb_arm_positive=pd_g.get("hb"),
                spin_rate=pd_g.get("spin_rate"),
                rel_height=rel_height,
                rel_side_arm=rel_side_arm,
                extension=extension,
                hand=hand,
                is_same_hand=is_same_hand_flag,
            )
            if row is None:
                continue
            X = pd.DataFrame([row], columns=feat_names)
            per_group[grp] = float(model.predict(X)[0])
        per_group_by_stand[stand_code] = per_group
        # Additive structural: arsenal = Σ usage_g × per_group_pred_g
        u_sum = sum(usages.get(g, 0.0) for g in per_group) or 1.0
        out_per_stand[stand_code] = sum(
            usages.get(g, 0.0) / u_sum * per_group[g] for g in per_group
        )
    if not out_per_stand:
        return None, {}, 0.0
    # Pairwise tunneling correction (avg over stands)
    lam_R = float(_usage_v3b.get("tunnel_lambda_vs_RHB", 0.0))
    lam_L = float(_usage_v3b.get("tunnel_lambda_vs_LHB", 0.0))
    lam_avg = (lam_R + lam_L) / 2.0
    sigmas = _usage_v3b.get("tunnel_sigmas", {})
    sig_v = float(sigmas.get("velo", 6.0))
    sig_i = float(sigmas.get("ivb",  8.0))
    sig_h = float(sigmas.get("hb",   8.0))
    tunnel_sum = 0.0
    grps = list(pitches_dict.keys())
    for i in range(len(grps)):
        for j in range(i + 1, len(grps)):
            g_i, g_j = grps[i], grps[j]
            u_i = usages.get(g_i, 0.0); u_j = usages.get(g_j, 0.0)
            if u_i <= 0 or u_j <= 0: continue
            v_i = float(pitches_dict[g_i].get("velo") or 90.0)
            v_j = float(pitches_dict[g_j].get("velo") or 90.0)
            i_i = float(pitches_dict[g_i].get("ivb")  or 0.0)
            i_j = float(pitches_dict[g_j].get("ivb")  or 0.0)
            h_i = float(pitches_dict[g_i].get("hb")   or 0.0)
            h_j = float(pitches_dict[g_j].get("hb")   or 0.0)
            k = (
                np.exp(-0.5 * ((v_i - v_j) / sig_v) ** 2)
                * np.exp(-0.5 * ((i_i - i_j) / sig_i) ** 2)
                * np.exp(-0.5 * ((h_i - h_j) / sig_h) ** 2)
            )
            tunnel_sum += 2.0 * u_i * u_j * k
    tunnel_corr = lam_avg * tunnel_sum
    pred_avg = (sum(out_per_stand.values()) / len(out_per_stand)) + tunnel_corr
    # Per-group output: average of R and L per-pitch predictions
    per_group_avg = {}
    all_groups = set(per_group_by_stand["R"]) | set(per_group_by_stand["L"])
    for g in all_groups:
        vals = []
        if g in per_group_by_stand["R"]: vals.append(per_group_by_stand["R"][g])
        if g in per_group_by_stand["L"]: vals.append(per_group_by_stand["L"][g])
        per_group_avg[g] = sum(vals) / len(vals)
    return pred_avg, per_group_avg, tunnel_corr





# Median values for v5 features — used to fill missing inputs at scoring time.
# Calibrated to 2017–2025 Statcast medians.
_V5_MEDIANS = {
    "start_speed": 91.0, "spin_rate": 2300.0, "extension": 6.4,
    # hb_arm_in is glove-side positive in the model (despite name). League-wide
    # median across all pitch types is roughly -2 (4-Seam ~-7, Sinker ~-15,
    # Slider ~+5, Curveball ~+9). Using a neutral default near 0.
    "ivb_in": 12.0, "hb_arm_in": -2.0,
    # v5c: VAA sign-fixed (descending = negative); HAA mean ~0
    "vaa": -5.0, "haa": 0.0,
    "vaa_aa": 0.0, "haa_aa": 0.0,    # residuals — centered on 0 by construction
    "rel_height": 5.8, "rel_side_arm": -1.7,
    # hb_diff is also in the model's convention. For a 4-Seam, hb_diff to itself = 0.
    # For a secondary pitch typed at the calculator without other pitches,
    # we'd default to median pitch HB minus median FB HB (~-2 vs -7 = +5).
    # Set hb_diff median in the model's sign (positive = pitch more glove-side than primary).
    "velo_diff": -2.0, "ivb_diff": -3.0, "hb_diff": 4.0,
    "ssw_magnitude": 0.0,
    "vaa_aa_x_velo": 0.0,            # v5c (replaces vaa_x_velo)
    "rel_height_x_velo": 528.0, "rel_side_x_typeint": -5.1,
    "active_spin_rate": 2200.0, "rel_quadrant": -9.9,
}


# v5e+: MLB-average arsenal lookup for single-pitch calculator inputs.
# Boundaries match those used by the trainer (compute_arsenal_defaults).
_DEFAULT_RELEASE_HEIGHT_BUCKETS = [
    ("submarine", 0.0, 5.0),
    ("low_3_4",   5.0, 5.7),
    ("high_3_4",  5.7, 6.3),
    ("overhand",  6.3, 99.0),
]
_ARSENAL_DEFAULT_FEATS = [
    "velo_diff_secondary", "arsenal_size",
    "arsenal_ivb_spread", "arsenal_hb_spread",
    "arsenal_ivb_max_other", "arsenal_ivb_min_other",
    "arsenal_hb_max_other", "arsenal_hb_min_other",
    "nearest_other_velo_diff", "nearest_other_ivb_diff", "nearest_other_hb_diff",
    # v8c: also fill these for single-pitch arsenals — they have no
    # natural value when there's only one pitch.
    "release_pt_arsenal_spread_h", "release_pt_arsenal_spread_v",
    "movement_arc_to_primary",
    "perceived_velo_diff_primary",
]


def _compute_runtime_arsenal_defaults_from_scalers():
    """v8c fix: if the bundle doesn't ship pre-computed arsenal_defaults,
    derive per-pitch-type median values from the bundle's RobustScaler
    `center_` attribute. RobustScaler.center_ stores the training median
    per feature, so using these for single-pitch arsenals produces ~0
    scaled values (= league baseline) instead of huge outliers.
    Returns dict {pitch_type_int: {feat_name: median_value}}.
    """
    if not _V5_AVAILABLE or _v5_bundle is None:
        return {}
    FEATURES_ALL = _v5_bundle.get("features", [])
    CAT = _v5_bundle.get("cat_features", [])
    cont_feats = [f for f in FEATURES_ALL if f not in CAT]
    per_type = _v5_bundle.get("per_type_scalers", {}) or {}
    defaults = {}
    for pt_int, scaler in per_type.items():
        try:
            centers = list(scaler.center_)
            defaults[int(pt_int)] = {
                cont_feats[i]: float(centers[i]) for i in range(len(cont_feats))
            }
        except Exception:
            continue
    # Global fallback from fallback_scaler
    try:
        fb_scaler = _v5_bundle.get("fallback_scaler") or _v5_bundle.get("scaler")
        if fb_scaler is not None:
            defaults["__global__"] = {
                cont_feats[i]: float(fb_scaler.center_[i]) for i in range(len(cont_feats))
            }
    except Exception:
        pass
    return defaults


_RUNTIME_ARSENAL_DEFAULTS = _compute_runtime_arsenal_defaults_from_scalers() if _V5_AVAILABLE else {}


def _release_height_bucket(rh):
    """Map release height (feet) to bucket name."""
    if rh is None:
        return "high_3_4"
    try:
        rh_v = float(rh)
    except (TypeError, ValueError):
        return "high_3_4"
    for name, lo, hi in _DEFAULT_RELEASE_HEIGHT_BUCKETS:
        if lo <= rh_v < hi:
            return name
    return "high_3_4"


def _lookup_arsenal_defaults(is_lefty: int, rh: float, pitch_type_int: int):
    """Return dict of MLB-average arsenal-context features for the given
    (hand, release-height-bucket, pitch_type_int). Falls back through
    (hand × pitch_type) → global, returning None if no defaults available.
    """
    defaults = (_v5_bundle or {}).get("arsenal_defaults") if _V5_AVAILABLE else None
    if not defaults:
        return None
    bucket = _release_height_bucket(rh)
    by_cell = defaults.get("by_cell", {}) or {}
    by_hand = defaults.get("by_hand", {}) or {}
    by_global = defaults.get("by_global", {}) or {}
    # Try most specific lookup
    cell = by_cell.get((int(is_lefty), bucket, int(pitch_type_int)))
    if cell is not None:
        return cell
    # Fall back to hand+pitch_type
    cell = by_hand.get((int(is_lefty), int(pitch_type_int)))
    if cell is not None:
        return cell
    # Final fallback: global means
    return by_global if by_global else None


# v5e+: Primary fastball defaults for single-pitch calculator inputs.
# Used when the user enters a breaking/offspeed pitch WITHOUT a fastball —
# instead of computing velo_diff/ivb_diff/hb_diff against the pitch itself
# (which gives 0 and produces low scores), we use these MLB-typical primary
# fastball values to compute proper differentials.
#
# Values computed from pitcher_profiles.csv (median per cell) — see
# the conversation for the derivation. HB values are in the MODEL'S internal
# convention (glove-side positive, which is how profiles store them).
_PRIMARY_FB_DEFAULTS = {
    # (is_lefty, rel_height_bucket): typical primary FB stats
    (0, "high_3_4"):  {"velo": 94.3, "ivb": 16.1, "hb": -7.7},
    (0, "low_3_4"):   {"velo": 94.1, "ivb": 14.7, "hb": -9.6},
    (0, "overhand"):  {"velo": 93.8, "ivb": 17.0, "hb": -5.8},
    (0, "submarine"): {"velo": 92.0, "ivb": 10.2, "hb": -12.3},
    (1, "high_3_4"):  {"velo": 92.6, "ivb": 15.8, "hb": -8.1},
    (1, "low_3_4"):   {"velo": 92.4, "ivb": 13.9, "hb": -10.3},
    (1, "overhand"):  {"velo": 92.7, "ivb": 16.5, "hb": -7.0},
    (1, "submarine"): {"velo": 90.5, "ivb": 5.2,  "hb": -16.6},
}
_PRIMARY_FB_HAND_DEFAULTS = {
    0: {"velo": 94.1, "ivb": 15.6, "hb": -8.3},  # RHP all
    1: {"velo": 92.5, "ivb": 15.3, "hb": -8.9},  # LHP all
}
_PRIMARY_FB_GLOBAL = {"velo": 93.7, "ivb": 15.5, "hb": -8.4}


def _lookup_primary_fb_defaults(is_lefty: int, rh: float):
    """Return dict of typical {velo, ivb, hb} for the primary fastball of an
    MLB pitcher of this hand and arm slot. Used to compute velo_diff/ivb_diff/
    hb_diff when the calculator user enters a non-fastball pitch alone.

    HB is in the model's internal convention (glove-side positive).
    """
    bucket = _release_height_bucket(rh)
    cell = _PRIMARY_FB_DEFAULTS.get((int(is_lefty), bucket))
    if cell is not None:
        return cell
    return _PRIMARY_FB_HAND_DEFAULTS.get(int(is_lefty), _PRIMARY_FB_GLOBAL)


# ── VAA approximation from shape + slot + velo ─────────────────────────────
# Fit from pitcher_profiles.csv (~22K pitcher-seasons across 8 pitch types).
# VAA is in model convention (descending = negative).
# Form: vaa = a + b_ivb*IVB + b_velo*velo + b_rh*rel_height
# R² per type: 0.77-0.90 — good enough that the model gets a meaningful
# IVB-dependent VAA signal instead of a constant.
# Critically: higher IVB → flatter (less negative) VAA. The previous calculator
# code used a constant -5.0 regardless of IVB, which meant the model couldn't
# tell that 20 IVB / 5.0' had a flatter VAA than 12 IVB / 5.0' — so it scored
# them similarly even though the 20-IVB pitch should be much more deceptive.
_VAA_APPROX = {
    "4-Seam":        (-6.7212, 0.10167, 0.06952, -1.07965),
    "2-Seam/Sinker": (-7.4854, 0.12215, 0.07360, -1.10859),
    "Cutter":        (-8.0730, 0.14477, 0.08624, -1.22067),
    "Slider":        (-9.8377, 0.12300, 0.09425, -1.06066),
    "Sweeper":       (-8.7510, 0.11873, 0.07816, -0.99131),
    "Curveball":    (-11.0888, 0.10997, 0.10133, -0.95120),
    "Splitter":      (-9.9527, 0.11755, 0.09057, -1.02412),
    "Changeup":      (-9.9411, 0.11545, 0.09193, -1.02518),
}


def _approximate_vaa(pitch_group: str, ivb: float, velo: float, rh: float) -> float:
    """Estimate VAA (degrees, model convention: descending = negative) from
    pitch shape + slot + velocity. Returns None if pitch group is unknown.
    """
    params = _VAA_APPROX.get(pitch_group)
    if params is None:
        return None
    a, b_ivb, b_velo, b_rh = params
    return a + b_ivb * float(ivb) + b_velo * float(velo) + b_rh * float(rh)


def _approximate_haa(pitch_group: str, hb_arm_positive: float, hand: str) -> float:
    """Estimate HAA (horizontal approach angle, catcher's-view degrees) from
    pitch type, hand, and arm-side-positive HB.

    Method: anchor at the per-(pitch_group, hand) HAA league mean computed
    from pitcher_profiles, then linearly shift by the pitch's HB deviation
    from the type-typical HB.

    Physics-derived slope: a pitch travels ~660" release-to-plate; an extra
    inch of horizontal break at the plate ≈ arctan(1/660) ≈ 0.087° of
    additional horizontal approach angle. RHP arm-side break shifts HAA
    glove-side (more negative in catcher's view); LHP arm-side break
    shifts HAA arm-side (more positive). Hand sign handles this.

    Returns 0.0 if no league baseline available (degrades gracefully).
    """
    if hb_arm_positive is None:
        hb_arm_positive = 0.0
    hand_code = "R" if str(hand).upper().startswith("R") else "L"
    # Look up league baseline; _vaa_haa_league is populated at module-load time.
    baseline = (_vaa_haa_league.get(f"{pitch_group}_{hand_code}")
                  or _vaa_haa_league.get(pitch_group)
                  or {})
    haa_mu = float(baseline.get("haa_mu", 0.0))
    typical_hb = float(_MLB_PITCH_MEDIANS.get(pitch_group, {}).get("hb", 0.0))
    SLOPE = 0.087   # degrees per inch (physics-derived, see docstring)
    # RHP: more arm-side HB → HAA more negative (catcher's view points left)
    # LHP: more arm-side HB → HAA more positive
    hand_sign = +1.0 if hand_code == "L" else -1.0
    return haa_mu + (float(hb_arm_positive) - typical_hb) * SLOPE * hand_sign


def _score_v5_arsenal(pitches: dict, rel_height: float = None,
                       rel_side: float = None, extension: float = None,
                       hand: str = "R") -> dict:
    """Score a pitch arsenal through the v5 model.

    `pitches`: {group_name: {velo, ivb, hb, spin_rate, ...}}
       Missing values are imputed from training medians; only velo is required.
    `rel_height`, `rel_side`, `extension`: arsenal-wide release info. If None,
       use medians. rel_side here is in the "arm-side positive" convention.

    Returns dict {group_name: {"stuff_plus": float, "imputed": [list of fields]}}.
    """
    if not _V5_AVAILABLE:
        return {}

    import numpy as _np
    import pandas as _pd

    FEATURES     = _v5_bundle["features"]
    CAT_FEATURES = _v5_bundle["cat_features"]
    GROUP_TO_INT = _v5_bundle["group_to_int"]
    model        = _v5_bundle["model"]
    per_type_scalers = _v5_bundle.get("per_type_scalers", {}) or {}
    fallback_scaler  = _v5_bundle.get("fallback_scaler") or _v5_bundle.get("scaler")
    # v5c: VAA/HAA baselines for residualization
    vaa_haa_baselines = _v5_bundle.get("vaa_haa_baselines", {}) or {}
    if fallback_scaler is None:
        return {}

    # Fill release-profile defaults
    rh  = float(rel_height) if rel_height is not None else _V5_MEDIANS["rel_height"]
    ext = float(extension)  if extension  is not None else _V5_MEDIANS["extension"]
    # rel_side sign-convention bridge between calculator input and v8c training.
    #
    # CALCULATOR INPUT  : user enters a POSITIVE value for arm-side release
    #                     (per the field label "Rel Side — arm side").
    # V8C TRAINING      : `df["rel_side_arm"] = np.where(is_lhp, -pfx_x, pfx_x)`,
    #                     which produces a NEGATIVE value for arm-side release
    #                     for BOTH hands (see train_stuff_plus_v8c.py L883).
    #                     The variable is misleadingly named — it is actually
    #                     glove-side-positive in the v8c bundle.
    #
    # The earlier RHP-only flip left LHP user input with the wrong sign, which
    # propagated into haa_aa residualisation, rel_quadrant, and rel_side_x_typeint
    # and biased LHP Stuff+ by a few points. Fix: flip the sign for both hands.
    rs_arm = float(rel_side) if rel_side is not None else _V5_MEDIANS["rel_side_arm"]
    if rs_arm > 0:
        rs_arm = -rs_arm

    arsenal_imputed_release = []
    if rel_height is None: arsenal_imputed_release.append("rel_height")
    if rel_side   is None: arsenal_imputed_release.append("rel_side")
    if extension  is None: arsenal_imputed_release.append("extension")

    # Primary fastball for differentials (priority order: 4-Seam, 2-Seam, Cutter)
    # NOTE: User-entered HB is in "arm-side positive" convention; model wants
    # glove-side positive. Negate when ingesting.
    is_lefty = 1 if hand == "L" else 0
    FB_PRIORITY = ["4-Seam", "2-Seam/Sinker", "Cutter"]
    primary_velo = primary_ivb = primary_hb = None
    primary_was_imputed = False  # track for imputed-list display
    for fb in FB_PRIORITY:
        if fb in pitches and pitches[fb].get("velo") is not None:
            primary_velo = float(pitches[fb]["velo"])
            primary_ivb  = (float(pitches[fb]["ivb"]) if pitches[fb].get("ivb") is not None
                             else _V5_MEDIANS["ivb_in"])
            primary_hb   = ((-float(pitches[fb]["hb"])) if pitches[fb].get("hb") is not None
                             else _V5_MEDIANS["hb_arm_in"])
            break
    if primary_velo is None:
        # No fastball entered. Use MLB-average primary FB stats for this
        # hand × release-height bucket. This avoids the "single slider scored
        # against itself produces velo_diff=0" bug, which produced absurdly
        # low scores for breaking balls entered alone.
        # HB defaults are ALREADY in the model's glove-side-positive convention
        # (computed from profiles directly), so no sign flip here.
        fb_defaults = _lookup_primary_fb_defaults(is_lefty, rh)
        primary_velo = float(fb_defaults["velo"])
        primary_ivb  = float(fb_defaults["ivb"])
        primary_hb   = float(fb_defaults["hb"])
        primary_was_imputed = True

    rows, keys, imputed_per_pitch = [], [], {}

    for grp, m in pitches.items():
        if grp not in GROUP_TO_INT:
            continue
        velo = m.get("velo")
        if velo is None:
            continue  # velo is required
        velo = float(velo)

        imputed = list(arsenal_imputed_release)
        if primary_was_imputed:
            imputed.append("primary FB (MLB avg)")
        # Per-pitch metric fills
        ivb = m.get("ivb")
        if ivb is None:
            ivb = _V5_MEDIANS["ivb_in"]; imputed.append("ivb")
        else:
            ivb = float(ivb)
        hb_arm = m.get("hb")
        if hb_arm is None:
            hb_arm = _V5_MEDIANS["hb_arm_in"]; imputed.append("hb")
        else:
            # User enters HB in "arm-side positive" convention (matches leaderboard
            # display, matches TrackMan/Rapsodo). Model's internal `hb_arm_in` is
            # actually glove-side positive (named misleadingly — it equals pfx_x*12
            # for RHP, which is negative for arm-side). Negate user input.
            hb_arm = -float(hb_arm)
        spin_rate = m.get("spin_rate")
        if spin_rate is None:
            # Use pitch-type-specific median (splitter ~1370, sweeper ~2571, etc.)
            spin_rate = float(
                (_MLB_PITCH_MEDIANS.get(grp) or {}).get("spin_rate")
                or _V5_MEDIANS["spin_rate"]
            )
            imputed.append("spin_rate")
        else:
            spin_rate = float(spin_rate)

        # v5c: reclassify shape-ambiguous breaking balls so we use the right
        # per-type scaler. Must mirror training-time logic exactly.
        scored_group = grp
        if grp == "Slider" and hb_arm >= 10.0 and velo <= 87.0:
            scored_group = "Sweeper"
        elif grp == "Sweeper" and hb_arm <= 8.0:
            scored_group = "Slider"
        pt_int = GROUP_TO_INT.get(scored_group, GROUP_TO_INT[grp])

        # VAA: approximated from shape + slot + velocity using a regression fit
        # from real pitcher data. This is the critical signal for low-slot
        # rising fastballs — without it, the model can't tell 20 IVB / 5.0'
        # from 12 IVB / 5.0' because raw IVB alone doesn't carry the flatness
        # signal the model was trained to read.
        # HAA is harder to approximate without spin axis, so we still use the
        # constant median there.
        vaa_approx = _approximate_vaa(scored_group, ivb, velo, rh)
        if vaa_approx is not None:
            vaa = vaa_approx
            imputed.append("vaa (estimated from shape)")
        else:
            vaa = _V5_MEDIANS["vaa"]
            imputed.append("vaa")
        # Model-input HAA stays at the trained median (changing it would
        # shift predictions in untested ways). For DISPLAY we compute a
        # physically-motivated estimate from shape + hand so the per-pitch
        # card shows a non-zero, pitch-appropriate value.
        haa = _V5_MEDIANS["haa"]; imputed.append("haa")
        if hand == "L":
            haa = -haa
        # hb_arm here is glove-side-positive (model internal); convert to
        # arm-side-positive before passing to _approximate_haa.
        haa_display = _approximate_haa(scored_group, -hb_arm, hand)

        # v5c: residualize VAA/HAA against release geometry using bundle baselines
        bl_v = vaa_haa_baselines.get(("vaa", int(pt_int), int(is_lefty)))
        bl_h = vaa_haa_baselines.get(("haa", int(pt_int), int(is_lefty)))
        vaa_aa = (vaa - (bl_v[0] + bl_v[1] * rh))     if bl_v else vaa
        haa_aa = (haa - (bl_h[0] + bl_h[1] * rs_arm)) if bl_h else haa

        # SSW magnitude — use user-provided tilt-derived hint if available,
        # otherwise default to 0 (model treats as fully Magnus).
        _ssw_hint = m.get("ssw_magnitude_hint")
        if _ssw_hint is not None:
            ssw_mag = float(_ssw_hint)
        else:
            ssw_mag = 0.0
            imputed.append("ssw_magnitude")

        velo_diff = velo  - primary_velo if primary_velo is not None else _V5_MEDIANS["velo_diff"]
        ivb_diff  = ivb   - primary_ivb  if primary_ivb  is not None else _V5_MEDIANS["ivb_diff"]
        hb_diff   = hb_arm - primary_hb  if primary_hb  is not None else _V5_MEDIANS["hb_diff"]

        # v5c interaction features (vaa_aa_x_velo replaces v4's vaa_x_velo)
        vaa_aa_x_velo     = vaa_aa * velo
        rel_height_x_velo = rh  * velo
        rel_side_x_typeint = rs_arm * pt_int
        # active_spin_rate ≈ spin_rate × (1 - ssw_fraction); ssw=0 → active = spin_rate
        pfx_x_ft = (hb_arm / 12.0) * (1 if hand == "R" else -1)
        pfx_z_ft = ivb / 12.0
        total_break_ft = (pfx_x_ft**2 + pfx_z_ft**2) ** 0.5
        if total_break_ft < 0.01:
            active_spin = spin_rate
        else:
            ssw_frac = min(max(ssw_mag / total_break_ft, 0.0), 1.0)
            active_spin = spin_rate * (1.0 - ssw_frac)
        rel_quadrant = rh * rs_arm

        row = {
            "start_speed":   velo,
            "spin_rate":     spin_rate,
            "ivb_in":        ivb,
            "hb_arm_in":     hb_arm,
            # v5c: vaa_aa / haa_aa replace raw vaa / haa
            "vaa_aa":        vaa_aa,
            "haa_aa":        haa_aa,
            # Raw (un-residualized) VAA / HAA for display purposes only.
            # These are the model's *estimated* approach angles given shape,
            # slot, and velo — not used as input features (vaa_aa/haa_aa are).
            "vaa_raw":       vaa,
            "haa_raw":       haa_display,
            "rel_height":    rh,
            "rel_side_arm":  rs_arm,
            "extension":     ext,
            "velo_diff":     velo_diff,
            "ivb_diff":      ivb_diff,
            "hb_diff":       hb_diff,
            "pitch_type_int": pt_int,
            "is_lefty":      is_lefty,
            "is_same_hand":  0,
            "ssw_magnitude": ssw_mag,
            # v5c: vaa_aa_x_velo replaces v4's vaa_x_velo
            "vaa_aa_x_velo":      vaa_aa_x_velo,
            "rel_height_x_velo":  rel_height_x_velo,
            "rel_side_x_typeint": rel_side_x_typeint,
            "active_spin_rate":   active_spin,
            "rel_quadrant":       rel_quadrant,
            # v5d: arsenal-context placeholders (filled after all rows built)
            "velo_diff_secondary":     float("nan"),
            "arsenal_size":            float("nan"),
            "arsenal_ivb_spread":      float("nan"),
            "arsenal_hb_spread":       float("nan"),
            "arsenal_ivb_max_other":   float("nan"),
            "arsenal_ivb_min_other":   float("nan"),
            "arsenal_hb_max_other":    float("nan"),
            "arsenal_hb_min_other":    float("nan"),
            "nearest_other_velo_diff": float("nan"),
            "nearest_other_ivb_diff":  float("nan"),
            "nearest_other_hb_diff":   float("nan"),
        }
        # v6++ #2: infer spin_axis_sin/cos from movement direction.
        # Statcast spin_axis follows: pfx_x = total * sin(axis), pfx_z = total * -cos(axis).
        # Inverting: axis_rad = atan2(pfx_x, -pfx_z). For LHP, hb_arm_in maps to -pfx_x.
        if total_break_ft > 0.01:
            axis_rad = _np.arctan2(pfx_x_ft, -pfx_z_ft)
            row["spin_axis_sin"] = float(_np.sin(axis_rad))
            row["spin_axis_cos"] = float(_np.cos(axis_rad))
        else:
            row["spin_axis_sin"] = 0.0
            row["spin_axis_cos"] = -1.0
        # v6++ #4 Bauer Units (spin/velo). Universal feature.
        row["bauer_units"] = float(spin_rate / max(velo, 60.0))

        # ─── v8c calculator-derivable physics features ────────────────────
        # arm_angle in degrees (0 = sidearm, 45 = 3/4, 80+ = over-top)
        _vert = rh - 5.0
        _horiz = max(abs(rs_arm), 0.01)
        row["arm_angle"] = float(_np.degrees(_np.arctan2(_vert, _horiz)))
        # Interactions
        row["velo_x_typeint"]         = float(velo * pt_int)
        row["rel_quadrant_x_velo"]    = float(rel_quadrant * velo)
        row["rel_quadrant_x_typeint"] = float(rel_quadrant * pt_int)
        # Perceived velocity (extension-adjusted)
        row["perceived_velo"] = float(velo * (1.0 + (ext - 6.5) / 55.0))
        # Raw products
        row["velo_x_spin_rate"] = float(velo * spin_rate)
        row["velo_x_ivb"]       = float(velo * ivb)
        # vaa_aa_x_velo (uses already-residualized vaa_aa)
        row["vaa_aa_x_velo"] = float(vaa_aa * velo)
        # Movement angle as unit vector (v8c: replaces movement_angle_deg)
        _total_mv = max(_np.sqrt(ivb**2 + hb_arm**2), 0.01)
        row["movement_angle_sin"] = float(ivb / _total_mv)
        row["movement_angle_cos"] = float(hb_arm / _total_mv)
        row["total_movement"]     = float(_total_mv)
        # Per-spin efficiency metrics
        _safe_spin = max(spin_rate, 100.0)
        row["ivb_per_spin"]    = float(ivb / _safe_spin * 1000.0)
        row["hb_per_spin"]     = float(hb_arm / _safe_spin * 1000.0)
        row["active_spin_pct"] = float(min(max(active_spin / _safe_spin, 0.0), 1.0))

        # ─── K-means soft cluster scores (v6++/v8c) ───────────────────────
        # v8+ bundles use bb_kmeans_state + fb_kmeans_state (split by FB vs BB).
        # v6/v7 bundles use a single kmeans_state for Slider/Sweeper.
        bb_km = _v5_bundle.get("bb_kmeans_state") or _v5_bundle.get("kmeans_state") if _V5_AVAILABLE else None
        fb_km = _v5_bundle.get("fb_kmeans_state") if _V5_AVAILABLE else None

        def _soft_cluster_score(km_state, target_cluster_key, row_dict):
            """Compute softmax over -d² from input row to k-means centroids,
            return probability of the named cluster."""
            try:
                cf = km_state["cluster_feats"]
                means = km_state["feature_means"]; stds = km_state["feature_stds"]
                vec = _np.array([
                    (row_dict.get(c, 0.0) - means[c]) / max(stds[c], 1e-6) for c in cf
                ]).reshape(1, -1)
                centroids = km_state["centroids"]
                dists = _np.sqrt(((vec - centroids) ** 2).sum(axis=1))
                neg_d2 = -(dists ** 2)
                shifted = neg_d2 - neg_d2.max()
                probs = _np.exp(shifted); probs = probs / probs.sum()
                return float(probs[km_state[target_cluster_key]])
            except Exception:
                return float("nan")

        _slider_int = _v5_bundle["group_to_int"].get("Slider", -1)
        _sweeper_int = _v5_bundle["group_to_int"].get("Sweeper", -1)
        _4seam_int = _v5_bundle["group_to_int"].get("4-Seam", -1)
        _sinker_int = _v5_bundle["group_to_int"].get("2-Seam/Sinker", -1)

        # Cluster scores: training data hard-coded 1.0 for Statcast-tagged
        # pitches (~95%+ of rows), so the per-type scaler was fit on values
        # near 1.0/0.0. At inference we trust the user's pitch-type label
        # and assign hard values (matching the training-data convention),
        # NOT softmax probabilities — softmax values like 0.5 are ~28 SDs
        # below the training median and produce wildly wrong scores.
        # sweeper_cluster_score: 1.0 for Sweepers, 0.0 for Sliders, NaN for others
        if pt_int == _sweeper_int:
            row["sweeper_cluster_score"] = 1.0
        elif pt_int == _slider_int:
            row["sweeper_cluster_score"] = 0.0
        else:
            row["sweeper_cluster_score"] = float("nan")

        # four_seam_cluster_score: 1.0 for 4-Seam, 0.0 for Sinker, NaN for others
        if pt_int == _4seam_int:
            row["four_seam_cluster_score"] = 1.0
        elif pt_int == _sinker_int:
            row["four_seam_cluster_score"] = 0.0
        else:
            row["four_seam_cluster_score"] = float("nan")

        rows.append(row)
        # Use the scored group (post-reclassification) for the norms lookup,
        # but keep the user-facing key as what they entered so the result
        # displays as the pitch they actually input.
        keys.append((grp, scored_group))
        imputed_per_pitch[grp] = imputed

    if not rows:
        return {}

    # v5d: fill arsenal-context features now that all rows are built
    if len(rows) >= 2:
        velos = _np.array([r["start_speed"] for r in rows])
        ivbs  = _np.array([r["ivb_in"]      for r in rows])
        hbs   = _np.array([r["hb_arm_in"]   for r in rows])
        pts   = _np.array([r["pitch_type_int"] for r in rows])
        for i, r in enumerate(rows):
            r["arsenal_size"] = float(len(rows))
            other_mask = pts != pts[i]
            if not other_mask.any():
                continue
            ov = velos[other_mask]; oi = ivbs[other_mask]; oh = hbs[other_mask]
            r["arsenal_ivb_spread"]    = float(oi.max() - oi.min())
            r["arsenal_hb_spread"]     = float(oh.max() - oh.min())
            r["arsenal_ivb_max_other"] = float(oi.max())
            r["arsenal_ivb_min_other"] = float(oi.min())
            r["arsenal_hb_max_other"]  = float(oh.max())
            r["arsenal_hb_min_other"]  = float(oh.min())
            my_v = velos[i]
            slower = ov[ov < my_v]
            if len(slower) > 0:
                r["velo_diff_secondary"] = float(my_v - slower.max())
            else:
                r["velo_diff_secondary"] = float(my_v - ov.max())
            gaps = _np.abs(ov - my_v)
            j = int(gaps.argmin())
            r["nearest_other_velo_diff"] = float(my_v - ov[j])
            r["nearest_other_ivb_diff"]  = float(ivbs[i] - oi[j])
            r["nearest_other_hb_diff"]   = float(hbs[i]  - oh[j])
    else:
        # v5e+: Single-pitch input → fill with MLB-average arsenal defaults
        # for this hand × release-height-bucket × pitch_type. This makes
        # calculator scores more comparable to leaderboard scores, since
        # the model expects arsenal context.
        r = rows[0]
        defaults = _lookup_arsenal_defaults(
            is_lefty=is_lefty,
            rh=rh,
            pitch_type_int=int(r["pitch_type_int"]),
        )
        if defaults:
            for f in _ARSENAL_DEFAULT_FEATS:
                if f in defaults:
                    r[f] = float(defaults[f])
            for grp in imputed_per_pitch:
                imputed_per_pitch[grp].append("arsenal (MLB avg)")
        else:
            # v8c fix: use runtime defaults derived from per-type scaler
            # medians instead of zeros. Zeros are wildly off from training
            # medians for features like arsenal_ivb_max_other (which is
            # typically ~14 for 4-Seam pitchers) → 0 produces huge negative
            # z-scores that blow up predictions.
            pt_int_for_default = int(r["pitch_type_int"])
            type_defaults = _RUNTIME_ARSENAL_DEFAULTS.get(pt_int_for_default,
                            _RUNTIME_ARSENAL_DEFAULTS.get("__global__", {}))
            for f in _ARSENAL_DEFAULT_FEATS:
                if f in type_defaults:
                    r[f] = float(type_defaults[f])
                elif f == "arsenal_size":
                    r[f] = 1.0
                else:
                    r[f] = 0.0
            for grp in imputed_per_pitch:
                imputed_per_pitch[grp].append("arsenal (training median)")

    # ─── v8c arsenal-context features (computed after all rows built) ────
    # Only compute these from arsenal context when there are MULTIPLE pitches.
    # For single-pitch arsenals, leave whatever the defaults block set
    # (training medians) — computing them as 0.0 would clobber realistic
    # defaults and produce wildly off-distribution scaled values.
    if len(rows) >= 2:
        _rh_arr = _np.array([r["rel_height"]   for r in rows], dtype=float)
        _rs_arr = _np.array([r["rel_side_arm"] for r in rows], dtype=float)
        _spread_h = float(_np.std(_rs_arr, ddof=0))
        _spread_v = float(_np.std(_rh_arr, ddof=0))
        for r in rows:
            _ivb_d = r["ivb_in"]    - (primary_ivb if primary_ivb is not None else _V5_MEDIANS["ivb_in"])
            _hb_d  = r["hb_arm_in"] - (primary_hb  if primary_hb  is not None else _V5_MEDIANS["hb_arm_in"])
            r["movement_arc_to_primary"] = float(_np.sqrt(_ivb_d**2 + _hb_d**2))
            _primary_perceived = (primary_velo if primary_velo is not None else _V5_MEDIANS["start_speed"]) \
                                  * (1.0 + (r["extension"] - 6.5) / 55.0)
            r["perceived_velo_diff_primary"] = float(r["perceived_velo"] - _primary_perceived)
            r["release_pt_arsenal_spread_h"] = _spread_h
            r["release_pt_arsenal_spread_v"] = _spread_v
    # else: single-pitch case — these features were set to runtime medians
    # in the defaults block above (lines ~1606-1622)

    # Defensive: fill any FEATURES still missing with NaN before slice.
    # Protects against future bundle versions adding features the app
    # doesn't know about — LightGBM handles NaN natively for those.
    df = _pd.DataFrame(rows)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = float("nan")
    df = df[FEATURES]
    pt_arr = df["pitch_type_int"].to_numpy()

    # Per-type scaling
    cont_idx = [i for i, c in enumerate(FEATURES) if c not in CAT_FEATURES]
    X = df.values.astype(_np.float64)
    X_out = X.copy()
    for grp_int in _np.unique(pt_arr):
        mask = pt_arr == grp_int
        scaler = per_type_scalers.get(int(grp_int), fallback_scaler)
        sub_cont = X[mask][:, cont_idx]
        sub_scaled = scaler.transform(sub_cont)
        rows_idx = _np.where(mask)[0]
        for i, col in enumerate(cont_idx):
            X_out[rows_idx, col] = sub_scaled[:, i]

    try:
        raw = model.predict(X_out)
    except Exception as e:
        return {}

    # v6++ #5 Confidence intervals via quantile boosters (if bundle has them)
    raw_q10 = raw_q90 = None
    q10_model = _v5_bundle.get("q10_model")
    q90_model = _v5_bundle.get("q90_model")
    if q10_model is not None and q90_model is not None:
        try:
            raw_q10 = q10_model.predict(X_out)
            raw_q90 = q90_model.predict(X_out)
        except Exception:
            raw_q10 = raw_q90 = None

    # v6++ #4 Platoon-split scoring: re-score with is_same_hand flipped to get
    # vs-LHB and vs-RHB explicit numbers.
    raw_vs_rhb = raw_vs_lhb = None
    if "is_same_hand" in FEATURES:
        ish_idx = FEATURES.index("is_same_hand")
        # Same-hand for RHP = batter is R. For LHP = batter is L.
        # So is_same_hand=1 means same hand as pitcher; for RHP-vs-RHB, ish=1.
        X_rhb = X_out.copy()
        X_lhb = X_out.copy()
        X_rhb[:, ish_idx] = 1 if hand == "R" else 0
        X_lhb[:, ish_idx] = 0 if hand == "R" else 1
        try:
            raw_vs_rhb = model.predict(X_rhb)
            raw_vs_lhb = model.predict(X_lhb)
        except Exception:
            raw_vs_rhb = raw_vs_lhb = None

    # OOD ranges + NN reference from bundle
    ood_ranges = _v5_bundle.get("ood_ranges", {}) or {}
    nn_ref     = _v5_bundle.get("nn_reference")    # pandas DataFrame or None

    by_type = _v5_norms.get("by_type", {})
    overall = _v5_norms.get("overall", {"mean": 0.0, "sd": 1.0})
    out = {}
    int_to_grp = {v: k for k, v in GROUP_TO_INT.items()}
    for i, (display_grp, scored_grp) in enumerate(keys):
        params = by_type.get(scored_grp, overall)
        m_, s_ = params["mean"], params["sd"]
        # Keep raw[i] (is_same_hand=0) as the normed score — arsenal norms were
        # calibrated against this distribution, so raw_pred must stay unchanged.
        sp = 100.0 + ((raw[i] - m_) / max(s_, 1e-6)) * 10.0
        result = {
            "stuff_plus": round(float(sp), 1),
            "raw_pred":   float(raw[i]),           # keep original for arsenal norms
            "raw_vs_rhb": float(raw_vs_rhb[i]) if raw_vs_rhb is not None else None,
            "raw_vs_lhb": float(raw_vs_lhb[i]) if raw_vs_lhb is not None else None,
            "imputed":    imputed_per_pitch.get(display_grp, []),
            "shape_row":  rows[i],
        }
        # #5 Confidence interval
        if raw_q10 is not None:
            sp_lo = 100.0 + ((raw_q10[i] - m_) / max(s_, 1e-6)) * 10.0
            sp_hi = 100.0 + ((raw_q90[i] - m_) / max(s_, 1e-6)) * 10.0
            # Clamp ordering (q10 should be <= q90, but quantile crossing happens)
            result["stuff_plus_p10"] = round(float(min(sp_lo, sp_hi)), 1)
            result["stuff_plus_p90"] = round(float(max(sp_lo, sp_hi)), 1)
        # #4 Platoon split — also compute averaged overall for display
        if raw_vs_rhb is not None:
            sp_rhb = 100.0 + ((raw_vs_rhb[i] - m_) / max(s_, 1e-6)) * 10.0
            sp_lhb = 100.0 + ((raw_vs_lhb[i] - m_) / max(s_, 1e-6)) * 10.0
            result["stuff_plus_vs_rhb"] = round(float(sp_rhb), 1)
            result["stuff_plus_vs_lhb"] = round(float(sp_lhb), 1)
            # Overall = simple average of both sides; displayed instead of the
            # opposite-hand-only default so the headline reflects a true overall.
            result["stuff_plus_overall"] = round((sp_rhb + sp_lhb) / 2.0, 1)
        # #3 OOD warning: flag features outside training 5th/95th percentile
        ranges_for_pt = ood_ranges.get(scored_grp, {})
        if ranges_for_pt:
            r_row = rows[i]
            ood_flags = []
            for feat, fr in ranges_for_pt.items():
                v = r_row.get(feat)
                if v is None: continue
                # Convert hb_arm_in back to arm-side-positive for display
                v_disp = v
                if feat == "hb_arm_in":
                    v_disp = -v
                lo, hi = fr.get("p05"), fr.get("p95")
                if v < fr.get("p01", -1e9) or v > fr.get("p99", 1e9):
                    ood_flags.append({"feat": feat, "value": round(v_disp,2),
                                       "range": [round(lo,2), round(hi,2)],
                                       "severity": "extreme"})
                elif v < lo or v > hi:
                    ood_flags.append({"feat": feat, "value": round(v_disp,2),
                                       "range": [round(lo,2), round(hi,2)],
                                       "severity": "mild"})
            if ood_flags:
                result["ood_warnings"] = ood_flags
        # #9 Nearest-neighbor pitcher lookup
        if nn_ref is not None:
            try:
                pt_int_match = int(rows[i]["pitch_type_int"])
                sub = nn_ref[nn_ref["pitch_type_int"] == pt_int_match]
                if len(sub) > 0:
                    target = _np.array([
                        rows[i]["start_speed"], rows[i]["spin_rate"],
                        rows[i]["ivb_in"],     rows[i]["hb_arm_in"],
                        rows[i]["rel_height"], rows[i]["rel_side_arm"],
                        rows[i]["extension"],
                    ])
                    cols  = ["start_speed","spin_rate","ivb_in","hb_arm_in",
                             "rel_height","rel_side_arm","extension"]
                    cols  = [c for c in cols if c in sub.columns]
                    avail = sub[cols].to_numpy(dtype=_np.float64)
                    # Per-feature scale (use training feature scale = std)
                    scales = _np.maximum(sub[cols].std().to_numpy(), 1e-3)
                    target_v = _np.array([
                        {"start_speed": rows[i]["start_speed"],
                         "spin_rate": rows[i]["spin_rate"],
                         "ivb_in": rows[i]["ivb_in"],
                         "hb_arm_in": rows[i]["hb_arm_in"],
                         "rel_height": rows[i]["rel_height"],
                         "rel_side_arm": rows[i]["rel_side_arm"],
                         "extension": rows[i]["extension"]}[c] for c in cols
                    ])
                    diffs = (avail - target_v) / scales
                    distances = _np.sqrt((diffs ** 2).sum(axis=1))
                    j = int(_np.argmin(distances))
                    closest = sub.iloc[j]
                    # ivb_in is arm-side neutral; hb_arm_in is glove-side positive.
                    # Convert hb to arm-side positive (= -hb_arm_in) for plot consumers.
                    _nn_ivb = float(closest["ivb_in"]) if "ivb_in" in closest else None
                    _nn_hb  = -float(closest["hb_arm_in"]) if "hb_arm_in" in closest else None
                    result["nearest_pitcher"] = {
                        "name": str(closest["player_name"]),
                        "year": int(closest["year"]),
                        "distance": round(float(distances[j]), 2),
                        "ivb": _nn_ivb,
                        "hb":  _nn_hb,    # arm-side-positive
                        "velo": float(closest["start_speed"]) if "start_speed" in closest else None,
                    }
            except Exception as _nn_err:
                import sys as _sys
                print(f"[nearest_pitcher:{display_grp}] failed: {_nn_err}",
                      file=_sys.stderr)
        out[display_grp] = result
    return out


# ── Arsenal Stuff+ aggregation (raw-prediction approach) ─────────────────
# MLB-typical usage% per pitch type (fallback when user doesn't enter usage).
_MLB_USAGE_FALLBACK = {
    "4-Seam": 35.0, "2-Seam/Sinker": 18.0, "Cutter": 8.0,
    "Slider": 16.0, "Sweeper": 6.0, "Curveball": 9.0,
    "Splitter": 3.0, "Changeup": 5.0, "Knuckleball": 0.1,
}


def _score_arsenal_combined(scores: dict, usage: dict = None) -> dict:
    """Compute Arsenal Stuff+ via raw-prediction aggregation.

    Y_arsenal = sum(raw_pred_i × usage_i)   (usage weights normalized to 1)
    Stuff+    = 100 + (Y_arsenal - μ_arsenal) / σ_arsenal × 10

    Where μ_arsenal, σ_arsenal are league-wide pitcher-season anchors
    stored in bundle["norms"]["arsenal"] (computed by compute_arsenal_norms.py).

    Parameters
    ----------
    scores : dict from _score_v5_arsenal — must contain "raw_pred",
             "raw_vs_rhb", "raw_vs_lhb" per pitch
    usage  : dict {pitch_group: usage_pct} (optional — uses MLB fallback)

    Returns
    -------
    dict with:
      arsenal_stuff_plus       — headline (vs LHB default for RHP)
      arsenal_stuff_plus_vs_rhb
      arsenal_stuff_plus_vs_lhb
      pitch_contributions      — per-pitch {usage_pct, raw, contribution_to_Y}
      method                   — "raw_aggregation" or "fallback_avg"
    """
    if not scores:
        return {}

    # Get arsenal norms from the loaded bundle
    arsenal_norms = ((_v5_bundle or {}).get("norms", {}).get("arsenal") if _V5_AVAILABLE else None)
    if not arsenal_norms or "mean" not in arsenal_norms or "sd" not in arsenal_norms:
        # Bundle predates arsenal_norms — fall back to simple average
        sp_vals = [s["stuff_plus"] for s in scores.values() if "stuff_plus" in s]
        avg = sum(sp_vals) / len(sp_vals) if sp_vals else 100.0
        return {
            "arsenal_stuff_plus": round(avg, 1),
            "arsenal_stuff_plus_vs_rhb": None,
            "arsenal_stuff_plus_vs_lhb": None,
            "pitch_contributions": {},
            "method": "fallback_avg",
            "note": "bundle missing norms.arsenal — run compute_arsenal_norms.py",
        }

    mu = float(arsenal_norms["mean"])
    sd = max(float(arsenal_norms["sd"]), 1e-6)

    # Resolve usage % per pitch (fallback to MLB-typical median if blank)
    usage = usage or {}
    weights = {}
    for grp in scores.keys():
        u = usage.get(grp)
        if u is None or (isinstance(u, float) and (u != u)):  # NaN or missing
            u = _MLB_USAGE_FALLBACK.get(grp, 5.0)
        weights[grp] = max(float(u), 0.01)
    w_sum = sum(weights.values())
    norm_weights = {g: w / w_sum for g, w in weights.items()}

    # Compute Y_arsenal for each platoon
    def _aggregate(raw_key: str):
        Y = 0.0
        contribs = {}
        for grp, data in scores.items():
            raw = data.get(raw_key)
            if raw is None:
                continue
            w = norm_weights[grp]
            contrib = raw * w
            Y += contrib
            contribs[grp] = {"usage_pct": weights[grp], "raw": raw,
                             "contribution_to_Y": contrib}
        sp = 100.0 + (Y - mu) / sd * 10.0
        return round(sp, 1), Y, contribs

    sp_default, Y_default, contribs = _aggregate("raw_pred")
    sp_rhb = sp_lhb = None
    if all(scores[g].get("raw_vs_rhb") is not None for g in scores):
        sp_rhb, _, _ = _aggregate("raw_vs_rhb")
    if all(scores[g].get("raw_vs_lhb") is not None for g in scores):
        sp_lhb, _, _ = _aggregate("raw_vs_lhb")

    return {
        "arsenal_stuff_plus":         sp_default,
        "arsenal_stuff_plus_vs_rhb":  sp_rhb,
        "arsenal_stuff_plus_vs_lhb":  sp_lhb,
        "pitch_contributions":        contribs,
        "method":                     "raw_aggregation",
        "Y_arsenal":                  round(Y_default, 4),
        "norms_used":                 {"mean": mu, "sd": sd},
        "weights":                    norm_weights,
    }


# ── Zone-conditional Stuff+ scorer ────────────────────────────────────────
_ALL_ZONES = list(range(1, 10)) + list(range(11, 27))


# Zone v5 trained on Statcast native zones (1-9 in-zone, 11-14 outer quadrants).
# The app's heatmap renders a 5×5 grid (zones 1-26). Map each app-zone to its
# v5 equivalent + provide accurate spatial coordinates so the model can use
# its plate_x_norm / plate_z_norm features (rank 2 and 8 importance).
# Coordinates in feet (Statcast convention): plate_x ∈ [-1.5, 1.5], plate_z ∈ [0.5, 4.5].
# normalized: plate_x_norm = plate_x / 0.83, plate_z_norm = (plate_z - 2.5) / 1.0
_ZONE_TO_V5_AND_COORDS = {
    # In-zone 3x3 (1-9): pass through, use inside grid centers
    1:  (1, -0.55,  3.20),  2:  (2, 0.0,  3.20),  3:  (3, 0.55,  3.20),
    4:  (4, -0.55,  2.50),  5:  (5, 0.0,  2.50),  6:  (6, 0.55,  2.50),
    7:  (7, -0.55,  1.80),  8:  (8, 0.0,  1.80),  9:  (9, 0.55,  1.80),
    # Outer ring — top (above zone):
    11: (11, -1.30, 3.80),  12: (11, -0.55, 3.80),
    13: (11,  0.0,  3.80),  14: (12,  0.55, 3.80),  15: (12, 1.30, 3.80),
    # Outer ring — left side (top-to-bottom):
    16: (11, -1.30, 3.20),  17: (11, -1.30, 2.50),  18: (13, -1.30, 1.80),
    # Outer ring — right side (top-to-bottom):
    19: (12,  1.30, 3.20),  20: (12,  1.30, 2.50),  21: (14,  1.30, 1.80),
    # Outer ring — bottom (below zone):
    22: (13, -1.30, 1.20),  23: (13, -0.55, 1.20),
    24: (13,  0.0,  1.20),  25: (14,  0.55, 1.20),  26: (14, 1.30, 1.20),
}


def _score_zone_grid(shape_row: dict, pitcher_hand: str = "R") -> dict:
    """Predict Stuff+ for every (zone, platoon) given a single pitch shape row.

    shape_row : dict with the same keys as `_score_v5_arsenal` builds per
                pitch (start_speed, ivb_in, hb_arm_in, vaa_aa, ..., plus
                arsenal-context features filled or NaN).
    pitcher_hand : "R" or "L".

    Returns dict {"vs_rhb": {1: 102.3, ...}, "vs_lhb": {1: 99.4, ...}}
            or None if zone model not loaded.
    """
    if not _ZONE_AVAILABLE:
        return None
    import numpy as _np
    import pandas as _pd

    FEATURES     = _zone_bundle["features"]
    CAT_FEATURES = _zone_bundle["cat_features"]
    NORMS        = _zone_bundle["norms"]
    model        = _zone_bundle["model"]
    per_type     = _zone_bundle["per_type_scalers"]
    fallback     = _zone_bundle["fallback_scaler"]
    GROUP_TO_INT = _zone_bundle["group_to_int"]

    # Impute NaN arsenal features using per-pitch-type medians from the bundle.
    # Without this, LightGBM routes NaN values to high-prediction branches,
    # inflating all zone scores by ~13 points for single-pitch calculator inputs.
    _arsenal_medians = _zone_bundle.get("arsenal_medians", {})
    _arsenal_feats   = _zone_bundle.get("arsenal_feats", [])
    if _arsenal_medians and _arsenal_feats:
        _pt_key = int(shape_row.get("pitch_type_int", -1))
        _med = _arsenal_medians.get(_pt_key, _arsenal_medians.get("overall", {}))
        shape_row = dict(shape_row)  # don't mutate caller's dict
        for _af in _arsenal_feats:
            val = shape_row.get(_af)
            if val is None or (isinstance(val, float) and _np.isnan(val)):
                shape_row[_af] = _med.get(_af, _np.nan)

    # Build 52 rows: 26 zones × 2 platoons.
    # For each app-zone (1-26), map to the v5-zone (1-14) for `zone_int` and
    # provide accurate plate_x/z normalized coords for v5's spatial features.
    # Count features (mean_balls, mean_strikes, pct_2k) use league-average
    # defaults so the model gives a count-neutral prediction.
    import datetime as _dt
    _default_season    = _dt.datetime.now().year
    _LEAGUE_MEAN_BALLS   = 1.05   # avg balls-at-time-of-pitch league-wide
    _LEAGUE_MEAN_STRIKES = 0.95   # avg strikes-at-time-of-pitch
    _LEAGUE_PCT_2K       = 0.30   # fraction of pitches thrown in 2-strike counts
    rows = []
    keys = []
    for plat_key, batter_hand in (("vs_rhb", "R"), ("vs_lhb", "L")):
        is_same_hand = 1 if pitcher_hand == batter_hand else 0
        for zone in _ALL_ZONES:
            r = dict(shape_row)
            # Map app-zone → v5 zone + zone-center coordinates
            v5_zone, _px, _pz = _ZONE_TO_V5_AND_COORDS.get(zone, (zone, 0.0, 2.5))
            r["zone_int"]      = int(v5_zone)
            r["is_same_hand"]  = is_same_hand
            # v5++ spatial features (rank 2 and 8 in importance)
            r["plate_x_norm"]  = _px / 0.83
            r["plate_z_norm"]  = (_pz - 2.5) / 1.0
            r["in_zone"]       = int(abs(r["plate_x_norm"]) <= 1.0 and
                                       abs(r["plate_z_norm"]) <= 1.0)
            # Shape × zone interactions — use the app-zone INDEX (1-26) so
            # interactions remain expressive across the full grid; the model
            # treated these as continuous features rather than categoricals.
            r["ivb_x_zone"]          = r.get("ivb_in", 0)           * zone
            r["hb_x_zone"]           = r.get("hb_arm_in", 0)        * zone
            r["velo_x_zone"]         = r.get("start_speed", 0)      * zone
            r["rel_height_x_zone"]   = r.get("rel_height", 0)       * zone
            r["vaa_aa_x_zone"]       = r.get("vaa_aa", 0)           * zone
            r["rel_side_x_zone"]     = r.get("rel_side_arm", 0)     * zone
            r["active_spin_x_zone"]  = r.get("active_spin_rate", 0) * zone
            # v5++ count features (count-neutral defaults)
            r["mean_balls"]   = _LEAGUE_MEAN_BALLS
            r["mean_strikes"] = _LEAGUE_MEAN_STRIKES
            r["pct_2k"]       = _LEAGUE_PCT_2K
            # v3 legacy features (filled if bundle includes them)
            r["n_pitches"] = 40.0
            r["season"]    = _default_season
            rows.append(r)
            keys.append((plat_key, zone))

    df = _pd.DataFrame(rows)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = _np.nan
    X = df[FEATURES]
    pt_int = df["pitch_type_int"].to_numpy()

    # Apply per-pitch-type scaling
    cont_idx = [i for i, c in enumerate(FEATURES) if c not in CAT_FEATURES]
    X_vals = X.values.astype(_np.float64)
    X_out = X_vals.copy()
    for grp_int in _np.unique(pt_int):
        m = pt_int == grp_int
        scaler = per_type.get(int(grp_int), fallback)
        sub_cont = X_vals[m][:, cont_idx]
        sub_scaled = scaler.transform(sub_cont)
        rows_idx = _np.where(m)[0]
        for i, col in enumerate(cont_idx):
            X_out[rows_idx, col] = sub_scaled[:, i]

    raw = model.predict(X_out)

    # Standardize to Stuff+ scale using per-type norms
    pt = int(shape_row.get("pitch_type_int", -1))
    group_name = None
    for gname, gint in GROUP_TO_INT.items():
        if gint == pt:
            group_name = gname
            break
    if group_name and group_name in NORMS.get("by_type", {}):
        params = NORMS["by_type"][group_name]
    else:
        params = NORMS.get("overall", {"mean": 0, "sd": 1})
    m_, s_ = params["mean"], max(params["sd"], 1e-6)
    stuff_plus = 100.0 + ((raw - m_) / s_) * 10.0

    out = {"vs_rhb": {}, "vs_lhb": {}}
    for (plat_key, zone), val in zip(keys, stuff_plus):
        out[plat_key][zone] = round(float(val), 1)
    return out


def _zone_color(stuff_plus_val: float, is_inside_zone: bool = True) -> str:
    """Map Stuff+ value to a heatmap color. Inside-zone cells are saturated;
    outside-zone cells are dimmed by 35% alpha overlay.
    """
    if stuff_plus_val is None:
        return "#1a2030"
    v = float(stuff_plus_val)
    # Smooth gradient from cold blue (low) to gold (high)
    # Anchor points: 80=cold, 100=neutral, 120=hot
    if v >= 130:
        rgb = (212, 168, 72)   # gold
    elif v >= 120:
        # 120-130 → gold-amber
        t = (v - 120) / 10
        rgb = (int(184 + (212-184)*t), int(148 + (168-148)*t), int(88 + (72-88)*t))
    elif v >= 110:
        # 110-120 → amber-tan
        t = (v - 110) / 10
        rgb = (int(160 + (184-160)*t), int(140 + (148-140)*t), int(110 + (88-110)*t))
    elif v >= 100:
        # 100-110 → neutral-tan
        t = (v - 100) / 10
        rgb = (int(120 + (160-120)*t), int(130 + (140-130)*t), int(140 + (110-140)*t))
    elif v >= 90:
        # 90-100 → cool blue-neutral
        t = (v - 90) / 10
        rgb = (int(90 + (120-90)*t), int(115 + (130-115)*t), int(155 + (140-155)*t))
    elif v >= 80:
        # 80-90 → blue-cool
        t = (v - 80) / 10
        rgb = (int(70 + (90-70)*t), int(100 + (115-100)*t), int(160 + (155-160)*t))
    else:
        rgb = (60, 90, 160)    # deep blue
    if not is_inside_zone:
        # Dim outside-zone cells by mixing with dark background
        rgb = tuple(int(c * 0.55) for c in rgb)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


# Layout: 5x5 grid. Each cell maps to a zone number (or None for the central
# strike-zone area which is already rendered by zones 1-9).
# Reading: row 0 is top of view (high pitches), col 0 is leftmost.
# This is the BATTER'S view, so for a RHB the inside is on the right side of
# the grid (col 4) and outside is on the left (col 0).
# We use the pitcher's perspective for the grid (matches Statcast convention):
# col 0 = catcher's view left = batter's right side.
_ZONE_GRID = [
    # row 0: above zone
    [11, 12, 13, 14, 15],
    # row 1: top of zone (gr=1)  — outer-left=16, inner=1,2,3, outer-right=19
    [16,  1,  2,  3, 19],
    # row 2: middle of zone (gr=2) — outer-left=17, inner=4,5,6, outer-right=20
    [17,  4,  5,  6, 20],
    # row 3: bottom of zone (gr=3) — outer-left=18, inner=7,8,9, outer-right=21
    [18,  7,  8,  9, 21],
    # row 4: below zone
    [22, 23, 24, 25, 26],
]
_INSIDE_ZONES = set(range(1, 10))  # 1-9


def _render_zone_heatmap_svg(zone_scores: dict, title: str,
                                width: int = 220, cell_size: int = 38,
                                zone_coverage: dict = None) -> str:
    """Build an inline SVG heatmap from {zone_int: stuff_plus} dict.

    zone_coverage: optional {zone_int: int} = training-sample count per cell.
        If provided, cells with low coverage are rendered with reduced opacity
        to signal lower confidence (v6++ improvement #8).

    Returns HTML string suitable for st.markdown(unsafe_allow_html=True).
    """
    # 5×5 grid + label area
    grid_w = cell_size * 5
    grid_h = cell_size * 5
    pad_top = 28      # title row
    pad_bottom = 4
    total_h = pad_top + grid_h + pad_bottom

    svg_parts = [
        f'<svg width="{grid_w}" height="{total_h}" '
        f'viewBox="0 0 {grid_w} {total_h}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="display:block;margin:0 auto">',
        # Title
        f'<text x="{grid_w//2}" y="16" font-family="JetBrains Mono,monospace" '
        f'font-size="11" font-weight="700" fill="#8aaab8" '
        f'letter-spacing="1.5" text-anchor="middle">{title}</text>',
    ]

    for ri, row in enumerate(_ZONE_GRID):
        for ci, zone in enumerate(row):
            x = ci * cell_size
            y = pad_top + ri * cell_size
            sp = zone_scores.get(zone) if zone_scores else None
            is_inside = zone in _INSIDE_ZONES
            fill = _zone_color(sp, is_inside)
            stroke = "#2a3a50" if is_inside else "#1a2a40"
            stroke_w = 1.0
            # v6++ #8: opacity based on training coverage of this cell
            opacity = 1.0
            if zone_coverage is not None:
                cov = zone_coverage.get(zone, 0)
                if cov < 100:
                    opacity = 0.35
                elif cov < 500:
                    opacity = 0.65
                elif cov < 2000:
                    opacity = 0.85
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'opacity="{opacity}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_w}"/>'
            )
            if sp is not None:
                # Display value, scaled font
                txt_color = "#0a1018" if (sp >= 105 and is_inside) else \
                              ("#d4dae0" if is_inside else "#8a98a8")
                font_w = 700 if is_inside else 500
                svg_parts.append(
                    f'<text x="{x + cell_size//2}" y="{y + cell_size//2 + 4}" '
                    f'font-family="Inter,sans-serif" font-size="12" '
                    f'font-weight="{font_w}" fill="{txt_color}" '
                    f'text-anchor="middle">{int(round(sp))}</text>'
                )

    # Strike zone boundary (between rows 1-3 and cols 1-3)
    sz_x = cell_size
    sz_y = pad_top + cell_size
    sz_w = cell_size * 3
    sz_h = cell_size * 3
    svg_parts.append(
        f'<rect x="{sz_x}" y="{sz_y}" width="{sz_w}" height="{sz_h}" '
        f'fill="none" stroke="#c0a878" stroke-width="2"/>'
    )

    svg_parts.append('</svg>')
    return "".join(svg_parts)


@st.cache_resource
def load_dmstuff_model():
    if _os.path.exists(_MODEL_PATH):
        try:
            return _joblib.load(_MODEL_PATH)  # sklearn Pipeline: RobustScaler + LightGBM
        except Exception:
            return None
    return None

_dm_model = load_dmstuff_model()

# Exact 10 features extracted from model binary
_DMSP_FEATURES = ["start_speed","spin_rate","extension","az","ax","x0","z0",
                  "speed_diff","az_diff","ax_diff"]

# RobustScaler medians from model binary — correct defaults for missing features
_DMSP_MEDIANS = {
    "start_speed": 89.8, "spin_rate": 2271.0, "extension": 6.36,
    "az": -23.51, "ax": -7.19, "x0": 1.67, "z0": 5.74,
    "speed_diff": -2.07, "az_diff": -3.23, "ax_diff": 4.17,
}

def score_dmstuff(pitches: dict, rel_height: float = 5.74,
                  rel_side: float = None, extension: float = 6.36,
                  hand: str = "R") -> dict:
    """Score entered pitches through DMStuff+ pipeline.
    Exact model medians used as defaults (extracted from RobustScaler binary).
    x0 sign: RHP = positive rel_side, LHP = negative.
    """
    if _dm_model is None:
        return {}
    # x0 = raw Statcast release_pos_x: RHP ≈ -2ft (right side), LHP ≈ +2ft (left side)
    # rel_side in our CSV is already in this coordinate system — pass through unchanged
    if rel_side is None:
        x0 = -1.89 if hand == "R" else 2.08  # hand-specific population means
    else:
        x0 = rel_side

    # Primary pitch = highest velo for differential features
    primary = max(pitches.items(), key=lambda kv: kv[1].get("velo", 0), default=(None, {}))
    p_speed = primary[1].get("velo") or _DMSP_MEDIANS["start_speed"]
    _p_az_raw = primary[1].get("az")
    _p_ax_raw = primary[1].get("ax")
    p_az    = float(_p_az_raw) if (_p_az_raw is not None and not (isinstance(_p_az_raw, float) and math.isnan(_p_az_raw))) else _DMSP_MEDIANS["az"]
    _p_ax_r = float(_p_ax_raw) if (_p_ax_raw is not None and not (isinstance(_p_ax_raw, float) and math.isnan(_p_ax_raw))) else _DMSP_MEDIANS["ax"]
    p_ax    = _p_ax_r if hand == "R" else -_p_ax_r

    rows, keys = [], []
    for grp, m in pitches.items():
        velo = m.get("velo")
        if velo is None:
            continue
        az   = m.get("az") if m.get("az") is not None else _DMSP_MEDIANS["az"]
        ax_r = m.get("ax") if m.get("ax") is not None else _DMSP_MEDIANS["ax"]
        # Normalize ax to arm-side negative convention (RHP as-is, flip LHP)
        ax   = ax_r if hand == "R" else -ax_r
        rows.append({
            "start_speed": velo,
            "spin_rate":   m.get("spin_rate") or _DMSP_MEDIANS["spin_rate"],
            "extension":   extension or _DMSP_MEDIANS["extension"],
            "az":          az,
            "ax":          ax,
            "x0":          x0,
            "z0":          rel_height or _DMSP_MEDIANS["z0"],
            "speed_diff":  velo - p_speed,
            "az_diff":     az   - p_az,
            "ax_diff":     ax   - p_ax,
        })
        keys.append(grp)
    if not rows:
        return {}
    import pandas as _pd
    X = _pd.DataFrame(rows)[_DMSP_FEATURES].fillna(0)
    try:
        xrv = _dm_model.predict(X) * 100  # model outputs xRV/pitch → xRV/100
    except Exception:
        return {}
    xrv_mean, xrv_sd = 0.0, 0.68
    dmsp = 100.0 - ((xrv - xrv_mean) / xrv_sd) * 10.0
    return {k: round(float(v), 1) for k, v in zip(keys, dmsp)}


# ── Load profiles AFTER first render so health check doesn't time out ─────────
@st.cache_data(show_spinner=False)
def load_profiles() -> pd.DataFrame:
    return pd.read_csv("pitcher_profiles.csv")

try:
    profiles = load_profiles()
    data_ok  = True
except FileNotFoundError:
    data_ok  = False
    profiles = None
# ── Load zone stats (optional — shows "—" gracefully if not yet built) ────────
@st.cache_data(show_spinner=False)
def load_zone_stats() -> pd.DataFrame:
    """Load pitch_zone_stats from the fastest available source.

    Priority: parquet (4 MB, ~0.2s) → gzip CSV (smaller than raw)
    → raw CSV (47 MB, slowest). Parquet is regenerated whenever the
    raw CSV is newer than the parquet file.
    """
    import os
    _PARQ = "pitch_zone_stats.parquet"
    _CSV  = "pitch_zone_stats.csv"
    _GZ   = "pitch_zone_stats.csv.gz"
    # If parquet exists and is at least as fresh as raw CSV → use it.
    if os.path.exists(_PARQ):
        try:
            if not os.path.exists(_CSV) or \
                    os.path.getmtime(_PARQ) >= os.path.getmtime(_CSV):
                return pd.read_parquet(_PARQ)
        except Exception as _e:
            import sys
            print(f"[load_zone_stats] parquet read failed, falling back: {_e}",
                  file=sys.stderr)
    # Otherwise read CSV (gz preferred) then write parquet for next time.
    if os.path.exists(_GZ):
        df = pd.read_csv(_GZ, compression="gzip")
    else:
        df = pd.read_csv(_CSV)
    try:
        df.to_parquet(_PARQ, engine="pyarrow", compression="snappy", index=False)
    except Exception:
        pass   # parquet write is a cache optimization; not fatal if it fails
    return df

try:
    zone_stats = load_zone_stats()
    zone_stats["zone"] = zone_stats["zone"].astype(int)
    # Pre-compute league-wide means and stds per stat for z-score coloring
    # Filter to stand='all' to avoid inflating league means from splits
    _zs_all = zone_stats[zone_stats["stand"] == "all"] if "stand" in zone_stats.columns else zone_stats
    league_csw   = (_zs_all.groupby("zone")["csw_pct"].agg(["mean","std"])
                    .rename(columns={"mean":"csw_mu","std":"csw_sd"}))
    # Zone-level xwOBA baselines remain BIP-mean (these power per-zone heatmap
    # coloring; they show "if hit here, expected wOBA on contact").
    league_xwoba = (_zs_all.groupby("zone")["xwoba_mean"].agg(["mean","std"])
                    .rename(columns={"mean":"xw_mu","std":"xw_sd"}))
    if "whiff_pct" in _zs_all.columns:
        league_whiff = (_zs_all.groupby("zone")["whiff_pct"].agg(["mean","std"])
                        .rename(columns={"mean":"whiff_mu","std":"whiff_sd"}))
        zone_league = league_csw.join(league_xwoba).join(league_whiff)
    else:
        zone_league = league_csw.join(league_xwoba)
    # League avg CSW% and xwOBA per pitch group for card gradient coloring.
    # xwOBA here is per-PA (matches Savant); falls back to legacy BIP-weighted
    # if the CSV doesn't have pa_xwoba_sum/n_pa yet.
    _has_pa_cols = ("pa_xwoba_sum" in _zs_all.columns and "n_pa" in _zs_all.columns)
    if _has_pa_cols:
        _xw_grp = (_zs_all.groupby("pitch_group")
                   .apply(lambda g: g["pa_xwoba_sum"].sum() / max(g["n_pa"].sum(), 1),
                          include_groups=False)
                   .rename("xw_mu_w"))
    else:
        _xw_grp = (_zs_all.assign(xw_weighted=_zs_all["xwoba_mean"].fillna(0) * _zs_all["n_pitches"])
                   .groupby("pitch_group")
                   .apply(lambda g: g["xw_weighted"].sum() / max(g["n_pitches"].sum(), 1),
                          include_groups=False)
                   .rename("xw_mu_w"))
    pitch_grp_league = (
        _zs_all.groupby("pitch_group").agg(
            csw_mu=("csw_pct",    "mean"),
            csw_sd=("csw_pct",    "std"),
            xw_sd =("xwoba_mean", "std"),
        ).fillna(0)
        .join(_xw_grp)
        .rename(columns={"xw_mu_w": "xw_mu"})
    )
    zone_stats_ok = True
except (FileNotFoundError, Exception) as _zone_err:
    zone_stats        = pd.DataFrame()
    zone_league       = pd.DataFrame()
    pitch_grp_league  = pd.DataFrame()
    zone_stats_ok     = False


# ── Per-PA xwOBA helper (matches Statcast Savant) ─────────────────────────────
# pitch_zone_stats may have pa_xwoba_sum/n_pa from the v3 build; if not, fall
# back to the legacy BIP-weighted method (won't match Savant but won't crash).
_HAS_PA_XWOBA = (
    not zone_stats.empty
    and "pa_xwoba_sum" in zone_stats.columns
    and "n_pa" in zone_stats.columns
)

def per_pa_xwoba(df_subset):
    """Compute per-PA xwOBA from a zone_stats subset.
    Returns None if no PAs (rare; <=3-pitch arsenals).
    """
    if df_subset is None or df_subset.empty:
        return None
    if _HAS_PA_XWOBA:
        n_pa = df_subset["n_pa"].sum()
        if n_pa > 0:
            return float(df_subset["pa_xwoba_sum"].sum() / n_pa)
        return None
    # Legacy fallback: BIP-mean weighted by n_pitches (the old buggy method
    # — kept only so app doesn't crash on stale CSVs).
    total_n = df_subset["n_pitches"].sum()
    if total_n > 0:
        return float((df_subset["xwoba_mean"].fillna(0) * df_subset["n_pitches"]).sum() / total_n)
    return None


# ── VAA/HAA per-pitch-group league baselines (computed from profiles CSV) ─────
# profiles may not have vaa_/haa_ cols until rebuilt — degrade gracefully
# VAA/HAA baselines keyed by (pitch_group, hand) — HAA differs significantly by hand
# RHP 4-seam HAA ≈ -1.3, LHP 4-seam HAA ≈ +1.5 — must separate or coloring is wrong
_vaa_haa_league = {}
if data_ok and profiles is not None:
    for _grp in list(PITCH_GROUPS.keys()):
        _vc = f"vaa_{_grp}"
        _hc = f"haa_{_grp}"
        if _vc not in profiles.columns:
            continue
        # Store combined (for backward compat) AND per-hand baselines
        for _h_filter, _key in [(None, _grp), ("R", f"{_grp}_R"), ("L", f"{_grp}_L")]:
            _sub = profiles if _h_filter is None else profiles[profiles["hand"] == _h_filter]
            _vs = _sub[_vc].dropna()
            _hs = _sub[_hc].dropna() if _hc in profiles.columns else _vs[:0]
            if len(_vs) > 10:
                _vaa_haa_league[_key] = {
                    "vaa_mu": float(_vs.mean()), "vaa_sd": float(_vs.std()),
                    "haa_mu": float(_hs.mean()) if len(_hs) > 10 else 0.0,
                    "haa_sd": float(_hs.std())  if len(_hs) > 10 else 1.0,
                }


# ── Zone heatmap SVG renderer ─────────────────────────────────────────────────
# Zone layout (catcher's view):
#   Inside 1-9 (3x3, top-left = zone 1 = up-and-in from catcher's perspective)
#   Outside 10-17 (8 surrounding cells)
#
#   10 | 11 | 12
#   13 | 1  2  3 | 14
#      | 4  5  6 |
#   15 | 7  8  9 | 16 — wait, need to map correctly
#   15 | 16 | 17
#
# We render as a 5x5 grid where corners of outside are single cells:
# Row 0: [10][11][11][12] → actually 4 col header row
# Correct layout: 5 cols x 5 rows
# Col 0=left-out, 1=inner-left, 2=inner-mid, 3=inner-right, 4=right-out
# Row 0=top-out, 1=inner-top, 2=inner-mid, 3=inner-bot, 4=bot-out

# Inside-only 3×3 zone grid (zones 1-9, catcher's perspective)
# Zone numbering: 1=up-left  2=up-mid  3=up-right
#                 4=mid-left 5=center  6=mid-right
#                 7=dn-left  8=dn-mid  9=dn-right
# ── Zone grid layout (5×5 grid, PITCHER's perspective facing catcher) ────────
# Pitcher POV: glove side is on the LEFT of the display.
# For a RHP: arm side (right) = LEFT of image; glove side (left) = RIGHT of image.
# This is the standard baseball zone diagram as seen from the mound.
# Outside pitch to a RHB = far left column (glove side of RHP).
#
# Grid coordinates: (zone_id, grid_row, grid_col) — 5×5 grid
#   grid_row 0=top-out, 1-3=inner rows, 4=bot-out
#   grid_col 0=pitcher's LEFT (arm side RHP / glove side LHP)
#            4=pitcher's RIGHT (glove side RHP / arm side LHP)
#
# Inside zones numbered 1-9 reading left-to-right from pitcher's view:
#   1=up-arm-side  2=up-mid  3=up-glove-side
#   4=mid-arm      5=center  6=mid-glove
#   7=dn-arm       8=dn-mid  9=dn-glove

INSIDE_ZONES = [
    (1,1,1),(2,1,2),(3,1,3),
    (4,2,1),(5,2,2),(6,2,3),
    (7,3,1),(8,3,2),(9,3,3),
]

# Outside zones: 16 cells — 5 top, 3 left, 3 right, 5 bottom
OUTSIDE_ZONES = [
    # top row (row 0) — 5 cells
    (11,0,0),(12,0,1),(13,0,2),(14,0,3),(15,0,4),
    # mid-left (col 0, arm-side RHP) rows 1-3
    (16,1,0),(17,2,0),(18,3,0),
    # mid-right (col 4, glove-side RHP) rows 1-3
    (19,1,4),(20,2,4),(21,3,4),
    # bottom row (row 4) — 5 cells
    (22,4,0),(23,4,1),(24,4,2),(25,4,3),(26,4,4),
]

ALL_ZONES = INSIDE_ZONES + OUTSIDE_ZONES

def _lerp_color(z_score, stat_type):
    """Blue (low) → Grey (avg) → Red (high). Invert t for stats where high=bad."""
    z = max(-2.5, min(2.5, z_score if z_score == z_score else 0))
    if stat_type in ("csw", "whiff"):
        t = (z + 2.5) / 5.0        # high CSW/whiff = good = red
    else:
        t = (-z + 2.5) / 5.0       # high xwOBA = bad = red
    # Blue=(30,80,220)  Grey=(120,130,140)  Red=(220,35,35)
    if t < 0.5:
        s = t * 2
        r = int(30  + (120 - 30)  * s)
        g = int(80  + (120 - 80)  * s)
        b = int(128 + (120 - 128) * s)
    else:
        s = (t - 0.5) * 2
        r = int(120 + (220 - 120) * s)
        g = int(120 + (35  - 120) * s)
        b = int(120 + (35  - 120) * s)
    return f"rgb({max(0,min(255,r))},{max(0,min(255,g))},{max(0,min(255,b))})"


def stat_gradient_color(val, mu, sd, invert=False):
    """Blue→grey→red gradient based on z-score. invert=True for stats where high=bad."""
    if val is None or (isinstance(val, float) and val != val) or sd == 0:
        return "#2a4a5a"
    z = max(-2.0, min(2.0, (val - mu) / max(sd, 0.001)))
    if invert:
        z = -z
    t = (z + 2.0) / 4.0
    if t < 0.5:
        s = t * 2
        r = int(30  + (120 - 30)  * s)
        g = int(80  + (130 - 80)  * s)
        b = int(220 + (140 - 220) * s)
    else:
        s = (t - 0.5) * 2
        r = int(120 + (220 - 120) * s)
        g = int(130 + (35  - 130) * s)
        b = int(140 + (35  - 140) * s)
    return f"rgb({max(0,min(255,r))},{max(0,min(255,g))},{max(0,min(255,b))})"


def render_zone_heatmap(pitcher_zone_df, stat_col, stat_type, title, fmt=".1%"):
    """
    Render a 5×5 full-zone strike zone heatmap as SVG.
    Center 3×3 = inside zones 1-9. Surrounding 16 cells = outside zones 11-26.
    Each cell colored by z-score vs league mean/sd.
    """
    CW, CH   = 40, 34    # cell size (inside and outside same size)
    PAD_TOP  = 16
    COLS     = 5
    ROWS     = 5
    TOTAL_W  = CW * COLS + 2
    TOTAL_H  = CH * ROWS + PAD_TOP + 2

    pdata = {}
    if not pitcher_zone_df.empty:
        for _, row in pitcher_zone_df.iterrows():
            try:
                pdata[int(row["zone"])] = row
            except (ValueError, KeyError):
                pass

    svg = (
        f"<svg viewBox='0 0 {TOTAL_W} {TOTAL_H}' "
        f"xmlns='http://www.w3.org/2000/svg' "
        f"style='width:100%;max-width:240px;display:block;margin:0 auto'>"
        f"<rect width='{TOTAL_W}' height='{TOTAL_H}' fill='#0a0e18' rx='6'/>"
        f"<text x='{TOTAL_W//2}' y='11' text-anchor='middle' "
        f"font-family='Inter,sans-serif' font-size='8' fill='#4a7090' "
        f"letter-spacing='0.5'>{title}</text>"
    )

    mu_col = {"csw_pct": "csw_mu", "xwoba_mean": "xw_mu", "whiff_pct": "whiff_mu"}.get(stat_col)
    sd_col = {"csw_pct": "csw_sd", "xwoba_mean": "xw_sd", "whiff_pct": "whiff_sd"}.get(stat_col)

    for (zone_id, gr, gc) in ALL_ZONES:
        # Pitcher's POV: flip horizontally so arm-side is on left
        x = (4 - gc) * CW + 1
        y = gr * CH + PAD_TOP
        is_inside = zone_id <= 9

        row_data = pdata.get(zone_id)
        val = None
        z   = 0.0
        if row_data is not None:
            raw = row_data.get(stat_col)
            if raw is not None and raw == raw:
                val = float(raw)
                if (mu_col and sd_col and not zone_league.empty
                        and zone_id in zone_league.index):
                    mu = zone_league.loc[zone_id, mu_col]
                    sd = zone_league.loc[zone_id, sd_col]
                    z  = (val - mu) / max(sd, 0.001) if (sd == sd and sd > 0) else 0.0

        if val is not None:
            fill = _lerp_color(z, stat_type)
        elif is_inside:
            fill = "#0e1828"
        else:
            fill = "#0a0e18"   # outer empty cells slightly darker

        txt_fill = "#000000" if val is not None else ("#1e3a5a" if is_inside else "#141e2e")

        display = (f"{val:.0%}" if fmt == ".1%" else f"{val:.3f}") if val is not None else "—"
        font_size = "10" if is_inside else "8"

        svg += (
            f"<rect x='{x}' y='{y}' width='{CW}' height='{CH}' "
            f"fill='{fill}' stroke='#141e2e' stroke-width='0.5'/>"
            f"<text x='{x + CW//2}' y='{y + CH//2 + 4}' "
            f"text-anchor='middle' font-family='Inter,sans-serif' "
            f"font-size='{font_size}' font-weight='700' fill='{txt_fill}'>{display}</text>"
        )

    # Strike zone border (center 3×3)
    inner_x = 1 * CW + 1
    inner_y = 1 * CH + PAD_TOP
    sz_w    = CW * 3
    sz_h    = CH * 3
    # Subtle tint behind the strike zone cells (drawn before border, behind text)
    svg += (
        f"<rect x='{inner_x}' y='{inner_y}' width='{sz_w}' height='{sz_h}' "
        f"fill='#d4a84806' rx='1'/>"
    )
    # Bold strike zone outline
    svg += (
        f"<rect x='{inner_x}' y='{inner_y}' width='{sz_w}' height='{sz_h}' "
        f"fill='none' stroke='#d4a848' stroke-width='2' rx='1'/>"
    )
    # Corner tick marks for extra clarity
    _tk = 5  # tick length
    for cx, cy in [
        (inner_x,          inner_y),
        (inner_x + sz_w,   inner_y),
        (inner_x,          inner_y + sz_h),
        (inner_x + sz_w,   inner_y + sz_h),
    ]:
        _dx = _tk if cx == inner_x else -_tk
        _dy = _tk if cy == inner_y else -_tk
        svg += (
            f"<line x1='{cx}' y1='{cy}' x2='{cx+_dx}' y2='{cy}' "
            f"stroke='#d4a848' stroke-width='2'/>"
            f"<line x1='{cx}' y1='{cy}' x2='{cx}' y2='{cy+_dy}' "
            f"stroke='#d4a848' stroke-width='2'/>"
        )
    svg += "</svg>"
    return svg


def pitcher_zone_data(pitcher_name, year, pitch_group):
    """Look up zone stats for one pitcher-season-pitch combo (all batters)."""
    if not zone_stats_ok or zone_stats.empty:
        return pd.DataFrame()
    mask = (
        (zone_stats["player_name"] == pitcher_name) &
        (zone_stats["year"] == int(year)) &
        (zone_stats["pitch_group"] == pitch_group)
    )
    # Filter to stand='all' to avoid double-counting same+opp splits
    if "stand" in zone_stats.columns:
        mask &= (zone_stats["stand"] == "all")
    sub = zone_stats[mask].copy()
    if sub.empty:
        return sub
    sub = sub.set_index("zone").join(zone_league, how="left").reset_index()
    sub = sub.rename(columns={"index": "zone"})
    return sub


def overall_pitcher_zone_data(pitcher_name, year):
    """Zone stats aggregated across all pitch types for overall heatmap."""
    if not zone_stats_ok or zone_stats.empty:
        return pd.DataFrame()
    mask = (
        (zone_stats["player_name"] == pitcher_name) &
        (zone_stats["year"] == int(year))
    )
    if "stand" in zone_stats.columns:
        mask &= (zone_stats["stand"] == "all")
    sub = zone_stats[mask]
    if sub.empty:
        return sub
    # Weighted average for csw (weight by n_pitches), simple mean for others
    sub2 = sub.copy()
    sub2["csw_weighted"] = sub2["csw_pct"] * sub2["n_pitches"]
    agg = sub2.groupby("zone").agg(
        n_pitches    = ("n_pitches",    "sum"),
        csw_weighted = ("csw_weighted", "sum"),
        xwoba_mean   = ("xwoba_mean",   "mean"),
    ).reset_index()
    agg["csw_pct"] = agg["csw_weighted"] / agg["n_pitches"].clip(lower=1)
    agg = agg.drop(columns=["csw_weighted"])
    if not zone_league.empty:
        agg = agg.set_index("zone").join(zone_league, how="left").reset_index()
    return agg


def pitcher_zone_data_by_stand(pitcher_name, year, pitch_group, stand):
    """Zone stats for one pitcher filtered by batter handedness ('same','opp','all')."""
    if not zone_stats_ok or zone_stats.empty:
        return pd.DataFrame()
    mask = (
        (zone_stats["player_name"] == pitcher_name) &
        (zone_stats["year"]        == int(year)) &
        (zone_stats["pitch_group"] == pitch_group)
    )
    if "stand" in zone_stats.columns:
        mask &= (zone_stats["stand"] == stand)
    sub = zone_stats[mask].copy()
    if sub.empty:
        return sub
    sub = sub.set_index("zone").join(zone_league, how="left").reset_index()
    sub = sub.rename(columns={"index": "zone"})
    return sub


def overall_pitcher_zone_data_by_stand(pitcher_name, year, stand):
    """Overall zone stats (all pitch types) filtered by batter handedness."""
    if not zone_stats_ok or zone_stats.empty:
        return pd.DataFrame()
    mask = (
        (zone_stats["player_name"] == pitcher_name) &
        (zone_stats["year"]        == int(year))
    )
    if "stand" in zone_stats.columns:
        mask &= (zone_stats["stand"] == stand)
    sub = zone_stats[mask]
    if sub.empty:
        return sub
    sub2 = sub.copy()
    sub2["csw_weighted"]   = sub2["csw_pct"] * sub2["n_pitches"]
    has_whiff = "whiff_pct" in sub2.columns
    if has_whiff:
        sub2["whiff_weighted"] = sub2["whiff_pct"] * sub2["n_pitches"]
    agg_dict = dict(
        n_pitches    = ("n_pitches",    "sum"),
        csw_weighted = ("csw_weighted", "sum"),
        xwoba_mean   = ("xwoba_mean",   "mean"),
    )
    if has_whiff:
        agg_dict["whiff_weighted"] = ("whiff_weighted", "sum")
    agg = sub2.groupby("zone").agg(**agg_dict).reset_index()
    agg["csw_pct"] = agg["csw_weighted"] / agg["n_pitches"].clip(lower=1)
    agg = agg.drop(columns=["csw_weighted"])
    if has_whiff:
        agg["whiff_pct"] = agg["whiff_weighted"] / agg["n_pitches"].clip(lower=1)
        agg = agg.drop(columns=["whiff_weighted"])
    if not zone_league.empty:
        agg = agg.set_index("zone").join(zone_league, how="left").reset_index()
    return agg


def comp_zone_data(results, pitch_group=None, stand="all"):
    """
    Build an average zone heatmap across all comp pitchers.
    pitch_group=None → overall all-pitch; stand='all'|'same'|'opp'.
    """
    if not zone_stats_ok or zone_stats.empty or not results:
        return pd.DataFrame()
    has_stand = "stand" in zone_stats.columns
    frames = []
    for r in results:
        name = r["Pitcher"]
        year = int(r["Year"])
        if pitch_group:
            if has_stand and stand != "all":
                df = pitcher_zone_data_by_stand(name, year, pitch_group, stand)
            else:
                df = pitcher_zone_data(name, year, pitch_group)
        else:
            if has_stand and stand != "all":
                df = overall_pitcher_zone_data_by_stand(name, year, stand)
            else:
                df = overall_pitcher_zone_data(name, year)
        if not df.empty:
            # Include whiff_pct if available
            keep_cols = ["zone", "csw_pct", "xwoba_mean", "n_pitches"]
            if "whiff_pct" in df.columns:
                keep_cols.append("whiff_pct")
            frames.append(df[keep_cols])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Weighted averages
    combined["csw_weighted"] = combined["csw_pct"] * combined["n_pitches"]
    has_whiff = "whiff_pct" in combined.columns
    if has_whiff:
        combined["whiff_weighted"] = combined["whiff_pct"] * combined["n_pitches"]
    agg_dict = dict(
        n_pitches    = ("n_pitches",    "sum"),
        csw_weighted = ("csw_weighted", "sum"),
        xwoba_mean   = ("xwoba_mean",   "mean"),
    )
    if has_whiff:
        agg_dict["whiff_weighted"] = ("whiff_weighted", "sum")
    agg = combined.groupby("zone").agg(**agg_dict).reset_index()
    agg["csw_pct"] = agg["csw_weighted"] / agg["n_pitches"].clip(lower=1)
    agg = agg.drop(columns=["csw_weighted"])
    if has_whiff:
        agg["whiff_pct"] = agg["whiff_weighted"] / agg["n_pitches"].clip(lower=1)
        agg = agg.drop(columns=["whiff_weighted"])
    # Require at least 30 total pitches across comp set per zone for reliability
    agg = agg[agg["n_pitches"] >= 30]
    if not zone_league.empty:
        agg = agg.set_index("zone").join(zone_league, how="left").reset_index()
    return agg


def comp_aggregate_stats(results, pitch_group=None):
    """
    Compute aggregate stats (velo, ivb, hb, vaa, haa, stuff+, csw, xwoba)
    across the comp set. pitch_group=None means overall; otherwise per-pitch-type.
    Returns a dict of stat -> (mean_value, n).
    """
    if not results:
        return {}
    vals = {k: [] for k in ["velo","ivb","hb","vaa","haa","stuff_plus","csw","whiff","xwoba"]}

    for r in results:
        row = r["_row"]
        grp = pitch_group or r.get("Matched Pitch")  # single-pitch uses matched type

        if pitch_group:
            # Arsenal mode: pull per-pitch metrics for this group
            velo = row.get(f"velo_{pitch_group}")
            ivb  = row.get(f"ivb_{pitch_group}")
            hb   = row.get(f"hb_{pitch_group}")
            vaa  = row.get(f"vaa_{pitch_group}")
            haa  = row.get(f"haa_{pitch_group}")
            sp   = row.get(f"sp_{pitch_group}")
        else:
            # Overall/single-pitch mode
            grp2 = r.get("Matched Pitch") if not pitch_group else pitch_group
            if grp2:
                velo = row.get(f"velo_{grp2}")
                ivb  = row.get(f"ivb_{grp2}")
                hb   = row.get(f"hb_{grp2}")
                vaa  = row.get(f"vaa_{grp2}")
                haa  = row.get(f"haa_{grp2}")
                sp   = row.get(f"sp_{grp2}")
            else:
                velo = ivb = hb = vaa = haa = sp = None

        hand_r = r.get("Hand", "R")
        def _safe(val):
            return val is not None and not (isinstance(val, float) and val != val)
        # Average in arm-side positive (TrackMan) display convention.
        # VAA: negate (Statcast positive → display negative).
        # HB: pfx_x is identical for same shape regardless of hand, so a single
        # negation converts pfx_x → arm-side positive for BOTH hands.
        vaa_display = (-float(vaa)) if _safe(vaa) else None
        hb_arm = (-float(hb)) if _safe(hb) else None
        for k, v in [("velo",velo),("ivb",ivb),("hb",hb_arm),("vaa",vaa_display),("haa",float(haa) if _safe(haa) else None),("stuff_plus",sp)]:
            if v is not None and not (isinstance(v, float) and v != v):
                vals[k].append(float(v))

        # CSW, Whiff%, and xwOBA from zone_stats
        if zone_stats_ok and not zone_stats.empty:
            mask = (zone_stats["player_name"] == r["Pitcher"]) & (zone_stats["year"] == int(r["Year"]))
            if pitch_group:
                mask &= (zone_stats["pitch_group"] == pitch_group)
            # Filter to stand=="all" if column exists to avoid double-counting splits
            if "stand" in zone_stats.columns:
                mask &= (zone_stats["stand"] == "all")
            sub = zone_stats[mask]
            if not sub.empty:
                n = sub["n_pitches"].sum()
                if n > 0:
                    vals["csw"].append((sub["csw_pct"] * sub["n_pitches"]).sum() / n)
                    if "whiff_pct" in sub.columns:
                        vals["whiff"].append((sub["whiff_pct"] * sub["n_pitches"]).sum() / n)
                # Per-PA xwOBA (matches Savant). Falls back to BIP-weighted-by-pitches
                # via per_pa_xwoba helper for older CSVs without n_pa columns.
                _xw = per_pa_xwoba(sub)
                if _xw is not None:
                    vals["xwoba"].append(_xw)

    result = {}
    for k, v in vals.items():
        if v:
            result[k] = (sum(v)/len(v), len(v))
    return result


if not data_ok:
    st.error("**`pitcher_profiles.csv` not found.** Run `build_profiles.py` locally, then commit the CSV to your repo.")
    st.stop()

yr_min = int(profiles["year"].min())
yr_max = int(profiles["year"].max())
st.markdown(
    f"<div class='status-bar'>"
    f"<span>✓ {profiles['year'].nunique()} SEASONS ({yr_min}–{yr_max})</span>"
    f"<span>·</span><span>{len(profiles):,} PITCHER-SEASONS</span>"
    f"<span>·</span><span>{profiles['player_name'].nunique():,} PITCHERS</span>"
    f"<span>·</span><span>⚡ INSTANT SEARCH</span>"
    f"</div>",
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_real(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))

def vn(v):
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

def pf(s):
    """Parse a text_input string to float, returning None if blank/invalid."""
    if s is None: return None
    s = str(s).strip()
    if s == "" or s == "-": return None
    try: return float(s)
    except ValueError: return None

def hb_to_csv(user_hb, pitcher_hand=None):
    """Convert user HB (arm-side positive, TrackMan convention) to Statcast pfx_x.
    Always negates: arm-side positive → pfx_x negative (toward 3B for arm-side pitches).
    Since pfx_x is identical for same-shape pitches regardless of hand
    (RHP slider pfx_x = +4.9, LHP slider pfx_x = +5.0), this single conversion
    correctly matches both RHP and LHP comps for any input.
    Hand arg kept for API compatibility but ignored.
    """
    if user_hb is None: return None
    return -float(user_hb)

def gaussian_sim(val_a, val_b, sigma):
    """
    Gaussian decay similarity: 1.0 when identical, falls off exponentially.
    sim = exp(-0.5 * ((a - b) / σ)²)
    At d=σ  → 0.607   (still a solid match)
    At d=2σ → 0.135   (noticeably worse)
    At d=3σ → 0.011   (essentially no match)
    """
    d = abs(val_a - val_b)
    return math.exp(-0.5 * (d / sigma) ** 2)

def gaussian_sim_asym(mv, val, sigma_down, sigma_up):
    """
    Asymmetric Gaussian decay for velocity.
    mv is comp velo, val is user velo.
    If comp throws harder (mv > val), we penalize aggressively -> use sigma_up
    If comp throws slower (mv < val), we penalize less -> use sigma_down
    """
    d = mv - val
    sigma = sigma_up if d > 0 else sigma_down
    return math.exp(-0.5 * (abs(d) / sigma) ** 2)

def sim_color(s):
    # Geometric mean model thresholds:
    # 80+ = very tight match (near-perfect across all dimensions)
    # 65+ = strong match (within ~1σ average)
    # 45+ = solid comp (some dimensions off)
    # <45  = loose match
    if s >= 80: return "#06d6a0"
    if s >= 65: return "#c9a84c"
    if s >= 45: return "#f4a261"
    return "#e06060"


# ── Dynamic velo weight (#5) ──────────────────────────────────────────────────

# ── STUFF+ — looked up from pitcher_profiles.csv (pre-baked by build_profiles.py)
# DM Stuff+ scores are pitcher-season level; stored as "stuff_plus" column in CSV.
# Per-pitch Stuff+ is not available from FanGraphs — we show overall pitcher Stuff+.

def stuff_color(s):
    """
    Vivid gradient for DM Stuff+ display.
    Scale: 100 = MLB avg, 15 pts = 1 SD. Clamps at ±2 SD (70 / 130).
    Pure saturated blue → grey → pure saturated red.
    Gets maximally vivid at the extremes, not washed out.
    """
    if s is None or (isinstance(s, float) and s != s):
        return "#2a4a5a"
    z = max(-2.0, min(2.0, (s - 100.0) / 15.0))
    t = (z + 2.0) / 4.0   # 0=worst, 1=best
    if t < 0.5:
        # Pure blue #0055ff → neutral grey #6a7a8a
        s2 = t * 2          # 0→1
        r = int(0   + (106 - 0)   * s2)
        g = int(85  + (122 - 85)  * s2)
        b = int(255 - (255 - 138) * s2)
    else:
        # Neutral grey #6a7a8a → pure red #ff2020
        s2 = (t - 0.5) * 2  # 0→1
        r = int(106 + (255 - 106) * s2)
        g = int(122 - (122 - 32)  * s2)
        b = int(138 - (138 - 32)  * s2)
    return f"rgb({max(0,min(255,r))},{max(0,min(255,g))},{max(0,min(255,b))})"

def stuff_grade_label(s):
    """Descriptive label for DM Stuff+ display."""
    if s is None:   return "—"
    if s >= 130:    return "Elite"
    if s >= 115:    return "Plus"
    if s >= 105:    return "Avg+"
    if s >= 95:     return "Avg"
    if s >= 85:     return "Below"
    return "Poor"

# DM Stuff+ per-pitch column names in pitcher_profiles.csv
# Built by build_profiles.py from FanGraphs type=36 leaderboard
FG_SP_COL = {
    "4-Seam":        "sp_4-Seam",
    "2-Seam/Sinker": "sp_2-Seam/Sinker",
    "Cutter":        "sp_Cutter",
    "Slider":        "sp_Slider",
    "Sweeper":       "sp_Sweeper",
    "Curveball":     "sp_Curveball",
    "Splitter":      "sp_Splitter",
    "Changeup":      "sp_Changeup",
}


def velo_sigma(user_velo):
    """
    For harder throwers, tighten the velo σ so 1 mph difference matters more.
    At 95 mph → σ=1.5 (standard), at 102+ mph → σ=0.8 (very tight).
    """
    if user_velo is None or user_velo <= VELO_BOOST_THRESHOLD:
        return SIGMA["velo"]
    frac = min(user_velo - VELO_BOOST_THRESHOLD, 7.0) / 7.0
    return SIGMA["velo"] - (SIGMA["velo"] - VELO_BOOST_MIN_SIGMA) * frac


# ── Similarity scoring — Gaussian decay model ────────────────────────────────
# Score = weighted geometric mean of per-dimension Gaussian similarities.
# Each dimension contributes: sim_d = exp(-0.5 * (delta/σ)²) ∈ [0,1]
# Final score = weighted average of all sim_d values × 100.
# Hand mismatch → score = 0 (hard filter).
# Missing MLB value → dimension skipped (not penalized).

def score_row(user, pitch_inputs, row):
    """
    Weighted geometric mean of per-dimension Gaussian similarities.

    Rules:
    1. Handedness mismatch → hard zero.
    2. If user entered pitch metrics, the MLB pitcher MUST have at least one
       matching pitch type. If zero pitch types match → hard zero.
    3. Missing pitch type (pitcher doesn't throw it) → near-zero sim (0.02),
       which strongly penalises pitchers lacking a pitch the user threw.
       This is much harsher than 0.4 — it ensures pitch-type coverage matters.
    4. The more pitch types match, the higher the score naturally, because
       each matched pitch contributes a real Gaussian sim vs near-zero.
    """
    # Hard filter: handedness
    if user.get("hand") and row["hand"] != user["hand"]:
        return 0.0

    # Hard filter: must have at least one matching pitch type (if user entered any)
    if pitch_inputs:
        matched_groups = [
            g for g in pitch_inputs
            if is_real(row.get(f"velo_{g}")) or
               any(is_real(row.get(f"velo_{a}")) for a in PITCH_ALIASES.get(g, []))
        ]
        if not matched_groups:
            return 0.0
        # Coverage ratio: how many of the user's pitches this pitcher has
        # Used to soften the missing-pitch penalty when 2+ pitches match
        n_matched  = len(matched_groups)
        n_total    = len(pitch_inputs)
        coverage   = n_matched / max(n_total, 1)  # 0.0 – 1.0
        # Missing pitch sim scales up with coverage: base 0.05, max 0.30 at full coverage
        # So a pitcher with 3/4 pitches gets sim=0.22 on the missing one vs 0.05 for 0/4
        missing_sim = 0.05 + 0.25 * coverage
    else:
        missing_sim = 0.05
        coverage    = 1.0

    log_sum = 0.0   # Σ w_d * ln(sim_d)
    total_w = 0.0   # Σ w_d

    # ── Release profile ────────────────────────────────────────────────────
    for key in ("rel_height", "rel_side", "extension"):
        val = user.get(key)
        if val is None:
            continue
        mv = row.get(key)
        if not is_real(mv):
            sim = 0.4   # missing release metric — moderate penalty
        else:
            # rel_side: compare in arm-side distance (abs). User input is arm-side
            # positive; profile rel_side is raw Statcast (RHP negative, LHP positive),
            # so abs() of both gives a hand-agnostic distance from the rubber centerline.
            if key == "rel_side":
                sim = gaussian_sim(abs(mv), abs(val), SIGMA[key])
            else:
                sim = gaussian_sim(mv, val, SIGMA[key])
        w = WEIGHTS[key]
        log_sum += w * math.log(max(sim, 1e-9))
        total_w += w

    # ── Per-pitch metrics ──────────────────────────────────────────────────
    for group, metrics in pitch_inputs.items():
        sv  = velo_sigma(metrics.get("velo"))
        has_pitch = is_real(row.get(f"velo_{group}"))

        # Cutter/Slider cross-search: if pitcher lacks this pitch,
        # check alias groups (e.g. Slider for Cutter) before penalizing
        alias_group = None
        if not has_pitch:
            for alias in PITCH_ALIASES.get(group, []):
                if is_real(row.get(f"velo_{alias}")):
                    alias_group = alias
                    has_pitch = True
                    break

        # Hard shape cutoff: if pitch exists but HB or iVB is more than
        # 6" or 5" off from user input, don't count it as a match
        if has_pitch:
            col_group = alias_group if alias_group else group
            user_hb_raw = metrics.get("hb")   # arm-side positive (TrackMan)
            user_ivb    = metrics.get("ivb")
            mv_hb       = row.get(f"hb_{col_group}")
            mv_ivb      = row.get(f"ivb_{col_group}")
            # Convert user HB (arm-side positive TrackMan) to Statcast pfx_x.
            # pfx_x is identical for same shape regardless of hand, so a single
            # negation works for both RHP and LHP comps.
            user_hb = hb_to_csv(user_hb_raw)
            if user_hb is not None and is_real(mv_hb):
                if abs(float(mv_hb) - float(user_hb)) > 6.0:
                    has_pitch = False
                    alias_group = None
            if has_pitch and user_ivb is not None and is_real(mv_ivb):
                if abs(float(mv_ivb) - float(user_ivb)) > 5.0:
                    has_pitch = False
                    alias_group = None

        for metric, sigma in [("ivb",  SIGMA["ivb"]),
                               ("hb",   SIGMA["hb"]),
                               ("velo", sv)]:
            val = metrics.get(metric)
            if val is None:
                continue
            # HB: compare in pfx_x convention (convert user's arm-side input).
            # iVB/velo: no sign convention difference, compare directly.
            if metric == "hb":
                val = hb_to_csv(val)
                if val is None:
                    continue
            if not has_pitch:
                # Pitcher doesn't throw this pitch type or any alias,
                # or shape is too far off — treat as missing.
                sim = missing_sim
            else:
                # Use alias column if primary missing
                col_group = alias_group if alias_group else group
                mv  = row.get(f"{metric}_{col_group}")
                if metric == "velo" and is_real(mv):
                    # Asymmetric velocity decay
                    sim = gaussian_sim_asym(mv, val, sigma * 1.5, sigma * 0.7)
                else:
                    sim = gaussian_sim(mv, val, sigma) if is_real(mv) else 0.4
            
            # Dynamic weighting: 2x for primary fastballs
            w = WEIGHTS.get(metric, 1.0)
            if group in ["4-Seam", "2-Seam/Sinker"]:
                w *= 2.0
            log_sum += w * math.log(max(sim, 1e-9))
            total_w += w

    # ── Arsenal Velocity Separation Scoring ───────────────────────────
    if pitch_inputs and len(pitch_inputs) > 1:
        primary_fb = "4-Seam" if "4-Seam" in pitch_inputs else ("2-Seam/Sinker" if "2-Seam/Sinker" in pitch_inputs else None)
        if primary_fb:
            u_fb_velo = pitch_inputs[primary_fb].get("velo")
            c_fb_velo = row.get(f"velo_{primary_fb}")
            if u_fb_velo and is_real(c_fb_velo):
                for group, metrics in pitch_inputs.items():
                    if group == primary_fb: continue
                    u_sec_velo = metrics.get("velo")
                    
                    alias_group = None
                    has_pitch = is_real(row.get(f"velo_{group}"))
                    if not has_pitch:
                        for alias in PITCH_ALIASES.get(group, []):
                            if is_real(row.get(f"velo_{alias}")):
                                alias_group = alias
                                has_pitch = True
                                break
                                
                    if has_pitch and u_sec_velo:
                        col_group = alias_group if alias_group else group
                        c_sec_velo = row.get(f"velo_{col_group}")
                        
                        u_diff = u_fb_velo - u_sec_velo
                        c_diff = c_fb_velo - c_sec_velo
                        # Compare the separations
                        sim = gaussian_sim(c_diff, u_diff, 2.0)
                        
                        w = 3.0 # Weight for velo separation
                        log_sum += w * math.log(max(sim, 1e-9))
                        total_w += w

    if total_w == 0:
        return 0.0

    return round(math.exp(log_sum / total_w) * 100, 1)


def sample_confidence(n_pitches):
    """
    Soft confidence multiplier based on pitch count sample size.
    f(n) = 1 - exp(-n / 300)
    At 100 pitches → 0.284  (small discount — valid pitcher season)
    At 300 pitches → 0.632  (moderate)
    At 500 pitches → 0.811  (good)
    At 1000+       → 0.965  (near full)
    At 2000+       → 0.999  (full)
    Halflife=300 is realistic — a full season starter throws 3000+,
    a reliever 200-400. We want 200+ to still rank well.
    """
    if n_pitches is None or not is_real(n_pitches) or n_pitches <= 0:
        return 0.70   # unknown sample — conservative default (~250 pitch equiv)
    return 1.0 - math.exp(-float(n_pitches) / 300.0)


# Precompute profile dicts once for fast lookup during search
@st.cache_data(show_spinner=False)
def _get_profile_dicts(profiles_hash: int) -> list:
    return [dict(row) for _, row in profiles.iterrows()]

def run_search(user, pitch_inputs, top_n):
    _pdicts = _get_profile_dicts(len(profiles))
    rows = []
    for r in _pdicts:
        s = score_row(user, pitch_inputs, r)
        if s > 0:
            # Apply sample-size confidence multiplier
            n = r.get("total_pitches")
            s = round(s * sample_confidence(n), 1)
        rows.append({
            "Similarity":    s,
            "Pitcher":       r["player_name"],
            "Year":          int(r["year"]),
            "Hand":          r["hand"],
            "Rel Height":    round(r["rel_height"], 2),
            "Rel Side":      round(abs(r["rel_side"]), 2),
            "Extension":     round(r["extension"],  2) if is_real(r.get("extension")) else None,
            "Total Pitches": int(r["total_pitches"]) if r.get("total_pitches") else 0,
            "_row":          dict(r),
        })
    return sorted(rows, key=lambda x: -x["Similarity"])[:top_n]


# ── TrackMan parser ───────────────────────────────────────────────────────────
def find_col(df_cols, candidates):
    """Find the first matching column name (case-insensitive)."""
    lc = {c.lower().replace(" ","").replace("_",""): c for c in df_cols}
    for cand in candidates:
        key = cand.lower().replace(" ","").replace("_","")
        if key in lc:
            return lc[key]
    return None

def sniff_data_source(file_bytes, filename) -> str:
    """Identify TrackMan / Rapsodo / Hawk-Eye from an uploaded file.

    Strategy by type:
      • CSV: examine column names. TrackMan uses CamelCase ("RelSpeed",
        "InducedVertBreak", "TaggedPitchType"); Statcast/Hawk-Eye uses
        snake_case ("release_speed", "pfx_x", "release_spin_rate");
        Rapsodo uses its own conventions ("Velocity Result", "VB", etc.).
      • PDF/Image: search the text for source brand strings.

    Returns one of the keys in _DATA_SOURCES, defaulting to
    "Hawk-Eye / Statcast" when the signal is ambiguous.
    """
    _fl = filename.lower()
    try:
        if _fl.endswith(".csv"):
            df_head = pd.read_csv(io.BytesIO(file_bytes), nrows=1)
            cols = {str(c).strip() for c in df_head.columns}
            cols_lower = {c.lower() for c in cols}
            # TrackMan distinctive columns
            tm_markers = {"RelSpeed", "InducedVertBreak", "HorzBreak",
                          "TaggedPitchType", "AutoPitchType", "PitchUID",
                          "VertRelAngle", "HorzRelAngle", "TmPitcherId"}
            if tm_markers & cols:
                return "TrackMan"
            # Statcast / Hawk-Eye distinctive columns (snake_case)
            statcast_markers = {"release_speed", "pfx_x", "pfx_z",
                                  "release_spin_rate", "release_pos_x"}
            if statcast_markers & cols_lower:
                return "Hawk-Eye / Statcast"
            # Rapsodo: includes "Velocity Result" / "VB" / "HB" / "Spin Rate"
            rapsodo_markers = {"Velocity Result", "Strike Zone Side",
                                 "Strike Zone Height"}
            if rapsodo_markers & cols:
                return "Rapsodo"
        elif _fl.endswith(".pdf"):
            # Quick text scan
            try:
                import re as _re
                text = ""
                try:
                    import pdfplumber as _pp
                    with _pp.open(io.BytesIO(file_bytes)) as pdf:
                        for page in pdf.pages[:2]:
                            t = page.extract_text() or ""
                            text += t
                except ImportError:
                    try:
                        from pypdf import PdfReader as _PdR
                        rr = _PdR(io.BytesIO(file_bytes))
                        for p in rr.pages[:2]:
                            text += p.extract_text() or ""
                    except ImportError:
                        text = ""
                tl = text.lower()
                if "trackman" in tl: return "TrackMan"
                if "rapsodo"  in tl: return "Rapsodo"
                if "hawk-eye" in tl or "hawkeye" in tl or "statcast" in tl:
                    return "Hawk-Eye / Statcast"
            except Exception:
                pass
        elif _fl.endswith((".jpg", ".jpeg", ".png")):
            # OCR the brand text — cheap re-OCR using tesseract via PIL
            try:
                import pytesseract as _pt
                from PIL import Image as _PImg
                img = _PImg.open(io.BytesIO(file_bytes))
                txt = _pt.image_to_string(img).lower()
                if "trackman" in txt: return "TrackMan"
                if "rapsodo"  in txt: return "Rapsodo"
            except Exception:
                pass
    except Exception:
        pass
    return "Hawk-Eye / Statcast"


def parse_trackman(file_bytes, filename) -> dict:
    """
    Parse a TrackMan CSV or PDF, return dict of
    {group: {velo, ivb, hb, extension, rel_height, rel_side, vaa, haa}}
    """
    results = {}

    # ── CSV ──────────────────────────────────────────────────────────────────
    if filename.lower().endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            return {"_error": str(e)}

        # Find pitch type column
        pt_col = find_col(df.columns, TM_COL_MAP["pitch_type"])
        if pt_col is None:
            return {"_error": "Could not find a pitch type column. Check that your CSV has AutoPitchType or PitchType."}

        # Find metric columns
        col_map = {}
        for metric, candidates in TM_COL_MAP.items():
            col_map[metric] = find_col(df.columns, candidates)

        # Group rows by pitch type
        df["_group"] = df[pt_col].astype(str).str.lower().str.strip().map(
            lambda x: TM_PITCH_MAP.get(x)
        )

        for group, gdf in df[df["_group"].notna()].groupby("_group"):
            entry = {}
            for metric, col in col_map.items():
                if col and col in gdf.columns:
                    vals = pd.to_numeric(gdf[col], errors="coerce").dropna()
                    if len(vals):
                        entry[metric] = round(vals.mean(), 2)
            if entry:
                results[group] = entry

    # ── PDF (text extraction via pdfplumber → pypdf fallback) ────────────────
    elif filename.lower().endswith(".pdf"):
        try:
            import re
            text_lines = []
            _pdf_lib = None

            try:
                import pdfplumber
                _pdf_lib = "pdfplumber"
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_lines.extend(t.split("\n"))
            except ImportError:
                pass

            if not text_lines:
                try:
                    from pypdf import PdfReader
                    _pdf_lib = "pypdf"
                    reader = PdfReader(io.BytesIO(file_bytes))
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text_lines.extend(t.split("\n"))
                except ImportError:
                    pass

            if not text_lines and _pdf_lib is None:
                return {"_error": "PDF parsing unavailable — neither pdfplumber nor pypdf is installed. Try uploading a PNG/JPG screenshot instead."}

            if not text_lines:
                return {"_error": "PDF contained no extractable text."}

            # Normalize Unicode minus signs (Tread PDFs use − U+2212)
            text_lines = [l.replace("\u2212", "-") for l in text_lines]

            # Remove thousand-separator commas from numbers like 2,346
            text_lines = [re.sub(r"(\d),(\d{3})\b", r"\1\2", l) for l in text_lines]

            # Match decimals and negatives; require decimal point for movement values
            num_re = re.compile(r"-?\d+\.\d+|-?\d{2,}")

            parsed_any = False
            for line in text_lines:
                ll = line.lower().strip()
                if not ll:
                    continue

                group = None
                for key, grp in TM_PITCH_MAP.items():
                    if key in ll:
                        group = grp
                        break
                if group is None:
                    continue

                nums = num_re.findall(line)
                floats = []
                for n in nums:
                    try: floats.append(float(n))
                    except ValueError: pass

                if len(floats) < 3:
                    continue

                # Tread column order: [PitchCount] AVG_VELO MAX_VELO VERT HB SPIN HT SIDE EXT
                # Drop pitch count (integer 1–50), drop spin rate (integer > 500)
                metric_nums = [
                    f for f in floats
                    if not (f == int(f) and 1 <= f <= 50 and "." not in str(f))  # not pitch count (integer only)
                    and not (f > 500)                        # not spin rate or any large number
                ]

                if len(metric_nums) < 2:
                    continue

                # First plausible velocity
                velo = next((f for f in metric_nums if 60 <= f <= 105), None)
                if velo is None:
                    continue
                vi = metric_nums.index(velo)

                # Movement values follow velo: next two plausible break values
                remaining = metric_nums[vi+1:]
                # Skip a second velo-range value (max velo)
                if remaining and 60 <= remaining[0] <= 105:
                    remaining = remaining[1:]

                ivb = remaining[0] if len(remaining) > 0 and abs(remaining[0]) <= 35 else None
                hb  = remaining[1] if len(remaining) > 1 and abs(remaining[1]) <= 35 else None

                # Release values: 3.5–8.5 ft
                release = [f for f in metric_nums if 3.5 <= f <= 8.5]

                entry = {"velo": velo}
                if ivb is not None: entry["ivb"] = ivb
                if hb  is not None: entry["hb"]  = hb
                if len(release) >= 1: entry["rel_height"] = release[0]
                if len(release) >= 2: entry["rel_side"]   = release[1]
                if len(release) >= 3: entry["extension"]  = release[2]

                results[group] = entry
                parsed_any = True

            if not parsed_any and text_lines:
                return {"_error": "Could not parse pitch data from PDF. Check that it is a TrackMan/Tread pitch metrics report."}

        except Exception as e:
            return {"_error": f"PDF parse error: {e}"}

    return results


# ── Image parser — Tesseract OCR (free, runs locally) ────────────────────────
_TREAD_COLS = ["avg velo","max velo","vert","horz","avg spin","height","side","extension"]

def _detect_tread_layout(text_lines):
    for line in text_lines:
        ll = line.lower()
        hits = sum(1 for col in _TREAD_COLS if col in ll)
        if hits >= 3:
            return True
        if "pitch metrics" in ll:
            return True
    return False


def _parse_tread_layout(text_lines):
    import re
    num_re = re.compile(r"(?<![A-Za-z\-])(-?\d+\.\d+|-?\d+)(?![A-Za-z\-])")
    results = {}
    for line in text_lines:
        ll = line.lower().strip()
        if not ll:
            continue
        group = None
        matched_key = ""
        for key, grp in TM_PITCH_MAP.items():
            if key in ll:
                group = grp
                matched_key = key
                break
        if group is None:
            continue
        line_stripped = re.sub(re.escape(matched_key), " ", ll, flags=re.IGNORECASE)
        line_stripped = re.sub(r"^[^0-9\-]+", " ", line_stripped)
        raw_nums = num_re.findall(line_stripped)
        floats = []
        for n in raw_nums:
            try:
                floats.append(float(n))
            except ValueError:
                pass
        if len(floats) < 3:
            continue
        filtered = []
        dropped_count = False
        for n, f in zip(raw_nums, floats):
            if not dropped_count and f == int(f) and 1 <= f <= 999 and "." not in n:
                dropped_count = True
                continue
            filtered.append(f)
        no_spin = [f for f in filtered if not (f > 500 and f == int(f))]
        spin    = next((f for f in filtered if f > 500 and f == int(f)), None)
        if len(no_spin) < 2:
            continue
        avg_velo  = no_spin[0] if len(no_spin) > 0 else None
        vert      = no_spin[2] if len(no_spin) > 2 else None
        horz      = no_spin[3] if len(no_spin) > 3 else None
        height    = no_spin[4] if len(no_spin) > 4 else None
        side      = no_spin[5] if len(no_spin) > 5 else None
        extension = no_spin[6] if len(no_spin) > 6 else None
        if avg_velo is None or not (50 <= avg_velo <= 110):
            continue
        entry = {"velo": avg_velo}
        if vert      is not None and abs(vert) <= 40:        entry["ivb"]        = vert
        if horz      is not None and abs(horz) <= 40:        entry["hb"]         = horz
        if height    is not None and 2.0 <= height <= 9.0:   entry["rel_height"] = height
        if side      is not None and abs(side) <= 5.0:       entry["rel_side"]   = side
        if extension is not None and 3.0 <= extension <= 9.0:entry["extension"]  = extension
        if spin      is not None:                            entry["spin_rate"]  = spin
        results[group] = entry
    return results


def parse_trackman_image(file_bytes: bytes, filename: str) -> dict:
    """Parse a TrackMan/Rapsodo screenshot using Tesseract OCR (free, no API)."""
    try:
        import pytesseract
        from PIL import Image, ImageFilter, ImageEnhance
    except ImportError:
        return {"_error": "Image parsing requires pytesseract and Pillow. Check requirements.txt and packages.txt."}
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        w, h = img.size
        if w < 1400:
            scale = 1400 / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img = img.convert("L")
        img = ImageEnhance.Contrast(img).enhance(2.5)
        img = img.filter(ImageFilter.SHARPEN)
        img = img.filter(ImageFilter.SHARPEN)
        raw_text = pytesseract.image_to_string(img, config="--psm 6")
    except Exception as e:
        return {"_error": f"OCR error: {e}. Make sure tesseract-ocr is in packages.txt."}
    if not raw_text.strip():
        return {"_error": "OCR returned no text. Try a clearer, higher-resolution screenshot."}
    import re
    text_lines = raw_text.split("\n")
    text_lines = [l.replace("\u2212", "-").replace("\u2013", "-") for l in text_lines]
    text_lines = [re.sub(r"(\d),(\d{3})\b", r"\1\2", l) for l in text_lines]
    if _detect_tread_layout(text_lines):
        results = _parse_tread_layout(text_lines)
        if results:
            return results
    # Heuristic fallback
    num_re = re.compile(r"-?\d+\.\d+|-?\d{2,}")
    results = {}
    parsed_any = False
    for line in text_lines:
        ll = line.lower().strip()
        if not ll:
            continue
        group = None
        for key, grp in TM_PITCH_MAP.items():
            if key in ll:
                group = grp
                break
        if group is None:
            continue
        nums = num_re.findall(line)
        floats = []
        for n in nums:
            try: floats.append(float(n))
            except ValueError: pass
        if len(floats) < 2:
            continue
        metric_nums = [f for f in floats
                       if not (f == int(f) and 1 <= f <= 50 and "." not in str(f))
                       and not (f > 500)]
        if len(metric_nums) < 2:
            continue
        velo = next((f for f in metric_nums if 60 <= f <= 105), None)
        if velo is None:
            continue
        vi = metric_nums.index(velo)
        remaining = metric_nums[vi + 1:]
        if remaining and 60 <= remaining[0] <= 105:
            remaining = remaining[1:]
        ivb = remaining[0] if len(remaining) > 0 and abs(remaining[0]) <= 35 else None
        hb  = remaining[1] if len(remaining) > 1 and abs(remaining[1]) <= 35 else None
        release = [f for f in metric_nums if 3.5 <= f <= 8.5]
        entry = {"velo": velo}
        if ivb is not None:   entry["ivb"]        = ivb
        if hb  is not None:   entry["hb"]         = hb
        if len(release) >= 1: entry["rel_height"]  = release[0]
        if len(release) >= 2: entry["rel_side"]    = release[1]
        if len(release) >= 3: entry["extension"]   = release[2]
        results[group] = entry
        parsed_any = True
    if not parsed_any:
        return {"_error": "Could not parse pitch data from image. Make sure it is a TrackMan or Rapsodo report with readable text."}
    return results


def run_search_single_pitch(user, velo, ivb, hb_csv, top_n, pitch_type_filter=None):
    """
    Single-pitch mode: compare one pitch's metrics against every individual
    pitch type in every pitcher-season profile. Returns the top N matches
    as (pitcher, year, matched_pitch_type, similarity_score).
    pitch_type_filter: if set, only scores against that specific pitch group.
    """
    sv = velo_sigma(velo)
    search_groups = [pitch_type_filter] if pitch_type_filter else list(PITCH_GROUPS.keys())
    # When no filter: score ALL pitch types per pitcher and return each as separate result
    # (not just the single best per pitcher), so results are ranked across pitch types
    multi_pitch_mode = (pitch_type_filter is None)
    rows = []
    _pdicts = _get_profile_dicts(len(profiles))
    for r in _pdicts:
        # Hard filter: handedness
        if user.get("hand") and r["hand"] != user["hand"]:
            continue
        # Score release profile
        log_sum = total_w = 0.0
        for key in ("rel_height", "rel_side", "extension"):
            val = user.get(key)
            if val is None:
                continue
            mv = r.get(key)
            if not is_real(mv):
                sim = 0.4
            else:
                # rel_side: compare in arm-side distance (abs). User input is arm-side
                # positive; profile rel_side is raw Statcast (RHP negative, LHP positive),
                # so abs() of both gives a hand-agnostic distance from the rubber.
                if key == "rel_side":
                    sim = gaussian_sim(abs(mv), abs(val), SIGMA[key])
                else:
                    sim = gaussian_sim(mv, val, SIGMA[key])
            w = WEIGHTS[key]
            log_sum += w * math.log(max(sim, 1e-9))
            total_w += w

        # Score against each pitch type individually — keep the best match
        best_pitch = None
        best_pitch_score = -1.0
        for group in search_groups:
            mv_velo = r.get(f"velo_{group}")
            mv_ivb  = r.get(f"ivb_{group}")
            mv_hb   = r.get(f"hb_{group}")
            if not is_real(mv_velo):
                continue
            # Hard shape cutoff: skip if iVB or HB is too far off from user input
            # Single-pitch mode uses tighter thresholds since pitch type is unknown
            _ivb_thresh = 4.0  # tighter — prevents sinker showing for 4-seam input
            _hb_thresh  = 5.0
            if ivb is not None and is_real(mv_ivb):
                if abs(float(mv_ivb) - float(ivb)) > _ivb_thresh:
                    continue
            # Convert user HB (arm-side positive TrackMan) to Statcast pfx_x.
            # pfx_x is identical for same shape regardless of hand, so a single
            # negation works for both RHP and LHP comps.
            _hb_use = hb_to_csv(hb_csv)
            if _hb_use is not None and is_real(mv_hb):
                if abs(float(mv_hb) - float(_hb_use)) > _hb_thresh:
                    continue
            # Score this pitch
            p_log = log_sum
            p_w   = total_w
            for metric, mv, sigma in [
                ("velo", mv_velo, sv),
                ("ivb",  mv_ivb,  SIGMA["ivb"]),
                ("hb",   mv_hb,   SIGMA["hb"]),
            ]:
                if metric == "velo" and velo is None:
                    continue
                if metric == "ivb"  and ivb  is None:
                    continue
                if metric == "hb"   and hb_csv is None:
                    continue
                user_val = {"velo": velo, "ivb": ivb, "hb": _hb_use}[metric]
                sim = gaussian_sim(mv, user_val, sigma) if is_real(mv) else 0.4
                w   = WEIGHTS.get(metric, 1.0)
                p_log += w * math.log(max(sim, 1e-9))
                p_w   += w
            if p_w == 0:
                continue
            score = math.exp(p_log / p_w) * 100
            if score > best_pitch_score:
                best_pitch_score = score
                best_pitch = group

        if best_pitch is None:
            continue

        n = r.get("total_pitches")
        conf = sample_confidence(n)
        ext_val = r.get("extension")

        if multi_pitch_mode:
            # Append ALL pitch types that passed shape cutoff, ordered by score
            for group in search_groups:
                mv_ivb_g = r.get(f"ivb_{group}")
                mv_hb_g  = r.get(f"hb_{group}")
                if not is_real(r.get(f"velo_{group}")):
                    continue
                # Apply same hard shape cutoff — tighter for single-pitch mode
                _ivb_thresh = 4.0
                _hb_thresh  = 5.0
                if ivb is not None and is_real(mv_ivb_g):
                    if abs(float(mv_ivb_g) - float(ivb)) > _ivb_thresh:
                        continue
                if _hb_use is not None and is_real(mv_hb_g):
                    if abs(float(mv_hb_g) - float(_hb_use)) > _hb_thresh:
                        continue
                # Re-score this specific pitch type
                p_log = log_sum
                p_w   = total_w
                for metric, mv, sigma in [
                    ("velo", r.get(f"velo_{group}"), sv),
                    ("ivb",  mv_ivb_g,               SIGMA["ivb"]),
                    ("hb",   mv_hb_g,                SIGMA["hb"]),
                ]:
                    user_val = {"velo": velo, "ivb": ivb, "hb": _hb_use}[metric]
                    if user_val is None:
                        continue
                    sim = gaussian_sim(mv, user_val, sigma) if is_real(mv) else 0.4
                    w   = WEIGHTS.get(metric, 1.0)
                    p_log += w * math.log(max(sim, 1e-9))
                    p_w   += w
                if p_w == 0:
                    continue
                score = round(math.exp(p_log / p_w) * 100 * conf, 1)
                if score < 20:
                    continue
                rows.append({
                    "Similarity":   score,
                    "Pitcher":      r["player_name"],
                    "Year":         int(r["year"]),
                    "Hand":         r["hand"],
                    "Rel Height":   r["rel_height"],
                    "Rel Side":     abs(r["rel_side"]),
                    "Extension":    float(ext_val) if is_real(ext_val) else None,
                    "Total Pitches":r.get("total_pitches"),
                    "Matched Pitch":group,
                    "_row":         dict(r),
                })
        else:
            final_score = round(best_pitch_score * conf, 1)
            rows.append({
                "Similarity":   final_score,
                "Pitcher":      r["player_name"],
                "Year":         int(r["year"]),
                "Hand":         r["hand"],
                "Rel Height":   r["rel_height"],
                "Rel Side":     abs(r["rel_side"]),
                "Extension":    float(ext_val) if is_real(ext_val) else None,
                "Total Pitches":r.get("total_pitches"),
                "Matched Pitch":best_pitch,
                "_row":         dict(r),
            })

    return sorted(rows, key=lambda x: -x["Similarity"])[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: TITLE — mode selection landing page
# ══════════════════════════════════════════════════════════════════════════════
# ── Loading screen — shown during search computation ──────────────────────────
if st.session_state.get("computing", False):
    _, lc, _ = st.columns([1, 4, 1])
    with lc:
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='text-align:center'>"
            "<div style='font-family:Rajdhani,sans-serif;font-size:28px;font-weight:700;"
            "color:#c9a84c;letter-spacing:3px;text-transform:uppercase;margin-bottom:20px'>"
            "⚾ Searching…</div>"
            "<div style='font-family:monospace;font-size:11px;color:#6a90a8'>"
            f"Scoring against {len(profiles):,} pitcher-seasons</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.stop()

elif st.session_state.screen == "title":

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

    # #13 audit: About / Help expander (top-right)
    _, _, _help_col = st.columns([6, 1, 1.4])
    with _help_col:
        with st.expander("ⓘ  About", expanded=False):
            st.markdown(
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                "color:#a0c0d4;line-height:1.7'>"
                f"<b style='color:{_BRAND_GOLD}'>Pitcher Similarity</b> trains a "
                "Gaussian-decay similarity model on every MLB pitcher-season from "
                f"{_DATA_YEAR_RANGE} Statcast data.<br><br>"
                f"<b style='color:{_BRAND_GOLD}'>Stuff+</b> is a per-pitch quality "
                "score (mean = 100, SD = 10). 115+ = elite, 95–105 = average. "
                "Computed via a LightGBM model on per-pitch shape features.<br><br>"
                f"<b style='color:{_BRAND_GOLD}'>Zone Stuff+</b> conditions Stuff+ "
                "on (zone, batter-stand) — heatmaps show the predicted score for "
                "every (zone, platoon) combination.<br><br>"
                f"<b style='color:{_BRAND_GOLD}'>Arsenal Grade</b> is the "
                "usage-weighted average Stuff+ across all entered pitches. "
                "A+ = 120+, A = 112+, B+ = 107+, B = 102+, C+ = 97+, C = 92+, D = below."
                "</div>",
                unsafe_allow_html=True,
            )

    _, tc, _ = st.columns([1, 6, 1])
    with tc:
        # Title
        st.markdown(
            "<div style='text-align:center;margin-bottom:40px'>"
            "<div style='font-family:Inter,sans-serif;font-size:13px;font-weight:600;"
            "color:#7aaac0;letter-spacing:6px;text-transform:uppercase;margin-bottom:12px'>"
            "STATCAST ANALYTICS</div>"
            "<div style='font-family:Inter,sans-serif;font-size:52px;font-weight:800;"
            "color:#e8e0d0;letter-spacing:2px;text-transform:uppercase;line-height:1.05'>"
            "Pitcher</div>"
            "<div style='font-family:Inter,sans-serif;font-size:52px;font-weight:800;"
            "background:linear-gradient(135deg,#d4a848,#e8c868,#d4a848);-webkit-background-clip:text;"
            "-webkit-text-fill-color:transparent;letter-spacing:2px;text-transform:uppercase;"
            "line-height:1.05'>Similarity</div>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6a90a8;"
            "margin-top:16px;letter-spacing:1.5px'>"
            "2017–2025 &nbsp;·&nbsp; Gaussian Scoring &nbsp;·&nbsp; Factor-Matched Comps"
            "</div></div>",
            unsafe_allow_html=True,
        )

        # Mode cards — pure Streamlit layout, styled with CSS around the buttons
        card_l, card_r = st.columns(2)

        with card_l:
            st.markdown(
                "<div style='background:linear-gradient(165deg,#0e1828 0%,#0c1420 100%);"
                "border:1px solid #d4a84830;border-radius:14px;"
                "padding:36px 28px 24px 28px;text-align:center;margin-bottom:4px;"
                "box-shadow:0 0 30px #d4a84808;transition:border-color 0.3s,box-shadow 0.3s'>"
                "<div style='font-size:40px;margin-bottom:14px'>⚾</div>"
                "<div style='font-family:Inter,sans-serif;font-size:18px;font-weight:800;"
                "color:#d4a848;letter-spacing:3px;text-transform:uppercase;margin-bottom:12px'>"
                "Full Arsenal</div>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a0c0d4;"
                "line-height:1.8;letter-spacing:0.3px'>"
                "Match an entire pitch mix to find your closest<br>"
                "MLB pitcher comp by arm slot, velocity,<br>"
                "and pitch shape across all pitches."
                "</div></div>",
                unsafe_allow_html=True,
            )
            if st.button("Enter Full Arsenal →", key="btn_arsenal", width='stretch'):
                st.session_state.mode   = "arsenal"
                st.session_state.screen = "input"
                st.rerun()

        with card_r:
            st.markdown(
                "<div style='background:linear-gradient(165deg,#0e1828 0%,#0c1420 100%);"
                "border:1px solid #3d6a8a30;border-radius:14px;"
                "padding:36px 28px 24px 28px;text-align:center;margin-bottom:4px;"
                "box-shadow:0 0 30px #3d6a8a08;transition:border-color 0.3s,box-shadow 0.3s'>"
                "<div style='font-size:40px;margin-bottom:14px'>🎯</div>"
                "<div style='font-family:Inter,sans-serif;font-size:18px;font-weight:800;"
                "color:#8aadcc;letter-spacing:3px;text-transform:uppercase;margin-bottom:12px'>"
                "Single Pitch</div>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a0c0d4;"
                "line-height:1.8;letter-spacing:0.3px'>"
                "Enter one pitch's metrics and find the<br>"
                "most similar individual pitches across<br>"
                "all MLB pitchers and pitch types."
                "</div></div>",
                unsafe_allow_html=True,
            )
            if st.button("Enter Single Pitch →", key="btn_single", width='stretch'):
                st.session_state.mode   = "single"
                st.session_state.screen = "input"
                st.rerun()

        # Bottom row — Leaderboard + DM Stuff+ Calculator side by side
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        card_bl, card_br = st.columns(2)
        with card_bl:
            st.markdown(
                "<div style='background:linear-gradient(165deg,#0e1828 0%,#0c1420 100%);"
                "border:1px solid #2a7a5a30;border-radius:14px;"
                "padding:28px 28px 20px 28px;text-align:center;margin-bottom:4px;"
                "box-shadow:0 0 30px #2a7a5a08'>"
                "<div style='font-size:36px;margin-bottom:12px'>📊</div>"
                "<div style='font-family:Inter,sans-serif;font-size:18px;font-weight:800;"
                "color:#5ac8a0;letter-spacing:3px;text-transform:uppercase;margin-bottom:10px'>"
                "Pitch Leaderboard</div>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a0c0d4;"
                "line-height:1.8;letter-spacing:0.3px'>"
                "Browse and filter every pitch type across all MLB pitchers — "
                "sortable by velo, movement, CSW%, xwOBA, and more."
                "</div></div>",
                unsafe_allow_html=True,
            )
            if st.button("Open Pitch Leaderboard →", key="btn_leaderboard", width='stretch'):
                st.session_state.screen = "leaderboard"
                st.rerun()

        with card_br:
            # Border tint depends on whether DM Stuff+ model is available
            _dm_avail = _V5_AVAILABLE
            _dm_border = "#c4914830" if _dm_avail else "#3a5a7830"
            _dm_color  = "#c49148" if _dm_avail else "#5a7a90"
            _dm_glow   = "#c4914808" if _dm_avail else "#3a5a7808"
            _dm_version = (_v5_bundle or {}).get("version", "model") if _dm_avail else "model"
            _dm_sub    = ("Compute your own pitch's Stuff+ score from any "
                          "combination of velo, movement, spin rate, and release "
                          "data — missing fields use league medians.") if _dm_avail else \
                         (f"DM Stuff+ model not deployed yet. Run training and "
                          f"add the bundle to enable this calculator.")
            st.markdown(
                f"<div style='background:linear-gradient(165deg,#0e1828 0%,#0c1420 100%);"
                f"border:1px solid {_dm_border};border-radius:14px;"
                f"padding:28px 28px 20px 28px;text-align:center;margin-bottom:4px;"
                f"box-shadow:0 0 30px {_dm_glow}'>"
                f"<div style='font-size:36px;margin-bottom:12px'>🧮</div>"
                f"<div style='font-family:Inter,sans-serif;font-size:18px;font-weight:800;"
                f"color:{_dm_color};letter-spacing:3px;text-transform:uppercase;margin-bottom:10px'>"
                f"DM Stuff+ Calc</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a0c0d4;"
                f"line-height:1.8;letter-spacing:0.3px'>"
                f"{_dm_sub}"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            if st.button("Open DM Stuff+ Calc →", key="btn_dmstuff_calc",
                         width='stretch', disabled=not _dm_avail):
                st.session_state.screen = "dmstuff"
                st.rerun()

    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: COMPUTING — clean loading screen during search
# ══════════════════════════════════════════════════════════════════════════════
# (computing flag handled inline — st.spinner covers it)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: LOADING — compute results, then flip to results screen
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "loading":
    snap = st.session_state.get("user_snapshot", {})
    mode = snap.get("mode", "arsenal")

    # Force scroll to top so the user sees the loading card, not whatever
    # was at the bottom of the previous screen. Streamlit preserves browser
    # scroll position across reruns by default.
    st.markdown(
        "<script>window.scrollTo({top: 0, behavior: 'instant'});</script>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    _, lc, _ = st.columns([1, 4, 1])
    with lc:
        st.markdown(
            "<div style='text-align:center;padding:32px;"
            "background:linear-gradient(165deg,#0e1828,#0a1218);"
            "border:1px solid #1a2a40;border-radius:16px'>"
            "<div style='font-size:36px;margin-bottom:16px'>⚾</div>"
            "<div style='font-family:Inter,sans-serif;font-size:18px;font-weight:800;"
            "color:#d4a848;letter-spacing:3px;text-transform:uppercase;margin-bottom:8px'>"
            "Finding Comps</div>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#4a6880;"
            "margin-bottom:24px;letter-spacing:1px'>"
            f"Scoring against {len(profiles):,} pitcher-seasons</div>"
            "<div style='display:flex;flex-direction:column;gap:8px;text-align:left;"
            "background:#0a0e16;border-radius:8px;padding:12px 16px'>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#d4a848;letter-spacing:0.5px'>▸ Loading pitcher profiles…</div>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#3a5a78;letter-spacing:0.5px'>▸ Applying Gaussian similarity model…</div>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#3a5a78;letter-spacing:0.5px'>▸ Computing DM Stuff+ scores…</div>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            "color:#3a5a78;letter-spacing:0.5px'>▸ Ranking matches…</div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        prog = st.progress(0, text="")
    prog.progress(30, text="Applying similarity model…")

    if mode == "arsenal":
        results = run_search(
            snap["user"], snap["pitch_inputs"], snap["top_n"]
        )
    else:
        results = run_search_single_pitch(
            snap["user"],
            snap.get("sp_velo"), snap.get("sp_ivb"), snap.get("sp_hb_csv"),
            snap["top_n"],
            pitch_type_filter=snap.get("sp_pitch_type"),
        )

    prog.progress(80, text="Computing DMStuff+…")

    # ── Score user's entered pitches through DMStuff+ model ──────────────────
    # az/ax inference strategy (best available, in priority order):
    #   1. Physics formula from velo+iVB+HB (Newton's equations, R²≈0.97)
    #   2. kNN cross-check: median az/ax of top-10 closest result matches
    #      per pitch type (uses real Statcast data from profiles after rebuild)
    #   3. Blend physics + kNN when both available, weight kNN 60/40
    _user_dmsp = {}
    if _dm_model is not None and data_ok:
        try:
            import math as _m

            def _infer_az_ax(velo, ivb_in, hb_in):
                """Physics-based az/ax from velo + iVB + HB.
                From Statcast trajectory model:
                  pfx = accel * t² / 2  where t = 55 / (velo * 1.467)
                  az = pfx_z * 2/t² - 32.174  (gravity correction)
                  ax = pfx_x * 2/t²           (no gravity horizontally)
                """
                if velo is None or velo <= 0:
                    return _DMSP_MEDIANS["az"], _DMSP_MEDIANS["ax"]
                t   = 55.0 / (velo * 1.467)
                t2h = t * t / 2.0
                az  = (ivb_in / 12.0) / t2h - 32.174 if ivb_in is not None else _DMSP_MEDIANS["az"]
                ax  = (hb_in  / 12.0) / t2h           if hb_in  is not None else _DMSP_MEDIANS["ax"]
                # Clamp to physically plausible range
                az = max(-55.0, min(5.0,  az))
                ax = max(-30.0, min(30.0, ax))
                return float(az), float(ax)

            def _knn_az_ax(grp, velo, ivb_in, hb_in, all_results, n=10):
                """Find n closest pitches of this type from ALL profiles using
                vectorized numpy — fast even on 5k+ rows.
                Returns (weighted_az, weighted_ax, mean_dist) or (None, None, None).
                """
                import numpy as _np
                az_col = f"az_{grp}"
                ax_col = f"ax_{grp}"
                v_col  = f"velo_{grp}"
                i_col  = f"ivb_{grp}"
                h_col  = f"hb_{grp}"
                if az_col not in profiles.columns:
                    return None, None, None
                # Vectorized: pull all four columns at once, drop NaN rows
                sub = profiles[[v_col, i_col, h_col, az_col, ax_col]].dropna()
                if sub.empty:
                    return None, None, None
                vv  = sub[v_col].values.astype(float)
                iv  = sub[i_col].values.astype(float)
                hv  = sub[h_col].values.astype(float)
                azv = sub[az_col].values.astype(float)
                axv = sub[ax_col].values.astype(float)
                # Euclidean distance in velo/iVB/HB space, normalised by SD
                dists = _np.sqrt(
                    ((velo   - vv) / 3.0) ** 2 +
                    ((ivb_in - iv) / 3.0) ** 2 +
                    ((hb_in  - hv) / 3.0) ** 2
                )
                # Top-n closest
                idx   = _np.argpartition(dists, min(n, len(dists)-1))[:n]
                top_d = dists[idx];  top_az = azv[idx];  top_ax = axv[idx]
                # Inverse-distance weighted average
                w = 1.0 / _np.maximum(top_d, 0.01)
                w /= w.sum()
                return float((w * top_az).sum()), float((w * top_ax).sum()), float(top_d.mean())

            def _best_az_ax(grp, velo, ivb_in, hb_in, all_results, hand="R"):
                """Blend physics (60%) + kNN (40%).
                hand: used to normalize kNN ax (flip LHP to match model convention).
                When kNN matches start to drift, reduce kNN weight toward 0%.
                """
                phys_az, phys_ax = _infer_az_ax(velo, ivb_in, hb_in)
                knn_az, knn_ax, mean_dist = _knn_az_ax(
                    grp, velo, ivb_in, hb_in, all_results
                )
                # Normalize kNN ax for hand — model expects arm-side negative convention
                if knn_ax is not None and hand == "L":
                    knn_ax = -knn_ax
                if knn_az is None:
                    return phys_az, phys_ax  # no az/ax cols yet — physics only

                # Distance-adaptive blending:
                # At dist=0 (perfect match): 60% physics, 40% kNN
                # At dist=4 (moderate drift): 80% physics, 20% kNN
                # At dist=8+ (poor match):    100% physics, 0% kNN
                knn_weight = max(0.0, 0.40 * (1.0 - mean_dist / 8.0))
                phys_weight = 1.0 - knn_weight
                az = phys_weight * phys_az + knn_weight * knn_az
                ax = phys_weight * phys_ax + knn_weight * knn_ax
                return float(az), float(ax)

            # Build pitch dict — kNN searches ALL profiles per pitch type internally
            _score_pitches = {}
            # League-average iVB/HB per pitch type as fallback when not entered
            # Used so physics formula gets a reasonable seed instead of 0
            _lg_ivb = {}
            _lg_hb  = {}
            for _g in ["4-Seam","2-Seam/Sinker","Cutter","Slider","Sweeper",
                       "Curveball","Splitter","Changeup","Knuckleball"]:
                _ic = f"ivb_{_g}"; _hc = f"hb_{_g}"
                if _ic in profiles.columns:
                    _v = profiles[_ic].dropna()
                    if len(_v): _lg_ivb[_g] = float(_v.mean())
                if _hc in profiles.columns:
                    _v = profiles[_hc].dropna()
                    if len(_v): _lg_hb[_g] = float(_v.mean())

            _hand = snap["user"].get("hand") or "R"

            if mode == "arsenal":
                for _grp, _pm in snap["pitch_inputs"].items():
                    if _pm.get("velo") is None:
                        continue
                    _velo = _pm["velo"]
                    # Use entered value if present, else league avg for pitch type
                    _ivb  = _pm.get("ivb")  if _pm.get("ivb")  is not None else _lg_ivb.get(_grp, 0.0)
                    _hb   = _pm.get("hb")   if _pm.get("hb")   is not None else _lg_hb.get(_grp,  0.0)
                    _az, _ax = _best_az_ax(_grp, _velo, _ivb, _hb, results, hand=_hand)
                    _score_pitches[_grp] = {
                        "velo": _velo, "az": _az, "ax": _ax, "spin_rate": _DMSP_MEDIANS["spin_rate"],
                    }
            else:
                _sp_type = snap.get("sp_pitch_type") or "4-Seam"
                _velo    = snap.get("sp_velo")
                if _velo is not None:
                    _ivb = snap.get("sp_ivb")    if snap.get("sp_ivb")    is not None else _lg_ivb.get(_sp_type, 0.0)
                    _hb  = snap.get("sp_hb_csv") if snap.get("sp_hb_csv") is not None else _lg_hb.get(_sp_type,  0.0)
                    _az, _ax = _best_az_ax(_sp_type, _velo, _ivb, _hb, results, hand=_hand)
                    _score_pitches[_sp_type] = {
                        "velo": _velo, "az": _az, "ax": _ax, "spin_rate": _DMSP_MEDIANS["spin_rate"],
                    }

            _rh  = snap["user"].get("rel_height") or _DMSP_MEDIANS["z0"]
            _rs  = snap["user"].get("rel_side") or (-1.89 if _hand == "R" else 2.08)
            _ext = snap["user"].get("extension")  or _DMSP_MEDIANS["extension"]

            if _score_pitches:
                _user_dmsp = score_dmstuff(
                    _score_pitches,
                    rel_height=_rh,
                    rel_side=_rs,
                    extension=_ext,
                    hand=_hand,
                )
        except Exception as _e:
            import traceback as _tb
            print(f"[DMStuff+ scoring error] {_e}\n{_tb.format_exc()}")
            _user_dmsp = {}

    prog.progress(100, text="Done!")
    st.session_state.results      = results
    st.session_state.user_dmsp    = _user_dmsp
    st.session_state.screen       = "results"
    st.rerun()


# SCREEN: INPUT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "input":

    # Guard: if results already computed and we somehow ended up here, redirect
    if st.session_state.get("results") is not None and st.session_state.screen == "input":
        st.session_state.screen = "results"
        st.rerun()

    mode = st.session_state.mode   # "arsenal" or "single"

    # ── Back to title ─────────────────────────────────────────────────────────
    if st.button("← Back", key="back_to_title"):
        st.session_state.screen = "title"
        st.session_state.pop("_arsenal_pitches", None)
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Page header ───────────────────────────────────────────────────────────
    if mode == "arsenal":
        hdr_label = "⚾  Compare Full Arsenal"
        hdr_sub   = "Leave any field blank = open filter &nbsp;·&nbsp; Fill only the pitches you throw"
    else:
        hdr_label = "🎯  Find Similar Pitches"
        hdr_sub   = "Enter one pitch's metrics — app searches across all pitchers and pitch types"

    st.markdown(
        f"<div style='text-align:center;max-width:680px;margin:0 auto 20px auto;padding:0 20px'>"
        f"<div style='font-family:Inter,sans-serif;font-size:22px;font-weight:700;"
        f"color:#d4a848;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px'>"
        f"{hdr_label}</div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#6a90a8'>{hdr_sub}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _, main_col, _ = st.columns([0.3, 11, 0.3])
    with main_col:

        # ── RELEASE PROFILE ────────────────────────────────────────────────
        st.markdown(
            "<div class='section-h'>● Release Profile</div>",
            unsafe_allow_html=True,
        )
        rp1, rp2, rp3, rp4, rp5 = st.columns([2, 2, 2, 2, 2])

        with rp1:
            st.markdown("<div class='field-label'>Throwing Hand</div>", unsafe_allow_html=True)
            hand_choice = st.radio("_hand", ["Any","RHP","LHP"], horizontal=True,
                                   index=0, key="hand_r", label_visibility="collapsed")
        with rp2:
            st.markdown("<div class='field-label'>Rel Height (ft)</div>", unsafe_allow_html=True)
            rel_height_v = st.number_input(" ", min_value=3.0, max_value=8.0,
                                            value=None, step=0.01, format="%.2f",
                                            placeholder=_PLACEHOLDER_REL_HEIGHT, key="rh",
                                            label_visibility="collapsed")
        with rp3:
            st.markdown("<div class='field-label'>Rel Side — arm side (ft)</div>", unsafe_allow_html=True)
            rel_side_v = st.number_input(" ", min_value=0.0, max_value=5.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder=_PLACEHOLDER_REL_SIDE,
                                          key="rs", label_visibility="collapsed")
        with rp4:
            st.markdown("<div class='field-label'>Extension (ft)</div>", unsafe_allow_html=True)
            extension_v = st.number_input(" ", min_value=4.0, max_value=8.0,
                                           value=None, step=0.01, format="%.2f",
                                           placeholder=_PLACEHOLDER_EXTENSION, key="ext",
                                           label_visibility="collapsed")
        with rp5:
            st.markdown("<div class='field-label'>Top N Results</div>", unsafe_allow_html=True)
            top_n = st.slider("_topn", 5, 50, 20, 5, key="topn",
                               label_visibility="collapsed")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── TRACKMAN UPLOAD ────────────────────────────────────────────────
        st.markdown(
            "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
            "color:#d4a848;letter-spacing:2px;text-transform:uppercase;"
            "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
            "● TrackMan / Rapsodo Auto-Fill "
            "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>(optional)</span></div>",
            unsafe_allow_html=True,
        )
        tm_file = st.file_uploader(
            "Upload TrackMan CSV, PDF, or screenshot (JPG/PNG) to auto-fill pitch metrics below",
            type=["csv","pdf","jpg","jpeg","png"], key="tm_upload",
            label_visibility="visible",
        )

        tm_data = {}
        if tm_file is not None:
            # Use file name+size as cache key to avoid re-parsing on every rerun
            file_id = f"{tm_file.name}_{tm_file.size}"
            if st.session_state.get("_tm_file_id") != file_id:
                # New file uploaded — parse and write into session state
                fname_lower = tm_file.name.lower()
                file_bytes  = tm_file.read()
                if fname_lower.endswith((".jpg", ".jpeg", ".png")):
                    with st.spinner("Reading image with OCR…"):
                        parsed = parse_trackman_image(file_bytes, tm_file.name)
                else:
                    parsed = parse_trackman(file_bytes, tm_file.name)
                if "_error" in parsed:
                    st.warning(f"TrackMan parse issue: {parsed['_error']}")
                    st.session_state["_tm_file_id"] = None
                else:
                    st.session_state["_tm_file_id"] = file_id
                    st.session_state["_tm_parsed"]  = parsed
                    # Write parsed values directly into widget session state keys
                    # so number_input picks them up immediately
                    for grp, vals in parsed.items():
                        key_prefix = f"a_{grp}"
                        if vals.get("velo") is not None:
                            st.session_state[f"{key_prefix}_velo"] = float(vals["velo"])
                        if vals.get("ivb") is not None:
                            st.session_state[f"{key_prefix}_ivb"]  = float(vals["ivb"])
                        if vals.get("hb") is not None:
                            st.session_state[f"{key_prefix}_hb"]   = float(vals["hb"])
                        # Release profile (use values from first pitch that has them)
                        if vals.get("rel_height") is not None and "rh" not in st.session_state:
                            st.session_state["rh"]  = float(vals["rel_height"])
                        if vals.get("rel_side") is not None and "rs" not in st.session_state:
                            st.session_state["rs"]  = float(vals["rel_side"])
                        if vals.get("extension") is not None and "ext" not in st.session_state:
                            st.session_state["ext"] = float(vals["extension"])

            tm_data = st.session_state.get("_tm_parsed", {})
            if tm_data:
                found = ", ".join(f"**{g}**" for g in tm_data)
                st.success(f"Parsed: {found} — metrics pre-filled below. Edit any value as needed.")
        else:
            # File removed — clear cached parse
            st.session_state["_tm_file_id"] = None
            st.session_state["_tm_parsed"]  = {}

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── PITCH INPUT — conditional on mode ───────────────────────────────
        hint_html = (
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a0c0d4;"
            "background:linear-gradient(165deg,#0e1828,#0c1420);border:1px solid #162236;"
            "border-left:3px solid #d4a84830;"
            "border-radius:10px;padding:10px 16px;margin-bottom:14px;letter-spacing:0.3px'>"
            "HB: arm-side positive (RHP sinker = +14, RHP slider = -5) &nbsp;·&nbsp; iVB: positive = rise"
            "</div>"
        )

        def pitch_inputs_widget(group, key_prefix, tm_data, show_placeholder=False):
            """Render velo/iVB/HB inputs for one pitch group. Returns (v, i, h_csv) or Nones.
            show_placeholder: show 'e.g.' hints (only for first pitch / single-pitch mode).
            """
            color   = PITCH_COLORS[group]
            tm_vals = tm_data.get(group, {})
            ph_v = ""
            ph_i = ""
            ph_h = ""
            st.markdown(
                f"<div class='pitch-card'>"
                f"<div class='pitch-card-title' style='color:{color}'>● {group}</div>",
                unsafe_allow_html=True,
            )
            _vc, _cc = st.columns([8, 1])
            with _cc:
                def _clear_widget():
                    st.session_state.pop(f"{key_prefix}_velo", None)
                    st.session_state.pop(f"{key_prefix}_ivb",  None)
                    st.session_state.pop(f"{key_prefix}_hb",   None)
                st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
                st.button("✕", key=f"_clr_{key_prefix}", on_click=_clear_widget,
                          help="Clear all fields")
            with _vc:
                st.markdown("<div class='field-label'>Velocity (mph)</div>", unsafe_allow_html=True)
                velo_def_s = f"{float(tm_vals['velo']):.1f}" if tm_vals.get("velo") is not None else ""
                velo_s = st.text_input(" ", value=velo_def_s, key=f"{key_prefix}_velo",
                                       label_visibility="collapsed")
                st.markdown("<div class='field-label'>iVB (in)</div>", unsafe_allow_html=True)
                ivb_def_s = f"{float(tm_vals['ivb']):.1f}" if tm_vals.get("ivb") is not None else ""
                ivb_s = st.text_input(" ", value=ivb_def_s, key=f"{key_prefix}_ivb",
                                      label_visibility="collapsed")
                st.markdown("<div class='field-label'>HB — arm-side + (in)</div>", unsafe_allow_html=True)
                hb_def_s = f"{float(tm_vals['hb']):.1f}" if tm_vals.get("hb") is not None else ""
                hb_s = st.text_input(" ", value=hb_def_s, key=f"{key_prefix}_hb",
                                     label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
            v, i, h = pf(velo_s), pf(ivb_s), pf(hb_s)
            return v, i, h  # HB stored as-entered (Statcast convention: RHP arm-side = negative)

        pitch_inputs_raw = {}
        sp_velo = sp_ivb = sp_hb_csv = None
        sp_pitch_type = None   # selected pitch type for single mode

        if mode == "arsenal":
            # ── FULL ARSENAL MODE ─────────────────────────────────────────
            st.markdown(
                "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                "color:#d4a848;letter-spacing:2px;text-transform:uppercase;"
                "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                "● Pitch Arsenal "
                "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>add the pitches you throw</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(hint_html, unsafe_allow_html=True)

            all_groups = list(PITCH_GROUPS.keys())

            # ── Session state: list of added pitch groups ─────────────────
            # Pre-populate from TrackMan parse if available
            if "_arsenal_pitches" not in st.session_state:
                st.session_state["_arsenal_pitches"] = []

            # If TrackMan data was just parsed, add those pitches automatically
            if tm_data:
                for grp in tm_data:
                    if grp in all_groups and grp not in st.session_state["_arsenal_pitches"]:
                        st.session_state["_arsenal_pitches"].append(grp)

            added = st.session_state["_arsenal_pitches"]
            remaining = [g for g in all_groups if g not in added]

            # ── Add Pitch row ─────────────────────────────────────────────
            def _on_add_pitch():
                choice = st.session_state.get("_add_pitch_sel", "")
                if choice and choice not in st.session_state["_arsenal_pitches"]:
                    st.session_state["_arsenal_pitches"].append(choice)
                # Reset back to placeholder via index — widget value is set
                # in the callback before the widget re-renders, so this is safe
                st.session_state["_add_pitch_sel"] = ""

            add_col, _ = st.columns([2, 5])
            with add_col:
                if remaining:
                    st.selectbox(
                        "_add_pitch",
                        options=[""] + remaining,
                        format_func=lambda x: "＋ Add a pitch…" if x == "" else f"● {x}",
                        key="_add_pitch_sel",
                        label_visibility="collapsed",
                        on_change=_on_add_pitch,
                    )
                else:
                    st.markdown(
                        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
"color:#3a5a78;padding:8px 0'>All pitch types added.</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # ── Render added pitch cards ──────────────────────────────────
            for group in list(added):  # list() so removal mid-loop is safe
                color = PITCH_COLORS[group]
                tm_vals = tm_data.get(group, {})

                # Card header with remove button
                card_hdr, remove_col = st.columns([8, 1])
                with card_hdr:
                    st.markdown(
                        f"<div style='font-family:Inter,sans-serif;font-size:12px;"
                        f"font-weight:700;color:{color};letter-spacing:2px;"
                        f"text-transform:uppercase;padding:6px 0 4px 0'>● {group}</div>",
                        unsafe_allow_html=True,
                    )
                with remove_col:
                    if st.button("✕", key=f"_remove_{group}", help=f"Remove {group}"):
                        st.session_state["_arsenal_pitches"].remove(group)
                        # Clear stored values for this pitch
                        for suffix in ["_velo", "_ivb", "_hb"]:
                            st.session_state.pop(f"a_{group}{suffix}", None)
                        st.rerun()

                # Three metric inputs side by side
                vc, ic, hc = st.columns(3)
                with vc:
                    st.markdown("<div class='field-label'>Velocity (mph)</div>", unsafe_allow_html=True)
                    velo_def_s = f"{float(tm_vals['velo']):.1f}" if tm_vals.get("velo") is not None else ""
                    velo_s = st.text_input(" ", value=velo_def_s, key=f"a_{group}_velo",
                                           label_visibility="collapsed")
                with ic:
                    st.markdown("<div class='field-label'>iVB (in)</div>", unsafe_allow_html=True)
                    ivb_def_s = f"{float(tm_vals['ivb']):.1f}" if tm_vals.get("ivb") is not None else ""
                    ivb_s = st.text_input(" ", value=ivb_def_s, key=f"a_{group}_ivb",
                                          label_visibility="collapsed")
                with hc:
                    st.markdown("<div class='field-label'>HB — arm-side + (in)</div>", unsafe_allow_html=True)
                    hb_def_s = f"{float(tm_vals['hb']):.1f}" if tm_vals.get("hb") is not None else ""
                    hb_s = st.text_input(" ", value=hb_def_s, key=f"a_{group}_hb",
                                         label_visibility="collapsed")

                st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                v, i, h = pf(velo_s), pf(ivb_s), pf(hb_s)
                if any(x is not None for x in [v, i, h]):
                    pitch_inputs_raw[group] = {"velo": v, "ivb": i, "hb": h}

            if not added:
                st.markdown(
                    "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
"color:#3a5a78;padding:4px 0 12px 0'>"
"Use the dropdown above to add your pitches.</div>",
                    unsafe_allow_html=True,
                )

        else:
            # ── SINGLE PITCH MODE ─────────────────────────────────────────
            st.markdown(
                "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                "color:#d4a848;letter-spacing:2px;text-transform:uppercase;"
                "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                "● Single Pitch "
                "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>enter your pitch metrics</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(hint_html, unsafe_allow_html=True)

            # Pitch type dropdown (optional — "All Pitches" searches everything)
            pt_options = ["All Pitches"] + list(PITCH_GROUPS.keys())
            st.markdown("<div class='field-label' style='margin-bottom:4px'>Search within pitch type (optional)</div>", unsafe_allow_html=True)
            sp_pitch_type_choice = st.selectbox(
                "_sp_pitch_type",
                options=pt_options,
                index=0,
                key="sp_pitch_type_sel",
                label_visibility="collapsed",
            )
            sp_pitch_type = None if sp_pitch_type_choice == "All Pitches" else sp_pitch_type_choice

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Single pitch card (use first TM pitch if available)
            first_tm = next(iter(tm_data.values()), {}) if tm_data else {}
            sp_col_w, _ = st.columns([3, 6])
            with sp_col_w:
                # Card title: show selected type or generic "Your Pitch"
                card_label = sp_pitch_type if sp_pitch_type else None
                card_color = PITCH_COLORS.get(sp_pitch_type, "#8aadcc") if sp_pitch_type else "#8aadcc"
                if card_label:
                    st.markdown(
                        f"<div class='pitch-card'>"
                        f"<div class='pitch-card-title' style='color:{card_color}'>● {card_label}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='pitch-card'>"
                        "<div class='pitch-card-title' style='color:#8aadcc'>● Your Pitch</div>",
                        unsafe_allow_html=True,
                    )
                # Velo / iVB / HB inputs (no group title — rendered above)
                tm_vals = first_tm
                _sp_vc, _sp_cc = st.columns([8, 1])
                with _sp_cc:
                    def _clear_sp():
                        st.session_state.pop("sp_velo", None)
                        st.session_state.pop("sp_ivb",  None)
                        st.session_state.pop("sp_hb",   None)
                    st.markdown("<div style='margin-top:18px'></div>", unsafe_allow_html=True)
                    st.button("✕", key="_clr_sp", on_click=_clear_sp, help="Clear all fields")
                with _sp_vc:
                    st.markdown("<div class='field-label'>Velocity (mph)</div>", unsafe_allow_html=True)
                    velo_def_s = f"{float(tm_vals['velo']):.1f}" if tm_vals.get("velo") is not None else ""
                    sp_velo_s = st.text_input(" ", value=velo_def_s, key="sp_velo",
                                       label_visibility="collapsed")
                    st.markdown("<div class='field-label'>iVB (in)</div>", unsafe_allow_html=True)
                    ivb_def_s = f"{float(tm_vals['ivb']):.1f}" if tm_vals.get("ivb") is not None else ""
                    sp_ivb_s = st.text_input(" ", value=ivb_def_s, key="sp_ivb",
                                      label_visibility="collapsed")
                    st.markdown("<div class='field-label'>HB — arm-side + (in)</div>", unsafe_allow_html=True)
                    hb_def_s = f"{float(tm_vals['hb']):.1f}" if tm_vals.get("hb") is not None else ""
                    sp_hb_s = st.text_input(" ", value=hb_def_s, key="sp_hb",
                                     label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
                sp_velo   = pf(sp_velo_s)
                sp_ivb    = pf(sp_ivb_s)
                sp_hb_csv = pf(sp_hb_s)  # HB as-entered (Statcast convention)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── RUN BUTTON ─────────────────────────────────────────────────────
        _, btn_col, _ = st.columns([3, 4, 3])
        with btn_col:
            st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)
            btn_label = "⚾  Find My MLB Comps" if mode == "arsenal" else "🎯  Find Similar Pitches"
            run = st.button(btn_label, key="run_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        if run:
            _hand_key = hand_choice[0] if hand_choice != "Any" else None
            # rel_side: user enters arm-side distance (positive). Scoring uses abs()
            # for hand-agnostic comparison, so no conversion needed here.
            _rs_csv = vn(rel_side_v)
            user = {
                "hand":       _hand_key,
                "rel_height": vn(rel_height_v),
                "rel_side":   _rs_csv,
                "extension":  vn(extension_v),
            }
            if mode == "arsenal":
                if not any(v is not None for v in user.values()) and not pitch_inputs_raw:
                    st.error("Enter at least one metric to search.")
                else:
                    st.session_state.user_snapshot = {
                        "user": user, "pitch_inputs": pitch_inputs_raw,
                        "top_n": top_n, "hand_label": hand_choice,
                        "mode": "arsenal",
                    }
                    st.session_state.screen = "loading"
                    st.rerun()
            else:
                if sp_velo is None and sp_ivb is None and sp_hb_csv is None and not any(v is not None for v in user.values()):
                    st.error("Enter at least one pitch metric to search.")
                else:
                    st.session_state.user_snapshot = {
                        "user": user, "pitch_inputs": {},
                        "top_n": top_n, "hand_label": hand_choice,
                        "mode": "single",
                        "sp_velo": sp_velo, "sp_ivb": sp_ivb, "sp_hb_csv": sp_hb_csv,
                        "sp_pitch_type": sp_pitch_type,
                    }
                    spinner_msg = (
                        f"Searching {sp_pitch_type} pitches…" if sp_pitch_type
                        else "Searching all pitch types…"
                    )
                    st.session_state.screen = "loading"
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "leaderboard":

    # Back button
    if st.button("← Back", key="lb_back"):
        st.session_state.screen = "title"
        st.rerun()

    st.markdown(
        "<div style='font-family:Inter,sans-serif;font-size:22px;font-weight:800;"
        "color:#5ac8a0;letter-spacing:3px;text-transform:uppercase;margin:8px 0 4px 0'>"
        "📊 Pitch Leaderboard</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7aaac0;"
        "margin-bottom:16px'>All pitch types · Statcast 2017–2025 · Click column headers to sort</div>",
        unsafe_allow_html=True,
    )

    # ── Build the flat pitch-level dataframe ─────────────────────────────────
    @st.cache_data(show_spinner=False)
    def build_leaderboard(profiles_hash: int) -> pd.DataFrame:
        """Melt profiles into one row per pitcher-season-pitch_type."""
        rows = []
        _pdicts_lb = _get_profile_dicts(len(profiles))
        for r in _pdicts_lb:
            name = r["player_name"]
            # Convert "Last, First" → "First Last"
            if "," in str(name):
                p = str(name).split(",", 1)
                display_name = f"{p[1].strip()} {p[0].strip()}"
            else:
                display_name = str(name)
            yr   = int(r["year"])
            hand = r["hand"]
            rh   = r.get("rel_height")
            rs   = r.get("rel_side")
            ext  = r.get("extension")

            for grp in PITCH_GROUPS:
                velo = r.get(f"velo_{grp}")
                if not is_real(velo):
                    continue
                ivb  = r.get(f"ivb_{grp}")
                hb   = r.get(f"hb_{grp}")
                # Flip HB sign for display (arm-side positive, TrackMan convention).
                # pfx_x is identical for same shape regardless of hand → single negation.
                if is_real(hb):
                    hb_disp = -float(hb)
                else:
                    hb_disp = None
                vaa  = r.get(f"vaa_{grp}")
                haa  = r.get(f"haa_{grp}")
                _sp_pg = r.get(f"sp_{grp}")
                sp = _sp_pg if (_sp_pg is not None and not (isinstance(_sp_pg, float) and math.isnan(_sp_pg))) else None

                # Pitch count: prefer n_{grp} col, fallback to pct * total
                n_val = r.get(f"n_{grp}")
                if not is_real(n_val):
                    pct = r.get(f"pct_{grp}")
                    tot = r.get("total_pitches")
                    n_val = round(float(pct) * float(tot)) if is_real(pct) and is_real(tot) else None

                rows.append({
                    "Pitcher":     display_name,
                    "Year":        yr,
                    "Hand":        hand,
                    "Pitch":       grp,
                    "Velo":        round(float(velo), 1) if is_real(velo) else None,
                    "iVB":         round(float(ivb),  1) if is_real(ivb)  else None,
                    "HB":          round(hb_disp,     1) if hb_disp is not None else None,
                    "VAA":         round(-float(vaa), 1) if is_real(vaa)  else None,
                    "HAA":         round(float(haa),  1) if is_real(haa)  else None,
                    "DM Stuff+":      round(float(sp),   0) if is_real(sp)   else None,
                    "Rel Ht":      round(float(rh),   2) if is_real(rh)   else None,
                    "Rel Side":    round(abs(float(rs)), 2) if is_real(rs)   else None,
                    "Extension":   round(float(ext),  2) if is_real(ext)  else None,
                    "N":           int(n_val) if n_val is not None else None,
                })

        lb = pd.DataFrame(rows)

        # Join zone stats for CSW% and xwOBA per pitcher-pitch
        if zone_stats_ok and not zone_stats.empty:
            zs = zone_stats.copy()
            if "stand" in zs.columns:
                zs = zs[zs["stand"] == "all"]
            zs["_csw_w"] = zs["csw_pct"] * zs["n_pitches"]
            agg_dict = dict(
                n_pitches    = ("n_pitches",  "sum"),
                csw_weighted = ("_csw_w",     "sum"),
            )
            # Per-PA xwOBA if available, else legacy BIP-mean
            if _HAS_PA_XWOBA:
                agg_dict["pa_xwoba_sum"] = ("pa_xwoba_sum", "sum")
                agg_dict["n_pa"]         = ("n_pa",         "sum")
            else:
                # Legacy: weight BIP-mean by total pitches (the old method)
                zs["_xw_w"] = zs["xwoba_mean"].fillna(0) * zs["n_pitches"]
                agg_dict["xw_weighted"] = ("_xw_w",     "sum")
            agg = zs.groupby(["player_name", "year", "pitch_group"]).agg(**agg_dict).reset_index()
            agg["CSW%"]  = (agg["csw_weighted"] / agg["n_pitches"].clip(lower=1) * 100).round(1)
            if _HAS_PA_XWOBA:
                agg["xwOBA"] = (agg["pa_xwoba_sum"] / agg["n_pa"].clip(lower=1)).round(3)
                # Hide xwOBA when there's no PA data (e.g. only 1-2 PAs in arsenal entry)
                agg.loc[agg["n_pa"] < 1, "xwOBA"] = float("nan")
            else:
                agg["xwOBA"] = (agg["xw_weighted"] / agg["n_pitches"].clip(lower=1)).round(3)
            if "whiff_pct" in zs.columns:
                # Use swing_count if available (proper Whiff% = whiffs/swings)
                # Fall back to n_pitches weighting (SwStr% = whiffs/pitches)
                if "swing_count" in zs.columns:
                    zs["_whiff_w"] = zs["whiff_pct"] * zs["swing_count"]
                    _w_denom = "swing_count"
                else:
                    zs["_whiff_w"] = zs["whiff_pct"] * zs["n_pitches"]
                    _w_denom = "n_pitches"
                wagg = zs.groupby(["player_name", "year", "pitch_group"]).agg(
                    whiff_wsum = ("_whiff_w",  "sum"),
                    n_total    = (_w_denom,    "sum"),
                ).reset_index()
                wagg["whiff_w"] = wagg["whiff_wsum"] / wagg["n_total"].clip(lower=1)
                agg = agg.merge(wagg, on=["player_name", "year", "pitch_group"], how="left")
                agg["Whiff%"] = (agg["whiff_w"] * 100).round(1)

            # Normalize pitcher name for join
            def _norm(n):
                if "," in str(n):
                    p = str(n).split(",",1)
                    return f"{p[1].strip()} {p[0].strip()}"
                return str(n)
            agg["Pitcher"] = agg["player_name"].apply(_norm)
            agg["Year"]    = agg["year"].astype(int)
            agg["Pitch"]   = agg["pitch_group"]
            merge_cols = ["Pitcher", "Year", "Pitch", "CSW%", "xwOBA"]
            if "Whiff%" in agg.columns:
                merge_cols.append("Whiff%")
            lb = lb.merge(agg[merge_cols], on=["Pitcher", "Year", "Pitch"], how="left")

        return lb

    # ── Read query param from HM button click ────────────────────────────────
    _qp = st.query_params.get("lbr")
    if _qp is not None:
        try:
            _new_idx = int(_qp)
            if _new_idx == st.session_state.get("_lb_hm_idx"):
                st.session_state["_lb_hm_idx"] = None
            else:
                st.session_state["_lb_hm_idx"] = _new_idx
        except (ValueError, TypeError):
            pass
        st.query_params.clear()
        st.rerun()

    lb_df = build_leaderboard(len(profiles))

    # Add Whiff% overall column if zone stats available (uses all-pitch denominator)
    if zone_stats_ok and not zone_stats.empty and "whiff_pct_overall" in zone_stats.columns:
        zs_all = zone_stats[zone_stats.get("stand","all") == "all"] if "stand" in zone_stats.columns else zone_stats
        wh_agg = zs_all.groupby(["player_name","year","pitch_group"]).agg(
            tw = ("n_pitches","sum"),
            ww = ("whiff_count","sum") if "whiff_count" in zs_all.columns else ("n_pitches","sum"),
        ).reset_index() if "whiff_count" in zs_all.columns else None

    # ── Filter controls ───────────────────────────────────────────────────────
    METRIC_COLS = ["Velo","iVB","HB","VAA","HAA","DM Stuff+","CSW%","Whiff%","xwOBA","Rel Ht","Rel Side","Extension","N"]

    # Header row: labels for dropdowns
    lbl1, lbl2, lbl3 = st.columns([3, 1, 1])
    with lbl1:
        st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px'>Pitch Type</div>", unsafe_allow_html=True)
    with lbl2:
        st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px'>Handedness</div>", unsafe_allow_html=True)
    with lbl3:
        st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px'>Season</div>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([3, 1, 1])
    with fc1:
        all_pitches = list(PITCH_GROUPS.keys())
        selected_pitches = st.multiselect(
            " ", options=all_pitches, default=all_pitches,
            key="lb_pitches", label_visibility="collapsed",
        )
        if not selected_pitches:
            selected_pitches = all_pitches
    with fc2:
        hand_filter = st.selectbox(" ", ["All","RHP","LHP"], key="lb_hand", label_visibility="collapsed")
    with fc3:
        year_options = ["All"] + sorted(lb_df["Year"].unique().tolist(), reverse=True)
        year_filter = st.selectbox(" ", year_options, key="lb_year", label_visibility="collapsed")









    # ── Build display columns ─────────────────────────────────────────────────
    view = lb_df.copy()
    filter_cols = [c for c in METRIC_COLS if c in view.columns]
    DISPLAY_COLS = ["Pitcher","Year","Hand","Pitch"] + filter_cols
    SORTABLE     = filter_cols  # only numeric cols are sortable + Pitcher

    # ── Apply pitch/hand/year filters ────────────────────────────────────────
    if selected_pitches:
        view = view[view["Pitch"].isin(selected_pitches)]
    if hand_filter == "RHP":
        view = view[view["Hand"] == "R"]
    elif hand_filter == "LHP":
        view = view[view["Hand"] == "L"]
    if year_filter != "All":
        view = view[view["Year"] == int(year_filter)]

    # ── Apply range filters from stable _lbf dict ────────────────────────────
    _lbf = st.session_state.get("_lbf", {})
    for col in filter_cols:
        mn = _lbf.get(f"min_{col}")
        mx = _lbf.get(f"max_{col}")
        if mn is not None and not (isinstance(mn, float) and mn != mn) and col in view.columns:
            view = view[view[col].fillna(-9999) >= float(mn)]
        if mx is not None and not (isinstance(mx, float) and mx != mx) and col in view.columns:
            view = view[view[col].fillna(9999) <= float(mx)]

    total_rows   = len(view)
    view_display = view.head(500)  # default; overwritten after sort buttons

    # ── Gradient baselines (overall, for non-VAA/HAA cols) ──────────────────
    lb_baselines = {}
    for col in filter_cols:
        if col in lb_df.columns:
            vals = lb_df[col].dropna()
            if len(vals) > 10:
                # VAA is negated for display; store positive mean so -val comparison works
                if col == "VAA":
                    lb_baselines[col] = (float((-vals).mean()), float(vals.std()))
                else:
                    lb_baselines[col] = (float(vals.mean()), float(vals.std()))

    # ── Per-pitch VAA/HAA baselines — blue=near avg, red=outlier either dir ──
    lb_vaa_haa = _vaa_haa_league  # backward compat for results screen

    @st.cache_data(show_spinner=False)
    def _pitch_hand_baselines(profiles_hash: int):
        """Per (pitch_group, hand) mean/sd for Velo, iVB, HB, VAA, HAA."""
        if profiles is None or profiles.empty:
            return {}
        out = {}
        for grp in PITCH_GROUPS:
            for hand_f, key_suffix in [("R", "_R"), ("L", "_L"), (None, "")]:
                sub  = profiles if hand_f is None else profiles[profiles["hand"] == hand_f]
                key  = f"{grp}{key_suffix}"
                entry = {}
                for metric, col_name in [
                    ("velo", f"velo_{grp}"),
                    ("ivb",  f"ivb_{grp}"),
                    ("hb",   f"hb_{grp}"),
                    ("vaa",  f"vaa_{grp}"),
                    ("haa",  f"haa_{grp}"),
                ]:
                    if col_name not in sub.columns:
                        continue
                    vals = sub[col_name].dropna()
                    if metric == "hb":
                        vals = -vals   # pfx_x → arm-side positive (both hands)
                    if len(vals) > 10:
                        entry[f"{metric}_mu"] = float(vals.mean())
                        entry[f"{metric}_sd"] = float(vals.std())
                if entry:
                    out[key] = entry
        return out

    lb_pitch_bl = _pitch_hand_baselines(len(profiles))

    # ── Handedness-normalized Rel Ht / Rel Side baselines ────────────────────
    # Rel Side: RHP releases arm-side (negative), LHP arm-side (positive) — very different means
    # Rel Height: similar mean for both hands but still normalize separately
    @st.cache_data(show_spinner=False)
    def _hand_release_baselines(profiles_hash: int):
        if profiles is None or profiles.empty:
            return {}
        out = {}
        for hand in ("R", "L"):
            sub = profiles[profiles["hand"] == hand]
            for col, key in [("rel_height","ht"), ("rel_side","side"), ("extension","ext")]:
                if col in sub.columns:
                    vals = sub[col].dropna()
                    # Use abs for rel_side since it's displayed as arm-side distance
                    if key == "side":
                        vals = vals.abs()
                    if len(vals) > 10:
                        out[f"{hand}_{key}"] = (float(vals.mean()), float(vals.std()))
        return out

    lb_release_baselines = _hand_release_baselines(len(profiles))

    def _outlier_color(val, mu, sd):
        """Blue (near average) → grey → red (outlier in either direction)."""
        if sd <= 0:
            return "#1a3050", "#d8cbb4"
        z = abs(val - mu) / sd
        t = min(z / 2.0, 1.0)   # 0 = average, 1 = 2+ SD outlier
        # Interpolate: blue #1e5080 → grey #787878 → red #dc2323
        if t < 0.5:
            f = t * 2.0
            r = int(30  + (120 - 30)  * f)
            g = int(80  + (120 - 80)  * f)
            b = int(128 + (120 - 128) * f)
        else:
            f = (t - 0.5) * 2.0
            r = int(120 + (220 - 120) * f)
            g = int(120 + (35  - 120) * f)
            b = int(120 + (35  - 120) * f)
        c = f"rgb({r},{g},{b})"
        lum = (0.299*r + 0.587*g + 0.114*b) / 255
        return c, "#000000" if lum > 0.45 else "#e8dcc8"

    def _directional_color(val, mu, sd, higher_is_better=True):
        """Same ramp as _outlier_color but directional: above avg = red, below = blue.
        Uses identical endpoints so it blends visually with all other columns."""
        if sd <= 0:
            return "#1a3050", "#d8cbb4"
        z = (val - mu) / sd
        if not higher_is_better:
            z = -z
        # Map signed z to t: z<-2 → t=0 (deep blue), z=0 → t=0.5 (grey), z>+2 → t=1 (red)
        t = max(0.0, min(1.0, (z + 2.0) / 4.0))
        if t < 0.5:
            f = t * 2.0
            r = int(30  + (120 - 30)  * f)
            g = int(80  + (120 - 80)  * f)
            b = int(128 + (120 - 128) * f)
        else:
            f = (t - 0.5) * 2.0
            r = int(120 + (220 - 120) * f)
            g = int(120 + (35  - 120) * f)
            b = int(120 + (35  - 120) * f)
        c = f"rgb({r},{g},{b})"
        lum = (0.299*r + 0.587*g + 0.114*b) / 255
        return c, "#000000" if lum > 0.45 else "#e8dcc8"

    def cell_color(col, val, pitch_type=None, hand=None):
        if val is None or (isinstance(val, float) and val != val):
            return "#141e2e", "#7aaac0"

        # Best-match baseline: pitch+hand → pitch-only → global
        _pb = None
        if pitch_type:
            _pb = (lb_pitch_bl.get(f"{pitch_type}_{hand}") or
                   lb_pitch_bl.get(pitch_type))

        # VAA — display value is negated; baselines store raw positive → compare -val
        if col == "VAA":
            if _pb and "vaa_mu" in _pb:
                return _outlier_color(-val, _pb["vaa_mu"], max(_pb["vaa_sd"], 0.001))
            if col in lb_baselines:
                mu, sd = lb_baselines[col]
                return _outlier_color(-val, mu, max(sd, 0.001))
            return "#141e2e", "#d8cbb4"

        # HAA — pitch+hand baseline (LHP/RHP have opposite sign means)
        if col == "HAA":
            if _pb and "haa_mu" in _pb:
                return _outlier_color(val, _pb["haa_mu"], max(_pb["haa_sd"], 0.001))
            if col in lb_baselines:
                mu, sd = lb_baselines[col]
                return _outlier_color(val, mu, max(sd, 0.001))
            return "#141e2e", "#d8cbb4"

        # Velo — pitch+hand (fastball ~93, changeup ~84, curve ~79)
        if col == "Velo" and _pb and "velo_mu" in _pb:
            return _outlier_color(val, _pb["velo_mu"], max(_pb["velo_sd"], 0.001))

        # iVB — pitch+hand (4-seam ~+15", curve ~-12")
        if col == "iVB" and _pb and "ivb_mu" in _pb:
            return _outlier_color(val, _pb["ivb_mu"], max(_pb["ivb_sd"], 0.001))

        # HB — pitch+hand, arm-side positive (already display-flipped in lb_df)
        if col == "HB" and _pb and "hb_mu" in _pb:
            return _outlier_color(val, _pb["hb_mu"], max(_pb["hb_sd"], 0.001))

        # Rel Ht / Rel Side / Extension — hand-only baseline
        if col == "Rel Ht" and hand:
            key = f"{hand}_ht"
            if key in lb_release_baselines:
                mu, sd = lb_release_baselines[key]
                return _outlier_color(val, mu, max(sd, 0.001))
        if col == "Rel Side" and hand:
            key = f"{hand}_side"
            if key in lb_release_baselines:
                mu, sd = lb_release_baselines[key]
                return _outlier_color(val, mu, max(sd, 0.001))
        if col == "Extension" and hand:
            key = f"{hand}_ext"
            if key in lb_release_baselines:
                mu, sd = lb_release_baselines[key]
                return _outlier_color(val, mu, max(sd, 0.001))

        # All other columns — global baseline
        if col not in lb_baselines:
            return "#141e2e", "#d8cbb4"
        mu, sd = lb_baselines[col]
        invert = col in ("xwOBA",)
        c = stat_gradient_color(val, mu, max(sd, 0.001), invert=invert)
        if c.startswith("rgb"):
            try:
                parts = c[4:-1].split(",")
                r_v, g_v, b_v = int(parts[0]), int(parts[1]), int(parts[2])
            except Exception:
                r_v, g_v, b_v = 20, 30, 40
        else:
            r_v, g_v, b_v = 20, 30, 40
        lum = (0.299*r_v + 0.587*g_v + 0.114*b_v) / 255
        return c, "#000000" if lum > 0.45 else "#e8dcc8"

    # ── CONTROL PANEL: SORT / MAX / MIN — all share same column widths ────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if "lb_sort_col" not in st.session_state:
        st.session_state["lb_sort_col"] = "Velo"
    if "lb_sort_asc" not in st.session_state:
        st.session_state["lb_sort_asc"] = False
    if "_lbf" not in st.session_state:
        st.session_state["_lbf"] = {}

    CTRL_COLS  = filter_cols
    label_w    = 0.7
    metric_w   = 1.0
    col_widths = [label_w] + [metric_w] * len(CTRL_COLS)

    # ── Row 1: SORT buttons ───────────────────────────────────────────────────
    row_sort = st.columns(col_widths)
    with row_sort[0]:
        st.markdown(
            "<div style='font-family:monospace;font-size:9px;color:#7aaac0;"
            "text-transform:uppercase;letter-spacing:1px;padding:6px 8px 0 0;"
            "text-align:right'>SORT</div>",
            unsafe_allow_html=True,
        )
    # Short display labels for cramped sort buttons. The underlying column
    # name stays the same; only the visible button text changes.
    SORT_LABEL = {
        "DM Stuff+": "Stuff+",
        "Extension": "Ext",
    }
    for j, col in enumerate(CTRL_COLS):
        with row_sort[j+1]:
            is_active = (st.session_state["lb_sort_col"] == col)
            arrow = (" ↓" if not st.session_state["lb_sort_asc"] else " ↑") if is_active else ""
            label = SORT_LABEL.get(col, col)
            if st.button(
                f"{label}{arrow}",
                key=f"_lbsort_{col}",
                type="primary" if is_active else "secondary",
                width='stretch',
            ):
                if st.session_state["lb_sort_col"] == col:
                    st.session_state["lb_sort_asc"] = not st.session_state["lb_sort_asc"]
                else:
                    st.session_state["lb_sort_col"] = col
                    st.session_state["lb_sort_asc"] = False
                st.session_state["_lb_hm_idx"] = None
                st.rerun()

    # ── Row 2: MAX inputs ─────────────────────────────────────────────────────
    def _cb_max(c):
        val = pf(st.session_state.get(f"_w_max_{c}", ""))
        if val is None:
            st.session_state["_lbf"].pop(f"max_{c}", None)
        else:
            st.session_state["_lbf"][f"max_{c}"] = val
    def _cb_min(c):
        val = pf(st.session_state.get(f"_w_min_{c}", ""))
        if val is None:
            st.session_state["_lbf"].pop(f"min_{c}", None)
        else:
            st.session_state["_lbf"][f"min_{c}"] = val
    def _cb_clear_max(c):
        st.session_state["_lbf"].pop(f"max_{c}", None)
        st.session_state.pop(f"_w_max_{c}", None)
    def _cb_clear_min(c):
        st.session_state["_lbf"].pop(f"min_{c}", None)
        st.session_state.pop(f"_w_min_{c}", None)

    # Check if any filters are active
    _any_filters = any(
        st.session_state["_lbf"].get(f"max_{c}") is not None or
        st.session_state["_lbf"].get(f"min_{c}") is not None
        for c in CTRL_COLS
    )

    # MAX row: label col + one sub-col per metric (input + clear x)
    row_max = st.columns(col_widths)
    with row_max[0]:
        st.markdown(
            "<div style='font-family:monospace;font-size:9px;color:#7aaac0;"
            "text-transform:uppercase;letter-spacing:1px;padding:4px 8px 0 0;"
            "text-align:right'>MAX</div>",
            unsafe_allow_html=True,
        )
    def _make_clr_max(c):
        def _fn():
            st.session_state["_lbf"].pop(f"max_{c}", None)
            st.session_state.pop(f"_w_max_{c}", None)
            st.session_state["_lb_hm_idx"] = None
        return _fn
    def _cb_max_num(c):
        gen = st.session_state.get("_lbf_gen", 0)
        v = st.session_state.get(f"_w_max_{c}_{gen}")
        if v is None:
            st.session_state["_lbf"].pop(f"max_{c}", None)
        else:
            st.session_state["_lbf"][f"max_{c}"] = float(v)
    for j, col in enumerate(CTRL_COLS):
        with row_max[j+1]:
            _cur_max = st.session_state["_lbf"].get(f"max_{col}")
            _gen = st.session_state.get("_lbf_gen", 0)
            st.number_input(
                " ", value=_cur_max, step=0.1, format="%.1f",
                key=f"_w_max_{col}_{_gen}", on_change=_cb_max_num, args=(col,),
                label_visibility="collapsed",
            )

    # MIN row
    row_min = st.columns(col_widths)
    with row_min[0]:
        st.markdown(
            "<div style='font-family:monospace;font-size:9px;color:#7aaac0;"
            "text-transform:uppercase;letter-spacing:1px;padding:4px 8px 0 0;"
            "text-align:right'>MIN</div>",
            unsafe_allow_html=True,
        )
    def _make_clr_min(c):
        def _fn():
            st.session_state["_lbf"].pop(f"min_{c}", None)
            st.session_state.pop(f"_w_min_{c}", None)
            st.session_state["_lb_hm_idx"] = None
        return _fn
    def _cb_min_num(c):
        gen = st.session_state.get("_lbf_gen", 0)
        v = st.session_state.get(f"_w_min_{c}_{gen}")
        if v is None:
            st.session_state["_lbf"].pop(f"min_{c}", None)
        else:
            st.session_state["_lbf"][f"min_{c}"] = float(v)
    for j, col in enumerate(CTRL_COLS):
        with row_min[j+1]:
            _cur_min = st.session_state["_lbf"].get(f"min_{col}")
            _gen = st.session_state.get("_lbf_gen", 0)
            st.number_input(
                " ", value=_cur_min, step=0.1, format="%.1f",
                key=f"_w_min_{col}_{_gen}", on_change=_cb_min_num, args=(col,),
                label_visibility="collapsed",
            )

    # ── Row: per-cell clear "×" buttons (only shown if that column has any filter) ──
    row_clr = st.columns(col_widths)
    with row_clr[0]:
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    for j, col in enumerate(CTRL_COLS):
        with row_clr[j+1]:
            has_max = st.session_state["_lbf"].get(f"max_{col}") is not None
            has_min = st.session_state["_lbf"].get(f"min_{col}") is not None
            if has_max or has_min:
                if st.button("×", key=f"_lbf_clr_{col}",
                             help=f"Clear {col} filter",
                             width='stretch'):
                    st.session_state["_lbf"].pop(f"max_{col}", None)
                    st.session_state["_lbf"].pop(f"min_{col}", None)
                    # Bump generation so this column's widgets reset
                    st.session_state["_lbf_gen"] = st.session_state.get("_lbf_gen", 0) + 1
                    st.session_state["_lb_hm_idx"] = None
                    st.rerun()
            else:
                st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

    # Status line + discreet "clear all" link only when filters are active
    _status_cols = st.columns([8, 1]) if _any_filters else [st.container(), None]
    with _status_cols[0]:
        st.markdown(
            f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
            f"color:#7aaac0;margin:6px 0 8px 0'>"
            f"{total_rows:,} pitches · showing top 500</div>",
            unsafe_allow_html=True,
        )
    if _any_filters and _status_cols[1]:
        with _status_cols[1]:
            st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
            if st.button("clear ×", key="_lbf_clear_all",
                         help="Clear all MIN/MAX filters",
                         width='stretch'):
                # Clear filter store
                for c in CTRL_COLS:
                    st.session_state["_lbf"].pop(f"max_{c}", None)
                    st.session_state["_lbf"].pop(f"min_{c}", None)
                # Bump generation so number_input widgets get fresh keys
                # (Streamlit ignores `value` arg after first render — only a key
                # change forces a clean reset.)
                st.session_state["_lbf_gen"] = st.session_state.get("_lbf_gen", 0) + 1
                st.session_state["_lb_hm_idx"] = None
                st.rerun()

    # Apply sort + cap to 500
    _sort_by  = st.session_state["lb_sort_col"]
    _sort_asc = st.session_state["lb_sort_asc"]
    if _sort_by in view.columns:
        view = view.sort_values(_sort_by, ascending=_sort_asc, na_position="last")
    view_display = view.head(500)

    # ── Heatmap viewer ────────────────────────────────────────────────────────
    cur_idx = st.session_state.get("_lb_hm_idx", -1)
    _vd_len = len(view_display) if view_display is not None else 0
    _hm_idx = cur_idx if isinstance(cur_idx, int) and 0 <= cur_idx < _vd_len else None

    if _hm_idx is not None:
        sel_row_data = view_display.iloc[_hm_idx]
        sel_pitcher  = sel_row_data["Pitcher"]
        sel_year     = int(sel_row_data["Year"])
        sel_pitch    = sel_row_data["Pitch"]
        sel_hand     = sel_row_data.get("Hand", "R")
        pc = PITCH_COLORS.get(sel_pitch, "#8aadcc")
        _hm_title_col, _hm_close_col = st.columns([10, 1])
        with _hm_title_col:
            st.markdown(
                f"<div style='font-family:Inter,sans-serif;font-size:12px;font-weight:700;"
                f"color:{pc};letter-spacing:1.5px;text-transform:uppercase;margin:8px 0 6px 0'>"
                f"● {sel_pitcher}  {sel_year}  —  {sel_pitch}</div>",
                unsafe_allow_html=True,
            )
        with _hm_close_col:
            if st.button("✕", key="_lb_hm_close", help="Close zone heatmaps"):
                st.session_state["_lb_hm_idx"] = None
                st.rerun()
        if not zone_stats_ok or zone_stats.empty:
            st.markdown(
                "<div style='font-family:monospace;font-size:10px;color:#7aaac0;"
                "padding:12px;background:#0c1420;border-radius:8px;border:1px solid #1a2a40'>"
                "⚠ Zone heatmaps require <b>pitch_zone_stats.csv.gz</b> — "
                "run <code>build_profiles.py</code> locally and commit.</div>",
                unsafe_allow_html=True,
            )
        else:
            def _csv_name(dn):
                parts = dn.rsplit(" ", 1)
                return f"{parts[1]}, {parts[0]}" if len(parts) == 2 else dn
            csv_nm = _csv_name(sel_pitcher)
            hm_data = pitcher_zone_data(csv_nm, sel_year, sel_pitch)
            if not hm_data.empty:
                hc1, hc2, hc3 = st.columns(3)
                with hc1: st.markdown(render_zone_heatmap(hm_data,"csw_pct","csw","CSW% (All)",fmt=".1%"), unsafe_allow_html=True)
                with hc2: st.markdown(render_zone_heatmap(hm_data,"whiff_pct","whiff","Whiff% (All)",fmt=".1%"), unsafe_allow_html=True)
                with hc3: st.markdown(render_zone_heatmap(hm_data,"xwoba_mean","xwoba","xwOBA (All)",fmt=".3f"), unsafe_allow_html=True)
                if "stand" in zone_stats.columns:
                    same_data = pitcher_zone_data_by_stand(csv_nm, sel_year, sel_pitch, "same")
                    opp_data  = pitcher_zone_data_by_stand(csv_nm, sel_year, sel_pitch, "opp")
                    if not same_data.empty:
                        st.markdown(f"<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:8px 0 4px 0'>vs {sel_hand}HB</div>", unsafe_allow_html=True)
                        hs1,hs2,hs3 = st.columns(3)
                        with hs1: st.markdown(render_zone_heatmap(same_data,"csw_pct","csw","CSW%",fmt=".1%"), unsafe_allow_html=True)
                        with hs2: st.markdown(render_zone_heatmap(same_data,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                        with hs3: st.markdown(render_zone_heatmap(same_data,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
                    if not opp_data.empty:
                        st.markdown(f"<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:8px 0 4px 0'>vs {'L' if sel_hand == 'R' else 'R'}HB</div>", unsafe_allow_html=True)
                        ho1,ho2,ho3 = st.columns(3)
                        with ho1: st.markdown(render_zone_heatmap(opp_data,"csw_pct","csw","CSW%",fmt=".1%"), unsafe_allow_html=True)
                        with ho2: st.markdown(render_zone_heatmap(opp_data,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                        with ho3: st.markdown(render_zone_heatmap(opp_data,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div style='font-family:monospace;font-size:10px;color:#7aaac0;"
                    f"padding:10px;background:#0c1420;border-radius:8px;border:1px solid #1a2a40'>"
                    f"No zone data for {sel_pitcher} {sel_year} {sel_pitch}.</div>",
                    unsafe_allow_html=True,
                )

    # ── Dataframe with cached gradients ──────────────────────────────────────
    disp_cols = ["Pitcher","Year","Hand","Pitch"] + [c for c in filter_cols if c in view_display.columns]
    df_show   = view_display[disp_cols].copy().reset_index(drop=True)

    @st.cache_data(show_spinner=False)
    def _build_style_df(profiles_hash: int, df_json: str,
                        baselines_hash: int, vaa_hash: int) -> pd.DataFrame:
        """Compute CSS for ALL leaderboard rows once, keyed on profiles (not sort order).
        Returns a DataFrame indexed by original lb_df index."""
        import numpy as _np, io as _io
        _df = pd.read_json(_io.StringIO(df_json), orient="split")

        gradient_cols = [c for c in ["Velo","iVB","HB","VAA","HAA","DM Stuff+",
                                      "CSW%","Whiff%","xwOBA","Rel Ht","Rel Side",
                                      "Extension","N"] if c in _df.columns]
        _pa = _df["Pitch"].values if "Pitch" in _df.columns else None
        _ha = _df["Hand"].values  if "Hand"  in _df.columns else None

        _style = pd.DataFrame("", index=_df.index, columns=_df.columns)
        for _col in gradient_cols:
            _vals = pd.to_numeric(_df[_col], errors="coerce").values.astype(float)
            _valid = ~_np.isnan(_vals)
            _css = _np.full(len(_vals), "background-color:#141e2e;color:#7aaac0", dtype=object)
            if not _valid.any():
                _style[_col] = _css; continue
            if _col in ("VAA","HAA","Velo","iVB","HB"):
                for _i in _np.where(_valid)[0]:
                    _pt = _pa[_i] if _pa is not None else None
                    _hd = _ha[_i] if _ha is not None else None
                    _pb = (lb_pitch_bl.get(f"{_pt}_{_hd}") or lb_pitch_bl.get(_pt)) if _pt else None
                    if _col == "VAA":
                        # Display value is negated; baseline stores raw positive → compare -val
                        if _pb and "vaa_mu" in _pb:
                            _bg, _tx = _outlier_color(-_vals[_i], _pb["vaa_mu"], max(_pb["vaa_sd"],0.001))
                            _css[_i] = f"background-color:{_bg};color:{_tx}"
                        elif _col in lb_baselines:
                            _mu, _sd = lb_baselines[_col]
                            _bg, _tx = _outlier_color(-_vals[_i], _mu, max(_sd,0.001))
                            _css[_i] = f"background-color:{_bg};color:{_tx}"
                    elif _col == "HAA":
                        if _pb and "haa_mu" in _pb:
                            _bg, _tx = _outlier_color(_vals[_i], _pb["haa_mu"], max(_pb["haa_sd"],0.001))
                            _css[_i] = f"background-color:{_bg};color:{_tx}"
                    elif _col == "Velo":
                        if _pb and "velo_mu" in _pb:
                            _bg, _tx = _outlier_color(_vals[_i], _pb["velo_mu"], max(_pb["velo_sd"],0.001))
                            _css[_i] = f"background-color:{_bg};color:{_tx}"
                    elif _col == "iVB":
                        # Directional: high iVB = more rise = red, below avg = blue
                        # Uses same ramp as _outlier_color for visual consistency
                        if _pb and "ivb_mu" in _pb:
                            _bg, _tx = _directional_color(_vals[_i], _pb["ivb_mu"], max(_pb["ivb_sd"],0.001), higher_is_better=True)
                            _css[_i] = f"background-color:{_bg};color:{_tx}"
                    elif _col == "HB":
                        if _pb and "hb_mu" in _pb:
                            # HB is now always arm-side positive (both hands) — single convention.
                            # 2-Seam/Sinker: more positive = more arm-side run = red (directional).
                            # Other pitch types: symmetric outlier color (extremes either way).
                            if _pt == "2-Seam/Sinker":
                                _bg, _tx = _directional_color(_vals[_i], _pb["hb_mu"], max(_pb["hb_sd"],0.001), higher_is_better=True)
                                _css[_i] = f"background-color:{_bg};color:{_tx}"
                            else:
                                _bg, _tx = _outlier_color(_vals[_i], _pb["hb_mu"], max(_pb["hb_sd"],0.001))
                                _css[_i] = f"background-color:{_bg};color:{_tx}"
            elif _col in ("Rel Ht","Rel Side","Extension"):
                _lb = _hand_release_baselines(len(profiles))
                _kmap = {"Rel Ht":"ht","Rel Side":"side","Extension":"ext"}
                _k = _kmap[_col]
                for _i in _np.where(_valid)[0]:
                    _hk = f"{_ha[_i]}_{_k}" if _ha is not None else ""
                    if _hk in _lb:
                        _mu, _sd = _lb[_hk]
                        _bg, _tx = _outlier_color(_vals[_i], _mu, max(_sd,0.001))
                        _css[_i] = f"background-color:{_bg};color:{_tx}"
            else:
                if _col not in lb_baselines:
                    _style[_col] = _css; continue
                _mu, _sd = lb_baselines[_col]
                if _sd <= 0:
                    _style[_col] = _css; continue
                _inv = _col in ("xwOBA",)
                _z = _np.clip((_vals - _mu) / max(_sd, 0.001), -2.5, 2.5)
                if _inv: _z = -_z
                _t = (_z + 2.5) / 5.0
                _r = _np.where(_t<0.5, 30+(120-30)*(_t*2), 120+(220-120)*((_t-0.5)*2)).astype(int)
                _g = _np.where(_t<0.5, 80+(120-80)*(_t*2), 120+(35-120)*((_t-0.5)*2)).astype(int)
                _b = _np.where(_t<0.5, 128+(120-128)*(_t*2), 120+(35-120)*((_t-0.5)*2)).astype(int)
                _lum = (0.299*_r + 0.587*_g + 0.114*_b) / 255
                for _i in _np.where(_valid)[0]:
                    _tx = "#000000" if _lum[_i]>0.45 else "#e8dcc8"
                    _css[_i] = f"background-color:rgb({_r[_i]},{_g[_i]},{_b[_i]});color:{_tx}"
            _style[_col] = _css

        fmt = {}
        for c in _df.columns:
            if c in ("CSW%","Whiff%"):                                fmt[c] = "{:.1f}%"
            elif c == "xwOBA":                                        fmt[c] = "{:.3f}"
            elif c in ("DM Stuff+","N"):                              fmt[c] = "{:.0f}"
            elif c in ("HB","VAA","HAA"):                             fmt[c] = "{:+.1f}"
            elif c in ("Velo","iVB","Rel Ht","Rel Side","Extension"): fmt[c] = "{:.1f}"

        return _style, fmt

    import json as _json
    # Cache key = profiles hash (stable across sorts) — compute once per session
    _base_hash = hash(str(sorted(lb_baselines.items())) +
                      str(sorted(lb_release_baselines.items())))
    _vaa_hash  = hash(str(sorted(_vaa_haa_league.items())) +
                      str(sorted(lb_pitch_bl.items())))
    # Pass full unsorted df so cache is independent of sort order
    _full_disp  = lb_df[disp_cols].copy().reset_index(drop=True)
    _full_json  = _full_disp.to_json(orient="split")
    _full_style, _fmt = _build_style_df(len(profiles), _full_json, _base_hash, _vaa_hash)

    # Reindex the cached style to match current sort order
    # Map original lb_df rows to sorted view_display rows via Pitcher+Year+Pitch key
    _key_cols = ["Pitcher","Year","Pitch"]
    _full_key  = _full_disp[_key_cols].apply(tuple, axis=1)
    _show_key  = df_show[_key_cols].apply(tuple, axis=1)
    _key_to_style_idx = {k: i for i, k in enumerate(_full_key)}
    _style_rows = [_key_to_style_idx.get(k, 0) for k in _show_key]
    _style_df   = _full_style.iloc[_style_rows].reset_index(drop=True)
    _style_df.index = df_show.index

    pd.set_option("styler.render.max_elements", len(df_show)*len(df_show.columns)+1)
    styled = df_show.style.apply(lambda _: _style_df, axis=None).format(_fmt, na_rep="—")

    @st.fragment
    def _lb_dataframe(styled_df, cur_hm_idx):
        event = st.dataframe(
            styled_df,
            width='stretch',
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="lb_df",
        )
        sel_rows = event.selection.rows if event and hasattr(event, "selection") else []
        new_idx  = sel_rows[0] if sel_rows else -1
        if new_idx >= 0 and new_idx != cur_hm_idx:
            # New row selected — open heatmap
            st.session_state["_lb_hm_idx"] = new_idx
            st.rerun()
        elif new_idx < 0 and cur_hm_idx is not None and cur_hm_idx >= 0:
            # Row deselected (clicked again to uncheck) — close heatmap
            st.session_state["_lb_hm_idx"] = None
            st.rerun()

    _lb_dataframe(styled, cur_idx)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: DM STUFF+ CALCULATOR
# Standalone calculator that scores arsenal pitches through the v5 model.
# Tolerates missing fields by imputing from training medians.
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "dmstuff":

    # #9 audit: restore arsenal from URL query params (one-time only per session)
    if not st.session_state.get("_dm_url_loaded", False):
        try:
            _qp = st.query_params
            if any(k.startswith("p_") for k in _qp.keys()):
                # Restore release profile
                for _k_qp, _k_ss in [("hand","dm_hand_r"), ("rh","dm_rh"),
                                       ("rs","dm_rs"), ("ext","dm_ext")]:
                    if _k_qp in _qp:
                        st.session_state[_k_ss] = _qp[_k_qp]
                # Restore pitches
                _restored_pitches = []
                for _k, _v in _qp.items():
                    if not _k.startswith("p_"): continue
                    _grp = _k[2:]
                    if _grp in PITCH_GROUPS:
                        _restored_pitches.append(_grp)
                        _parts = (_v if isinstance(_v, str) else _v[0]).split(",")
                        _parts = (_parts + [""]*6)[:6]
                        for _suf, _val in zip(["_velo","_ivb","_hb","_spin","_usage","_tilt"], _parts):
                            if _val:
                                st.session_state[f"dm_{_grp}{_suf}"] = _val
                if _restored_pitches:
                    st.session_state["_dmsp_pitches"] = _restored_pitches
            st.session_state["_dm_url_loaded"] = True
        except Exception:
            st.session_state["_dm_url_loaded"] = True

    # ── Pending pitcher-load consumer ──────────────────────────────
    # The "Load from MLB Pitcher" and "Load this pitcher" buttons stash
    # the requested arsenal here so it can be applied BEFORE any widget
    # with a `key=` is instantiated. Streamlit forbids writing to a
    # widget's session_state key after the widget has been rendered on
    # the current run; doing so during the button handler throws
    # "st.session_state.<key> cannot be modified after the widget…".
    # Applying the load HERE (above all widgets) avoids that.
    if "_dm_pending_load" in st.session_state:
        try:
            _pending = st.session_state.pop("_dm_pending_load")
            # Hand + release
            if _pending.get("hand") is not None:
                st.session_state["dm_hand_r"] = _pending["hand"]
            for _src_key, _ss_key in [("rh", "dm_rh"), ("rs", "dm_rs"),
                                        ("ext", "dm_ext")]:
                if _pending.get(_src_key) is not None:
                    st.session_state[_ss_key] = float(_pending[_src_key])
            # Wipe + rebuild arsenal
            for _g_old in list(st.session_state.get("_dmsp_pitches", [])):
                for _suf in ["_velo","_ivb","_hb","_spin","_usage","_tilt"]:
                    st.session_state.pop(f"dm_{_g_old}{_suf}", None)
            st.session_state["_dmsp_pitches"] = []
            for _g_new, _vals in _pending.get("pitches", []):
                st.session_state["_dmsp_pitches"].append(_g_new)
                for _f, _suf in [("velo","_velo"), ("ivb","_ivb"),
                                  ("hb","_hb"),   ("spin_rate","_spin"),
                                  ("spin","_spin"), ("usage","_usage"),
                                  ("tilt","_tilt")]:
                    if _vals.get(_f) is not None:
                        st.session_state[f"dm_{_g_new}{_suf}"] = str(_vals[_f])
            if _pending.get("data_source") in _DATA_SOURCES:
                st.session_state["dm_data_source"] = _pending["data_source"]
            st.session_state.pop("_dm_cache", None)
            if _pending.get("toast"):
                st.success(_pending["toast"])
        except Exception as _pl_err:
            st.warning(f"Couldn't apply pitcher load: {_pl_err}")

    # ── Back to title ─────────────────────────────────────────────────────────
    if st.button("← Back", key="back_dmstuff_to_title"):
        st.session_state.screen = "title"
        st.session_state.pop("_dmsp_pitches", None)
        st.session_state.pop("_dm_url_loaded", None)
        st.session_state.pop("_dm_share_qs", None)
        st.session_state.pop("_dm_cache", None)
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    _dm_ver_str = (_v5_bundle or {}).get("version", "model") if _V5_AVAILABLE else "model"
    st.markdown(
        "<div style='text-align:center;max-width:680px;margin:0 auto 20px auto;padding:0 20px'>"
        "<div style='font-family:Inter,sans-serif;font-size:22px;font-weight:700;"
        "color:#c49148;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px'>"
        "🧮  DM Stuff+ Calculator</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#6a90a8'>"
        f"Score any pitch through DM Stuff+ ({_dm_ver_str}) — leave fields blank to use league medians"
        "</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#3a5a78;"
        "margin-top:8px;letter-spacing:1px'>"
        f"● MODEL TRAINED ON STATCAST {_DATA_YEAR_RANGE}  ·  HAWK-EYE (PRE-2020: TRACKMAN)  "
        f"·  PER-PITCH MEAN = 100, SD = 10"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if not _V5_AVAILABLE:
        st.markdown(
            "<div style='max-width:680px;margin:24px auto;padding:20px;"
            "border:1px solid #c4914830;border-radius:8px;background:#181818;"
            "font-family:JetBrains Mono,monospace;font-size:11px;color:#a0c0d4;"
            "line-height:1.7'>"
            "⚠ DM Stuff+ bundle not found. To enable this calculator: "
            "<br>1. Run <code>python train_stuff_plus_v6.py</code> to produce "
            "<code>models/dm_stuff_plus_v6.joblib</code><br>"
            "2. Push the bundle and norms file to the deployment."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        _, dm_main, _ = st.columns([0.3, 11, 0.3])
        with dm_main:

            # ── Release Profile ───────────────────────────────────────────────
            st.markdown(
                "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                "● Release Profile "
                "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>"
                "(optional — defaults to league medians)</span></div>",
                unsafe_allow_html=True,
            )
            dr1, dr2, dr3, dr4 = st.columns([2, 2, 2, 2])
            with dr1:
                st.markdown("<div class='field-label'>Throwing Hand</div>", unsafe_allow_html=True)
                dm_hand = st.radio("_dm_hand", ["RHP","LHP"], horizontal=True,
                                    index=0, key="dm_hand_r", label_visibility="collapsed")
            with dr2:
                st.markdown("<div class='field-label'>Rel Height (ft)</div>", unsafe_allow_html=True)
                dm_rh = st.number_input(" ", min_value=3.0, max_value=8.0,
                                         value=None, step=0.01, format="%.2f",
                                         placeholder=_PLACEHOLDER_REL_HEIGHT, key="dm_rh",
                                         label_visibility="collapsed")
            with dr3:
                st.markdown("<div class='field-label'>Rel Side — arm side (ft)</div>", unsafe_allow_html=True)
                dm_rs = st.number_input(" ", min_value=0.0, max_value=5.0,
                                         value=None, step=0.01, format="%.2f",
                                         placeholder=_PLACEHOLDER_REL_SIDE, key="dm_rs",
                                         label_visibility="collapsed")
            with dr4:
                st.markdown("<div class='field-label'>Extension (ft)</div>", unsafe_allow_html=True)
                dm_ext = st.number_input(" ", min_value=4.0, max_value=8.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder=_PLACEHOLDER_EXTENSION, key="dm_ext",
                                          label_visibility="collapsed")

            # ── Movement-data source selector ─────────────────────────────
            # Lets the coach declare where their IVB/HB numbers come from
            # so we can scale to Statcast-equivalent before model scoring.
            ds_col_l, ds_col_r = st.columns([2, 6])
            with ds_col_l:
                st.markdown(
                    "<div class='field-label' style='margin-top:14px'>"
                    "Movement Data Source"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.selectbox(
                    " ", options=_DATA_SOURCES, index=0,
                    key="dm_data_source", label_visibility="collapsed",
                    help=(
                        "Adjusts internal scoring to a Hawk-Eye (Statcast) "
                        "equivalent. The model was trained on Statcast data; "
                        "TrackMan typically reports ~1–1.5\" more induced "
                        "vertical break on fastballs than Hawk-Eye, and "
                        "Rapsodo lands a bit between the two. Your entered "
                        "numbers are not changed — only the model's "
                        "internal scoring is adjusted."
                    ),
                )
            with ds_col_r:
                _cur_src = st.session_state.get("dm_data_source",
                                                  _DATA_SOURCES[0])
                if _cur_src != _DATA_SOURCES[0]:
                    # Build a human-readable summary of the multiplicative
                    # scaling being applied. The adjustment is the % the
                    # entered movement gets shrunk by, with a concrete
                    # numeric example so the magnitude is intuitive.
                    _tbl = _DATA_SOURCE_SCALE.get(_cur_src, {})
                    _dflt = _tbl.get("default", {"ivb": 1.0, "hb": 1.0})
                    _ivb_pct = (1 - 1/_dflt["ivb"]) * 100
                    _hb_pct  = (1 - 1/_dflt["hb"])  * 100
                    # Example: 18" IVB shrinks to ...; 8" HB shrinks to ...
                    _ex_ivb_in  = 18.0
                    _ex_ivb_out = _ex_ivb_in / _dflt["ivb"]
                    _ex_hb_in   = 8.0
                    _ex_hb_out  = _ex_hb_in / _dflt["hb"]
                    st.markdown(
                        "<div style='font-family:JetBrains Mono,monospace;"
                        "font-size:10px;color:#7a9ab0;margin:38px 0 0 0;"
                        "padding:10px 14px;background:#0a1218;"
                        "border:1px solid #1a2a40;border-radius:6px;line-height:1.7'>"
                        f"Scoring divides entered iVB by "
                        f"<b style='color:#a0c0d4'>{_dflt['ivb']:.2f}</b> "
                        f"(≈ −{_ivb_pct:.0f}%) and HB by "
                        f"<b style='color:#a0c0d4'>{_dflt['hb']:.2f}</b> "
                        f"(≈ −{_hb_pct:.0f}%) for {_cur_src} → Hawk-Eye. "
                        f"Per-pitch-type scales override these defaults. "
                        f"Example: an entered <b>{_ex_ivb_in:.0f}\" iVB</b> "
                        f"scores as <b>{_ex_ivb_out:.1f}\"</b>; "
                        f"<b>{_ex_hb_in:.0f}\" HB</b> → <b>{_ex_hb_out:.1f}\"</b>. "
                        f"Absolute adjustment scales with movement magnitude. "
                        f"Your entered values stay as displayed."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div style='font-family:JetBrains Mono,monospace;"
                        "font-size:10px;color:#5a7a90;margin:38px 0 0 0;"
                        "padding:10px 14px;background:#0a1218;"
                        "border:1px solid #1a2a40;border-radius:6px;line-height:1.7'>"
                        "Hawk-Eye / Statcast is the model's native scale — "
                        "no adjustment applied. Use this if you don't know "
                        "your data source."
                        "</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

            # ── TrackMan / Rapsodo Auto-Fill (calculator) ─────────────────────
            st.markdown(
                "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                "● TrackMan / Rapsodo Auto-Fill "
                "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>"
                "(optional — CSV, PDF, or photo)</span></div>",
                unsafe_allow_html=True,
            )
            dm_tm_file = st.file_uploader(
                "Upload a TrackMan/Rapsodo CSV, PDF report, or screenshot — "
                "metrics will be pre-filled into the pitch arsenal below.",
                type=["csv","pdf","jpg","jpeg","png"], key="dm_tm_upload",
            )
            if dm_tm_file is not None:
                _dm_file_id = f"{dm_tm_file.name}_{dm_tm_file.size}"
                if st.session_state.get("_dm_tm_file_id") != _dm_file_id:
                    _fname_l = dm_tm_file.name.lower()
                    _fbytes  = dm_tm_file.read()
                    # Auto-detect movement-data source from the file content
                    _detected_src = sniff_data_source(_fbytes, dm_tm_file.name)
                    if _detected_src and _detected_src in _DATA_SOURCES:
                        st.session_state["dm_data_source"] = _detected_src
                        st.session_state["_dm_src_autodetected"] = _detected_src
                    if _fname_l.endswith((".jpg",".jpeg",".png")):
                        with st.spinner("Reading image with OCR…"):
                            _parsed = parse_trackman_image(_fbytes, dm_tm_file.name)
                    else:
                        _parsed = parse_trackman(_fbytes, dm_tm_file.name)
                    if "_error" in _parsed:
                        st.warning(f"Upload parse issue: {_parsed['_error']}")
                        st.session_state["_dm_tm_file_id"] = None
                    else:
                        st.session_state["_dm_tm_file_id"] = _dm_file_id
                        # Add parsed pitches to _dmsp_pitches and pre-fill widget keys
                        _added_new = []
                        for _grp, _vals in _parsed.items():
                            if _grp not in PITCH_GROUPS:
                                continue
                            if _grp not in st.session_state.get("_dmsp_pitches", []):
                                st.session_state.setdefault("_dmsp_pitches", []).append(_grp)
                                _added_new.append(_grp)
                            for _f, _suf in [("velo","_velo"),("ivb","_ivb"),
                                              ("hb","_hb"),("spin_rate","_spin")]:
                                if _vals.get(_f) is not None:
                                    st.session_state[f"dm_{_grp}{_suf}"] = f"{_vals[_f]}"
                            # Release profile
                            if _vals.get("rel_height") is not None and \
                                    st.session_state.get("dm_rh") is None:
                                st.session_state["dm_rh"] = float(_vals["rel_height"])
                            if _vals.get("rel_side") is not None and \
                                    st.session_state.get("dm_rs") is None:
                                st.session_state["dm_rs"] = float(_vals["rel_side"])
                            if _vals.get("extension") is not None and \
                                    st.session_state.get("dm_ext") is None:
                                st.session_state["dm_ext"] = float(_vals["extension"])
                        st.session_state.pop("_dm_cache", None)
                        _summary = ", ".join(_parsed.keys())
                        _detect_note = ""
                        _auto_src = st.session_state.pop("_dm_src_autodetected", None)
                        if _auto_src:
                            _detect_note = (
                                f"  ·  Detected data source: **{_auto_src}** "
                                f"(scoring adjusted accordingly)"
                            )
                        st.success(
                            f"Parsed: {_summary}.{_detect_note}  "
                            "Edit any pre-filled value below as needed."
                        )

            # ── Load from MLB pitcher ────────────────────────────────────────
            # Quick-fill from the existing profiles dataset. Picks the
            # pitcher's most-recent season (highest year).
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
            _lp_l, _lp_r = st.columns([3, 5])
            with _lp_l:
                st.markdown(
                    "<div class='field-label'>Load from MLB Pitcher</div>",
                    unsafe_allow_html=True,
                )
                _lp_options = ([""] + sorted(profiles["player_name"].dropna().unique().tolist())
                                if profiles is not None else [""])
                _lp_sel = st.selectbox(
                    " ", options=_lp_options, index=0,
                    key="dm_lp_pitcher",
                    format_func=lambda x: "— Pick a pitcher to auto-fill —" if not x else x,
                    label_visibility="collapsed",
                    help=(
                        "Pre-fills the entire arsenal + release profile "
                        "with this pitcher's most-recent-season averages "
                        "from the profiles dataset. Overwrites any current "
                        "values."
                    ),
                )
            with _lp_r:
                st.markdown("<div style='height:23px'></div>", unsafe_allow_html=True)
                if st.button("⤴ Apply pitcher", key="dm_lp_apply",
                              disabled=not _lp_sel):
                    # Build a pending-load packet and trigger a rerun.
                    # The widget-key writes happen at the top of the
                    # next run (see "Pending pitcher-load consumer"
                    # block above), avoiding the Streamlit error
                    # "session_state.dm_hand_r cannot be modified after
                    # the widget … is instantiated."
                    try:
                        _sub = profiles[profiles["player_name"] == _lp_sel]
                        if not _sub.empty:
                            _row = _sub.sort_values("year", ascending=False).iloc[0]
                            _hand_val = str(_row.get("hand", "R")).upper()[:1]
                            _pending_pitches = []
                            for _grp_lp in PITCH_GROUPS:
                                _n = _row.get(f"n_{_grp_lp}")
                                _v = _row.get(f"velo_{_grp_lp}")
                                if pd.notna(_n) and _n > 25 and pd.notna(_v):
                                    _vals_lp = {"velo": f"{float(_v):.1f}"}
                                    for _src, _f in [("ivb",       "ivb"),
                                                      ("hb",        "hb"),
                                                      ("spin_rate", "spin"),
                                                      ("pct",       "usage")]:
                                        _val = _row.get(f"{_src}_{_grp_lp}")
                                        if pd.notna(_val):
                                            if _src == "pct":
                                                _vals_lp[_f] = f"{float(_val)*100:.1f}"
                                            else:
                                                _vals_lp[_f] = f"{float(_val):.2f}"
                                    _pending_pitches.append((_grp_lp, _vals_lp))
                            st.session_state["_dm_pending_load"] = {
                                "hand":        "LHP" if _hand_val == "L" else "RHP",
                                "rh":          (float(_row["rel_height"]) if pd.notna(_row.get("rel_height")) else None),
                                "rs":          (float(_row["rel_side"])   if pd.notna(_row.get("rel_side"))   else None),
                                "ext":         (float(_row["extension"])  if pd.notna(_row.get("extension"))  else None),
                                "pitches":     _pending_pitches,
                                "data_source": _DATA_SOURCES[0],
                                "toast":       f"Loaded {_lp_sel} ({int(_row['year'])}). "
                                               f"Source set to Hawk-Eye/Statcast.",
                            }
                            st.rerun()
                    except Exception as _lp_err:
                        st.warning(f"Couldn't load pitcher: {_lp_err}")

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

            # ── Pitch Arsenal ─────────────────────────────────────────────────
            st.markdown(
                "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                "● Pitch Arsenal "
                "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>"
                "velocity required — other fields use medians if blank</span></div>",
                unsafe_allow_html=True,
            )

            dm_all_groups = list(PITCH_GROUPS.keys())
            if "_dmsp_pitches" not in st.session_state:
                st.session_state["_dmsp_pitches"] = []
            dm_added = st.session_state["_dmsp_pitches"]
            dm_remaining = [g for g in dm_all_groups if g not in dm_added]

            def _dm_on_add_pitch():
                ch = st.session_state.get("_dm_add_pitch_sel", "")
                if ch and ch not in st.session_state["_dmsp_pitches"]:
                    st.session_state["_dmsp_pitches"].append(ch)
                st.session_state["_dm_add_pitch_sel"] = ""
                st.session_state.pop("_dm_cache", None)

            dm_add_col, _ = st.columns([2, 5])
            with dm_add_col:
                if dm_remaining:
                    st.selectbox(
                        "_dm_add_pitch",
                        options=[""] + dm_remaining,
                        format_func=lambda x: "＋ Add a pitch…" if x == "" else f"● {x}",
                        key="_dm_add_pitch_sel",
                        label_visibility="collapsed",
                        on_change=_dm_on_add_pitch,
                    )
                else:
                    st.markdown(
                        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                        "color:#3a5a78;padding:8px 0'>All pitch types added.</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            # Render added pitch cards — 4 metric inputs per pitch
            for group in list(dm_added):
                color = PITCH_COLORS[group]
                hdr_col, rm_col = st.columns([8, 1])
                with hdr_col:
                    st.markdown(
                        f"<div style='font-family:Inter,sans-serif;font-size:12px;"
                        f"font-weight:700;color:{color};letter-spacing:2px;"
                        f"text-transform:uppercase;padding:6px 0 4px 0'>● {group}</div>",
                        unsafe_allow_html=True,
                    )
                with rm_col:
                    if st.button("✕", key=f"_dm_rm_{group}", help=f"Remove {group}"):
                        st.session_state["_dmsp_pitches"].remove(group)
                        for suf in ["_velo", "_ivb", "_hb", "_spin", "_usage", "_tilt"]:
                            st.session_state.pop(f"dm_{group}{suf}", None)
                        st.session_state.pop("_dm_cache", None)
                        st.rerun()

                # 6 columns now: velo, ivb, hb, spin, tilt, usage
                # Tilt promoted from a buried sub-row to an inline field so
                # coaches see it and use it. Tooltip explains why it matters.
                vc, ic, hc, sc, tc, uc = st.columns([2, 2, 2, 2, 1.5, 1])
                with vc:
                    st.markdown("<div class='field-label'>Velocity (mph) *</div>", unsafe_allow_html=True)
                    st.text_input(" ", value="", key=f"dm_{group}_velo",
                                   placeholder="required", label_visibility="collapsed")
                with ic:
                    st.markdown("<div class='field-label'>iVB (in)</div>", unsafe_allow_html=True)
                    st.text_input(" ", value="", key=f"dm_{group}_ivb",
                                   placeholder="optional", label_visibility="collapsed")
                with hc:
                    st.markdown("<div class='field-label'>HB — arm-side + (in)</div>", unsafe_allow_html=True)
                    st.text_input(" ", value="", key=f"dm_{group}_hb",
                                   placeholder="optional", label_visibility="collapsed")
                with sc:
                    st.markdown("<div class='field-label'>Spin (rpm)</div>", unsafe_allow_html=True)
                    st.text_input(" ", value="", key=f"dm_{group}_spin",
                                   placeholder="optional", label_visibility="collapsed")
                with tc:
                    st.markdown(
                        "<div class='field-label' title='Spin axis as a clock "
                        "tilt (1–12). 12 = pure backspin (RHP riding 4-seam), "
                        "9 = pure sidespin (LHP arm-side run), 6 = pure topspin "
                        "(12-6 curve). Half-hours allowed (e.g. 1.5 = 1:30). "
                        "Improves gyro/SSW estimate for breaking balls and "
                        "sinkers; leave blank if unknown.'>Tilt (1–12)</div>",
                        unsafe_allow_html=True,
                    )
                    st.text_input(" ", value="", key=f"dm_{group}_tilt",
                                   placeholder="e.g. 1.5",
                                   label_visibility="collapsed")
                with uc:
                    st.markdown("<div class='field-label'>Usage %</div>", unsafe_allow_html=True)
                    st.text_input(" ", value="", key=f"dm_{group}_usage",
                                   placeholder="e.g. 35", label_visibility="collapsed")

            # ── Live HB convention check (LHP only) ──────────────────────
            # Detect Trackman/catcher's-view HB signs at input time so the
            # coach can confirm before scoring. Mirrors the post-compute
            # detector but runs on live session_state values.
            _dm_hand_live = st.session_state.get("dm_hand_r", "RHP")
            if _dm_hand_live == "LHP":
                _LHP_EXP_LIVE = {
                    "4-Seam": +1, "2-Seam/Sinker": +1,
                    "Splitter": +1, "Changeup": +1,
                    "Slider": -1, "Sweeper": -1,
                }
                _live_wrong = _live_total = 0
                for _g_live in st.session_state.get("_dmsp_pitches", []):
                    _hb_raw = st.session_state.get(f"dm_{_g_live}_hb", "")
                    try:
                        _hb_val = float(str(_hb_raw).strip()) if str(_hb_raw).strip() else None
                    except (TypeError, ValueError):
                        _hb_val = None
                    _exp_live = _LHP_EXP_LIVE.get(_g_live)
                    if _hb_val is None or _exp_live is None or _hb_val == 0:
                        continue
                    _live_total += 1
                    if (1 if _hb_val > 0 else -1) != _exp_live:
                        _live_wrong += 1
                if _live_total >= 2 and _live_wrong > _live_total / 2:
                    st.markdown(
                        f"<div style='margin:10px 0 0 0;padding:10px 14px;"
                        f"background:#0a1420;border:1px solid #2a4a6a40;"
                        f"border-radius:6px;font-family:JetBrains Mono,monospace;"
                        f"font-size:10px;color:#5a8aaa;line-height:1.7'>"
                        f"ℹ <b>LHP HB convention check:</b> {_live_wrong} of "
                        f"{_live_total} entered HBs look like Trackman "
                        f"catcher's-view (LHP arm-side negative). "
                        f"<b>Auto-flip will apply at compute</b> so values are "
                        f"converted to arm-side-positive for the model. "
                        f"If this is unintended, edit the values to use "
                        f"pitcher's view (arm-side positive)."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # ── Usage sanity chip (#4) ────────────────────────────────────
            # Show only when the user has entered at least one usage value.
            _usage_total_live = 0.0
            _usage_count_live = 0
            for _g_chip in st.session_state.get("_dmsp_pitches", []):
                _u_raw = st.session_state.get(f"dm_{_g_chip}_usage", "")
                try:
                    _u_f = float(str(_u_raw).strip())
                    _usage_total_live += _u_f
                    _usage_count_live += 1
                except (ValueError, TypeError):
                    pass
            if _usage_count_live > 0:
                if 90.0 <= _usage_total_live <= 110.0:
                    _chip_bg, _chip_fg, _chip_msg = ("#0e2a1a", "#5ac8a0",
                        f"✓ Usage sums to {_usage_total_live:.0f}% — looks right.")
                elif _usage_total_live < 90.0:
                    _chip_bg, _chip_fg, _chip_msg = ("#2a1f0e", "#c0a878",
                        f"⚠ Usage sums to {_usage_total_live:.0f}%. "
                        f"Add usages for the rest of your pitches "
                        f"(the model still works but is calibrated for ~100%).")
                else:
                    _chip_bg, _chip_fg, _chip_msg = ("#2a1010", "#d48a8a",
                        f"⚠ Usage sums to {_usage_total_live:.0f}% — over 100%. "
                        f"Did you double-enter one of your pitches?")
                st.markdown(
                    f"<div style='margin:12px 0 0 0;padding:8px 14px;"
                    f"background:{_chip_bg};border:1px solid {_chip_fg}30;"
                    f"border-radius:6px;font-family:JetBrains Mono,monospace;"
                    f"font-size:10px;color:{_chip_fg};'>{_chip_msg}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

            # ── Score + Share buttons + Results (fragment prevents page greyout) ─
            @st.fragment
            def _dm_results_frag():
                # Read live state so add/remove pitch during fragment rerun is picked up
                _dm_added = st.session_state.get("_dmsp_pitches", [])
                _dm_hand  = st.session_state.get("dm_hand_r", "RHP")
                _dm_rh    = st.session_state.get("dm_rh")
                _dm_rs    = st.session_state.get("dm_rs")
                _dm_ext   = st.session_state.get("dm_ext")
                _auto_compute = st.session_state.pop("_dm_auto_compute", False)

                # ── Score + Share buttons ─────────────────────────────────────
                score_col, share_col, _ = st.columns([3, 2, 3])
                with score_col:
                    _do_score = st.button("Compute DM Stuff+ →", key="dm_score_btn",
                                           width='stretch', disabled=(len(_dm_added) == 0))
                with share_col:
                    if _dm_added:
                        import urllib.parse as _urlparse
                        _share_qs = {"hand": _dm_hand}
                        if _dm_rh  is not None: _share_qs["rh"]  = f"{_dm_rh:.2f}"
                        if _dm_rs  is not None: _share_qs["rs"]  = f"{_dm_rs:.2f}"
                        if _dm_ext is not None: _share_qs["ext"] = f"{_dm_ext:.2f}"
                        for g in _dm_added:
                            _vals = [
                                st.session_state.get(f"dm_{g}_velo")  or "",
                                st.session_state.get(f"dm_{g}_ivb")   or "",
                                st.session_state.get(f"dm_{g}_hb")    or "",
                                st.session_state.get(f"dm_{g}_spin")  or "",
                                st.session_state.get(f"dm_{g}_usage") or "",
                                st.session_state.get(f"dm_{g}_tilt")  or "",
                            ]
                            _share_qs[f"p_{g}"] = ",".join(str(v) for v in _vals)
                        _qs = _urlparse.urlencode(_share_qs)
                        if st.button("🔗 Copy share link", key="dm_share_btn", width='stretch'):
                            st.session_state["_dm_share_qs"] = _qs
                            st.rerun()
                    if st.session_state.get("_dm_share_qs"):
                        st.code(f"?{st.session_state['_dm_share_qs']}", language=None)
                        st.markdown(
                            "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                            "color:#5a7a90;margin-top:2px'>Append to your app URL to "
                            "restore this arsenal.</div>",
                            unsafe_allow_html=True,
                        )

                # ── Save & Compare arsenals (#7) ──────────────────────────
                # Persisted only for the current session. Coaches can save a
                # pitcher's current shape and compare against past versions
                # to track development between bullpens.
                _saved_arsenals = st.session_state.setdefault("_dm_saved_arsenals", [])
                if _dm_added and "_dm_cache" in st.session_state:
                    _save_l, _save_m, _save_r = st.columns([3, 3, 2])
                    with _save_l:
                        _save_label = st.text_input(
                            "Save current arsenal as:", value="",
                            key="_dm_save_label",
                            placeholder="e.g. Smith 2025-05-22",
                            label_visibility="visible",
                        )
                    with _save_m:
                        st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
                        if st.button("💾 Save", key="_dm_save_btn",
                                      disabled=not _save_label.strip()):
                            import time as _time
                            _C_save = st.session_state.get("_dm_cache", {})
                            _snapshot = {
                                "label": _save_label.strip(),
                                "ts":    _time.time(),
                                "hand":  _dm_hand,
                                "rh":    _dm_rh, "rs": _dm_rs, "ext": _dm_ext,
                                "pitches": [(g, {
                                    "velo":  st.session_state.get(f"dm_{g}_velo")  or "",
                                    "ivb":   st.session_state.get(f"dm_{g}_ivb")   or "",
                                    "hb":    st.session_state.get(f"dm_{g}_hb")    or "",
                                    "spin":  st.session_state.get(f"dm_{g}_spin")  or "",
                                    "usage": st.session_state.get(f"dm_{g}_usage") or "",
                                    "tilt":  st.session_state.get(f"dm_{g}_tilt")  or "",
                                }) for g in _dm_added],
                                "arsenal_sp": _C_save.get("display_arsenal_sp"),
                            }
                            _saved_arsenals.append(_snapshot)
                            st.success(f"Saved «{_snapshot['label']}»")
                    with _save_r:
                        st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)
                        st.markdown(
                            f"<div style='font-family:JetBrains Mono,monospace;"
                            f"font-size:10px;color:#5a7a90;padding-top:6px'>"
                            f"{len(_saved_arsenals)} saved this session</div>",
                            unsafe_allow_html=True,
                        )

                # ── Export / Import (collapsed expander) ─────────────
                # Wrapped in an expander so the import file-uploader
                # widget doesn't render orphaned at the bottom of the
                # page when there's nothing to export and no compute yet.
                with st.expander("📦  Export / Import results", expanded=False):
                    st.markdown(
                        "<div class='note-mono' style='margin-bottom:10px'>"
                        "Download a PDF of the current results, export "
                        "your session's saved arsenals as JSON, or import "
                        "a previously-exported JSON to restore them."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    _exp_pdf_col, _exp_json_col, _imp_json_col = st.columns([1, 1, 1])

                    # PDF export
                    with _exp_pdf_col:
                        if _PDF_AVAILABLE and "_dm_cache" in st.session_state:
                            _C_pdf = st.session_state["_dm_cache"]
                            try:
                                _pdf_bytes = build_calculator_pdf(_C_pdf, {
                                    "arsenal_sp": _C_pdf.get("display_arsenal_sp"),
                                    "grade":      _C_pdf.get("display_arsenal_grade"),
                                    "vs_rhb":     _C_pdf.get("display_arsenal_vs_rhb"),
                                    "vs_lhb":     _C_pdf.get("display_arsenal_vs_lhb"),
                                })
                                import time as _time_pdf
                                _pdf_fname = (
                                    f"dm_stuff_report_{_time_pdf.strftime('%Y%m%d_%H%M%S')}.pdf"
                                )
                                st.download_button(
                                    "📄  Report (PDF)",
                                    data=_pdf_bytes,
                                    file_name=_pdf_fname,
                                    mime="application/pdf",
                                    key="_dm_export_pdf",
                                    width="stretch",
                                    help="One-page PDF report — grade, "
                                         "per-pitch table, source.",
                                )
                            except Exception as _pdfu_err:
                                import sys as _sys_pdf
                                print(f"[pdf] export failed: {_pdfu_err}", file=_sys_pdf.stderr)
                        else:
                            st.markdown(
                                "<div class='note-mono' style='opacity:0.6;"
                                "padding:8px 0'>PDF available after compute.</div>",
                                unsafe_allow_html=True,
                            )

                    # JSON export of saved arsenals
                    with _exp_json_col:
                        if _saved_arsenals:
                            import json as _json_persist
                            _persist_payload = _json_persist.dumps(
                                _saved_arsenals, default=str, indent=2
                            )
                            st.download_button(
                                "📥  Saved (JSON)",
                                data=_persist_payload,
                                file_name="dm_stuff_saved_arsenals.json",
                                mime="application/json",
                                key="_dm_export_arsenals",
                                width="stretch",
                                help="Download all saved arsenals so they "
                                     "can be restored in a future session.",
                            )
                        else:
                            st.markdown(
                                "<div class='note-mono' style='opacity:0.6;"
                                "padding:8px 0'>No saved arsenals yet.</div>",
                                unsafe_allow_html=True,
                            )

                    # JSON import of saved arsenals
                    with _imp_json_col:
                        st.markdown(
                            "<div class='field-label'>Import saved (.json)</div>",
                            unsafe_allow_html=True,
                        )
                        _persist_upload = st.file_uploader(
                            " ",
                            type=["json"],
                            key="_dm_import_arsenals",
                            label_visibility="collapsed",
                            help="Restore previously-exported saved arsenals. "
                                 "Duplicates by label are skipped.",
                        )
                        if _persist_upload is not None:
                            _import_id = f"{_persist_upload.name}_{_persist_upload.size}"
                            if st.session_state.get("_dm_imp_id") != _import_id:
                                try:
                                    import json as _json_imp
                                    _imp_payload = _json_imp.loads(
                                        _persist_upload.read().decode("utf-8")
                                    )
                                    if isinstance(_imp_payload, list):
                                        _existing = {a.get("label") for a in _saved_arsenals}
                                        _added = 0
                                        for _entry in _imp_payload:
                                            if (isinstance(_entry, dict)
                                                    and _entry.get("label")
                                                    and _entry["label"] not in _existing):
                                                _saved_arsenals.append(_entry)
                                                _added += 1
                                        st.session_state["_dm_imp_id"] = _import_id
                                        st.success(
                                            f"Imported {_added} new arsenal(s) "
                                            f"(skipped duplicates)."
                                        )
                                    else:
                                        st.warning("File didn't contain an arsenal list.")
                                except Exception as _imp_err:
                                    st.warning(f"Couldn't import: {_imp_err}")

                if _saved_arsenals:
                    with st.expander(f"📚 Saved arsenals ({len(_saved_arsenals)})",
                                     expanded=False):
                        # Compare-against dropdown
                        _compare_options = [""] + [f"#{i+1} {a['label']}"
                                                    for i, a in enumerate(_saved_arsenals)]
                        _cmp_pick = st.selectbox(
                            "Compare current arsenal against a saved one:",
                            options=_compare_options, key="_dm_cmp_pick",
                        )

                        # List rows with Load / Delete
                        for _idx, _snap in enumerate(list(_saved_arsenals)):
                            _row_a, _row_b, _row_c, _row_d = st.columns([5, 2, 1, 1])
                            with _row_a:
                                _pitch_summary = ", ".join(g for g, _ in _snap.get("pitches", []))
                                _sp_str = (f"  ·  Arsenal {_snap['arsenal_sp']:.1f}"
                                            if _snap.get("arsenal_sp") else "")
                                st.markdown(
                                    f"<div style='font-family:Inter,sans-serif;"
                                    f"font-size:12px;color:#c0d8e8'>"
                                    f"<b>{_snap['label']}</b>  "
                                    f"<span style='color:#5a7a90'>({_snap.get('hand','RHP')})"
                                    f"{_sp_str}</span></div>"
                                    f"<div style='font-family:JetBrains Mono,monospace;"
                                    f"font-size:9px;color:#5a7a90'>{_pitch_summary}</div>",
                                    unsafe_allow_html=True,
                                )
                            with _row_b:
                                st.markdown("<div style='height:6px'></div>",
                                             unsafe_allow_html=True)
                            with _row_c:
                                if st.button("Load", key=f"_dm_load_{_idx}"):
                                    # Wipe current arsenal, then restore from snapshot
                                    for _g_old in list(st.session_state.get("_dmsp_pitches", [])):
                                        for _suf in ["_velo","_ivb","_hb","_spin","_usage","_tilt"]:
                                            st.session_state.pop(f"dm_{_g_old}{_suf}", None)
                                    st.session_state["_dmsp_pitches"] = []
                                    st.session_state["dm_hand_r"] = _snap.get("hand", "RHP")
                                    for _k in ["rh","rs","ext"]:
                                        if _snap.get(_k) is not None:
                                            st.session_state[f"dm_{_k}"] = _snap[_k]
                                    for _g_new, _vals in _snap.get("pitches", []):
                                        st.session_state["_dmsp_pitches"].append(_g_new)
                                        for _f, _suf in [("velo","_velo"),("ivb","_ivb"),
                                                          ("hb","_hb"),("spin","_spin"),
                                                          ("usage","_usage"),("tilt","_tilt")]:
                                            if _vals.get(_f):
                                                st.session_state[f"dm_{_g_new}{_suf}"] = _vals[_f]
                                    st.session_state.pop("_dm_cache", None)
                                    st.session_state["_dm_auto_compute"] = True
                                    st.rerun(scope="app")
                            with _row_d:
                                if st.button("✕", key=f"_dm_del_{_idx}",
                                              help=f"Delete {_snap['label']}"):
                                    _saved_arsenals.pop(_idx)
                                    st.rerun(scope="fragment")

                        # Render compare visualization
                        if _cmp_pick:
                            try:
                                _cmp_idx = _compare_options.index(_cmp_pick) - 1
                                _cmp_snap = _saved_arsenals[_cmp_idx]
                                _C_now = st.session_state.get("_dm_cache", {})
                                _now_sp = _C_now.get("display_arsenal_sp")
                                _cmp_sp = _cmp_snap.get("arsenal_sp")
                                _delta = (
                                    (_now_sp - _cmp_sp)
                                    if (_now_sp is not None and _cmp_sp is not None)
                                    else None
                                )
                                _delta_str = ""
                                if _delta is not None:
                                    _d_col = "#5ac8a0" if _delta > 0 else (
                                              "#d48a8a" if _delta < 0 else "#8a9aac")
                                    _delta_str = (
                                        f"<span style='color:{_d_col};font-weight:700'>"
                                        f"{'+' if _delta>=0 else ''}{_delta:.1f}</span>"
                                    )
                                _now_str = f"{_now_sp:.1f}" if _now_sp is not None else "—"
                                _cmp_str = f"{_cmp_sp:.1f}" if _cmp_sp is not None else "—"
                                _d_show  = _delta_str if _delta_str else "—"
                                st.markdown(
                                    f"<div style='margin-top:14px;padding:12px 16px;"
                                    f"background:#0a1218;border:1px solid #1a2a40;"
                                    f"border-radius:6px;font-family:JetBrains Mono,monospace;"
                                    f"font-size:11px;color:#c0d8e8'>"
                                    f"<b>Compare:</b> current ({_now_str}) ↔ "
                                    f"«{_cmp_snap['label']}» ({_cmp_str}) "
                                    f"&nbsp;&nbsp;Δ = {_d_show}"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                                # ── Side-by-side per-pitch diff ──────────
                                # Build {group: usage_pct} for both arsenals,
                                # then show one row per pitch with the
                                # current value, saved value, and the delta.
                                _now_use = {g: float(_pdict.get(g, {}).get("usage_pct") or 0)
                                            for g in _C_now.get("dm_added", [])
                                            if g in _C_now.get("scores", {})}
                                _cmp_use = {}
                                for _g_cmp, _v_cmp in _cmp_snap.get("pitches", []):
                                    try:
                                        _cmp_use[_g_cmp] = float(_v_cmp.get("usage") or 0)
                                    except (TypeError, ValueError):
                                        _cmp_use[_g_cmp] = 0.0
                                # Union of pitches across both arsenals
                                _all_grps_cmp = sorted(set(_now_use) | set(_cmp_use),
                                                         key=lambda g: -(_now_use.get(g, 0)
                                                                          + _cmp_use.get(g, 0)))
                                if _all_grps_cmp:
                                    _rows_html = []
                                    for _g_cmp_r in _all_grps_cmp:
                                        _now_u = _now_use.get(_g_cmp_r, 0.0)
                                        _cmp_u = _cmp_use.get(_g_cmp_r, 0.0)
                                        _du = _now_u - _cmp_u
                                        if abs(_du) < 0.5:
                                            _du_col = "#8a9aac"; _arrow = "·"
                                        elif _du > 0:
                                            _du_col = "#5ac8a0"; _arrow = "↑"
                                        else:
                                            _du_col = "#d48a8a"; _arrow = "↓"
                                        _gc = PITCH_COLORS.get(_g_cmp_r, "#aaaaaa")
                                        _rows_html.append(
                                            f"<tr>"
                                            f"<td style='padding:4px 10px;color:{_gc};"
                                            f"font-weight:600'>{_g_cmp_r}</td>"
                                            f"<td style='padding:4px 10px;text-align:right;"
                                            f"color:#a0c0d4'>{_now_u:.0f}%</td>"
                                            f"<td style='padding:4px 10px;text-align:right;"
                                            f"color:#7a9ab0'>{_cmp_u:.0f}%</td>"
                                            f"<td style='padding:4px 10px;text-align:right;"
                                            f"color:{_du_col};font-weight:700'>"
                                            f"{_arrow} {_du:+.0f}pp</td>"
                                            f"</tr>"
                                        )
                                    st.markdown(
                                        f"<div style='margin-top:10px;padding:8px 10px;"
                                        f"background:#0a1218;border:1px solid #1a2a40;"
                                        f"border-radius:6px'>"
                                        f"<table style='width:100%;border-collapse:collapse;"
                                        f"font-family:JetBrains Mono,monospace;font-size:10px'>"
                                        f"<thead><tr style='border-bottom:1px solid #1a2a40'>"
                                        f"<th style='padding:6px 10px;text-align:left;"
                                        f"color:#5a7a90;font-weight:600'>Pitch</th>"
                                        f"<th style='padding:6px 10px;text-align:right;"
                                        f"color:#5a7a90;font-weight:600'>Current</th>"
                                        f"<th style='padding:6px 10px;text-align:right;"
                                        f"color:#5a7a90;font-weight:600'>Saved</th>"
                                        f"<th style='padding:6px 10px;text-align:right;"
                                        f"color:#5a7a90;font-weight:600'>Δ usage</th>"
                                        f"</tr></thead>"
                                        f"<tbody>{''.join(_rows_html)}</tbody></table></div>",
                                        unsafe_allow_html=True,
                                    )
                                    # Note about pitches added/dropped
                                    _added_set = set(_now_use) - set(_cmp_use)
                                    _dropped_set = set(_cmp_use) - set(_now_use)
                                    _note_parts = []
                                    if _added_set:
                                        _note_parts.append(
                                            f"<b style='color:#5ac8a0'>Added:</b> "
                                            f"{', '.join(sorted(_added_set))}"
                                        )
                                    if _dropped_set:
                                        _note_parts.append(
                                            f"<b style='color:#d48a8a'>Dropped:</b> "
                                            f"{', '.join(sorted(_dropped_set))}"
                                        )
                                    if _note_parts:
                                        st.markdown(
                                            f"<div style='font-family:JetBrains Mono,monospace;"
                                            f"font-size:9px;color:#7a9ab0;margin-top:6px;"
                                            f"padding:0 6px'>"
                                            f"{'  ·  '.join(_note_parts)}</div>",
                                            unsafe_allow_html=True,
                                        )
                            except Exception as _cmp_err:
                                import sys as _sys
                                print(f"[compare-card] failed to render: {_cmp_err}",
                                      file=_sys.stderr)
                                st.markdown(
                                    "<div style='font-family:JetBrains Mono,monospace;"
                                    "font-size:10px;color:#a08070;padding:8px'>"
                                    "Compare view temporarily unavailable. "
                                    f"Reason: <code>{_cmp_err}</code>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )

                # ── Compute ───────────────────────────────────────────────────
                if (_do_score or _auto_compute) and _dm_added:
                    hand_code = "L" if _dm_hand == "LHP" else "R"
                    def _pf(s):
                        try:
                            if s is None or str(s).strip() == "":
                                return None
                            return float(str(s).strip())
                        except (TypeError, ValueError):
                            return None

                    pitches_dict = {}
                    missing_velo = []
                    # Movement-data source adjustment applied per pitch
                    _data_source = st.session_state.get("dm_data_source",
                                                          _DATA_SOURCES[0])
                    for group in _dm_added:
                        v = _pf(st.session_state.get(f"dm_{group}_velo"))
                        if v is None:
                            missing_velo.append(group)
                            continue
                        _tilt_in = _pf(st.session_state.get(f"dm_{group}_tilt"))
                        _user_ivb = _pf(st.session_state.get(f"dm_{group}_ivb"))
                        _user_hb  = _pf(st.session_state.get(f"dm_{group}_hb"))
                        # Convert from source to Hawk-Eye (Statcast) equivalent
                        _adj_ivb, _adj_hb = _apply_data_source_adjustment(
                            group, _user_ivb, _user_hb, _data_source
                        )
                        pitches_dict[group] = {
                            "velo":      v,
                            "ivb":       _adj_ivb,
                            "hb":        _adj_hb,
                            "spin_rate": _pf(st.session_state.get(f"dm_{group}_spin")),
                            "usage_pct": _pf(st.session_state.get(f"dm_{group}_usage")),
                            "tilt_hour": _tilt_in,
                            # Preserve user-entered (un-adjusted) values for display
                            "ivb_entered": _user_ivb,
                            "hb_entered":  _user_hb,
                        }
                        if _tilt_in is not None:
                            pitches_dict[group]["ssw_magnitude_hint"] = (
                                _ssw_fraction_from_clock(_tilt_in, group) * 1.5
                            )

                    # ── LHP catcher's-view HB detection ──────────────────────
                    # Model expects arm-side-positive (positive = arm-side regardless of
                    # hand). Trackman/Rapsodo reports LHP pitches in catcher's view, so
                    # coaches pull LHP arm-side pitches with NEGATIVE HB. Detect this and
                    # flip all HBs back to arm-side-positive before scoring.
                    hb_auto_flipped = False
                    if hand_code == "L":
                        # Expected sign in arm-side-positive convention
                        _LHP_SIGN_EXPECTED = {
                            "4-Seam": +1, "2-Seam/Sinker": +1,
                            "Splitter": +1, "Changeup": +1,
                            "Slider": -1, "Sweeper": -1,
                        }
                        _hb_wrong = _hb_total = 0
                        for _g_check, _pd_check in pitches_dict.items():
                            _hb_check = _pd_check.get("hb")
                            _exp_sign = _LHP_SIGN_EXPECTED.get(_g_check)
                            if _hb_check is None or _exp_sign is None or _hb_check == 0:
                                continue
                            _hb_total += 1
                            if (1 if _hb_check > 0 else -1) != _exp_sign:
                                _hb_wrong += 1
                        if _hb_total > 0 and _hb_wrong > _hb_total / 2:
                            for _g_flip in pitches_dict:
                                if pitches_dict[_g_flip].get("hb") is not None:
                                    pitches_dict[_g_flip]["hb"] = -pitches_dict[_g_flip]["hb"]
                            hb_auto_flipped = True

                    if pitches_dict:
                        with st.spinner("Scoring pitches…"):
                            scores = _score_v5_arsenal(
                                pitches=pitches_dict,
                                rel_height=_dm_rh,
                                rel_side=_dm_rs,
                                extension=_dm_ext,
                                hand=hand_code,
                            )
                        if scores:
                            st.session_state["_dm_cache"] = {
                                "scores":          scores,
                                "pitches_dict":    pitches_dict,
                                "hand_code":       hand_code,
                                "dm_rh":           _dm_rh,
                                "dm_rs":           _dm_rs,
                                "dm_ext":          _dm_ext,
                                "dm_added":        list(_dm_added),
                                "missing_velo":    missing_velo,
                                "hb_auto_flipped": hb_auto_flipped,
                                "data_source":     _data_source,
                                "plot":            {},
                                "plot_error":      None,
                                "top5":            [],
                                "baseline_mean":   100.0,
                                "scored_vals":     [scores[g]["stuff_plus"]
                                                    for g in _dm_added if g in scores],
                            }

                            # ── Movement (5×5 coarse + 5×5 fine = 50 calls/pitch) ──
                            try:
                                import matplotlib.pyplot as _plt
                                import matplotlib.patheffects as _pe

                                _PLT_COLORS = {
                                    "4-Seam": "#e63946", "2-Seam/Sinker": "#f4a261",
                                    "Cutter": "#2a9d8f", "Slider": "#457b9d",
                                    "Sweeper": "#a855f7", "Curveball": "#e9c46a",
                                    "Splitter": "#f4845f", "Changeup": "#90be6d",
                                    "Knuckleball": "#adb5bd",
                                }

                                # Collect user pts in arm-side-positive convention.
                                # pitches_dict["hb"] is already arm-side-positive.
                                # hb_arm_in in shape_row is glove-side-positive → negate.
                                _user_pts = {}
                                for _g in _dm_added:
                                    if _g not in scores:
                                        continue
                                    _sr = scores[_g].get("shape_row", {})
                                    _hb_val  = pitches_dict[_g].get("hb")  if _g in pitches_dict else None
                                    _ivb_val = pitches_dict[_g].get("ivb") if _g in pitches_dict else None
                                    if _hb_val is None:
                                        _hb_arm_in = _sr.get("hb_arm_in")
                                        _hb_val = (-_hb_arm_in) if _hb_arm_in is not None else None
                                    if _ivb_val is None: _ivb_val = _sr.get("ivb_in")
                                    if _hb_val is not None and _ivb_val is not None:
                                        _user_pts[_g] = (float(_hb_val), float(_ivb_val))

                                _SEARCH_BOUNDS = {
                                    "4-Seam":        {"ivb": ( 8, 25), "hb": ( 0, 20)},
                                    "2-Seam/Sinker": {"ivb": ( 2, 18), "hb": ( 5, 25)},
                                    "Cutter":        {"ivb": ( 1, 14), "hb": (-10,  8)},
                                    "Slider":        {"ivb": (-6, 10), "hb": (-20,  5)},
                                    "Sweeper":       {"ivb": (-5,  8), "hb": (-28, -3)},
                                    "Curveball":     {"ivb": (-20, 0), "hb": (-20,  5)},
                                    "Splitter":      {"ivb": (-3, 12), "hb": (  3, 20)},
                                    "Changeup":      {"ivb": ( 0, 15), "hb": (  5, 22)},
                                    "Knuckleball":   {"ivb": (-8,  8), "hb": (-10, 10)},
                                }

                                def _score_mean(pd_dict):
                                    _s = _score_v5_arsenal(
                                        pitches=pd_dict,
                                        rel_height=_dm_rh, rel_side=_dm_rs,
                                        extension=_dm_ext, hand=hand_code,
                                    )
                                    if not _s:
                                        return None
                                    _u = {g: pd_dict.get(g, {}).get("usage_pct")
                                          for g in pd_dict if g in _s}
                                    _info = _score_arsenal_combined(
                                        {g: _s[g] for g in pd_dict if g in _s}, usage=_u,
                                    )
                                    # Use overall (avg of both platoon sides) so optimisation
                                    # targets the same metric shown in the display.
                                    _rhb = _info.get("arsenal_stuff_plus_vs_rhb")
                                    _lhb = _info.get("arsenal_stuff_plus_vs_lhb")
                                    if _rhb is not None and _lhb is not None:
                                        return (_rhb + _lhb) / 2.0
                                    _asp = _info.get("arsenal_stuff_plus")
                                    if _asp is not None:
                                        return float(_asp)
                                    _vs = [_s[g]["stuff_plus"] for g in pd_dict if g in _s]
                                    return sum(_vs) / len(_vs) if _vs else None

                                # Penalty per inch of movement away from current position.
                                # Keeps targets close to current shape (least-change preference).
                                _DIST_PENALTY = 0.20

                                def _best_movement(grp, base_dict, max_delta=4.0):
                                    """Coarse-to-fine 2-pass grid search.

                                    Grid resolution chosen as a tradeoff between accuracy
                                    and wall-time: 4×4 coarse + 4×4 fine = 32 model calls
                                    per pitch (vs the previous 5×5+5×5 = 50). At a 4-inch
                                    range, the 4×4 step is ~1.33", which is finer than the
                                    plot's visual resolution can convey. Total speedup is
                                    ~36% on the optimisation spinner with no visible
                                    accuracy loss on the movement plot.
                                    """
                                    _GRID_N = 4
                                    _bd = _SEARCH_BOUNDS.get(grp, {"ivb": (-15, 25), "hb": (-25, 25)})
                                    _pd_m = base_dict[grp]
                                    _velo_m  = _pd_m.get("velo", 90)
                                    _spin_m  = _pd_m.get("spin_rate")
                                    _usage_m = _pd_m.get("usage_pct")
                                    _cur_ivb_m = _pd_m.get("ivb") or _V5_MEDIANS["ivb_in"]
                                    _cur_hb_m  = _pd_m.get("hb") or (-_V5_MEDIANS["hb_arm_in"])
                                    _iv_lo = max(_bd["ivb"][0], _cur_ivb_m - max_delta)
                                    _iv_hi = min(_bd["ivb"][1], _cur_ivb_m + max_delta)
                                    _hb_lo = max(_bd["hb"][0],  _cur_hb_m  - max_delta)
                                    _hb_hi = min(_bd["hb"][1],  _cur_hb_m  + max_delta)

                                    def _eval_grid(iv_lo, iv_hi, hb_lo, hb_hi):
                                        best = (-1e9, None, None)
                                        for _iv in np.linspace(iv_lo, iv_hi, _GRID_N):
                                            for _hb_v in np.linspace(hb_lo, hb_hi, _GRID_N):
                                                _trial = dict(base_dict)
                                                _trial[grp] = {"velo": _velo_m, "ivb": _iv,
                                                               "hb": _hb_v, "spin_rate": _spin_m,
                                                               "usage_pct": _usage_m}
                                                _sp_m = _score_mean(_trial)
                                                if _sp_m is None:
                                                    continue
                                                _dist = np.sqrt((_iv - _cur_ivb_m)**2
                                                                + (_hb_v - _cur_hb_m)**2)
                                                _obj = _sp_m - _DIST_PENALTY * _dist
                                                if _obj > best[0]:
                                                    best = (_obj, _iv, _hb_v)
                                        return best

                                    _best_obj, _best_iv_m, _best_hb_vm = _eval_grid(
                                        _iv_lo, _iv_hi, _hb_lo, _hb_hi
                                    )
                                    if _best_iv_m is None:
                                        return None, None
                                    # Fine pass around coarse optimum
                                    _iv_step = (_iv_hi - _iv_lo) / (_GRID_N - 1)
                                    _hb_step = (_hb_hi - _hb_lo) / (_GRID_N - 1)
                                    _fiv_lo = max(_iv_lo, _best_iv_m - _iv_step)
                                    _fiv_hi = min(_iv_hi, _best_iv_m + _iv_step)
                                    _fhb_lo = max(_hb_lo, _best_hb_vm - _hb_step)
                                    _fhb_hi = min(_hb_hi, _best_hb_vm + _hb_step)
                                    _obj2, _iv2, _hb2 = _eval_grid(_fiv_lo, _fiv_hi, _fhb_lo, _fhb_hi)
                                    if _obj2 > _best_obj:
                                        _best_iv_m, _best_hb_vm = _iv2, _hb2
                                    return _best_iv_m, _best_hb_vm

                                # Current score before any optimisation
                                _current_mean = _score_mean(pitches_dict) or 0.0

                                _added_pitches = []
                                if _current_mean >= 120.0:
                                    # Already A+ — no movement changes needed.
                                    # Target = current positions so no confusing arrows.
                                    _opt_dict = {g: dict(v) for g, v in pitches_dict.items()}
                                    _opt_mean = _current_mean
                                else:
                                    # Optimise movement — two passes so each pitch can react
                                    # to changes the previous pass made to its neighbours.
                                    with st.spinner("Optimising movement profile…"):
                                        _opt_dict = {g: dict(v) for g, v in pitches_dict.items()}
                                        for _pass in range(2):
                                            for _g in list(pitches_dict.keys()):
                                                _oi, _oh = _best_movement(_g, _opt_dict)
                                                if _oi is not None:
                                                    _opt_dict[_g]["ivb"] = _oi
                                                    _opt_dict[_g]["hb"]  = _oh

                                    # Non-regression guard: revert if optimisation degraded score
                                    _opt_mean = _score_mean(_opt_dict) or 0.0
                                    if _opt_mean < _current_mean - 0.1:
                                        _opt_dict = {g: dict(v) for g, v in pitches_dict.items()}
                                        _opt_mean = _current_mean

                                    # Try adding a pitch only if still below A+ after movement opt
                                    _missing_grps = [g for g in _MLB_PITCH_MEDIANS
                                                     if g not in pitches_dict]
                                    while _opt_mean < 120.0 and _missing_grps:
                                        _best_add_sp = _opt_mean
                                        _best_add_g  = None
                                        for _cand in _missing_grps:
                                            _try_d = dict(_opt_dict)
                                            _try_d[_cand] = dict(_MLB_PITCH_MEDIANS[_cand])
                                            _sp_a = _score_mean(_try_d)
                                            if _sp_a is not None and _sp_a > _best_add_sp:
                                                _best_add_sp = _sp_a; _best_add_g = _cand
                                        if _best_add_g is None:
                                            break
                                        _opt_dict[_best_add_g] = dict(_MLB_PITCH_MEDIANS[_best_add_g])
                                        _missing_grps.remove(_best_add_g)
                                        _added_pitches.append(_best_add_g)
                                        _opt_mean = _best_add_sp

                                # Build match_pts only for the user's own pitches
                                # (added pitches are annotated in the caption, not plotted)
                                _match_pts = {}
                                for _g in pitches_dict:
                                    _pd_mp = _opt_dict.get(_g, {})
                                    _oi_mp = _pd_mp.get("ivb"); _oh_mp = _pd_mp.get("hb")
                                    if _oi_mp is not None and _oh_mp is not None:
                                        _match_pts[_g] = (float(_oh_mp), float(_oi_mp))

                                if _current_mean >= 120.0:
                                    _aplus_label = f"A+ ✓ current shapes · {_opt_mean:.1f}"
                                else:
                                    _aplus_label = f"A+ target · {_opt_mean:.1f}"
                                    if _added_pitches:
                                        _aplus_label += f" (+ {', '.join(_added_pitches)})"

                                st.session_state["_dm_cache"]["plot"] = {
                                    "user_pts":      _user_pts,
                                    "match_pts":     _match_pts,
                                    "aplus_label":   _aplus_label,
                                    "added_pitches": _added_pitches,
                                    "opt_mean":      _opt_mean,
                                    "plt_colors":    _PLT_COLORS,
                                    "already_aplus": (_current_mean >= 120.0),
                                }
                            except Exception as _plot_err:
                                st.session_state["_dm_cache"]["plot_error"] = str(_plot_err)

                            # ── Suggestions ───────────────────────────────────
                            try:
                                _FB_GROUPS_S = {"4-Seam", "2-Seam/Sinker"}

                                def _rescore_mean(mod_pitches, rh=_dm_rh, rs=_dm_rs,
                                                  ext=_dm_ext, hand=hand_code):
                                    _s = _score_v5_arsenal(
                                        pitches=mod_pitches,
                                        rel_height=rh, rel_side=rs, extension=ext, hand=hand,
                                    )
                                    if not _s:
                                        return None
                                    _u = {g: mod_pitches.get(g, {}).get("usage_pct")
                                          for g in mod_pitches if g in _s}
                                    _info = _score_arsenal_combined(
                                        {g: _s[g] for g in mod_pitches if g in _s}, usage=_u,
                                    )
                                    # Use overall (avg of both platoon sides) to match display.
                                    _rhb = _info.get("arsenal_stuff_plus_vs_rhb")
                                    _lhb = _info.get("arsenal_stuff_plus_vs_lhb")
                                    if _rhb is not None and _lhb is not None:
                                        return (_rhb + _lhb) / 2.0
                                    _asp = _info.get("arsenal_stuff_plus")
                                    if _asp is not None:
                                        return float(_asp)
                                    vals = [_s[g]["stuff_plus"] for g in mod_pitches if g in _s]
                                    return sum(vals) / len(vals) if vals else None

                                _scored_vals_sugg = [scores[g]["stuff_plus"]
                                                     for g in _dm_added if g in scores]
                                _baseline_arsenal = _rescore_mean(pitches_dict)
                                _baseline_mean = (_baseline_arsenal if _baseline_arsenal is not None
                                                  else (sum(_scored_vals_sugg) / len(_scored_vals_sugg)
                                                        if _scored_vals_sugg else 100.0))

                                _suggestions = []

                                with st.spinner("Computing suggestions…"):
                                    # 1. Add each missing MLB pitch type
                                    _current_pitches = set(pitches_dict.keys())
                                    for _grp, _meds in _MLB_PITCH_MEDIANS.items():
                                        if _grp in _current_pitches:
                                            continue
                                        _try_d = dict(pitches_dict)
                                        _try_d[_grp] = dict(_meds)
                                        _new_mean = _rescore_mean(_try_d)
                                        if _new_mean is not None:
                                            _delta = _new_mean - _baseline_mean
                                            _suggestions.append((
                                                _delta,
                                                f"Add MLB-avg {_grp}",
                                                f"{_meds['velo']:.1f} mph · "
                                                f"{_meds['ivb']:+.1f}″ iVB · "
                                                f"{_meds['hb']:+.1f}″ HB",
                                                {"type": "add_pitch", "group": _grp,
                                                 "values": dict(_meds)},
                                            ))

                                    # 2. Movement tweaks (iVB ±2.5 in, HB ±2.5 in)
                                    for _grp, _pd_s in pitches_dict.items():
                                        _cur_ivb = _pd_s.get("ivb")
                                        _cur_hb  = _pd_s.get("hb")
                                        for _feat, _cur_val, _label_feat in [
                                            ("ivb", _cur_ivb, "iVB"),
                                            ("hb",  _cur_hb,  "HB"),
                                        ]:
                                            for _sign, _sign_str in [(+1, "+"), (-1, "−")]:
                                                _try_d = {g: dict(v) for g, v in pitches_dict.items()}
                                                _new_val = (_cur_val if _cur_val is not None else
                                                            (_V5_MEDIANS["ivb_in"] if _feat == "ivb"
                                                             else _V5_MEDIANS["hb_arm_in"])) + _sign * 2.5
                                                _try_d[_grp][_feat] = _new_val
                                                _new_mean = _rescore_mean(_try_d)
                                                if _new_mean is not None:
                                                    _delta = _new_mean - _baseline_mean
                                                    _suggestions.append((
                                                        _delta,
                                                        f"{_sign_str}2.5″ {_label_feat} on {_grp}",
                                                        f"{_new_val:+.1f}″ {_label_feat}",
                                                        {"type": "set_pitch_field", "group": _grp,
                                                         "field": _feat, "value": round(_new_val, 1)},
                                                    ))

                                    # 3. Velo tweaks on offspeed/breaking (+2 mph and +3 mph)
                                    for _grp, _pd_s in pitches_dict.items():
                                        if _grp in _FB_GROUPS_S:
                                            continue
                                        _cur_v = _pd_s.get("velo", 0)
                                        for _delta_v in [2.0, 3.0]:
                                            for _sign, _sign_str in [(+1, "+")]:
                                                _try_d = {g: dict(v) for g, v in pitches_dict.items()}
                                                _new_v = _cur_v + _sign * _delta_v
                                                _try_d[_grp]["velo"] = _new_v
                                                _new_mean = _rescore_mean(_try_d)
                                                if _new_mean is not None:
                                                    _delta = _new_mean - _baseline_mean
                                                    _suggestions.append((
                                                        _delta,
                                                        f"{_sign_str}{_delta_v:.0f} mph on {_grp}",
                                                        f"{_new_v:.1f} mph",
                                                        {"type": "set_pitch_field", "group": _grp,
                                                         "field": "velo", "value": round(_new_v, 1)},
                                                    ))

                                    # 4. Release profile tweaks
                                    _cur_rh  = _dm_rh  if _dm_rh  is not None else _V5_MEDIANS["rel_height"]
                                    _cur_rs  = _dm_rs  if _dm_rs  is not None else abs(_V5_MEDIANS["rel_side_arm"])
                                    _cur_ext = _dm_ext if _dm_ext is not None else _V5_MEDIANS["extension"]
                                    for _param, _cur, _lo, _hi, _label, _rh_arg, _rs_arg, _ext_arg, _ss_key in [
                                        ("rh",  _cur_rh,  3.0, 8.0, "Rel Height", None,    _cur_rs, _cur_ext, "dm_rh"),
                                        ("rs",  _cur_rs,  0.0, 5.0, "Rel Side",   _cur_rh, None,    _cur_ext, "dm_rs"),
                                        ("ext", _cur_ext, 4.0, 8.0, "Extension",  _cur_rh, _cur_rs, None,     "dm_ext"),
                                    ]:
                                        for _sign, _sign_str in [(+1, "+"), (-1, "−")]:
                                            _new_val = _cur + _sign * 0.25
                                            if not (_lo <= _new_val <= _hi):
                                                continue
                                            _rh_use  = _new_val if _param == "rh"  else _rh_arg
                                            _rs_use  = _new_val if _param == "rs"  else _rs_arg
                                            _ext_use = _new_val if _param == "ext" else _ext_arg
                                            _new_mean = _rescore_mean(
                                                pitches_dict, rh=_rh_use, rs=_rs_use, ext=_ext_use)
                                            if _new_mean is not None:
                                                _delta = _new_mean - _baseline_mean
                                                _suggestions.append((
                                                    _delta,
                                                    f"{_sign_str}0.25 ft {_label}",
                                                    f"{_new_val:.2f} ft",
                                                    {"type": "set_release", "key": _ss_key,
                                                     "value": round(_new_val, 2)},
                                                ))

                                    # 5. Usage shifts — REMOVED. Usage recommendations are
                                    # now served by the dedicated Pitch Usage Verdict
                                    # section (v3b model, ranking-based). The legacy
                                    # rescore-based usage suggestions duplicated that
                                    # signal less reliably (they used the v5 arsenal
                                    # scorer which doesn't have the structural form
                                    # needed for safe usage counterfactuals).

                                    # 6. Remove each pitch — but only when the removal is
                                    # STABLE under re-computation. Two known instabilities:
                                    #   (a) Removing the primary fastball (highest priority
                                    #       in [4-Seam, 2-Seam/Sinker, Cutter] that the user
                                    #       has) shifts velo_diff/ivb_diff/hb_diff for every
                                    #       other pitch — the prediction becomes unreliable
                                    #       and often disagrees with the actual recompute.
                                    #   (b) For LHP: removing a pitch can flip the majority
                                    #       in the catcher's-view HB-sign auto-detection,
                                    #       producing a different scored arsenal than the
                                    #       suggestion previewed.
                                    # Also require the result keeps ≥ 2 pitches and the
                                    # predicted delta exceeds a conservative bar.
                                    _FB_PRIORITY_S = ["4-Seam", "2-Seam/Sinker", "Cutter"]
                                    _user_primary_fb = next(
                                        (fb for fb in _FB_PRIORITY_S if fb in pitches_dict),
                                        None,
                                    )

                                    def _flip_majority_after_remove(grp_removed):
                                        """For LHP only — would removing `grp_removed` change
                                        the auto-flip decision? Returns True if removal would
                                        alter the majority, which makes the suggestion unsafe."""
                                        if hand_code != "L":
                                            return False
                                        _EXP = {
                                            "4-Seam": +1, "2-Seam/Sinker": +1,
                                            "Splitter": +1, "Changeup": +1,
                                            "Slider": -1, "Sweeper": -1,
                                        }
                                        def _majority(pd_in):
                                            wrong = total = 0
                                            for g, pd in pd_in.items():
                                                hb = pd.get("hb")
                                                exp = _EXP.get(g)
                                                if hb is None or exp is None or hb == 0:
                                                    continue
                                                total += 1
                                                if (1 if hb > 0 else -1) != exp:
                                                    wrong += 1
                                            return total > 0 and wrong > total / 2
                                        # NOTE: pitches_dict here is POST-FLIP. To test the
                                        # majority on raw inputs we'd need session-state HBs.
                                        # Conservative proxy: if any pitch in the expected-sign
                                        # map is being removed and the arsenal is small (≤4),
                                        # skip — the majority is fragile.
                                        if grp_removed in _EXP and len(pitches_dict) <= 4:
                                            return True
                                        return False

                                    if len(pitches_dict) > 2:
                                        for _grp in list(pitches_dict.keys()):
                                            # (a) Never suggest removing the primary FB
                                            if _grp == _user_primary_fb:
                                                continue
                                            # (b) Skip LHP cases where the flip decision is fragile
                                            if _flip_majority_after_remove(_grp):
                                                continue
                                            _try_d = {g: dict(v) for g, v in pitches_dict.items()
                                                      if g != _grp}
                                            _new_mean = _rescore_mean(_try_d)
                                            if _new_mean is None:
                                                continue
                                            _delta = _new_mean - _baseline_mean
                                            # Conservative bar (was > 1.0). Removal is a
                                            # major arsenal change — only suggest when the
                                            # gain is unambiguous.
                                            if _delta <= 2.5:
                                                continue
                                            _suggestions.append((
                                                _delta,
                                                f"Remove {_grp}",
                                                f"{len(pitches_dict) - 1}-pitch arsenal",
                                                {"type": "remove_pitch", "group": _grp},
                                            ))

                                _suggestions.sort(key=lambda x: -x[0])
                                st.session_state["_dm_cache"]["top5"] = [
                                    s for s in _suggestions if s[0] > 0.05
                                ][:5]
                                st.session_state["_dm_cache"]["baseline_mean"] = _baseline_mean
                            except Exception as _sug_err:
                                import sys as _sys, traceback as _tb_sug
                                print(f"[suggestions] failed: {_sug_err}\n"
                                      f"{_tb_sug.format_exc()}", file=_sys.stderr)
                                st.session_state["_dm_cache"]["top5"] = []
                                st.session_state["_dm_cache"]["suggestions_error"] = str(_sug_err)

                            if _auto_compute:
                                # Sync input widgets after an Apply → action
                                st.rerun(scope="app")

                # ── Render from cache ─────────────────────────────────────────
                if "_dm_cache" in st.session_state:
                    _C       = st.session_state["_dm_cache"]
                    _scores  = _C["scores"]
                    _pdict   = _C["pitches_dict"]
                    _hcode   = _C["hand_code"]
                    _c_rh    = _C["dm_rh"]
                    _c_rs    = _C["dm_rs"]
                    _c_ext   = _C["dm_ext"]
                    _c_added = _C["dm_added"]

                    if _C.get("missing_velo"):
                        st.markdown(
                            f"<div style='max-width:680px;margin:16px auto;padding:14px 18px;"
                            f"border:1px solid #c4914830;border-radius:6px;background:#1a1410;"
                            f"font-family:JetBrains Mono,monospace;font-size:11px;color:#c0a878'>"
                            f"⚠ Missing velocity for: {', '.join(_C['missing_velo'])} — these pitches were skipped."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    if _C.get("hb_auto_flipped"):
                        st.markdown(
                            "<div style='max-width:680px;margin:8px auto 16px auto;"
                            "padding:10px 16px;border:1px solid #2a4a6a40;"
                            "border-radius:6px;background:#0a1420;"
                            "font-family:JetBrains Mono,monospace;font-size:10px;color:#5a8aaa'>"
                            "ℹ HB signs auto-converted: entered values looked like catcher's-view "
                            "(Trackman) format where LHP arm-side is negative. Flipped to "
                            "arm-side-positive for the model. Plot shows catcher's view (LHP "
                            "arm-side appears on the left)."
                            "</div>",
                            unsafe_allow_html=True,
                        )

                    _src_label = _C.get("data_source", _DATA_SOURCES[0])
                    if _src_label and _src_label != _DATA_SOURCES[0]:
                        st.markdown(
                            f"<div style='max-width:680px;margin:8px auto 16px auto;"
                            f"padding:10px 16px;border:1px solid #2a4a6a40;"
                            f"border-radius:6px;background:#0a1420;"
                            f"font-family:JetBrains Mono,monospace;font-size:10px;color:#5a8aaa'>"
                            f"ℹ Movement-data source: <b style='color:#c0d8e8'>{_src_label}</b>. "
                            f"iVB/HB inputs were adjusted to Hawk-Eye/Statcast "
                            f"equivalents before scoring. Displayed values below are the "
                            f"adjusted (model-internal) numbers."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # Zone model status indicator
                    _zone_status = ""
                    if _ZONE_AVAILABLE:
                        _z_ver = (_zone_bundle or {}).get("version", "unknown")
                        _zone_status = (
                            f"<span style='font-family:JetBrains Mono,monospace;"
                            f"font-size:10px;color:#3a8a5a;margin-left:12px;"
                            f"letter-spacing:1px'>● zone heatmaps: {_z_ver}</span>"
                        )
                    else:
                        _zone_status = (
                            "<span style='font-family:JetBrains Mono,monospace;"
                            "font-size:10px;color:#7a5a3a;margin-left:12px;"
                            "letter-spacing:1px'>● zone heatmaps: not loaded "
                            "(no models/zone_stuff_*.joblib found)</span>"
                        )
                    st.markdown(
                        "<div style='font-family:Inter,sans-serif;font-size:13px;font-weight:700;"
                        "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                        "margin:24px 0 12px 0'>"
                        f"DM Stuff+ Results{_zone_status}</div>",
                        unsafe_allow_html=True,
                    )
    
                    _v6_help = ""
                    if any(("stuff_plus_p10" in _scores[g] or
                             "stuff_plus_vs_rhb" in _scores[g] or
                             "nearest_pitcher" in _scores[g] or
                             "ood_warnings" in _scores[g])
                            for g in _c_added if g in _scores):
                        _v6_help = (
                            "<br><span style='color:#5a7a90'>"
                            "range = 80% confidence interval (P10–P90) · "
                            "vs RHB/LHB = platoon-specific Stuff+ · "
                            "closest MLB shape = nearest pitcher by feature distance · "
                            "⚠ = input outside training distribution."
                            "</span>"
                        )
                    _hm_note_pre = (
                        "<br><span style='color:#5a7a90'>Heatmaps show predicted Stuff+ "
                        "by zone (catcher's view; gold = elite, blue = below avg).</span>"
                    ) if _ZONE_AVAILABLE else ""
                    st.markdown(
                        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                        "color:#3a5a78;margin:0 0 14px 0;line-height:1.7;padding:10px 14px;"
                        "background:#0a1218;border:1px solid #1a2a40;border-radius:6px'>"
                        f"<b style='color:{_BRAND_GOLD}'>120+</b> elite &nbsp;·&nbsp; "
                        "<b style='color:#a0c0d4'>105–115</b> above avg &nbsp;·&nbsp; "
                        "<b style='color:#8a9aac'>95–105</b> avg &nbsp;·&nbsp; "
                        "<b style='color:#6a7a8a'>&lt;95</b> below avg<br>"
                        f"Scale: per-pitch-type, mean=100, SD=10. Trained on {_DATA_YEAR_RANGE} Statcast."
                        + _hm_note_pre + _v6_help +
                        "</div>",
                        unsafe_allow_html=True,
                    )
    
                    for group in _c_added:
                        if group not in _scores:
                            continue
                        # Use overall (avg vs RHB + vs LHB) when available;
                        # fall back to opposite-hand default if platoon not scored.
                        sp_val  = _scores[group].get(
                            "stuff_plus_overall", _scores[group]["stuff_plus"]
                        )
                        imputed = _scores[group]["imputed"]
                        color   = PITCH_COLORS[group]
    
                        if sp_val >= 115:   sp_color = "#d4a848"
                        elif sp_val >= 105: sp_color = "#a0c0d4"
                        elif sp_val >= 95:  sp_color = "#8a9aac"
                        else:               sp_color = "#6a7a8a"
    
                        imputed_user = [f for f in imputed if f in
                                        ("ivb", "hb", "spin_rate", "rel_height",
                                         "rel_side", "extension")]
                        imp_str = (f"<span style='font-family:JetBrains Mono,monospace;"
                                    f"font-size:10px;color:#5a7a90'>"
                                    f"imputed: {', '.join(imputed_user)}</span>") \
                                   if imputed_user else \
                                   ("<span style='font-family:JetBrains Mono,monospace;"
                                    "font-size:10px;color:#5ac8a040'>all fields provided</span>")
    
                        _sd = _scores[group]
                        _ci_html = ""
                        if "stuff_plus_p10" in _sd and "stuff_plus_p90" in _sd:
                            _ci_html = (
                                f"<span style='font-family:JetBrains Mono,monospace;"
                                f"font-size:10px;color:#5a7a90;margin-left:8px'>"
                                f"range {_sd['stuff_plus_p10']}–{_sd['stuff_plus_p90']}</span>"
                            )
                        _plat_html = ""
                        if "stuff_plus_vs_rhb" in _sd and "stuff_plus_vs_lhb" in _sd:
                            _plat_html = (
                                f"<div style='font-family:JetBrains Mono,monospace;"
                                f"font-size:10px;color:#7a9ab0;margin-top:3px'>"
                                f"vs RHB: <b style='color:#a0c0d4'>{_sd['stuff_plus_vs_rhb']}</b>"
                                f" &nbsp;·&nbsp; vs LHB: <b style='color:#a0c0d4'>{_sd['stuff_plus_vs_lhb']}</b>"
                                f"</div>"
                            )
                        _nn_html = ""
                        if "nearest_pitcher" in _sd:
                            _nn = _sd["nearest_pitcher"]
                            _nn_html = (
                                f"<div style='font-family:JetBrains Mono,monospace;"
                                f"font-size:10px;color:#6a8a9a;margin-top:3px'>"
                                f"closest MLB shape: <b style='color:#c0d8e8'>"
                                f"{_nn['name']} ({_nn['year']})</b>"
                                f"</div>"
                            )
                        _ood_html = ""
                        if "ood_warnings" in _sd and _sd["ood_warnings"]:
                            _ood_items = []
                            for w in _sd["ood_warnings"][:3]:
                                _color = "#d4a848" if w["severity"] == "extreme" else "#c0a878"
                                _ood_items.append(
                                    f"<span style='color:{_color}'>"
                                    f"{w['feat']}={w['value']} (typical: "
                                    f"{w['range'][0]}–{w['range'][1]})</span>"
                                )
                            _ood_html = (
                                "<div style='font-family:JetBrains Mono,monospace;"
                                "font-size:10px;color:#c0a878;margin-top:3px;"
                                "border-top:1px dashed #c4914830;padding-top:4px'>"
                                "⚠ outside training range: "
                                + " &nbsp;·&nbsp; ".join(_ood_items)
                                + "</div>"
                            )

                        # ── Approach angles (VAA / HAA) ────────────────────
                        _vaa_html = ""
                        _shape_for_aa = _sd.get("shape_row", {})
                        _vaa_v = _shape_for_aa.get("vaa_raw")
                        _haa_v = _shape_for_aa.get("haa_raw")
                        if _vaa_v is not None and _haa_v is not None:
                            # VAA shown as descending angle (negative); flatter = closer to 0.
                            # HAA: positive = arm-side run from pitcher's POV.
                            _vaa_html = (
                                "<div style='font-family:JetBrains Mono,monospace;"
                                "font-size:10px;color:#7a9ab0;margin-top:3px'>"
                                f"VAA: <b style='color:#a0c0d4'>{_vaa_v:.1f}°</b>"
                                "  &nbsp;·&nbsp;  "
                                f"HAA: <b style='color:#a0c0d4'>{_haa_v:+.1f}°</b>"
                                "  <span style='color:#5a7a90'>(estimated)</span>"
                                "</div>"
                            )

                        # ── 1-line explainer (#8) ──────────────────────────
                        _explainer_html = ""
                        _explainer_txt = _pitch_explainer(
                            group, _sd.get("shape_row"), sp_val,
                            _sd.get("ood_warnings", []), imputed,
                        )
                        if _explainer_txt:
                            _explainer_html = (
                                "<div style='font-family:Inter,sans-serif;"
                                "font-size:11px;color:#9aaec0;margin-top:6px;"
                                "padding-top:6px;border-top:1px solid #1a2a4060;"
                                "line-height:1.5;font-style:italic'>"
                                f"{_explainer_txt}"
                                "</div>"
                            )

                        # ── Confidence band (#1) ───────────────────────────
                        # If we have P10/P90, render a subtle horizontal bar
                        # under the score number to visualize the uncertainty.
                        _band_html = ""
                        if "stuff_plus_p10" in _sd and "stuff_plus_p90" in _sd:
                            _p10 = float(_sd["stuff_plus_p10"])
                            _p90 = float(_sd["stuff_plus_p90"])
                            _band_w = max(2.0, min(60.0, (_p90 - _p10) * 1.8))
                            _band_html = (
                                "<div style='display:flex;justify-content:center;"
                                "margin-top:4px'>"
                                f"<div style='height:3px;width:{_band_w:.0f}px;"
                                f"background:linear-gradient(90deg,{sp_color}30,{sp_color}80,{sp_color}30);"
                                "border-radius:2px'></div></div>"
                            )

                        st.markdown(
                            f"<div style='display:flex;align-items:center;justify-content:space-between;"
                            f"padding:14px 20px;margin-bottom:8px;"
                            f"background:linear-gradient(165deg,#0e1828 0%,#0c1420 100%);"
                            f"border-left:3px solid {color};border-radius:6px'>"
                            f"<div style='display:flex;flex-direction:column;gap:4px;flex:1'>"
                            f"<div style='font-family:Inter,sans-serif;font-size:13px;"
                            f"font-weight:700;color:{color};letter-spacing:2px;"
                            f"text-transform:uppercase'>{group}</div>"
                            f"<div>{imp_str}{_ci_html}</div>"
                            f"{_plat_html}{_vaa_html}{_nn_html}{_ood_html}"
                            f"{_explainer_html}"
                            f"</div>"
                            f"<div style='display:flex;flex-direction:column;align-items:center;"
                            f"margin-left:16px'>"
                            f"<div style='font-family:Inter,sans-serif;font-size:32px;"
                            f"font-weight:800;color:{sp_color}'>{sp_val}</div>"
                            f"{_band_html}"
                            f"</div>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        # "Load this pitcher" quick button (#15) — pre-fills
                        # the calculator with the nearest-MLB-shape pitcher's
                        # arsenal in one click.
                        _nn_for_load = _sd.get("nearest_pitcher")
                        if _nn_for_load and profiles is not None:
                            _nn_nm = _nn_for_load.get("name")
                            _nn_yr = _nn_for_load.get("year")
                            _btn_key = f"_nn_load_{group}_{_nn_nm}_{_nn_yr}"
                            if st.button(
                                f"⤴  Load {_nn_nm} ({_nn_yr})'s full arsenal "
                                f"into the calculator",
                                key=_btn_key,
                            ):
                                # Build a pending-load packet; widget-key
                                # writes happen at the top of the next run
                                # to avoid Streamlit's "cannot modify after
                                # widget instantiated" error.
                                try:
                                    _sub_lp = profiles[
                                        (profiles["player_name"] == _nn_nm)
                                        & (profiles["year"] == _nn_yr)
                                    ]
                                    if not _sub_lp.empty:
                                        _row_lp = _sub_lp.iloc[0]
                                        _h_lp = str(_row_lp.get("hand","R")).upper()[:1]
                                        _pending_pitches_nn = []
                                        for _grp_lp in PITCH_GROUPS:
                                            _n = _row_lp.get(f"n_{_grp_lp}")
                                            _v = _row_lp.get(f"velo_{_grp_lp}")
                                            if pd.notna(_n) and _n > 25 and pd.notna(_v):
                                                _vals_nn = {"velo": f"{float(_v):.1f}"}
                                                for _src, _f in [
                                                        ("ivb",       "ivb"),
                                                        ("hb",        "hb"),
                                                        ("spin_rate", "spin"),
                                                        ("pct",       "usage")]:
                                                    _val = _row_lp.get(f"{_src}_{_grp_lp}")
                                                    if pd.notna(_val):
                                                        if _src == "pct":
                                                            _vals_nn[_f] = f"{float(_val)*100:.1f}"
                                                        else:
                                                            _vals_nn[_f] = f"{float(_val):.2f}"
                                                _pending_pitches_nn.append((_grp_lp, _vals_nn))
                                        st.session_state["_dm_pending_load"] = {
                                            "hand":        "LHP" if _h_lp == "L" else "RHP",
                                            "rh":          (float(_row_lp["rel_height"])
                                                              if pd.notna(_row_lp.get("rel_height")) else None),
                                            "rs":          (float(_row_lp["rel_side"])
                                                              if pd.notna(_row_lp.get("rel_side")) else None),
                                            "ext":         (float(_row_lp["extension"])
                                                              if pd.notna(_row_lp.get("extension")) else None),
                                            "pitches":     _pending_pitches_nn,
                                            "data_source": _DATA_SOURCES[0],
                                            "toast":       f"Loaded {_nn_nm} ({int(_nn_yr)}). "
                                                           f"Scroll up — entire arsenal is filled.",
                                        }
                                        st.rerun()
                                except Exception as _nn_load_err:
                                    st.warning(f"Couldn't load: {_nn_load_err}")

                        # Zone-Stuff+ heatmaps
                        if _ZONE_AVAILABLE and "shape_row" in _scores[group]:
                            shape_row = _scores[group]["shape_row"]
                            zone_grid = _score_zone_grid(shape_row, pitcher_hand=_hcode)
                            _z_cov_all = (_zone_bundle or {}).get("zone_coverage") or {}
                            _z_cov = _z_cov_all.get(group) if isinstance(_z_cov_all, dict) else None
                            if zone_grid:
                                hm_rhb = _render_zone_heatmap_svg(
                                    zone_grid["vs_rhb"], "vs RHB", zone_coverage=_z_cov,
                                )
                                hm_lhb = _render_zone_heatmap_svg(
                                    zone_grid["vs_lhb"], "vs LHB", zone_coverage=_z_cov,
                                )
                                hm_cols = st.columns([1, 1])
                                with hm_cols[0]:
                                    st.markdown(hm_rhb, unsafe_allow_html=True)
                                with hm_cols[1]:
                                    st.markdown(hm_lhb, unsafe_allow_html=True)
                                st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)



                    # ── Arsenal Grade ─────────────────────────────────────────────
                    _scored_vals = _C.get("scored_vals", [])
                    if _scored_vals:
                        _usage_dict = {
                            g: _pdict.get(g, {}).get("usage_pct")
                            for g in _c_added if g in _scores
                        }
                        _arsenal_info = _score_arsenal_combined(
                            {g: _scores[g] for g in _c_added if g in _scores},
                            usage=_usage_dict,
                        )
                        _arsenal_sp_rhb = _arsenal_info.get("arsenal_stuff_plus_vs_rhb")
                        _arsenal_sp_lhb = _arsenal_info.get("arsenal_stuff_plus_vs_lhb")
                        # Show true overall (avg vs RHB + vs LHB) when both platoon
                        # scores are available; fall back to opposite-hand default.
                        if _arsenal_sp_rhb is not None and _arsenal_sp_lhb is not None:
                            _arsenal_sp = round(
                                (_arsenal_sp_rhb + _arsenal_sp_lhb) / 2.0, 1
                            )
                        else:
                            _arsenal_sp = _arsenal_info.get("arsenal_stuff_plus", 100.0)
                        # Cache for save/compare/PDF (#7, #19, #20)
                        st.session_state["_dm_cache"]["display_arsenal_sp"]     = _arsenal_sp
                        st.session_state["_dm_cache"]["display_arsenal_vs_rhb"] = _arsenal_sp_rhb
                        st.session_state["_dm_cache"]["display_arsenal_vs_lhb"] = _arsenal_sp_lhb
                        _grade_subtitle = (
                            "Usage-weighted raw aggregation, league-standardized"
                            if _arsenal_info.get("method") == "raw_aggregation"
                            else "Unweighted avg (no arsenal norms in bundle)"
                        )
    
                        if _arsenal_sp >= 120:   _grade = "A+"
                        elif _arsenal_sp >= 112: _grade = "A"
                        elif _arsenal_sp >= 107: _grade = "B+"
                        elif _arsenal_sp >= 102: _grade = "B"
                        elif _arsenal_sp >= 97:  _grade = "C+"
                        elif _arsenal_sp >= 92:  _grade = "C"
                        else:                    _grade = "D"
                        st.session_state["_dm_cache"]["display_arsenal_grade"] = _grade

                        if _arsenal_sp >= 107:   _grade_color = _BRAND_GOLD
                        elif _arsenal_sp >= 97:  _grade_color = "#a0c0d4"
                        elif _arsenal_sp >= 87:  _grade_color = "#8a9aac"
                        else:                    _grade_color = "#6a7a8a"
    
                        _platoon_html = ""
                        if _arsenal_sp_rhb is not None and _arsenal_sp_lhb is not None:
                            _platoon_html = (
                                "<div style='font-family:JetBrains Mono,monospace;"
                                "font-size:10px;color:#7a9ab0;margin-top:3px'>"
                                f"vs RHB: <b style='color:#a0c0d4'>{_arsenal_sp_rhb}</b>"
                                f" &nbsp;·&nbsp; vs LHB: <b style='color:#a0c0d4'>{_arsenal_sp_lhb}</b>"
                                "</div>"
                            )

                        # Aggregate per-pitch confidence intervals to arsenal
                        # level using the same usage weighting that produces
                        # the headline number. Each per-pitch p10/p90 is on
                        # the same 100/10 scale, so the usage-weighted sum
                        # is the arsenal-level CI on the same scale.
                        _arsenal_p10 = _arsenal_p90 = None
                        _usage_for_ci = {
                            g: _pdict.get(g, {}).get("usage_pct")
                            for g in _c_added if g in _scores
                        }
                        # Fill in MLB-typical usage when missing so the
                        # weights normalize to 1 over the arsenal.
                        _w_sum_ci = 0.0
                        _u_ci = {}
                        for g, u in _usage_for_ci.items():
                            uv = (float(u) if u is not None
                                  else float(_MLB_USAGE_FALLBACK.get(g, 15.0)))
                            _u_ci[g] = max(uv, 0.01)
                            _w_sum_ci += _u_ci[g]
                        if _w_sum_ci > 0:
                            _norm_u = {g: v / _w_sum_ci for g, v in _u_ci.items()}
                            _p10s, _p90s = [], []
                            for g in _u_ci:
                                _s = _scores.get(g, {})
                                _p10g = _s.get("stuff_plus_p10")
                                _p90g = _s.get("stuff_plus_p90")
                                if _p10g is not None and _p90g is not None:
                                    _p10s.append((_norm_u[g], float(_p10g)))
                                    _p90s.append((_norm_u[g], float(_p90g)))
                            if len(_p10s) == len(_u_ci) and _p10s:
                                _arsenal_p10 = sum(w * v for w, v in _p10s)
                                _arsenal_p90 = sum(w * v for w, v in _p90s)
                        _ci_arsenal_html = ""
                        if _arsenal_p10 is not None and _arsenal_p90 is not None:
                            _half = (_arsenal_p90 - _arsenal_p10) / 2.0
                            _ci_arsenal_html = (
                                f"<div style='font-family:JetBrains Mono,monospace;"
                                f"font-size:10px;color:#7a9ab0;margin-top:6px'>"
                                f"80% CI: <b style='color:#a0c0d4'>"
                                f"{_arsenal_p10:.1f} – {_arsenal_p90:.1f}</b>"
                                f"  &nbsp;·&nbsp; ± {_half:.1f}"
                                f"</div>"
                            )

                        st.markdown(
                            "<div style='margin:28px 0 8px 0;padding:18px 24px;"
                            "background:linear-gradient(165deg,#0e1828,#0a1520);"
                            "border:1px solid #1a2a40;border-radius:8px;"
                            "display:flex;align-items:center;justify-content:space-between'>"
                            "<div>"
                            "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                            f"color:{_BRAND_GOLD};letter-spacing:2px;text-transform:uppercase;"
                            "margin-bottom:4px'>Arsenal Stuff+</div>"
                            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                            f"color:#3a5a78'>{_grade_subtitle}</div>"
                            f"{_platoon_html}"
                            f"{_ci_arsenal_html}"
                            "</div>"
                            "<div style='display:flex;align-items:baseline;gap:14px'>"
                            f"<div style='font-family:Inter,sans-serif;font-size:36px;"
                            f"font-weight:800;color:{_grade_color}'>{_arsenal_sp:.1f}</div>"
                            f"<div style='font-family:Inter,sans-serif;font-size:22px;"
                            f"font-weight:700;color:{_grade_color};opacity:0.7'>{_grade}</div>"
                            "</div></div>",
                            unsafe_allow_html=True,
                        )

                        # ── Grade strip (#17): horizontal gradient with MLB
                        # percentile markers and current pitcher's position.
                        # Scale spans D (≈80) → A+ (≈130). Map to 0-100% width.
                        _strip_lo, _strip_hi = 80.0, 130.0
                        def _pct(v):
                            return max(0.0, min(100.0,
                                       (v - _strip_lo) / (_strip_hi - _strip_lo) * 100.0))
                        # MLB percentile anchors (approx, from training distribution)
                        _mlb_anchors = [
                            (92.0,  "p25",  "#6a7a8a"),
                            (100.0, "p50",  "#8a9aac"),
                            (108.0, "p75",  "#a0c0d4"),
                            (118.0, "p95",  _BRAND_GOLD),
                        ]
                        _anchor_html = ""
                        for _val, _lbl, _col in _mlb_anchors:
                            _x = _pct(_val)
                            _anchor_html += (
                                f"<div style='position:absolute;left:{_x:.1f}%;"
                                f"top:50%;transform:translate(-50%,-50%);"
                                f"width:1px;height:14px;background:{_col}40'></div>"
                                f"<div style='position:absolute;left:{_x:.1f}%;"
                                f"top:18px;transform:translateX(-50%);"
                                f"font-family:JetBrains Mono,monospace;font-size:9px;"
                                f"color:{_col};white-space:nowrap'>"
                                f"{_lbl}<br><span style='color:#5a7a90'>{_val:.0f}</span></div>"
                            )
                        _here = _pct(_arsenal_sp)
                        _here_html = (
                            f"<div style='position:absolute;left:{_here:.1f}%;"
                            f"top:50%;transform:translate(-50%,-50%);"
                            f"width:14px;height:14px;border-radius:50%;"
                            f"background:{_grade_color};border:2px solid #0c1420;"
                            f"box-shadow:0 0 8px {_grade_color}80;z-index:5'></div>"
                            f"<div style='position:absolute;left:{_here:.1f}%;"
                            f"top:-20px;transform:translateX(-50%);"
                            f"font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                            f"color:{_grade_color};white-space:nowrap'>"
                            f"YOU · {_arsenal_sp:.1f}</div>"
                        )
                        st.markdown(
                            "<div style='margin:8px 0 36px 0;padding:0 24px'>"
                            "<div style='position:relative;height:10px;border-radius:5px;"
                            "background:linear-gradient(90deg,"
                            "#5a3a3a 0%,#6a7a8a 24%,#8a9aac 40%,"
                            "#a0c0d4 56%,#d4a848 76%,#f5d068 100%)'>"
                            f"{_anchor_html}{_here_html}"
                            "</div></div>",
                            unsafe_allow_html=True,
                        )
    
                    # ── Movement Plot ─────────────────────────────────────────────
                    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                        "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                        "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                        "● Movement Profile vs Closest A+ Arsenal</div>",
                        unsafe_allow_html=True,
                    )
                    _cplot = _C.get("plot", {})
                    if _cplot:
                        try:
                            import matplotlib.pyplot as _plt
                            import matplotlib.patheffects as _pe
                            from matplotlib.lines import Line2D as _L2D
    
                            _user_pts_r      = _cplot["user_pts"]
                            _match_pts_r     = _cplot["match_pts"]
                            _aplus_label     = _cplot["aplus_label"]
                            _added_pitches_r = _cplot["added_pitches"]
                            _PLT_COLORS_R    = _cplot["plt_colors"]

                            # user_pts / match_pts stored in arm-side-positive convention.
                            # Display in catcher's view: RHP arm-side → right (+),
                            # LHP arm-side → left (−), so flip HBs for LHP.
                            _plt_sign = -1 if _hcode == "L" else 1
                            _user_pts_disp  = {g: (_plt_sign * hb, ivb)
                                               for g, (hb, ivb) in _user_pts_r.items()}
                            _match_pts_disp = {g: (_plt_sign * hb, ivb)
                                               for g, (hb, ivb) in _match_pts_r.items()}
                            if _hcode == "L":
                                _x_label = "Horizontal Break — catcher's view  (← LHP arm-side)"
                            else:
                                _x_label = "Horizontal Break — arm-side + (in)"

                            _fig, _ax = _plt.subplots(figsize=(4, 3.2))
                            _fig.patch.set_facecolor("#0c1420")
                            _ax.set_facecolor("#0e1828")
                            for _spine in _ax.spines.values():
                                _spine.set_edgecolor("#1a2a40")
                            _ax.tick_params(colors="#5a7a90", labelsize=9)
                            _ax.set_xlabel(_x_label, color="#5a7a90", fontsize=10)
                            _ax.set_ylabel("Induced Vertical Break (in)", color="#5a7a90", fontsize=10)
                            _ax.set_xlim(-25, 25)
                            _ax.set_ylim(-25, 25)
                            _ax.axhline(0, color="#1a2a40", lw=1, zorder=0)
                            _ax.axvline(0, color="#1a2a40", lw=1, zorder=0)
                            _ax.grid(True, color="#1a2a40", lw=0.5, alpha=0.6, zorder=0)

                            # Minimum movement (in.) to show a diamond/arrow.
                            # Suppresses visual noise for already-optimal pitches.
                            _MIN_MOVE = 0.5
                            _already_aplus_r = _cplot.get("already_aplus", False)

                            # Diamonds — only where movement exceeds threshold
                            _has_targets = False
                            for _g, (_hb, _ivb) in _match_pts_disp.items():
                                if _g not in _user_pts_disp:
                                    continue
                                _ux0, _uy0 = _user_pts_disp[_g]
                                if np.sqrt((_hb - _ux0)**2 + (_ivb - _uy0)**2) < _MIN_MOVE:
                                    continue
                                _has_targets = True
                                _c = _PLT_COLORS_R.get(_g, "#aaaaaa")
                                _ax.scatter(_hb, _ivb, s=180, facecolors="none",
                                            edgecolors=_c, linewidths=2,
                                            marker="D", alpha=0.7, zorder=3)

                            # Nearest-MLB-shape comp stars were here. Removed
                            # by request so the movement plot stays uncluttered.
                            # Comp pitchers are still available as
                            # "⤴ Load <Name> ('YY)'s full arsenal" buttons
                            # under each per-pitch card.

                            # (#11) Usage-weighted dot size — primary pitches
                            # dominate visually, show-me pitches stay small.
                            _plotted = []
                            for _g, (_hb, _ivb) in _user_pts_disp.items():
                                _c = _PLT_COLORS_R.get(_g, "#aaaaaa")
                                _u = _pdict.get(_g, {}).get("usage_pct")
                                if _u is None:
                                    try:
                                        _u = _MLB_USAGE_FALLBACK.get(_g, 15.0)
                                    except NameError:
                                        _u = 15.0
                                # Map usage 0-100% → marker size 80-420
                                _dot_size = max(80.0, min(420.0, 80.0 + 3.4 * float(_u)))
                                _ax.scatter(_hb, _ivb, s=_dot_size, color=_c,
                                            edgecolors="white", linewidths=1.2,
                                            marker="o", zorder=5, label=_g)
                                _ax.annotate(_g, (_hb, _ivb),
                                             textcoords="offset points", xytext=(8, 5),
                                             fontsize=8, color=_c,
                                             path_effects=[_pe.withStroke(linewidth=2, foreground="#0e1828")])
                                _plotted.append(_g)

                            # Arrows — only for meaningful movement
                            for _g in _plotted:
                                if _g in _match_pts_disp:
                                    _ux, _uy = _user_pts_disp[_g]
                                    _mx, _my = _match_pts_disp[_g]
                                    if np.sqrt((_mx - _ux)**2 + (_my - _uy)**2) < _MIN_MOVE:
                                        continue
                                    _c = _PLT_COLORS_R.get(_g, "#aaaaaa")
                                    _ax.plot([_ux, _mx], [_uy, _my],
                                             color=_c, lw=1, linestyle="--", alpha=0.35, zorder=2)

                            _target_label = "A+ target" if _has_targets else "A+ current shape"
                            _legend_elems = [
                                _L2D([0],[0], marker="o", color="w", markerfacecolor="#aaaaaa",
                                     markersize=8, label="Your arsenal (size=usage)", linestyle="None"),
                            ]
                            if _has_targets:
                                _legend_elems.append(
                                    _L2D([0],[0], marker="D", color="w", markerfacecolor="none",
                                         markeredgecolor="#aaaaaa", markersize=8,
                                         label="A+ target shape", linestyle="None")
                                )
                            _ax.legend(handles=_legend_elems, facecolor="#0e1828",
                                       edgecolor="#1a2a40", labelcolor="#a0c0d4",
                                       fontsize=7, loc="best")
                            _plt.tight_layout(pad=1.0)
                            
                            import io as _io_plot
                            _img_buf = _io_plot.BytesIO()
                            _plt.savefig(_img_buf, format='png', bbox_inches='tight', dpi=200, facecolor=_fig.get_facecolor())
                            _C["plot_img_bytes"] = _img_buf.getvalue()
                            
                            st.pyplot(_fig, use_container_width=True)
                            _plt.close(_fig)

                            if _already_aplus_r:
                                _caption = (
                                    f"<b style='color:#a0c0d4'>{_aplus_label}</b> — "
                                    "your current shapes already achieve A+. "
                                    "No movement changes needed."
                                )
                            elif _has_targets:
                                _caption = (
                                    f"<b style='color:#a0c0d4'>{_aplus_label}</b> — "
                                    "diamonds show the minimum movement change per pitch "
                                    "(same velo &amp; release) toward A+. "
                                    "Pitches without diamonds are already in their optimal range."
                                )
                            else:
                                _caption = (
                                    f"<b style='color:#a0c0d4'>{_aplus_label}</b> — "
                                    "all pitches are within 0.5″ of their optimal shapes. "
                                    "Current movement profile is near-optimal."
                                )
                            if _added_pitches_r:
                                _caption += (
                                    f" To reach A+, consider adding: "
                                    f"<b style='color:#c49148'>{', '.join(_added_pitches_r)}</b>."
                                )
                            st.markdown(
                                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                f"color:#3a5a78;margin-top:6px;padding:0 4px;line-height:1.6'>"
                                f"{_caption}</div>",
                                unsafe_allow_html=True,
                            )
                        except Exception as _plot_err2:
                            st.markdown(
                                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                f"color:#5a3a3a;padding:8px'>Movement plot unavailable: {_plot_err2}</div>",
                                unsafe_allow_html=True,
                            )
                    elif _C.get("plot_error"):
                        st.markdown(
                            f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                            f"color:#5a3a3a;padding:8px'>Movement plot unavailable: {_C['plot_error']}</div>",
                            unsafe_allow_html=True,
                        )

                    # ── Pitch Usage Verdict (v3b ranking) ────────────────────────
                    # Tells the coach in plain language which pitches to throw
                    # MORE and which to throw LESS — based on the model's
                    # per-pitch quality ranking vs the pitcher's current usage
                    # ranking. No magnitudes, no point estimates: just clear
                    # directional verdicts. This is the most reliable signal
                    # the model can produce (CF pass rate = 100% — the
                    # ranking direction is mathematically guaranteed safe
                    # under the structural aggregation).
                    if _USAGE_V3B_AVAILABLE:
                        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                        st.markdown(
                            "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                            f"color:{_BRAND_GOLD};letter-spacing:2px;text-transform:uppercase;"
                            "margin:0 0 4px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                            "● Pitch Usage Verdict "
                            "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>"
                            "(usage allocation only — v3b model)</span></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                            "color:#5a7a90;margin:0 0 12px 0;padding:10px 14px;"
                            "background:#0a1218;border:1px solid #1a2a40;border-radius:6px;"
                            "line-height:1.7'>"
                            "Compares each pitch's <b style='color:#a0c0d4'>model quality rank</b> "
                            "to its <b style='color:#a0c0d4'>current usage rank</b> within your arsenal. "
                            "When the model thinks a pitch is better than you're treating it, the verdict is "
                            "<b style='color:#5ac8a0'>USE MORE</b>; when worse, "
                            "<b style='color:#d48a8a'>USE LESS</b>. "
                            "Direction is the reliable signal — exact percentages aren't."
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        try:
                            _v3b_pd = _pdict
                            # Compute per-pitch f() averaged across stands.
                            _v3b_quality = {}
                            for _g, _pd_g in _v3b_pd.items():
                                _vals = []
                                for _stand_code in ("R", "L"):
                                    _m = _usage_v3b.get(f"model_vs_{_stand_code}HB")
                                    if _m is None: continue
                                    _is_same = 1 if str(_hcode).upper().startswith(_stand_code) else 0
                                    _row = _v3b_per_pitch_feature_row(
                                        pitch_group=_g,
                                        velo=_pd_g.get("velo"),
                                        ivb=_pd_g.get("ivb"),
                                        hb_arm_positive=_pd_g.get("hb"),
                                        spin_rate=_pd_g.get("spin_rate"),
                                        rel_height=_c_rh,
                                        rel_side_arm=_c_rs,
                                        extension=_c_ext,
                                        hand=_hcode,
                                        is_same_hand=_is_same,
                                    )
                                    if _row is None: continue
                                    _X = pd.DataFrame([_row], columns=_usage_v3b["feature_names"])
                                    _vals.append(float(_m.predict(_X)[0]))
                                if _vals:
                                    _v3b_quality[_g] = sum(_vals) / len(_vals)

                            if len(_v3b_quality) < 2:
                                st.markdown(
                                    "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                                    "color:#5a7a90;padding:12px 0'>"
                                    "Need at least 2 pitches in the arsenal to compare quality rankings."
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                # Build rank dicts (1 = best/most). Use MLB-typical
                                # usage as a fallback for pitches with no usage entered.
                                _v3b_usage = {}
                                for _g in _v3b_quality:
                                    _u = (_v3b_pd[_g].get("usage_pct")
                                            if _g in _v3b_pd else None)
                                    if _u is None:
                                        _u = _MLB_USAGE_FALLBACK.get(_g, 15.0)
                                    _v3b_usage[_g] = float(_u)
                                # Rank by quality (high f = rank 1)
                                _q_sorted = sorted(_v3b_quality.items(),
                                                     key=lambda x: -x[1])
                                _q_rank = {g: i + 1 for i, (g, _) in enumerate(_q_sorted)}
                                # Rank by usage (high usage = rank 1)
                                _u_sorted = sorted(_v3b_usage.items(),
                                                     key=lambda x: -x[1])
                                _u_rank = {g: i + 1 for i, (g, _) in enumerate(_u_sorted)}

                                # Build per-pitch verdict rows, ordered by
                                # strongest disagreement (|rank delta|) so the
                                # most actionable items are at top.
                                _rows = []
                                for _g, _q_v in _v3b_quality.items():
                                    _qr = _q_rank[_g]
                                    _ur = _u_rank[_g]
                                    _delta_rank = _ur - _qr  # >0 → use MORE (under-used)
                                    _rows.append({
                                        "group":       _g,
                                        "quality_val": _q_v,
                                        "q_rank":      _qr,
                                        "u_rank":      _ur,
                                        "delta_rank":  _delta_rank,
                                        "usage_pct":   _v3b_usage[_g],
                                    })
                                _rows.sort(key=lambda r: (-abs(r["delta_rank"]),
                                                           -r["quality_val"]))

                                for _r in _rows:
                                    _g = _r["group"]
                                    _dr = _r["delta_rank"]
                                    _c = PITCH_COLORS.get(_g, "#aaaaaa")
                                    # Verdict logic
                                    if _dr >= 2:
                                        _verdict = "USE MORE"
                                        _vcolor = "#5ac8a0"
                                        _strength = "STRONG"
                                    elif _dr == 1:
                                        _verdict = "use more"
                                        _vcolor = "#a0c0d4"
                                        _strength = "mild"
                                    elif _dr == 0:
                                        _verdict = "about right"
                                        _vcolor = "#8a9aac"
                                        _strength = ""
                                    elif _dr == -1:
                                        _verdict = "use less"
                                        _vcolor = "#c8a878"
                                        _strength = "mild"
                                    else:
                                        _verdict = "USE LESS"
                                        _vcolor = "#d48a8a"
                                        _strength = "STRONG"
                                    # Build qualitative description
                                    _rank_q_word = _ordinal_word(_r["q_rank"])
                                    _rank_u_word = _ordinal_word(_r["u_rank"])
                                    _detail = (
                                        f"Quality rank {_rank_q_word}  ·  "
                                        f"usage rank {_rank_u_word}  "
                                        f"({_r['usage_pct']:.0f}% currently)"
                                    )
                                    _strength_chip = (
                                        f"<span style='font-family:JetBrains Mono,monospace;"
                                        f"font-size:9px;color:{_vcolor};letter-spacing:1px;"
                                        f"margin-left:8px;opacity:0.7'>{_strength}</span>"
                                    ) if _strength else ""
                                    st.markdown(
                                        f"<div style='display:flex;align-items:center;gap:14px;"
                                        f"padding:12px 18px;margin-bottom:6px;"
                                        f"background:linear-gradient(165deg,#0e1828,#0a1218);"
                                        f"border-left:3px solid {_c};border-radius:6px'>"
                                        f"<div style='font-family:Inter,sans-serif;font-size:13px;"
                                        f"font-weight:700;color:{_c};letter-spacing:2px;"
                                        f"text-transform:uppercase;min-width:140px'>{_g}</div>"
                                        f"<div style='flex:1;font-family:JetBrains Mono,monospace;"
                                        f"font-size:10px;color:#7a9ab0'>{_detail}</div>"
                                        f"<div style='font-family:Inter,sans-serif;font-size:14px;"
                                        f"font-weight:800;color:{_vcolor};text-align:right'>"
                                        f"{_verdict}{_strength_chip}</div>"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )
                                # Inline footnote
                                st.markdown(
                                    "<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                                    "color:#3a5a78;margin-top:6px;padding:0 4px;line-height:1.6'>"
                                    "A pitch with quality rank #1 but usage rank #3 means the model "
                                    "thinks it's your best per-pitch but you're throwing it third-most "
                                    "— a candidate to use more. STRONG = ≥2 rank places of disagreement; "
                                    "mild = 1 rank place. Model is correct on directional rank "
                                    "comparisons more often than not but is not a guarantee — "
                                    "platoon matchups, count tendencies, and pitcher feel still rule."
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                        except Exception as _v3b_err:
                            st.markdown(
                                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                f"color:#5a3a3a;padding:8px'>Usage model unavailable: {_v3b_err}</div>",
                                unsafe_allow_html=True,
                            )

                    # ── Top 5 Improvement Suggestions ────────────────────────────
                    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                        "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                        "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                        "● Top 5 Improvement Suggestions "
                        "<span style='color:#3a5a78;font-size:9px;font-weight:400;letter-spacing:1px'>"
                        "(shape / velo / release — NOT usage)</span></div>",
                        unsafe_allow_html=True,
                    )

                    def _apply_suggestion(action):
                        t = action["type"]
                        if t == "add_pitch":
                            grp = action["group"]
                            pitches_list = st.session_state.setdefault("_dmsp_pitches", [])
                            if grp not in pitches_list:
                                pitches_list.append(grp)
                            vals = action.get("values", {})
                            _SUFFIX = {"velo": "_velo", "ivb": "_ivb",
                                       "hb": "_hb", "spin_rate": "_spin"}
                            for _field, _suf in _SUFFIX.items():
                                if _field in vals and vals[_field] is not None:
                                    st.session_state[f"dm_{grp}{_suf}"] = f"{vals[_field]}"
                        elif t == "set_pitch_field":
                            grp = action["group"]
                            _suf_map = {"velo": "_velo", "ivb": "_ivb",
                                        "hb": "_hb", "spin_rate": "_spin",
                                        "usage": "_usage"}
                            suf = _suf_map[action["field"]]
                            st.session_state[f"dm_{grp}{suf}"] = f"{action['value']}"
                        elif t == "set_release":
                            st.session_state[action["key"]] = float(action["value"])
                        elif t == "remove_pitch":
                            grp = action["group"]
                            pl = st.session_state.get("_dmsp_pitches", [])
                            if grp in pl:
                                pl.remove(grp)
                            for _suf in ["_velo", "_ivb", "_hb", "_spin", "_usage", "_tilt"]:
                                st.session_state.pop(f"dm_{grp}{_suf}", None)
                            st.session_state.pop("_dm_cache", None)
                        st.session_state["_dm_auto_compute"] = True
    
                    _top5 = _C.get("top5", [])
                    _sug_err_msg = _C.get("suggestions_error")
                    if _sug_err_msg:
                        st.markdown(
                            f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                            f"color:#a08070;padding:10px 14px;background:#1a1410;"
                            f"border:1px solid #3a2820;border-radius:6px;margin:0 0 12px 0'>"
                            f"⚠ Suggestions failed to compute. "
                            f"Reason: <code>{_sug_err_msg}</code>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    if not _top5:
                        st.markdown(
                            "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                            "color:#3a5a78;padding:16px 0'>No tested changes improved the arsenal score.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        for _rank, (_delta, _lbl, _detail, _action) in enumerate(_top5, 1):
                            if _delta >= 2.0:   _d_color = _BRAND_GOLD
                            elif _delta >= 1.0: _d_color = "#a0c0d4"
                            else:               _d_color = "#8a9aac"
                            _row_l, _row_r = st.columns([10, 1.4])
                            with _row_l:
                                st.markdown(
                                    f"<div style='display:flex;align-items:center;gap:16px;"
                                    f"padding:12px 18px;margin-bottom:6px;"
                                    f"background:linear-gradient(165deg,#0e1828,#0a1218);"
                                    f"border-left:3px solid {_d_color};border-radius:6px'>"
                                    f"<div style='font-family:Inter,sans-serif;font-size:18px;"
                                    f"font-weight:800;color:{_d_color};min-width:24px'>#{_rank}</div>"
                                    f"<div style='flex:1'>"
                                    f"<div style='font-family:Inter,sans-serif;font-size:13px;"
                                    f"font-weight:700;color:#c8d8e8;margin-bottom:3px'>{_lbl}</div>"
                                    f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                    f"color:#5a7a90'>{_detail}</div>"
                                    f"</div>"
                                    f"<div style='font-family:Inter,sans-serif;font-size:16px;"
                                    f"font-weight:700;color:{_d_color}'>+{_delta:.1f}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            with _row_r:
                                st.markdown("<div style='margin-top:14px'></div>",
                                             unsafe_allow_html=True)
                                st.button(
                                    "Apply →",
                                    key=f"_apply_sugg_{_rank}",
                                    on_click=_apply_suggestion,
                                    args=(_action,),
                                    width='stretch',
                                )
                        st.markdown(
                            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                            "color:#3a5a78;margin-top:10px;padding:0 4px'>"
                            "Δ = change in arsenal Stuff+. "
                            "Movement ±2.5\", velo ±2/3 mph, release ±0.25 ft, usage ±20%, pitch removal also tested. "
                            "Click <b style='color:#a0c0d4'>Apply →</b> to update inputs and re-score."
                            "</div>",
                            unsafe_allow_html=True,
                        )

            _dm_results_frag()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "results":

    results       = st.session_state.results
    snap          = st.session_state.user_snapshot
    user          = snap["user"]
    pitch_inputs  = snap["pitch_inputs"]
    result_mode   = snap.get("mode", "arsenal")
    sp_pitch_type = snap.get("sp_pitch_type")
    user_dmsp     = st.session_state.get("user_dmsp", {})

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Empty state ───────────────────────────────────────────────────────────
    if not results:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        _, ec, _ = st.columns([1, 4, 1])
        with ec:
            st.markdown(
                "<div style='text-align:center;padding:48px 32px;"
                "background:linear-gradient(165deg,#0e1828,#0a1218);"
                "border:1px solid #1a2a40;border-radius:16px'>"
                "<div style='font-size:48px;margin-bottom:16px'>🔍</div>"
                "<div style='font-family:Inter,sans-serif;font-size:18px;font-weight:800;"
                "color:#d4a848;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px'>"
                "No Matches Found</div>"
                "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:#4a6880;line-height:1.8;margin-bottom:24px'>"
                "No similarity scores returned.<br>"
                "This usually means the search hasn't been run yet — "
                "<br>head back and enter at least one pitch metric to try again.</div>"
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        _, bc, _ = st.columns([2, 1, 2])
        with bc:
            st.markdown('<div class="back-btn-wrap">', unsafe_allow_html=True)
            if st.button("← Try Again"):
                st.session_state.screen  = "input"
                st.session_state.results = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_back, hdr_title = st.columns([1, 9])
    with hdr_back:
        st.markdown('<div class="back-btn-wrap">', unsafe_allow_html=True)
        if st.button("← New Search"):
            st.session_state.screen  = "title"
            st.session_state.results = None
            st.session_state.pop("_arsenal_pitches", None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with hdr_title:
        st.markdown(
            "<div style='font-family:Inter,sans-serif;font-size:22px;font-weight:800;"
            "color:#5ac8a0;letter-spacing:3px;text-transform:uppercase;margin:6px 0 2px 0'>"
            "🔍 Similarity Results</div>"
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7aaac0;"
            "margin-bottom:4px'>Statcast 2017–2025 · Gaussian similarity model</div>",
            unsafe_allow_html=True,
        )

    # ── Profile strip ─────────────────────────────────────────────────────────
    parts = []
    if user.get("hand"):       parts.append(f"<b style='color:#e8dcc8'>{snap['hand_label']}</b>")
    if user.get("rel_height"): parts.append(f"HT <b style='color:#e8dcc8'>{user['rel_height']:.2f}'</b>")
    if user.get("rel_side"):   parts.append(f"SIDE <b style='color:#e8dcc8'>{abs(user['rel_side']):.2f}'</b>")
    if user.get("extension"):  parts.append(f"EXT <b style='color:#e8dcc8'>{user['extension']:.2f}'</b>")
    for g, m in pitch_inputs.items():
        subs = []
        if m.get("velo"): subs.append(f"{m['velo']:.1f}")
        if m.get("ivb"):  subs.append(f"iVB {m['ivb']:.1f}\"")
        if m.get("hb") is not None:
            subs.append(f"HB {m['hb']:+.1f}\"")
        if subs: parts.append(f"<b style='color:{PITCH_COLORS[g]}'>{g}</b>: {', '.join(subs)}")
    if parts:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a0c0d4;"
            "background:linear-gradient(165deg,#0e1828,#0c1420);padding:10px 20px;"
            "border-radius:10px;border:1px solid #162236;"
            "text-align:center;margin:4px 0 14px 0'>"
            "<span style='color:#d4a848;font-family:Inter,sans-serif;font-weight:700;"
            "letter-spacing:2px;text-transform:uppercase;font-size:10px'>PROFILE </span>"
            + " &nbsp;·&nbsp; ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )

    # ── Summary stat cards ────────────────────────────────────────────────────
    def _full(raw):
        if "," in raw:
            p = raw.split(",", 1)
            return f"{p[1].strip()} {p[0].strip()}"
        return raw

    top_name  = _full(results[0]["Pitcher"]) if results else "—"
    top_score = results[0]["Similarity"] if results else 0
    top_sc    = sim_color(top_score)

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    for col, label, val, val_color in [
        (sc1, "RESULTS",    str(len(results)),                      "#e8dcc8"),
        (sc2, "SEASONS",    "2017–2025",                            "#7aaac0"),
        (sc3, "TOP MATCH",  top_name,                               "#c9a84c"),
        (sc4, "BEST SCORE", f"{top_score:.1f}" if results else "—", top_sc),
        (sc5, "POOL SIZE",  f"{profiles['player_name'].nunique():,}", "#7aaac0"),
    ]:
        with col:
            col.markdown(
                f"<div style='background:linear-gradient(165deg,#0e1828,#0c1420);"
                f"border:1px solid #1a2a40;border-radius:10px;padding:12px 14px;"
                f"text-align:center;margin-bottom:2px'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
                f"color:#6a90a8;text-transform:uppercase;letter-spacing:1.5px;"
                f"margin-bottom:6px'>{label}</div>"
                f"<div style='font-family:Inter,sans-serif;font-size:15px;"
                f"font-weight:700;color:{val_color};line-height:1.1'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # COMP PANEL
    # ══════════════════════════════════════════════════════════════════════════
    if result_mode == "single":
        _grp_scores = {}
        for r in results:
            mp = r.get("Matched Pitch")
            if mp:
                _grp_scores.setdefault(mp, []).append(r["Similarity"])
        comp_groups = sorted(
            _grp_scores.keys(),
            key=lambda g: -sum(_grp_scores[g]) / max(len(_grp_scores[g]), 1)
        )
    else:
        pg_order = list(PITCH_GROUPS.keys())
        comp_groups = sorted(
            list(pitch_inputs.keys()) if pitch_inputs else [],
            key=lambda g: pg_order.index(g) if g in pg_order else 99
        )

    def stat_pill(label, val, color, sub=None):
        sub_html = (
            f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
            f"color:#6a90a8;margin-top:3px;letter-spacing:0.3px'>{sub}</div>"
        ) if sub else ""
        return (
            f"<div style='text-align:center;"
            f"background:linear-gradient(160deg,#0d1a28,#0a1420);"
            f"border:1px solid {color}30;border-radius:8px;padding:8px 4px'>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
            f"color:{color};text-transform:uppercase;letter-spacing:1px;"
            f"margin-bottom:4px;font-weight:600;opacity:0.8'>{label}</div>"
            f"<div style='font-family:Inter,sans-serif;font-size:16px;"
            f"font-weight:700;color:{color};line-height:1'>{val}</div>"
            f"{sub_html}</div>"
        )

    def render_comp_section(title_label, title_color, groups_to_show):
        st.markdown(
            f"<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
            f"color:{title_color};letter-spacing:2px;text-transform:uppercase;"
            f"margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
            f"● {title_label}</div>",
            unsafe_allow_html=True,
        )

        for grp in groups_to_show:
            color = PITCH_COLORS.get(grp, "#8aadcc")
            agg   = comp_aggregate_stats(results, pitch_group=grp)
            zdf   = comp_zone_data(results, pitch_group=grp)

            def fv(k, fmt=""):
                if k not in agg: return "—"
                v, _ = agg[k]
                if fmt == "pct":  return f"{v:.1%}"
                if fmt == "f3":   return f"{v:.3f}"
                if fmt == "f1":   return f"{v:.1f}"
                if fmt == "i":    return f"{v:.0f}"
                return f"{v:.1f}"

            def _fv_hb():
                if "hb" not in agg: return "—"
                raw, _ = agg["hb"]
                # Already in TrackMan convention (converted per-pitcher before averaging)
                return f"{raw:+.1f}"

            def _fv_haa():
                if "haa" not in agg: return "—"
                raw, _ = agg["haa"]
                return f"{raw:+.1f}"

            sp_val,    _ = agg.get("stuff_plus", (None, 0))
            csw_val,   _ = agg.get("csw",        (None, 0))
            xw_val,    _ = agg.get("xwoba",      (None, 0))
            vaa_val,   _ = agg.get("vaa",         (None, 0))
            haa_val,   _ = agg.get("haa",         (None, 0))
            whiff_val, _ = agg.get("whiff",       (None, 0))

            sp_c = stuff_color(sp_val) if sp_val else "#4a6a80"
            _pg_bl  = pitch_grp_league.loc[grp] if (not pitch_grp_league.empty and
                       grp and grp in pitch_grp_league.index) else None
            _csw_mu  = float(_pg_bl["csw_mu"])  if _pg_bl is not None else 0.32
            _csw_sd  = float(_pg_bl["csw_sd"])  if _pg_bl is not None else 0.05
            _xw_mu   = float(_pg_bl["xw_mu"])   if _pg_bl is not None else 0.31
            _xw_sd   = 0.06
            _whiff_mu = float(_pg_bl["whiff_mu"]) if (_pg_bl is not None and "whiff_mu" in _pg_bl.index) else 0.22
            _whiff_sd = float(_pg_bl["whiff_sd"]) if (_pg_bl is not None and "whiff_sd" in _pg_bl.index) else 0.08
            csw_c    = stat_gradient_color(csw_val,   _csw_mu,   _csw_sd)              if csw_val   is not None else "#2a4a5a"
            xw_c     = stat_gradient_color(xw_val,    _xw_mu,    _xw_sd,  invert=True) if xw_val    is not None else "#2a4a5a"
            whiff_c  = stat_gradient_color(whiff_val, _whiff_mu, _whiff_sd)            if whiff_val is not None else "#2a4a5a"

            _grp_hands = [r.get("Hand","R") for r in results
                          if r.get("Matched Pitch") == grp or not grp]
            _dom_hand = "R" if not _grp_hands or _grp_hands.count("R") >= _grp_hands.count("L") else "L"
            _vl = (_vaa_haa_league.get(f"{grp}_{_dom_hand}") or _vaa_haa_league.get(grp, {}))
            _vaa_raw = -vaa_val if vaa_val is not None else None
            vaa_c = stat_gradient_color(_vaa_raw, _vl["vaa_mu"], _vl["vaa_sd"], invert=True) if (_vl and _vaa_raw is not None) else "#2a4a5a"
            if _vl and haa_val is not None:
                # haa_val is raw Statcast; haa_mu is also Statcast → compare directly
                _hz = min(abs((haa_val - _vl["haa_mu"]) / max(_vl["haa_sd"], 0.001)), 2.0)
                _ht = _hz / 2.0
                haa_c = f"rgb({int(120+(220-120)*_ht)},{int(130+(35-130)*_ht)},{int(140+(35-140)*_ht)})"
            else:
                haa_c = "#2a4a5a"

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px;margin:0 0 8px 0'>"
                f"<div style='width:3px;height:16px;background:{color};border-radius:2px;flex-shrink:0'></div>"
                f"<div style='font-family:Inter,sans-serif;font-size:12px;font-weight:700;"
                f"color:{color};letter-spacing:1.5px;text-transform:uppercase'>{grp}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#4a6880'>"
                f"comp average across {len(results)} matches</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            pills_html = (
                "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:14px'>"
                + stat_pill("VELO",   fv("velo","f1"),   color)
                + stat_pill("iVB",    fv("ivb","f1"),    "#8aadcc")
                + stat_pill("HB",     _fv_hb(),          "#8aadcc")
                + stat_pill("VAA",    fv("vaa","f1"),    vaa_c)
                + stat_pill("HAA",    _fv_haa(),         haa_c)
                + stat_pill("STUFF+",
                            fv("stuff_plus","i") if sp_val else (f"{user_dmsp[grp]:.0f}" if grp in user_dmsp else "—"),
                            sp_c if sp_val else (stuff_color(user_dmsp[grp]) if grp in user_dmsp else "#2a4a5a"),
                            (stuff_grade_label(sp_val) if sp_val
                             else (f"{user_dmsp[grp]:.0f} yours" if grp in user_dmsp else "—")))
                + stat_pill("Whiff%", fv("whiff","pct"), whiff_c)
                + stat_pill("CSW%",   fv("csw","pct"),   csw_c)
                + stat_pill("xwOBA",  fv("xwoba","f3"),  xw_c)
                + "</div>"
            )
            st.markdown(pills_html, unsafe_allow_html=True)

            # Radar Chart
            user_m = pitch_inputs.get(grp, {})
            if user_m and all(k in user_m for k in ["velo", "ivb", "hb"]):
                with st.expander(f"🕸️  {grp} Radar Comparison vs Comp", expanded=False):
                    u_velo = user_m.get("velo", 0)
                    u_ivb = abs(user_m.get("ivb", 0))
                    u_hb = abs(user_m.get("hb", 0))
                    u_sp = user_dmsp.get(grp, 100)
                    
                    c_velo, _ = agg.get("velo", (0, 0))
                    c_ivb, _ = agg.get("ivb", (0, 0))
                    c_ivb = abs(c_ivb)
                    c_hb, _ = agg.get("hb", (0, 0))
                    c_hb = abs(c_hb)
                    c_sp, _ = agg.get("stuff_plus", (100, 0))
                    
                    def radar_norm(val, max_val):
                        return min(100, max(0, (val / max_val) * 100)) if max_val else 0
                        
                    cats = ['Velocity', 'Abs Vertical Break', 'Abs Horizontal Break', 'Stuff+']
                    
                    u_data = [radar_norm(u_velo, 105), radar_norm(u_ivb, 25), radar_norm(u_hb, 25), radar_norm(u_sp, 150)]
                    c_data = [radar_norm(c_velo, 105), radar_norm(c_ivb, 25), radar_norm(c_hb, 25), radar_norm(c_sp, 150)]
                    
                    cats.append(cats[0])
                    u_data.append(u_data[0])
                    c_data.append(c_data[0])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=u_data, theta=cats, fill='toself', name='Your Pitch',
                        line_color='#d4a848', fillcolor='rgba(212, 168, 72, 0.4)'
                    ))
                    fig.add_trace(go.Scatterpolar(
                        r=c_data, theta=cats, fill='toself', name='Comp Average',
                        line_color='#7aaac0', fillcolor='rgba(122, 170, 192, 0.4)'
                    ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=False, range=[0, 100]),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=40, r=40, t=30, b=30),
                        height=350,
                        font=dict(color='#d8cbb4', family='JetBrains Mono', size=11),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander(f"📊  {grp} comp zone heatmaps", expanded=False):
                hm_col1, hm_col2, hm_col3 = st.columns(3)
                with hm_col1: st.markdown(render_zone_heatmap(zdf,"csw_pct",  "csw",  "CSW% (All)",  fmt=".1%"), unsafe_allow_html=True)
                with hm_col2: st.markdown(render_zone_heatmap(zdf,"whiff_pct","whiff","Whiff% (All)",fmt=".1%"), unsafe_allow_html=True)
                with hm_col3: st.markdown(render_zone_heatmap(zdf,"xwoba_mean","xwoba","xwOBA (All)", fmt=".3f"), unsafe_allow_html=True)
                has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                if has_stand_col:
                    zdf_same = comp_zone_data(results, pitch_group=grp, stand="same")
                    zdf_opp  = comp_zone_data(results, pitch_group=grp, stand="opp")
                    if not zdf_same.empty:
                        st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs Same Hand</div>", unsafe_allow_html=True)
                        hs1,hs2,hs3 = st.columns(3)
                        with hs1: st.markdown(render_zone_heatmap(zdf_same,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                        with hs2: st.markdown(render_zone_heatmap(zdf_same,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                        with hs3: st.markdown(render_zone_heatmap(zdf_same,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
                    if not zdf_opp.empty:
                        st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs Opposite Hand</div>", unsafe_allow_html=True)
                        ho1,ho2,ho3 = st.columns(3)
                        with ho1: st.markdown(render_zone_heatmap(zdf_opp,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                        with ho2: st.markdown(render_zone_heatmap(zdf_opp,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                        with ho3: st.markdown(render_zone_heatmap(zdf_opp,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if result_mode == "single" and comp_groups:
        render_comp_section(f"Comp Average — {comp_groups[0]}", "#c9a84c", comp_groups)
    elif comp_groups:
        render_comp_section("Comp Average — Per Pitch", "#c9a84c", comp_groups)
        overall_zdf = comp_zone_data(results, pitch_group=None)
        if not overall_zdf.empty:
            with st.expander("📊  Overall all-pitch comp zone heatmaps", expanded=False):
                oz1,oz2,oz3 = st.columns(3)
                with oz1: st.markdown(render_zone_heatmap(overall_zdf,"csw_pct",  "csw",  "CSW% (All)",  fmt=".1%"), unsafe_allow_html=True)
                with oz2: st.markdown(render_zone_heatmap(overall_zdf,"whiff_pct","whiff","Whiff% (All)",fmt=".1%"), unsafe_allow_html=True)
                with oz3: st.markdown(render_zone_heatmap(overall_zdf,"xwoba_mean","xwoba","xwOBA (All)", fmt=".3f"), unsafe_allow_html=True)
                has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                if has_stand_col:
                    ov_same = comp_zone_data(results, pitch_group=None, stand="same")
                    ov_opp  = comp_zone_data(results, pitch_group=None, stand="opp")
                    if not ov_same.empty:
                        st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs Same Hand</div>", unsafe_allow_html=True)
                        os1,os2,os3 = st.columns(3)
                        with os1: st.markdown(render_zone_heatmap(ov_same,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                        with os2: st.markdown(render_zone_heatmap(ov_same,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                        with os3: st.markdown(render_zone_heatmap(ov_same,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
                    if not ov_opp.empty:
                        st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs Opposite Hand</div>", unsafe_allow_html=True)
                        oo1,oo2,oo3 = st.columns(3)
                        with oo1: st.markdown(render_zone_heatmap(ov_opp,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                        with oo2: st.markdown(render_zone_heatmap(ov_opp,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                        with oo3: st.markdown(render_zone_heatmap(ov_opp,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)

    # ── Full-width band separating comp from pitcher list ─────────────────────
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background:linear-gradient(90deg,#080c14,#0d1828 30%,#0d1828 70%,#080c14);"
        f"border-top:1px solid #1a2a40;border-bottom:1px solid #1a2a40;"
        f"padding:10px 0;margin:0 0 16px 0;text-align:center'>"
        f"<span style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
        f"color:#5ac8a0;letter-spacing:3px;text-transform:uppercase'>"
        f"── TOP {snap['top_n']} MATCHES ──</span></div>",
        unsafe_allow_html=True,
    )

    # ── Pitcher cards ─────────────────────────────────────────────────────────
    for idx, r in enumerate(results):
        row       = r["_row"]
        sc        = r["Similarity"]
        sc_c      = sim_color(sc)
        raw_name  = r["Pitcher"]
        if "," in raw_name:
            parts     = raw_name.split(",", 1)
            full_name = f"{parts[1].strip()} {parts[0].strip()}"
        else:
            full_name = raw_name
        ext_str      = f"{r['Extension']:.2f}" if r["Extension"] else "—"
        hand         = r["Hand"]
        rank         = idx + 1
        matched_pitch = r.get("Matched Pitch", "")
        pitch_badge  = f"· {matched_pitch}" if result_mode == "single" and matched_pitch else ""
        header = (
            f"#{rank}  {full_name}  {r['Year']}  ({hand}HP)  {pitch_badge}  SIM {sc:.1f}"
            f"  ·  HT {r['Rel Height']:.2f}  SIDE {abs(r['Rel Side']):.2f}  EXT {ext_str}"
        )

        with st.expander(header, expanded=(idx == 0)):

            fg_overall = None
            for _fg_key in ["stuff_plus"]:
                _fg_v = row.get(_fg_key)
                if _fg_v is None: continue
                try:
                    _fg_f = float(_fg_v)
                    if not math.isnan(_fg_f) and _fg_f > 0:
                        fg_overall = _fg_f
                        break
                except (TypeError, ValueError):
                    continue
            fg_has  = fg_overall is not None
            fg_str  = f"{fg_overall:.0f}" if fg_has else "—"
            fg_col  = stuff_color(fg_overall) if fg_has else "#3a6a8a"
            fg_lbl  = stuff_grade_label(fg_overall) if fg_has else "—"

            rank_colors = {1: "#d4a848", 2: "#9ab0c0", 3: "#c87850"}
            rank_c = rank_colors.get(rank, "#4a6880")

            st.markdown(
                f"<div style='display:flex;align-items:center;"
                f"background:linear-gradient(165deg,#0e1828,#0a1218);"
                f"border:1px solid #1a2a40;border-radius:12px;"
                f"padding:14px 18px;margin-bottom:14px;overflow:hidden;position:relative'>"
                f"<div style='font-family:Inter,sans-serif;font-size:48px;"
                f"font-weight:900;color:{rank_c};opacity:0.08;position:absolute;"
                f"left:12px;top:50%;transform:translateY(-50%);line-height:1;"
                f"z-index:0;pointer-events:none;user-select:none'>#{rank}</div>"
                f"<div style='flex:1;min-width:0;position:relative;z-index:1'>"
                f"<div style='font-family:Inter,sans-serif;font-size:16px;"
                f"font-weight:800;color:#e8dcc8;margin-bottom:4px'>{full_name}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                f"color:#6a90a8;display:flex;gap:12px;flex-wrap:wrap'>"
                f"<span>{r['Year']}</span><span>{hand}HP</span>"
                f"<span>HT {r['Rel Height']:.2f}'</span>"
                f"<span>SIDE {abs(r['Rel Side']):.2f}'</span>"
                f"<span>EXT {ext_str}'</span>"
                + (f"<span>{int(row.get('total_pitches',0)):,} pitches</span>" if is_real(row.get('total_pitches')) else "")
                + f"</div></div>"
                f"<div style='text-align:center;padding:0 24px;border-left:1px solid #1a2a40;"
                f"border-right:1px solid #1a2a40;margin:0 16px;flex-shrink:0'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
                f"color:#4a6880;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Similarity</div>"
                f"<div style='font-family:Inter,sans-serif;font-size:24px;"
                f"font-weight:900;color:{sc_c};line-height:1'>{sc:.1f}</div>"
                f"</div>"
                f"<div style='text-align:center;flex-shrink:0;position:relative;z-index:1'>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
                f"color:#4a6880;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>DM Stuff+</div>"
                f"<div style='font-family:Inter,sans-serif;font-size:24px;"
                f"font-weight:900;color:{fg_col};line-height:1'>{fg_str}</div>"
                f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
                f"color:{fg_col};opacity:0.7;margin-top:2px'>{fg_lbl}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            def display_hb(raw_hb, pitcher_hand=None):
                """Convert Statcast pfx_x → arm-side positive (TrackMan) display.
                pfx_x is identical for same shape regardless of hand, so a
                single negation works for both RHP and LHP.
                Hand arg kept for API compat but ignored.
                """
                if not is_real(raw_hb): return None
                return -float(raw_hb)

            MIN_DISPLAY_N   = 50
            MIN_DISPLAY_PCT = 0.01
            def pitch_has_data(g):
                if not is_real(row.get(f"velo_{g}")): return False
                n   = row.get(f"n_{g}")
                pct = row.get(f"pct_{g}")
                if not is_real(n): return True
                return float(n) >= MIN_DISPLAY_N and float(pct) >= MIN_DISPLAY_PCT

            def pitch_pct(g):
                pct = row.get(f"pct_{g}")
                return float(pct) if is_real(pct) else 0.0

            def pitch_sort_key(g):
                matched_p = r.get("Matched Pitch")
                if result_mode == "single":
                    if g == matched_p: return (0, 0.0)
                    return (1, -pitch_pct(g))
                return (0, -pitch_pct(g))

            active = sorted([g for g in PITCH_GROUPS if pitch_has_data(g)], key=pitch_sort_key)

            if not active:
                st.markdown(
                    "<div style='color:#6a90a8;font-family:JetBrains Mono,monospace;"
                    "font-size:11px;padding:8px 0'>No pitch data for this season.</div>",
                    unsafe_allow_html=True,
                )
            else:
                snap_sp = st.session_state.get("user_snapshot", {})

                for group in active:
                    color   = PITCH_COLORS[group]
                    user_m  = pitch_inputs.get(group, {})
                    if result_mode == "single" and group == r.get("Matched Pitch") and not user_m:
                        sp_v = snap.get("sp_velo"); sp_i = snap.get("sp_ivb"); sp_h = snap.get("sp_hb_csv")
                        user_m = {k: v for k, v in [("velo",sp_v),("ivb",sp_i),("hb",sp_h)] if v is not None}

                    mv_velo   = row.get(f"velo_{group}")
                    mv_ivb    = row.get(f"ivb_{group}")
                    mv_hb_raw = row.get(f"hb_{group}")
                    mv_hb     = display_hb(mv_hb_raw, hand)
                    vaa_v     = row.get(f"vaa_{group}")
                    haa_v     = row.get(f"haa_{group}")

                    velo_s = f"{mv_velo:.1f}"   if is_real(mv_velo) else "—"
                    ivb_s  = f"{mv_ivb:.1f}\""  if is_real(mv_ivb)  else "—"
                    hb_s   = f"{mv_hb:+.1f}\"" if mv_hb is not None else "—"
                    vaa_s  = f"{-vaa_v:.1f}°"  if is_real(vaa_v)   else "—"
                    haa_s  = f"{haa_v:.1f}°"   if is_real(haa_v)   else "—"

                    _vl = (_vaa_haa_league.get(f"{group}_{hand}") or _vaa_haa_league.get(group, {}))
                    vaa_gc = stat_gradient_color(vaa_v, _vl["vaa_mu"], _vl["vaa_sd"], invert=True) if (_vl and is_real(vaa_v)) else "#1a3550"
                    if _vl and is_real(haa_v):
                        _hz = min(abs((haa_v - _vl["haa_mu"]) / max(_vl["haa_sd"], 0.001)), 2.0)
                        _ht = _hz / 2.0
                        haa_gc = f"rgb({int(120+(220-120)*_ht)},{int(130+(35-130)*_ht)},{int(140+(35-140)*_ht)})"
                    else:
                        haa_gc = "#1a3550"

                    u_velo = f"{user_m['velo']:.1f} you" if user_m.get("velo") is not None else ""
                    u_ivb  = f"{user_m['ivb']:.1f}\" you" if user_m.get("ivb") is not None else ""
                    _u_hb_raw = user_m.get("hb")
                    # User entered arm-side positive — show as-is
                    u_hb   = f"{_u_hb_raw:+.1f}\" you" if _u_hb_raw is not None else ""

                    sp_col = FG_SP_COL.get(group)
                    sp_val = None
                    for _sp_key in ([sp_col] if sp_col else []) + ["stuff_plus"]:
                        _sp_v = row.get(_sp_key) if _sp_key else None
                        if _sp_v is None: continue
                        try:
                            _sp_f = float(_sp_v)
                            if math.isnan(_sp_f) or _sp_f <= 0: continue
                            sp_val = _sp_f; break
                        except (TypeError, ValueError): continue
                    sp_has   = sp_val is not None
                    sp_str   = f"{sp_val:.0f}" if sp_has else "—"
                    sp_color = stuff_color(sp_val) if sp_has else "#2a4a5a"
                    sp_lbl   = stuff_grade_label(sp_val) if sp_has else "—"

                    n_val   = row.get(f"n_{group}")
                    pct_val = row.get(f"pct_{group}")
                    n_str   = f"{int(n_val):,}" if is_real(n_val) else "—"
                    pct_str = f"{pct_val:.0%}"  if is_real(pct_val) else "—"

                    pz_data = pitcher_zone_data(r["Pitcher"], r["Year"], group)
                    p_csw = p_xwoba = None

                    if not pz_data.empty and "n_pitches" in pz_data.columns:
                        total_n = pz_data["n_pitches"].sum()
                        if total_n > 0:
                            p_csw = (pz_data["csw_pct"] * pz_data["n_pitches"]).sum() / total_n
                        p_xwoba = per_pa_xwoba(pz_data)

                    if p_csw is None and zone_stats_ok and not zone_stats.empty:
                        mask = (
                            (zone_stats["player_name"] == r["Pitcher"]) &
                            (zone_stats["year"]        == int(r["Year"])) &
                            (zone_stats["pitch_group"] == group)
                        )
                        sub = zone_stats[mask]
                        if not sub.empty:
                            total_n = sub["n_pitches"].sum()
                            if total_n > 0:
                                p_csw = (sub["csw_pct"] * sub["n_pitches"]).sum() / total_n
                            p_xwoba = per_pa_xwoba(sub)

                    csw_str   = f"{p_csw:.1%}"   if p_csw   is not None else "—"
                    xwoba_str = f"{p_xwoba:.3f}"  if p_xwoba is not None else "—"

                    if not pitch_grp_league.empty and group in pitch_grp_league.index:
                        pg     = pitch_grp_league.loc[group]
                        csw_gc = stat_gradient_color(p_csw,   pg["csw_mu"], pg["csw_sd"], invert=False)
                        xw_gc  = stat_gradient_color(p_xwoba, pg["xw_mu"],  pg["xw_sd"],  invert=True)
                    else:
                        csw_gc = xw_gc = "#1a3550"

                    def _pc(lbl, val, c, sub=None):
                        sub_h = (
                            f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
                            f"color:#6a90a8;margin-top:3px;letter-spacing:0.3px'>{sub}</div>"
                        ) if sub else ""
                        return (
                            f"<div style='text-align:center;"
                            f"background:linear-gradient(160deg,#0d1a28,#0a1420);"
                            f"border:1px solid {c}30;border-radius:8px;padding:8px 4px'>"
                            f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;"
                            f"color:{c};text-transform:uppercase;letter-spacing:1px;"
                            f"margin-bottom:4px;font-weight:600;opacity:0.8'>{lbl}</div>"
                            f"<div style='font-family:Inter,sans-serif;font-size:16px;"
                            f"font-weight:700;color:{c};line-height:1'>{val}</div>"
                            f"{sub_h}</div>"
                        )

                    _pw_val = None
                    if not pz_data.empty and "whiff_pct" in pz_data.columns and "n_pitches" in pz_data.columns:
                        _denom = pz_data["swing_count"].sum() if "swing_count" in pz_data.columns else pz_data["n_pitches"].sum()
                        if _denom > 0:
                            _pw_val = (pz_data["whiff_pct"] * pz_data["n_pitches"]).sum() / _denom

                    card_html = (
                        f"<div style='background:linear-gradient(165deg,#0c1520,#090f1a);"
                        f"border:1px solid #1a2a40;border-left:3px solid {color};"
                        f"border-radius:10px;padding:12px 14px;margin-bottom:6px'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:center;margin-bottom:10px'>"
                        f"<div style='font-family:Inter,sans-serif;font-size:12px;"
                        f"font-weight:700;color:{color};letter-spacing:1.5px;"
                        f"text-transform:uppercase'>● {group}</div>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                        f"color:#4a6880'>{pct_str} · {n_str} pitches</div>"
                        f"</div>"
                        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:5px'>"
                        + _pc("VELO",   velo_s, color,    u_velo or None)
                        + _pc("iVB",    ivb_s,  "#8aadcc", u_ivb  or None)
                        + _pc("HB",     hb_s,   "#8aadcc", u_hb   or None)
                        + _pc("VAA",    vaa_s,  vaa_gc)
                        + _pc("HAA",    haa_s,  haa_gc)
                        + _pc("STUFF+",
                               sp_str if sp_has else (f"{user_dmsp[group]:.0f}" if group in user_dmsp else "—"),
                               sp_color if sp_has else (stuff_color(user_dmsp[group]) if group in user_dmsp else "#2a4a5a"),
                               (sp_lbl if sp_has else (f"{user_dmsp[group]:.0f} yours" if group in user_dmsp else "—")))
                        + _pc("Whiff%", f"{_pw_val:.1%}" if _pw_val is not None else "—", "#8aadcc")
                        + _pc("CSW%",   csw_str,   csw_gc)
                        + _pc("xwOBA",  xwoba_str, xw_gc)
                        + "</div></div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    if not pz_data.empty:
                        with st.expander(f"📊  {group} zone heatmaps", expanded=False):
                            hm1,hm2,hm3 = st.columns(3)
                            with hm1: st.markdown(render_zone_heatmap(pz_data,"csw_pct",  "csw",  "CSW% — All",  fmt=".1%"), unsafe_allow_html=True)
                            with hm2: st.markdown(render_zone_heatmap(pz_data,"whiff_pct","whiff","Whiff% — All",fmt=".1%"), unsafe_allow_html=True)
                            with hm3: st.markdown(render_zone_heatmap(pz_data,"xwoba_mean","xwoba","xwOBA — All", fmt=".3f"), unsafe_allow_html=True)
                            has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                            if has_stand_col:
                                pz_same = pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], group, "same")
                                pz_opp  = pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], group, "opp")
                                if not pz_same.empty:
                                    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs {hand}HB</div>", unsafe_allow_html=True)
                                    ps1,ps2,ps3 = st.columns(3)
                                    with ps1: st.markdown(render_zone_heatmap(pz_same,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                                    with ps2: st.markdown(render_zone_heatmap(pz_same,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                                    with ps3: st.markdown(render_zone_heatmap(pz_same,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
                                if not pz_opp.empty:
                                    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs {'L' if hand == 'R' else 'R'}HB</div>", unsafe_allow_html=True)
                                    po1,po2,po3 = st.columns(3)
                                    with po1: st.markdown(render_zone_heatmap(pz_opp,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                                    with po2: st.markdown(render_zone_heatmap(pz_opp,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                                    with po3: st.markdown(render_zone_heatmap(pz_opp,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)

                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

                overall_data  = overall_pitcher_zone_data(r["Pitcher"], r["Year"])
                has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                if not overall_data.empty or not zone_stats_ok:
                    with st.expander("📊  Overall zone profile — all pitches", expanded=False):
                        if not overall_data.empty:
                            oh1,oh2,oh3 = st.columns(3)
                            with oh1: st.markdown(render_zone_heatmap(overall_data,"csw_pct",  "csw",  "CSW% — All",  fmt=".1%"), unsafe_allow_html=True)
                            with oh2: st.markdown(render_zone_heatmap(overall_data,"whiff_pct","whiff","Whiff% — All",fmt=".1%"), unsafe_allow_html=True)
                            with oh3: st.markdown(render_zone_heatmap(overall_data,"xwoba_mean","xwoba","xwOBA — All", fmt=".3f"), unsafe_allow_html=True)
                            if has_stand_col:
                                ov_same = overall_pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], "same")
                                ov_opp  = overall_pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], "opp")
                                if not ov_same.empty:
                                    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs {hand}HB</div>", unsafe_allow_html=True)
                                    s1,s2,s3 = st.columns(3)
                                    with s1: st.markdown(render_zone_heatmap(ov_same,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                                    with s2: st.markdown(render_zone_heatmap(ov_same,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                                    with s3: st.markdown(render_zone_heatmap(ov_same,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
                                if not ov_opp.empty:
                                    st.markdown(f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>vs {'L' if hand == 'R' else 'R'}HB</div>", unsafe_allow_html=True)
                                    o1,o2,o3 = st.columns(3)
                                    with o1: st.markdown(render_zone_heatmap(ov_opp,"csw_pct",  "csw",  "CSW%",  fmt=".1%"), unsafe_allow_html=True)
                                    with o2: st.markdown(render_zone_heatmap(ov_opp,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                                    with o3: st.markdown(render_zone_heatmap(ov_opp,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
                        else:
                            st.markdown(
                                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6a90a8;"
                                "background:#0a0e18;border:1px solid #162236;border-radius:8px;padding:8px 12px'>"
                                "⚾ Zone heatmaps require rebuilding pitch_zone_stats.csv</div>",
                                unsafe_allow_html=True,
                            )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    export = [{k: v for k, v in r.items() if k != "_row"} for r in results]
    csv    = pd.DataFrame(export).to_csv(index=False).encode("utf-8")
    _, dl_col, _ = st.columns([2, 3, 2])
    with dl_col:
        st.download_button("⬇  Download Results CSV", data=csv,
                           file_name="pitcher_similarity_results.csv",
                           mime="text/csv", width='stretch')
