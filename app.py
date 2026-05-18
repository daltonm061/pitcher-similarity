import warnings
import math
import io
import numpy as np
import pandas as pd
import streamlit as st

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
}

/* ── PITCH CARDS ── */
.pitch-card {
    background: linear-gradient(165deg, #0c1420 0%, #0a1220 100%);
    border: 1px solid #162236;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.pitch-card:hover {
    border-color: #1e3250;
    box-shadow: 0 2px 12px #00000030;
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
    background: linear-gradient(165deg, #0c1420 0%, #0a1220 100%);
    border: 1px solid #162236; border-radius: 10px; padding: 12px 16px;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #1e3250; }
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
    background: linear-gradient(165deg, #0c1420, #0a1220);
    border: 1px solid #162236;
    border-top: 2px solid #d4a84830; border-radius: 10px; padding: 18px 20px;
    margin-bottom: 12px;
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
    <div class="app-bar-sub">STATCAST 2017–2024 · ARM-SIDE NORMALIZED · GAUSSIAN SCORING</div>
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

@st.cache_resource
def load_dm_v5():
    """Returns (bundle_dict, norms_dict) or (None, None) if not present."""
    if not _os.path.exists(_V5_BUNDLE_PATH):
        return None, None
    try:
        bundle = _joblib.load(_V5_BUNDLE_PATH)
    except Exception:
        return None, None
    norms = bundle.get("norms", {})
    if _os.path.exists(_V5_NORMS_PATH):
        try:
            import json as _json
            with open(_V5_NORMS_PATH) as _f:
                norms = _json.load(_f)
        except Exception:
            pass
    return bundle, norms

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
]


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
    # Convert rel_side to arm-side convention (input is already arm-side positive)
    rs_arm = float(rel_side) if rel_side is not None else _V5_MEDIANS["rel_side_arm"]
    if rs_arm > 0 and hand == "R":
        rs_arm = -rs_arm  # arm-side for RHP is negative in raw Statcast convention

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
            spin_rate = _V5_MEDIANS["spin_rate"]; imputed.append("spin_rate")
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
        haa = _V5_MEDIANS["haa"];  imputed.append("haa")
        if hand == "L":
            haa = -haa

        # v5c: residualize VAA/HAA against release geometry using bundle baselines
        bl_v = vaa_haa_baselines.get(("vaa", int(pt_int), int(is_lefty)))
        bl_h = vaa_haa_baselines.get(("haa", int(pt_int), int(is_lefty)))
        vaa_aa = (vaa - (bl_v[0] + bl_v[1] * rh))     if bl_v else vaa
        haa_aa = (haa - (bl_h[0] + bl_h[1] * rs_arm)) if bl_h else haa

        # SSW magnitude — without spin axis we can't compute, default to 0
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
            # Mark as imputed in the per-pitch metadata
            for grp in imputed_per_pitch:
                imputed_per_pitch[grp].append("arsenal (MLB avg)")
        else:
            # No defaults available (very old bundle) — single-pitch arsenal sentinel
            r["arsenal_size"] = 1.0

    df = _pd.DataFrame(rows)[FEATURES]
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

    # Standardize per pitch type using norms.
    # `keys` is a list of (display_group, scored_group) tuples.
    # Use scored_group for the norms lookup, display_group for the result key.
    by_type = _v5_norms.get("by_type", {})
    overall = _v5_norms.get("overall", {"mean": 0.0, "sd": 1.0})
    out = {}
    for i, (display_grp, scored_grp) in enumerate(keys):
        params = by_type.get(scored_grp, overall)
        m_, s_ = params["mean"], params["sd"]
        sp = 100.0 + ((raw[i] - m_) / max(s_, 1e-6)) * 10.0
        out[display_grp] = {
            "stuff_plus": round(float(sp), 1),
            "imputed":    imputed_per_pitch.get(display_grp, []),
            "shape_row":  rows[i],   # for zone-Stuff+ scoring downstream
        }
    return out


# ── Zone-conditional Stuff+ scorer ────────────────────────────────────────
_ALL_ZONES = list(range(1, 10)) + list(range(11, 27))


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

    # Build 52 rows: 26 zones × 2 platoons
    # For v3+ bundles: n_pitches (typical cell size ~40) and season (current
    # year, e.g. 2025) get filled with sensible defaults so the model doesn't
    # rely on NaN routing.
    import datetime as _dt
    _default_n_pitches = 40.0    # median cell size in training data
    _default_season    = _dt.datetime.now().year
    rows = []
    keys = []
    for plat_key, batter_hand in (("vs_rhb", "R"), ("vs_lhb", "L")):
        is_same_hand = 1 if pitcher_hand == batter_hand else 0
        for zone in _ALL_ZONES:
            r = dict(shape_row)
            r["zone_int"]     = zone
            r["is_same_hand"] = is_same_hand
            r["ivb_x_zone"]          = r.get("ivb_in", 0)           * zone
            r["hb_x_zone"]           = r.get("hb_arm_in", 0)        * zone
            r["velo_x_zone"]         = r.get("start_speed", 0)      * zone
            r["rel_height_x_zone"]   = r.get("rel_height", 0)       * zone
            r["vaa_aa_x_zone"]       = r.get("vaa_aa", 0)           * zone
            r["rel_side_x_zone"]     = r.get("rel_side_arm", 0)     * zone
            r["active_spin_x_zone"]  = r.get("active_spin_rate", 0) * zone
            # v3 features (filled with defaults if the bundle wants them)
            r["n_pitches"] = _default_n_pitches
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
                                width: int = 220, cell_size: int = 38) -> str:
    """Build an inline SVG heatmap from {zone_int: stuff_plus} dict.

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
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
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
    # Try gzip first (smaller), fall back to plain CSV
    import os
    if os.path.exists("pitch_zone_stats.csv.gz"):
        return pd.read_csv("pitch_zone_stats.csv.gz", compression="gzip")
    return pd.read_csv("pitch_zone_stats.csv")

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

def last_name(full):
    """Extract last name for display."""
    parts = full.strip().split(",")
    if len(parts) > 1:
        return parts[0].strip()
    parts = full.strip().split()
    return parts[-1] if parts else full


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
                sim = gaussian_sim(mv, val, sigma) if is_real(mv) else 0.4
            w = WEIGHTS.get(metric, 1.0)
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
                    except: pass

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
            except: pass
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
            "Scoring against 1,700+ pitcher-seasons</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.stop()

elif st.session_state.screen == "title":

    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

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
            "2017–2024 &nbsp;·&nbsp; Gaussian Scoring &nbsp;·&nbsp; Factor-Matched Comps"
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
            # Border tint depends on whether v5 model is available
            _dm_avail = _V5_AVAILABLE
            _dm_border = "#c4914830" if _dm_avail else "#3a5a7830"
            _dm_color  = "#c49148" if _dm_avail else "#5a7a90"
            _dm_glow   = "#c4914808" if _dm_avail else "#3a5a7808"
            _dm_sub    = ("Compute your own pitch's Stuff+ score from any "
                          "combination of velo, movement, spin rate, and release "
                          "data — missing fields use league medians.") if _dm_avail else \
                         ("DM Stuff+ v5 model not deployed yet. Run training and "
                          "add the bundle to enable this calculator.")
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
            "Scoring against 1,700+ pitcher-seasons</div>"
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
            "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
            "color:#d4a848;letter-spacing:2px;text-transform:uppercase;"
            "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
            "● Release Profile</div>",
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
                                            placeholder="e.g. 5.00", key="rh",
                                            label_visibility="collapsed")
        with rp3:
            st.markdown("<div class='field-label'>Rel Side — arm side (ft)</div>", unsafe_allow_html=True)
            rel_side_v = st.number_input(" ", min_value=0.0, max_value=5.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder="e.g. 1.90 (RHP) / 2.10 (LHP)",
                                          key="rs", label_visibility="collapsed")
        with rp4:
            st.markdown("<div class='field-label'>Extension (ft)</div>", unsafe_allow_html=True)
            extension_v = st.number_input(" ", min_value=4.0, max_value=8.0,
                                           value=None, step=0.01, format="%.2f",
                                           placeholder="e.g. 6.20", key="ext",
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
        "margin-bottom:16px'>All pitch types · Statcast 2017–2024 · Click column headers to sort</div>",
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

    # ── Back to title ─────────────────────────────────────────────────────────
    if st.button("← Back", key="back_dmstuff_to_title"):
        st.session_state.screen = "title"
        st.session_state.pop("_dmsp_pitches", None)
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;max-width:680px;margin:0 auto 20px auto;padding:0 20px'>"
        "<div style='font-family:Inter,sans-serif;font-size:22px;font-weight:700;"
        "color:#c49148;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px'>"
        "🧮  DM Stuff+ Calculator</div>"
        "<div style='font-family:JetBrains Mono,monospace;font-size:11px;color:#6a90a8'>"
        "Score any pitch through DM Stuff+ v5 — leave fields blank to use league medians"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if not _V5_AVAILABLE:
        st.markdown(
            "<div style='max-width:680px;margin:24px auto;padding:20px;"
            "border:1px solid #c4914830;border-radius:8px;background:#181818;"
            "font-family:JetBrains Mono,monospace;font-size:11px;color:#a0c0d4;"
            "line-height:1.7'>"
            "⚠ DM Stuff+ v5 bundle not found. To enable this calculator: "
            "<br>1. Run <code>python train_stuff_plus.py</code> to produce "
            "<code>models/dm_stuff_plus_v5.joblib</code><br>"
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
                                         placeholder="e.g. 5.80", key="dm_rh",
                                         label_visibility="collapsed")
            with dr3:
                st.markdown("<div class='field-label'>Rel Side — arm side (ft)</div>", unsafe_allow_html=True)
                dm_rs = st.number_input(" ", min_value=0.0, max_value=5.0,
                                         value=None, step=0.01, format="%.2f",
                                         placeholder="e.g. 1.90", key="dm_rs",
                                         label_visibility="collapsed")
            with dr4:
                st.markdown("<div class='field-label'>Extension (ft)</div>", unsafe_allow_html=True)
                dm_ext = st.number_input(" ", min_value=4.0, max_value=8.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder="e.g. 6.40", key="dm_ext",
                                          label_visibility="collapsed")

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
                        for suf in ["_velo", "_ivb", "_hb", "_spin"]:
                            st.session_state.pop(f"dm_{group}{suf}", None)
                        st.rerun()

                vc, ic, hc, sc = st.columns(4)
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

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

            # ── Score button ──────────────────────────────────────────────────
            score_col, _ = st.columns([3, 5])
            with score_col:
                _do_score = st.button("Compute DM Stuff+ →", key="dm_score_btn",
                                       width='stretch', disabled=(len(dm_added) == 0))

            # ── Results ───────────────────────────────────────────────────────
            if _do_score and dm_added:
                # Collect inputs
                hand_code = "L" if dm_hand == "LHP" else "R"
                def _pf(s):  # parse float, return None if blank/invalid
                    try:
                        if s is None or str(s).strip() == "":
                            return None
                        return float(str(s).strip())
                    except (TypeError, ValueError):
                        return None

                pitches_dict = {}
                missing_velo = []
                for group in dm_added:
                    v = _pf(st.session_state.get(f"dm_{group}_velo"))
                    if v is None:
                        missing_velo.append(group)
                        continue
                    pitches_dict[group] = {
                        "velo":      v,
                        "ivb":       _pf(st.session_state.get(f"dm_{group}_ivb")),
                        "hb":        _pf(st.session_state.get(f"dm_{group}_hb")),
                        "spin_rate": _pf(st.session_state.get(f"dm_{group}_spin")),
                    }

                if missing_velo:
                    st.markdown(
                        f"<div style='max-width:680px;margin:16px auto;padding:14px 18px;"
                        f"border:1px solid #c4914830;border-radius:6px;background:#1a1410;"
                        f"font-family:JetBrains Mono,monospace;font-size:11px;color:#c0a878'>"
                        f"⚠ Missing velocity for: {', '.join(missing_velo)} — these pitches were skipped."
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if pitches_dict:
                    scores = _score_v5_arsenal(
                        pitches=pitches_dict,
                        rel_height=dm_rh,
                        rel_side=dm_rs,
                        extension=dm_ext,
                        hand=hand_code,
                    )
                    if scores:
                        # Status indicator for zone model availability (helps debug
                        # "why aren't my heatmaps showing up" situations)
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

                        for group in dm_added:
                            if group not in scores:
                                continue
                            sp_val = scores[group]["stuff_plus"]
                            imputed = scores[group]["imputed"]
                            color = PITCH_COLORS[group]

                            # Color code Stuff+ score
                            if sp_val >= 115:
                                sp_color = "#d4a848"
                            elif sp_val >= 105:
                                sp_color = "#a0c0d4"
                            elif sp_val >= 95:
                                sp_color = "#8a9aac"
                            else:
                                sp_color = "#6a7a8a"

                            # Filter imputed list to user-visible fields
                            imputed_user = [f for f in imputed if f in
                                              ("ivb", "hb", "spin_rate", "rel_height",
                                                "rel_side", "extension")]
                            imp_str = (f"<span style='font-family:JetBrains Mono,monospace;"
                                       f"font-size:10px;color:#5a7a90'>"
                                       f"imputed: {', '.join(imputed_user)}</span>") \
                                       if imputed_user else \
                                       ("<span style='font-family:JetBrains Mono,monospace;"
                                        "font-size:10px;color:#5ac8a040'>all fields provided</span>")

                            st.markdown(
                                f"<div style='display:flex;align-items:center;justify-content:space-between;"
                                f"padding:14px 20px;margin-bottom:8px;"
                                f"background:linear-gradient(165deg,#0e1828 0%,#0c1420 100%);"
                                f"border-left:3px solid {color};border-radius:6px'>"
                                f"<div style='display:flex;flex-direction:column;gap:4px'>"
                                f"<div style='font-family:Inter,sans-serif;font-size:13px;"
                                f"font-weight:700;color:{color};letter-spacing:2px;"
                                f"text-transform:uppercase'>{group}</div>"
                                f"<div>{imp_str}</div>"
                                f"</div>"
                                f"<div style='font-family:Inter,sans-serif;font-size:32px;"
                                f"font-weight:800;color:{sp_color}'>{sp_val}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                            # ── Zone-Stuff+ heatmaps (only if zone model loaded) ──
                            if _ZONE_AVAILABLE and "shape_row" in scores[group]:
                                shape_row = scores[group]["shape_row"]
                                zone_grid = _score_zone_grid(shape_row, pitcher_hand=hand_code)
                                if zone_grid:
                                    hm_rhb = _render_zone_heatmap_svg(
                                        zone_grid["vs_rhb"], "vs RHB"
                                    )
                                    hm_lhb = _render_zone_heatmap_svg(
                                        zone_grid["vs_lhb"], "vs LHB"
                                    )
                                    hm_cols = st.columns([1, 1])
                                    with hm_cols[0]:
                                        st.markdown(hm_rhb, unsafe_allow_html=True)
                                    with hm_cols[1]:
                                        st.markdown(hm_lhb, unsafe_allow_html=True)
                                    st.markdown(
                                        "<div style='height:14px'></div>",
                                        unsafe_allow_html=True,
                                    )

                        # Legend
                        _hm_note = ("<br><span style='color:#5a7a90'>"
                                    "Heatmaps show predicted Stuff+ by zone (catcher's view; "
                                    "gold = elite, blue = below avg)."
                                    "</span>") if _ZONE_AVAILABLE else ""
                        st.markdown(
                            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                            "color:#3a5a78;margin-top:16px;line-height:1.7;padding:0 8px'>"
                            "<b style='color:#d4a848'>120+</b> elite &nbsp;·&nbsp; "
                            "<b style='color:#a0c0d4'>105-115</b> above avg &nbsp;·&nbsp; "
                            "<b style='color:#8a9aac'>95-105</b> avg &nbsp;·&nbsp; "
                            "<b style='color:#6a7a8a'>&lt;95</b> below avg<br>"
                            "Scale: per-pitch-type, mean=100, SD=10. Trained on 2017-2025 Statcast."
                            + _hm_note +
                            "</div>",
                            unsafe_allow_html=True,
                        )

                        # ── Arsenal Grade ─────────────────────────────────────
                        _scored_vals = [scores[g]["stuff_plus"] for g in dm_added if g in scores]
                        if _scored_vals:
                            _arsenal_sp = sum(_scored_vals) / len(_scored_vals)
                            if _arsenal_sp >= 120:   _grade, _grade_color = "A+", "#d4a848"
                            elif _arsenal_sp >= 112: _grade, _grade_color = "A",  "#d4a848"
                            elif _arsenal_sp >= 107: _grade, _grade_color = "B+", "#a0c0d4"
                            elif _arsenal_sp >= 102: _grade, _grade_color = "B",  "#a0c0d4"
                            elif _arsenal_sp >= 97:  _grade, _grade_color = "C+", "#8a9aac"
                            elif _arsenal_sp >= 92:  _grade, _grade_color = "C",  "#8a9aac"
                            else:                    _grade, _grade_color = "D",  "#6a7a8a"
                            st.markdown(
                                "<div style='margin:28px 0 8px 0;padding:18px 24px;"
                                "background:linear-gradient(165deg,#0e1828,#0a1520);"
                                "border:1px solid #1a2a40;border-radius:8px;"
                                "display:flex;align-items:center;justify-content:space-between'>"
                                "<div>"
                                "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                                "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                                "margin-bottom:4px'>Arsenal Stuff+</div>"
                                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                "color:#3a5a78'>Unweighted avg across entered pitches</div>"
                                "</div>"
                                "<div style='display:flex;align-items:baseline;gap:14px'>"
                                f"<div style='font-family:Inter,sans-serif;font-size:36px;"
                                f"font-weight:800;color:{_grade_color}'>{_arsenal_sp:.1f}</div>"
                                f"<div style='font-family:Inter,sans-serif;font-size:22px;"
                                f"font-weight:700;color:{_grade_color};opacity:0.7'>{_grade}</div>"
                                "</div></div>",
                                unsafe_allow_html=True,
                            )

                        # ── Movement Plot ─────────────────────────────────────
                        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                        st.markdown(
                            "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                            "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                            "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                            "● Movement Profile vs Closest A+ Arsenal</div>",
                            unsafe_allow_html=True,
                        )
                        try:
                            import matplotlib as _mpl
                            import matplotlib.pyplot as _plt
                            import matplotlib.patheffects as _pe

                            _GROUPS_ALL = ["4-Seam","2-Seam/Sinker","Cutter","Slider",
                                           "Sweeper","Curveball","Splitter","Changeup","Knuckleball"]
                            _PLT_COLORS = {
                                "4-Seam":        "#e63946",
                                "2-Seam/Sinker": "#f4a261",
                                "Cutter":        "#2a9d8f",
                                "Slider":        "#457b9d",
                                "Sweeper":       "#a855f7",
                                "Curveball":     "#e9c46a",
                                "Splitter":      "#f4845f",
                                "Changeup":      "#90be6d",
                                "Knuckleball":   "#adb5bd",
                            }

                            # Build user pitch points {grp: (hb, ivb)}
                            # Use entered values; fall back to imputed from scores shape_row
                            _user_pts = {}
                            for _g in dm_added:
                                if _g not in scores:
                                    continue
                                _sr = scores[_g].get("shape_row", {})
                                _hb_val  = pitches_dict[_g].get("hb")  if _g in pitches_dict else None
                                _ivb_val = pitches_dict[_g].get("ivb") if _g in pitches_dict else None
                                # Fall back to shape_row (which has imputed values)
                                if _hb_val  is None: _hb_val  = _sr.get("hb_arm_in")
                                if _ivb_val is None: _ivb_val = _sr.get("ivb_in")
                                if _hb_val is not None and _ivb_val is not None:
                                    _user_pts[_g] = (float(_hb_val), float(_ivb_val))

                            # ── Build A+ target via per-pitch movement optimisation ──
                            # IVB/HB search bounds per pitch type (user arm-side + convention)
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
                                """Score a pitches dict and return mean Stuff+, or None."""
                                _s = _score_v5_arsenal(
                                    pitches=pd_dict,
                                    rel_height=dm_rh, rel_side=dm_rs,
                                    extension=dm_ext, hand=hand_code,
                                )
                                if not _s:
                                    return None
                                _vs = [_s[g]["stuff_plus"] for g in pd_dict if g in _s]
                                return sum(_vs) / len(_vs) if _vs else None

                            def _best_movement(grp, base_dict):
                                """Coarse→fine grid search on IVB×HB for a single pitch."""
                                _bd = _SEARCH_BOUNDS.get(grp, {"ivb": (-15, 25), "hb": (-25, 25)})
                                _iv_lo, _iv_hi = _bd["ivb"]
                                _hb_lo, _hb_hi = _bd["hb"]
                                _pd = base_dict[grp]
                                _velo = _pd.get("velo", 90)
                                _spin = _pd.get("spin_rate")
                                _best_sp = -1e9
                                _best_iv, _best_hb_v = None, None
                                # Coarse: 8-step grid
                                for _iv in np.linspace(_iv_lo, _iv_hi, 8):
                                    for _hb_v in np.linspace(_hb_lo, _hb_hi, 8):
                                        _trial = dict(base_dict)
                                        _trial[grp] = {"velo": _velo, "ivb": _iv,
                                                       "hb": _hb_v, "spin_rate": _spin}
                                        _sp = _score_mean(_trial)
                                        if _sp is not None and _sp > _best_sp:
                                            _best_sp = _sp; _best_iv = _iv; _best_hb_v = _hb_v
                                if _best_iv is None:
                                    return None, None
                                # Fine: 7-step grid centred on coarse best (±step)
                                _iv_step = (_iv_hi - _iv_lo) / 7
                                _hb_step = (_hb_hi - _hb_lo) / 7
                                for _iv in np.linspace(_best_iv - _iv_step, _best_iv + _iv_step, 7):
                                    for _hb_v in np.linspace(_best_hb_v - _hb_step, _best_hb_v + _hb_step, 7):
                                        _trial = dict(base_dict)
                                        _trial[grp] = {"velo": _velo, "ivb": _iv,
                                                       "hb": _hb_v, "spin_rate": _spin}
                                        _sp = _score_mean(_trial)
                                        if _sp is not None and _sp > _best_sp:
                                            _best_sp = _sp; _best_iv = _iv; _best_hb_v = _hb_v
                                return _best_iv, _best_hb_v

                            # Step 1: optimise movement for each entered pitch
                            _opt_dict = {g: dict(v) for g, v in pitches_dict.items()}
                            for _g in list(pitches_dict.keys()):
                                _oi, _oh = _best_movement(_g, _opt_dict)
                                if _oi is not None:
                                    _opt_dict[_g]["ivb"] = _oi
                                    _opt_dict[_g]["hb"]  = _oh

                            _opt_mean = _score_mean(_opt_dict) or 0.0

                            # Step 2: if still below A+, add pitches one at a time
                            _added_pitches = []
                            _missing_grps = [g for g in _MLB_PITCH_MEDIANS
                                             if g not in pitches_dict]
                            while _opt_mean < 112.0 and _missing_grps:
                                _best_add_sp = _opt_mean
                                _best_add_g  = None
                                for _cand in _missing_grps:
                                    _try = dict(_opt_dict)
                                    _try[_cand] = dict(_MLB_PITCH_MEDIANS[_cand])
                                    _sp = _score_mean(_try)
                                    if _sp is not None and _sp > _best_add_sp:
                                        _best_add_sp = _sp; _best_add_g = _cand
                                if _best_add_g is None:
                                    break
                                _opt_dict[_best_add_g] = dict(_MLB_PITCH_MEDIANS[_best_add_g])
                                _missing_grps.remove(_best_add_g)
                                _added_pitches.append(_best_add_g)
                                _opt_mean = _best_add_sp

                            # Build match_pts for plotting (arm-side + convention, same as user)
                            _match_pts = {}
                            for _g, _pd in _opt_dict.items():
                                _oi = _pd.get("ivb"); _oh = _pd.get("hb")
                                if _oi is not None and _oh is not None:
                                    _match_pts[_g] = (float(_oh), float(_oi))

                            _aplus_label = f"A+ target · avg {_opt_mean:.1f}"
                            if _added_pitches:
                                _aplus_label += f" (added: {', '.join(_added_pitches)})"

                            # Draw plot
                            _fig, _ax = _plt.subplots(figsize=(4, 3.2))
                            _fig.patch.set_facecolor("#0c1420")
                            _ax.set_facecolor("#0e1828")
                            for _spine in _ax.spines.values():
                                _spine.set_edgecolor("#1a2a40")
                            _ax.tick_params(colors="#5a7a90", labelsize=9)
                            _ax.set_xlabel("Horizontal Break — arm-side + (in)", color="#5a7a90", fontsize=10)
                            _ax.set_ylabel("Induced Vertical Break (in)", color="#5a7a90", fontsize=10)
                            _ax.axhline(0, color="#1a2a40", lw=1, zorder=0)
                            _ax.axvline(0, color="#1a2a40", lw=1, zorder=0)
                            _ax.grid(True, color="#1a2a40", lw=0.5, alpha=0.6, zorder=0)

                            # Plot matched A+ pitcher first (background, open markers)
                            for _g, (_hb, _ivb) in _match_pts.items():
                                _c = _PLT_COLORS.get(_g, "#aaaaaa")
                                _ax.scatter(_hb, _ivb, s=180, facecolors="none",
                                            edgecolors=_c, linewidths=2,
                                            marker="D", alpha=0.7, zorder=3)

                            # Plot user pitches (foreground, filled)
                            _plotted = []
                            for _g, (_hb, _ivb) in _user_pts.items():
                                _c = _PLT_COLORS.get(_g, "#aaaaaa")
                                _sc = _ax.scatter(_hb, _ivb, s=220, color=_c,
                                                  edgecolors="white", linewidths=1.2,
                                                  marker="o", zorder=5, label=_g)
                                _ax.annotate(_g, (_hb, _ivb),
                                             textcoords="offset points", xytext=(8, 5),
                                             fontsize=8, color=_c,
                                             path_effects=[_pe.withStroke(linewidth=2, foreground="#0e1828")])
                                _plotted.append(_g)

                            # Draw lines connecting user ↔ matched for shared pitches
                            for _g in _plotted:
                                if _g in _match_pts:
                                    _ux, _uy = _user_pts[_g]
                                    _mx, _my = _match_pts[_g]
                                    _c = _PLT_COLORS.get(_g, "#aaaaaa")
                                    _ax.plot([_ux, _mx], [_uy, _my],
                                             color=_c, lw=1, linestyle="--", alpha=0.35, zorder=2)

                            # Legend
                            from matplotlib.lines import Line2D as _L2D
                            _legend_elems = [
                                _L2D([0],[0], marker="o", color="w", markerfacecolor="#aaaaaa",
                                     markersize=8, label="Your arsenal", linestyle="None"),
                                _L2D([0],[0], marker="D", color="w", markerfacecolor="none",
                                     markeredgecolor="#aaaaaa", markersize=8,
                                     label="A+ target", linestyle="None"),
                            ]
                            _ax.legend(handles=_legend_elems, facecolor="#0e1828",
                                       edgecolor="#1a2a40", labelcolor="#a0c0d4",
                                       fontsize=7, loc="best")
                            _plt.tight_layout(pad=1.0)
                            st.pyplot(_fig, use_container_width=True)
                            _plt.close(_fig)

                            _caption = (
                                f"<b style='color:#a0c0d4'>{_aplus_label}</b> — "
                                "diamonds show the optimised IVB/HB for your pitch types "
                                "(same velo &amp; release) that maximises Stuff+."
                            )
                            if _added_pitches:
                                _caption += (
                                    f" Reaching A+ required adding: "
                                    f"<b style='color:#c49148'>{', '.join(_added_pitches)}</b>."
                                )
                            st.markdown(
                                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                f"color:#3a5a78;margin-top:6px;padding:0 4px;line-height:1.6'>"
                                f"{_caption}</div>",
                                unsafe_allow_html=True,
                            )
                        except Exception as _plot_err:
                            st.markdown(
                                f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                f"color:#5a3a3a;padding:8px'>Movement plot unavailable: {_plot_err}</div>",
                                unsafe_allow_html=True,
                            )

                        # ── Top 5 Improvement Suggestions ─────────────────────
                        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
                        st.markdown(
                            "<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                            "color:#c49148;letter-spacing:2px;text-transform:uppercase;"
                            "margin:0 0 12px 0;padding-bottom:8px;border-bottom:1px solid #1a2a40'>"
                            "● Top 5 Improvement Suggestions</div>",
                            unsafe_allow_html=True,
                        )

                        # MLB medians for adding a new pitch type
                        _MLB_PITCH_MEDIANS = {
                            "4-Seam":        {"velo": 93.8, "ivb": 16.0, "hb": -7.7,  "spin_rate": 2274},
                            "2-Seam/Sinker": {"velo": 93.2, "ivb":  9.3, "hb": -15.2, "spin_rate": 2160},
                            "Cutter":        {"velo": 88.7, "ivb":  7.7, "hb":   2.3, "spin_rate": 2365},
                            "Slider":        {"velo": 85.3, "ivb":  1.8, "hb":   4.4, "spin_rate": 2391},
                            "Sweeper":       {"velo": 82.1, "ivb":  1.0, "hb":  14.0, "spin_rate": 2571},
                            "Curveball":     {"velo": 79.4, "ivb": -9.9, "hb":   8.6, "spin_rate": 2503},
                            "Splitter":      {"velo": 86.1, "ivb":  3.9, "hb": -10.8, "spin_rate": 1370},
                            "Changeup":      {"velo": 85.5, "ivb":  6.6, "hb": -14.3, "spin_rate": 1740},
                        }
                        _FB_GROUPS = {"4-Seam", "2-Seam/Sinker"}

                        def _rescore_mean(mod_pitches, rh=dm_rh, rs=dm_rs, ext=dm_ext, hand=hand_code):
                            """Re-run _score_v5_arsenal with modified inputs and return mean Stuff+."""
                            _s = _score_v5_arsenal(
                                pitches=mod_pitches,
                                rel_height=rh, rel_side=rs, extension=ext, hand=hand,
                            )
                            if not _s:
                                return None
                            vals = [_s[g]["stuff_plus"] for g in mod_pitches if g in _s]
                            return sum(vals) / len(vals) if vals else None

                        _baseline_mean = sum(_scored_vals) / len(_scored_vals) if _scored_vals else 100.0

                        _suggestions = []  # list of (delta, label, detail)

                        # 1. Add each missing MLB pitch type (excluding FB upgrades which aren't relevant)
                        _current_pitches = set(pitches_dict.keys())
                        for _grp, _meds in _MLB_PITCH_MEDIANS.items():
                            if _grp in _current_pitches:
                                continue
                            _try = dict(pitches_dict)
                            _try[_grp] = dict(_meds)
                            _new_mean = _rescore_mean(_try)
                            if _new_mean is not None:
                                _delta = _new_mean - _baseline_mean
                                _suggestions.append((_delta, f"Add MLB-avg {_grp}", f"{_meds['velo']:.0f} mph · {_meds['ivb']:+.0f}″ iVB · {_meds['hb']:+.0f}″ HB"))

                        # 2. Movement tweaks per pitch (iVB ±2.5 in, HB ±2.5 in)
                        for _grp, _pd in pitches_dict.items():
                            _cur_ivb = _pd.get("ivb")
                            _cur_hb  = _pd.get("hb")
                            for _feat, _cur_val, _label_feat in [
                                ("ivb", _cur_ivb, "iVB"),
                                ("hb",  _cur_hb,  "HB"),
                            ]:
                                for _sign, _sign_str in [(+1, "+"), (-1, "−")]:
                                    _try = {g: dict(v) for g, v in pitches_dict.items()}
                                    _new_val = (_cur_val if _cur_val is not None else
                                                (_V5_MEDIANS["ivb_in"] if _feat == "ivb" else _V5_MEDIANS["hb_arm_in"])) + _sign * 2.5
                                    _try[_grp][_feat] = _new_val
                                    _new_mean = _rescore_mean(_try)
                                    if _new_mean is not None:
                                        _delta = _new_mean - _baseline_mean
                                        _suggestions.append((_delta, f"{_sign_str}2.5″ {_label_feat} on {_grp}", f"{_new_val:+.1f}″ {_label_feat}"))

                        # 3. Velo tweaks on non-fastball pitches (±3 mph)
                        for _grp, _pd in pitches_dict.items():
                            if _grp in _FB_GROUPS:
                                continue
                            _cur_v = _pd.get("velo", 0)
                            for _sign, _sign_str in [(+1, "+"), (-1, "−")]:
                                _try = {g: dict(v) for g, v in pitches_dict.items()}
                                _try[_grp]["velo"] = _cur_v + _sign * 3.0
                                _new_mean = _rescore_mean(_try)
                                if _new_mean is not None:
                                    _delta = _new_mean - _baseline_mean
                                    _suggestions.append((_delta, f"{_sign_str}3 mph on {_grp}", f"{_cur_v + _sign * 3.0:.0f} mph"))

                        # 4. Release profile tweaks (apply to whole arsenal)
                        _cur_rh  = dm_rh  if dm_rh  is not None else _V5_MEDIANS["rel_height"]
                        _cur_rs  = dm_rs  if dm_rs  is not None else abs(_V5_MEDIANS["rel_side_arm"])
                        _cur_ext = dm_ext if dm_ext is not None else _V5_MEDIANS["extension"]
                        for _param, _cur, _lo, _hi, _label, _rh_arg, _rs_arg, _ext_arg in [
                            ("rh",  _cur_rh,  3.0, 8.0, "Rel Height",    None, _cur_rs, _cur_ext),
                            ("rs",  _cur_rs,  0.0, 5.0, "Rel Side",      _cur_rh, None, _cur_ext),
                            ("ext", _cur_ext, 4.0, 8.0, "Extension",     _cur_rh, _cur_rs, None),
                        ]:
                            for _sign, _sign_str in [(+1, "+"), (-1, "−")]:
                                _new_val = _cur + _sign * 0.25
                                if not (_lo <= _new_val <= _hi):
                                    continue
                                _rh_use  = _new_val if _param == "rh"  else _rh_arg
                                _rs_use  = _new_val if _param == "rs"  else _rs_arg
                                _ext_use = _new_val if _param == "ext" else _ext_arg
                                _new_mean = _rescore_mean(pitches_dict, rh=_rh_use, rs=_rs_use, ext=_ext_use)
                                if _new_mean is not None:
                                    _delta = _new_mean - _baseline_mean
                                    _suggestions.append((_delta, f"{_sign_str}0.25 ft {_label}", f"{_new_val:.2f} ft"))

                        # Sort and take top 5 positive deltas
                        _suggestions.sort(key=lambda x: -x[0])
                        _top5 = [s for s in _suggestions if s[0] > 0.05][:5]

                        if not _top5:
                            st.markdown(
                                "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                                "color:#3a5a78;padding:16px 0'>No tested changes improved the arsenal score.</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            for _rank, (_delta, _lbl, _detail) in enumerate(_top5, 1):
                                if _delta >= 2.0:   _d_color = "#d4a848"
                                elif _delta >= 1.0: _d_color = "#a0c0d4"
                                else:               _d_color = "#8a9aac"
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
                            st.markdown(
                                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;"
                                "color:#3a5a78;margin-top:10px;padding:0 4px'>"
                                "Δ = change in unweighted mean Stuff+ across arsenal. "
                                "Movement tweaks ±2.5\", velo tweaks ±3 mph, release tweaks ±0.25 ft."
                                "</div>",
                                unsafe_allow_html=True,
                            )


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
                "Your metrics may be outside the range of the<br>"
                "2017–2024 MLB pitcher database.<br>"
                "Try loosening your inputs or removing some fields.</div>"
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
            "margin-bottom:4px'>Statcast 2017–2024 · Gaussian similarity model</div>",
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
        (sc2, "SEASONS",    "2017–2024",                            "#7aaac0"),
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
