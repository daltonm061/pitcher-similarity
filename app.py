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
.stNumberInput > div > div > input {
    background: #0c1220 !important; color: #d8cbb4 !important;
    border: 1px solid #1a2a40 !important; border-radius: 8px !important;
    font-size: 14px !important; font-family: 'JetBrains Mono', monospace !important;
    padding: 9px 12px !important; font-weight: 500 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #d4a848 !important;
    box-shadow: 0 0 0 2px #d4a84818, 0 0 16px #d4a84810 !important;
}
[data-testid="InputInstructions"] { display: none !important; }
[data-baseweb="tooltip"] { display: none !important; }
[role="tooltip"] { display: none !important; }
.stNumberInput button {
    background: #0e1624 !important; border-color: #1a2a40 !important;
    border-radius: 6px !important; transition: background 0.15s !important;
}
.stNumberInput button:hover { background: #162236 !important; }

/* ── RADIO ── */
.stRadio > label { display: none !important; }
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #b8c8d8 !important; font-size: 14px !important; font-weight: 500 !important;
}

/* ── SLIDER ── */
.stSlider > label { color: #7aaac0 !important; font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 500; }

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
    "Slider":  ["Cutter"],
    "Sweeper": ["Slider"],
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
    league_csw   = (zone_stats.groupby("zone")["csw_pct"].agg(["mean","std"])
                    .rename(columns={"mean":"csw_mu","std":"csw_sd"}))
    league_xwoba = (zone_stats.groupby("zone")["xwoba_mean"].agg(["mean","std"])
                    .rename(columns={"mean":"xw_mu","std":"xw_sd"}))
    if "whiff_pct" in zone_stats.columns:
        league_whiff = (zone_stats.groupby("zone")["whiff_pct"].agg(["mean","std"])
                        .rename(columns={"mean":"whiff_mu","std":"whiff_sd"}))
        zone_league = league_csw.join(league_xwoba).join(league_whiff)
    else:
        zone_league = league_csw.join(league_xwoba)
    # League avg CSW% and xwOBA per pitch group for card gradient coloring
    pitch_grp_league = (
        zone_stats.groupby("pitch_group").agg(
            csw_mu=("csw_pct",    "mean"),
            csw_sd=("csw_pct",    "std"),
            xw_mu =("xwoba_mean", "mean"),
            xw_sd =("xwoba_mean", "std"),
        ).fillna(0)
    )
    zone_stats_ok = True
except (FileNotFoundError, Exception) as _zone_err:
    zone_stats        = pd.DataFrame()
    zone_league       = pd.DataFrame()
    pitch_grp_league  = pd.DataFrame()
    zone_stats_ok     = False


# ── VAA/HAA per-pitch-group league baselines (computed from profiles CSV) ─────
# profiles may not have vaa_/haa_ cols until rebuilt — degrade gracefully
_vaa_haa_league = {}
if data_ok and profiles is not None:
    for _grp in list(PITCH_GROUPS.keys()):
        _vc = f"vaa_{_grp}"
        _hc = f"haa_{_grp}"
        if _vc in profiles.columns:
            _vs = profiles[_vc].dropna()
            _hs = profiles[_hc].dropna() if _hc in profiles.columns else _vs[:0]
            if len(_vs) > 10:
                _vaa_haa_league[_grp] = {
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
        g = int(80  + (130 - 80)  * s)
        b = int(220 + (140 - 220) * s)
    else:
        s = (t - 0.5) * 2
        r = int(120 + (220 - 120) * s)
        g = int(130 + (35  - 130) * s)
        b = int(140 + (35  - 140) * s)
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
    svg += (
        f"<rect x='{inner_x}' y='{inner_y}' width='{CW*3}' height='{CH*3}' "
        f"fill='none' stroke='#d4a84880' stroke-width='1.5'/>"
    )
    svg += "</svg>"
    return svg


def pitcher_zone_data(pitcher_name, year, pitch_group):
    """Look up zone stats for one pitcher-season-pitch combo."""
    if not zone_stats_ok or zone_stats.empty:
        return pd.DataFrame()
    mask = (
        (zone_stats["player_name"] == pitcher_name) &
        (zone_stats["year"] == int(year)) &
        (zone_stats["pitch_group"] == pitch_group)
    )
    sub = zone_stats[mask].copy()
    if sub.empty:
        return sub
    # Join league means/stds
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

        for k, v in [("velo",velo),("ivb",ivb),("hb",hb),("vaa",vaa),("haa",haa),("stuff_plus",sp)]:
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
                xw_vals = sub["xwoba_mean"].dropna()
                if not xw_vals.empty:
                    vals["xwoba"].append(xw_vals.mean())

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
# FanGraphs Stuff+ is pitcher-season level; stored as "stuff_plus" column in CSV.
# Per-pitch Stuff+ is not available from FanGraphs — we show overall pitcher Stuff+.

def stuff_color(s):
    """
    Vivid gradient for FG Stuff+ display.
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
    """Descriptive label for Stuff+ display."""
    if s is None:   return "—"
    if s >= 130:    return "Elite"
    if s >= 115:    return "Plus"
    if s >= 105:    return "Avg+"
    if s >= 95:     return "Avg"
    if s >= 85:     return "Below"
    return "Poor"

# FanGraphs per-pitch Stuff+ column names in pitcher_profiles.csv
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
            ref = abs(val) if key == "rel_side" else val
            cmp = abs(mv)  if key == "rel_side" else mv
            sim = gaussian_sim(cmp, ref, SIGMA[key])
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
            user_hb  = metrics.get("hb")   # already negated to CSV convention
            user_ivb = metrics.get("ivb")
            mv_hb    = row.get(f"hb_{col_group}")
            mv_ivb   = row.get(f"ivb_{col_group}")
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
    f(n) = 1 - exp(-n / 80)
    At 200 pitches → 0.918  (small discount)
    At 500 pitches → 0.998  (essentially full)
    At 1000+       → 1.000  (no discount)
    Effect: an 80 with 2000 pitches just beats an 85 with 200 pitches.
    """
    if n_pitches is None or not is_real(n_pitches) or n_pitches <= 0:
        return 0.90   # unknown sample — apply modest default discount
    return 1.0 - math.exp(-float(n_pitches) / 80.0)


def run_search(user, pitch_inputs, top_n):
    rows = []
    for _, r in profiles.iterrows():
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
            "Rel Side":      round(r["rel_side"],   2),
            "Extension":     round(r["extension"],  2) if is_real(r.get("extension")) else None,
            "Total Pitches": int(r["total_pitches"]),
            "_row":          r,
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

            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_lines.extend(t.split("\n"))
            except ImportError:
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(file_bytes))
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text_lines.extend(t.split("\n"))
                except ImportError:
                    return {"_error": "PDF parsing requires pdfplumber. It is now in requirements.txt — redeploy and try again."}

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
    for _, r in profiles.iterrows():
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
                ref = abs(val) if key == "rel_side" else val
                cmp = abs(mv)  if key == "rel_side" else mv
                sim = gaussian_sim(cmp, ref, SIGMA[key])
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
            if hb_csv is not None and is_real(mv_hb):
                if abs(float(mv_hb) - float(hb_csv)) > _hb_thresh:
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
                user_val = {"velo": velo, "ivb": ivb, "hb": hb_csv}[metric]
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
                if hb_csv is not None and is_real(mv_hb_g):
                    if abs(float(mv_hb_g) - float(hb_csv)) > _hb_thresh:
                        continue
                # Re-score this specific pitch type
                p_log = log_sum
                p_w   = total_w
                for metric, mv, sigma in [
                    ("velo", r.get(f"velo_{group}"), sv),
                    ("ivb",  mv_ivb_g,               SIGMA["ivb"]),
                    ("hb",   mv_hb_g,                SIGMA["hb"]),
                ]:
                    user_val = {"velo": velo, "ivb": ivb, "hb": hb_csv}[metric]
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
                    "Rel Side":     r["rel_side"],
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
                "Rel Side":     r["rel_side"],
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
            if st.button("Enter Full Arsenal →", key="btn_arsenal", use_container_width=True):
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
            if st.button("Enter Single Pitch →", key="btn_single", use_container_width=True):
                st.session_state.mode   = "single"
                st.session_state.screen = "input"
                st.rerun()

        # Third card — full width below
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
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
        if st.button("Open Pitch Leaderboard →", key="btn_leaderboard", use_container_width=True):
            st.session_state.screen = "leaderboard"
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

    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
    _, lc, _ = st.columns([1, 4, 1])
    with lc:
        st.markdown(
            "<div style='text-align:center'>"
            "<div style='font-family:Rajdhani,sans-serif;font-size:28px;font-weight:700;"
            "color:#c9a84c;letter-spacing:3px;text-transform:uppercase;margin-bottom:20px'>"
            "⚾  Searching…</div>"
            "<div style='font-family:monospace;font-size:11px;color:#7aaac0'>"
            "Scoring against 1,700+ pitcher seasons</div>"
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

    prog.progress(100, text="Done!")
    st.session_state.results = results
    st.session_state.screen  = "results"
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
        st.markdown('<div class="sec-label">Release Profile</div>', unsafe_allow_html=True)
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
            st.markdown("<div class='field-label'>Rel Side (ft)</div>", unsafe_allow_html=True)
            rel_side_v = st.number_input(" ", min_value=-5.0, max_value=5.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder="e.g. 2.80", key="rs",
                                          label_visibility="collapsed")
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
        st.markdown('<div class="sec-label">Auto-Fill from TrackMan Export (optional)</div>',
                    unsafe_allow_html=True)
        tm_file = st.file_uploader(
            "Upload TrackMan CSV or PDF to auto-fill pitch metrics below",
            type=["csv","pdf"], key="tm_upload",
            label_visibility="visible",
        )

        tm_data = {}
        if tm_file is not None:
            # Use file name+size as cache key to avoid re-parsing on every rerun
            file_id = f"{tm_file.name}_{tm_file.size}"
            if st.session_state.get("_tm_file_id") != file_id:
                # New file uploaded — parse and write into session state
                parsed = parse_trackman(tm_file.read(), tm_file.name)
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
            "HB: positive = arm-side &nbsp;·&nbsp; iVB: positive = rise &nbsp;·&nbsp; Enter raw Trackman/Rapsodo values"
            "</div>"
        )

        def pitch_inputs_widget(group, key_prefix, tm_data, show_placeholder=False):
            """Render velo/iVB/HB inputs for one pitch group. Returns (v, i, h_csv) or Nones.
            show_placeholder: show 'e.g.' hints (only for first pitch / single-pitch mode).
            """
            color   = PITCH_COLORS[group]
            tm_vals = tm_data.get(group, {})
            ph_v = "e.g. 93.5" if show_placeholder else ""
            ph_i = "e.g. 18.0" if show_placeholder else ""
            ph_h = "e.g. +14"  if show_placeholder else ""
            st.markdown(
                f"<div class='pitch-card'>"
                f"<div class='pitch-card-title' style='color:{color}'>● {group}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='field-label'>Velocity (mph)</div>", unsafe_allow_html=True)
            velo_def = float(tm_vals["velo"]) if tm_vals.get("velo") is not None else None
            velo = st.number_input(" ", min_value=60.0, max_value=105.0,
                                   value=velo_def, step=0.1, format="%.1f",
                                   placeholder=ph_v, key=f"{key_prefix}_velo",
                                   label_visibility="collapsed")
            st.markdown("<div class='field-label'>iVB (in)</div>", unsafe_allow_html=True)
            ivb_def = float(tm_vals["ivb"]) if tm_vals.get("ivb") is not None else None
            ivb = st.number_input(" ", min_value=-30.0, max_value=30.0,
                                  value=ivb_def, step=0.1, format="%.1f",
                                  placeholder=ph_i, key=f"{key_prefix}_ivb",
                                  label_visibility="collapsed")
            st.markdown("<div class='field-label'>HB (in)</div>", unsafe_allow_html=True)
            hb_def = float(tm_vals["hb"]) if tm_vals.get("hb") is not None else None
            hb = st.number_input(" ", min_value=-30.0, max_value=30.0,
                                 value=hb_def, step=0.1, format="%.1f",
                                 placeholder=ph_h, key=f"{key_prefix}_hb",
                                 label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)
            v, i, h = vn(velo), vn(ivb), vn(hb)
            h_csv = (-h) if h is not None else None
            return v, i, h_csv

        pitch_inputs_raw = {}
        sp_velo = sp_ivb = sp_hb_csv = None
        sp_pitch_type = None   # selected pitch type for single mode

        if mode == "arsenal":
            # ── FULL ARSENAL MODE ─────────────────────────────────────────
            st.markdown(
                '<div class="sec-label">Pitch Arsenal — fill only the pitches you throw</div>',
                unsafe_allow_html=True,
            )
            st.markdown(hint_html, unsafe_allow_html=True)
            all_groups = list(PITCH_GROUPS.keys())
            for row_groups in [all_groups[:4], all_groups[4:7], all_groups[7:]]:
                if not row_groups:
                    continue
                cols = st.columns(len(row_groups))
                for col, group in zip(cols, row_groups):
                    with col:
                        first_pitch = (group == all_groups[0])
                        v, i, h_csv = pitch_inputs_widget(group, f"a_{group}", tm_data,
                                                           show_placeholder=first_pitch)
                        if any(x is not None for x in [v, i, (-h_csv if h_csv is not None else None)]):
                            pitch_inputs_raw[group] = {"velo": v, "ivb": i, "hb": h_csv}

        else:
            # ── SINGLE PITCH MODE ─────────────────────────────────────────
            st.markdown(
                '<div class="sec-label">Single Pitch — enter your pitch metrics</div>',
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
                st.markdown("<div class='field-label'>Velocity (mph)</div>", unsafe_allow_html=True)
                velo_def = float(tm_vals["velo"]) if tm_vals.get("velo") is not None else None
                sp_velo_raw = st.number_input(" ", min_value=60.0, max_value=105.0,
                                   value=velo_def, step=0.1, format="%.1f",
                                   placeholder="e.g. 93.5", key="sp_velo",
                                   label_visibility="collapsed")
                st.markdown("<div class='field-label'>iVB (in)</div>", unsafe_allow_html=True)
                ivb_def = float(tm_vals["ivb"]) if tm_vals.get("ivb") is not None else None
                sp_ivb_raw = st.number_input(" ", min_value=-30.0, max_value=30.0,
                                  value=ivb_def, step=0.1, format="%.1f",
                                  placeholder="e.g. 18.0", key="sp_ivb",
                                  label_visibility="collapsed")
                st.markdown("<div class='field-label'>HB (in)</div>", unsafe_allow_html=True)
                hb_def = float(tm_vals["hb"]) if tm_vals.get("hb") is not None else None
                sp_hb_raw = st.number_input(" ", min_value=-30.0, max_value=30.0,
                                 value=hb_def, step=0.1, format="%.1f",
                                 placeholder="e.g. +14", key="sp_hb",
                                 label_visibility="collapsed")
                st.markdown("</div>", unsafe_allow_html=True)
                sp_velo = vn(sp_velo_raw)
                sp_ivb  = vn(sp_ivb_raw)
                _sp_h   = vn(sp_hb_raw)
                sp_hb_csv = (-_sp_h) if _sp_h is not None else None

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
            user = {
                "hand":       None if hand_choice == "Any" else hand_choice[0],
                "rel_height": vn(rel_height_v),
                "rel_side":   vn(rel_side_v),
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
        for _, r in profiles.iterrows():
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
                # Flip HB sign for display (positive = arm-side)
                if is_real(hb):
                    hb_disp = -float(hb) if hand == "R" else float(hb)
                else:
                    hb_disp = None
                vaa  = r.get(f"vaa_{grp}")
                haa  = r.get(f"haa_{grp}")
                sp   = r.get(f"sp_{grp}") or r.get("stuff_plus")

                rows.append({
                    "Pitcher":     display_name,
                    "Year":        yr,
                    "Hand":        hand,
                    "Pitch":       grp,
                    "Velo":        round(float(velo), 1) if is_real(velo) else None,
                    "iVB":         round(float(ivb),  1) if is_real(ivb)  else None,
                    "HB":          round(hb_disp,     1) if hb_disp is not None else None,
                    "VAA":         round(float(vaa),  1) if is_real(vaa)  else None,
                    "HAA":         round(float(haa),  1) if is_real(haa)  else None,
                    "Stuff+":      round(float(sp),   0) if is_real(sp)   else None,
                    "Rel Ht":      round(float(rh),   2) if is_real(rh)   else None,
                    "Rel Side":    round(float(rs),   2) if is_real(rs)   else None,
                    "Extension":   round(float(ext),  2) if is_real(ext)  else None,
                })

        lb = pd.DataFrame(rows)

        # Join zone stats for CSW% and xwOBA per pitcher-pitch
        if zone_stats_ok and not zone_stats.empty:
            zs = zone_stats.copy()
            if "stand" in zs.columns:
                zs = zs[zs["stand"] == "all"]
            zs["_csw_w"] = zs["csw_pct"] * zs["n_pitches"]
            agg = zs.groupby(["player_name", "year", "pitch_group"]).agg(
                n_pitches    = ("n_pitches",  "sum"),
                csw_weighted = ("_csw_w",     "sum"),
                xwoba_mean   = ("xwoba_mean", "mean"),
            ).reset_index()
            agg["CSW%"]  = (agg["csw_weighted"] / agg["n_pitches"].clip(lower=1) * 100).round(1)
            agg["xwOBA"] = agg["xwoba_mean"].round(3)
            if "whiff_pct" in zs.columns:
                zs["_whiff_w"] = zs["whiff_pct"] * zs["n_pitches"]
                wagg = zs.groupby(["player_name", "year", "pitch_group"]).agg(
                    whiff_wsum = ("_whiff_w",   "sum"),
                    n_total    = ("n_pitches",  "sum"),
                ).reset_index()
                wagg["whiff_w"] = wagg["whiff_wsum"] / wagg["n_total"].clip(lower=1)
                wagg = wagg.rename(columns={"player_name": "player_name", "year": "year", "pitch_group": "pitch_group"})
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

    lb_df = build_leaderboard(len(profiles))

    # Add Whiff% overall column if zone stats available (uses all-pitch denominator)
    if zone_stats_ok and not zone_stats.empty and "whiff_pct_overall" in zone_stats.columns:
        zs_all = zone_stats[zone_stats.get("stand","all") == "all"] if "stand" in zone_stats.columns else zone_stats
        wh_agg = zs_all.groupby(["player_name","year","pitch_group"]).agg(
            tw = ("n_pitches","sum"),
            ww = ("whiff_count","sum") if "whiff_count" in zs_all.columns else ("n_pitches","sum"),
        ).reset_index() if "whiff_count" in zs_all.columns else None

    # ── Filter controls ───────────────────────────────────────────────────────
    METRIC_COLS = ["N","Velo","iVB","HB","VAA","HAA","Stuff+","CSW%","Whiff%","xwOBA","Rel Ht","Rel Side","Extension"]

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
        col_ms, col_clr = st.columns([5, 1])
        with col_ms:
            selected_pitches = st.multiselect(
                " ", options=all_pitches, default=all_pitches,
                key="lb_pitches", label_visibility="collapsed",
            )
        with col_clr:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("✕ All", key="lb_clear_pitches", help="Select all pitch types"):
                st.session_state["lb_pitches"] = all_pitches
                st.rerun()
        if not selected_pitches:
            selected_pitches = all_pitches
    with fc2:
        hand_filter = st.selectbox(" ", ["All","RHP","LHP"], key="lb_hand", label_visibility="collapsed")
    with fc3:
        year_options = ["All"] + sorted(lb_df["Year"].unique().tolist(), reverse=True)
        year_filter = st.selectbox(" ", year_options, key="lb_year", label_visibility="collapsed")

    # ── Sort state ────────────────────────────────────────────────────────────
    if "lb_sort_col" not in st.session_state:
        st.session_state["lb_sort_col"] = "Velo"
        st.session_state["lb_sort_asc"] = False

    sort_by  = st.session_state["lb_sort_col"]
    sort_asc = st.session_state["lb_sort_asc"]

    # ── Apply filters ─────────────────────────────────────────────────────────
    view = lb_df.copy()
    filter_cols = [c for c in METRIC_COLS if c in view.columns]

    if selected_pitches:
        view = view[view["Pitch"].isin(selected_pitches)]
    if hand_filter == "RHP":
        view = view[view["Hand"] == "R"]
    elif hand_filter == "LHP":
        view = view[view["Hand"] == "L"]
    if year_filter != "All":
        view = view[view["Year"] == int(year_filter)]

    # ── Sort then top-500 ─────────────────────────────────────────────────────
    if sort_by in view.columns:
        view = view.sort_values(sort_by, ascending=sort_asc, na_position="last")
    total_rows = len(view)
    view_display = view.head(500)

    # ── Gradient baselines ────────────────────────────────────────────────────
    lb_baselines = {}
    for col in filter_cols:
        if col in lb_df.columns:
            vals = lb_df[col].dropna()
            if len(vals) > 10:
                lb_baselines[col] = (float(vals.mean()), float(vals.std()))

    def cell_color(col, val):
        if val is None or (isinstance(val, float) and val != val):
            return "#141e2e", "#7aaac0"
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

    DISPLAY_COLS = ["Pitcher","Year","Hand","Pitch"] + [c for c in filter_cols if c in view_display.columns]
    NUMERIC_COLS = [c for c in DISPLAY_COLS if c in filter_cols]

    # ── Build the HTML table with inline range inputs above headers ───────────
    # Range filter state
    range_filters = {}
    for col in NUMERIC_COLS:
        mn_key = f"lb_min_{col}"
        mx_key = f"lb_max_{col}"
        mn = st.session_state.get(mn_key)
        mx = st.session_state.get(mx_key)
        if mn is not None or mx is not None:
            range_filters[col] = (mn, mx)

    # Apply range filters
    for metric, (mn, mx) in range_filters.items():
        if metric in view_display.columns:
            if mn is not None:
                view_display = view_display[view_display[metric].fillna(-999) >= mn]
            if mx is not None:
                view_display = view_display[view_display[metric].fillna(999) <= mx]

    st.markdown(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7aaac0;margin:8px 0 4px 0'>"
        f"{total_rows:,} pitches · showing top {min(500,total_rows):,} by {sort_by} "
        f"({'↑' if sort_asc else '↓'})</div>",
        unsafe_allow_html=True,
    )

    # ── Range filter inputs — one row per 4 numeric cols ─────────────────────
    with st.expander("🔧  Range Filters — click to expand", expanded=False):
        for i in range(0, len(NUMERIC_COLS), 4):
            batch = NUMERIC_COLS[i:i+4]
            rcols = st.columns(len(batch) * 2)
            for j, col in enumerate(batch):
                with rcols[j*2]:
                    st.markdown(f"<div style='font-family:monospace;font-size:8px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>{col} min</div>", unsafe_allow_html=True)
                    mn_v = st.number_input(" ", value=None, key=f"lb_min_{col}", label_visibility="collapsed", format="%.2f")
                with rcols[j*2+1]:
                    st.markdown(f"<div style='font-family:monospace;font-size:8px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>{col} max</div>", unsafe_allow_html=True)
                    mx_v = st.number_input(" ", value=None, key=f"lb_max_{col}", label_visibility="collapsed", format="%.2f")

    # ── Sortable HTML table — headers use sendPrompt to trigger sort ──────────
    # Each header th contains a form that posts the column name back via st button
    # Since we can't do JS→Python directly in static HTML, we use Streamlit buttons
    # rendered as a fake header row above the HTML table

    # Render sort button row styled to look like table headers
    header_btn_html = (
        "<div style='display:grid;grid-template-columns:"
        + " ".join(["2fr" if c in ("Pitcher",) else "0.7fr" if c in ("Year","Hand") else "1.2fr" if c == "Pitch" else "1fr" for c in DISPLAY_COLS])
        + ";gap:1px;background:#0c1420;border-radius:10px 10px 0 0;"
        "border:1px solid #1a2a40;border-bottom:none;overflow:hidden'>"
    )
    for col in DISPLAY_COLS:
        is_sort = col == sort_by
        arrow = (" ↓" if not sort_asc else " ↑") if is_sort else ""
        bg = "#162236" if is_sort else "#0c1420"
        color = "#d4a848" if is_sort else "#7aaac0"
        header_btn_html += (
            f"<div style='padding:7px 6px;text-align:{'left' if col in ('Pitcher','Pitch') else 'center'};"
            f"font-family:JetBrains Mono,monospace;font-size:9px;color:{color};"
            f"text-transform:uppercase;letter-spacing:1px;background:{bg};"
            f"cursor:pointer;white-space:nowrap;font-weight:{'700' if is_sort else '500'}'>"
            f"{col}{arrow}</div>"
        )
    header_btn_html += "</div>"
    st.markdown(header_btn_html, unsafe_allow_html=True)

    # Actual sort buttons — invisible, triggered by user clicking a col name
    # Use a compact button row below the visual header
    sort_cols_ui = st.columns(len(DISPLAY_COLS))
    for idx, col in enumerate(DISPLAY_COLS):
        with sort_cols_ui[idx]:
            if st.button(col, key=f"lbsort_{col}", use_container_width=True,
                         help=f"Sort by {col}"):
                if st.session_state.get("lb_sort_col") == col:
                    st.session_state["lb_sort_asc"] = not st.session_state.get("lb_sort_asc", False)
                else:
                    st.session_state["lb_sort_col"] = col
                    st.session_state["lb_sort_asc"] = False
                st.rerun()

    # Hide the actual buttons visually but keep them functional
    st.markdown("""
    <style>
    /* Make sort buttons invisible but clickable — they sit under the visual header */
    div[data-testid="stHorizontalBlock"]:has(button[kind="secondary"][data-testid="baseButton-secondary"]) {
        margin-top: -42px !important;
        opacity: 0 !important;
        height: 36px !important;
        overflow: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Data rows ─────────────────────────────────────────────────────────────
    rows_html = ""
    for _, row in view_display.iterrows():
        row_cells = ""
        for col in DISPLAY_COLS:
            val = row.get(col)
            if col == "Pitcher":
                row_cells += f"<td style='padding:5px 8px;font-family:Inter,sans-serif;font-size:11px;font-weight:600;color:#d8cbb4;white-space:nowrap'>{val}</td>"
            elif col == "Year":
                row_cells += f"<td style='padding:5px 8px;text-align:center;font-family:JetBrains Mono,monospace;font-size:10px;color:#8ab0c8'>{int(val) if val else '—'}</td>"
            elif col == "Hand":
                c = "#e8a060" if val == "R" else "#6ab0e8"
                row_cells += f"<td style='padding:5px 8px;text-align:center;font-family:JetBrains Mono,monospace;font-size:10px;color:{c};font-weight:700'>{val}HP</td>"
            elif col == "Pitch":
                pc = PITCH_COLORS.get(str(val), "#8aadcc")
                row_cells += f"<td style='padding:5px 8px;font-family:Inter,sans-serif;font-size:10px;font-weight:700;color:{pc};white-space:nowrap'>● {val}</td>"
            elif col == "N":
                fmt = f"{int(val):,}" if val is not None and not (isinstance(val,float) and val!=val) else "—"
                row_cells += f"<td style='padding:5px 8px;text-align:center;font-family:JetBrains Mono,monospace;font-size:10px;color:#8ab0c8'>{fmt}</td>"
            else:
                bg, txt = cell_color(col, val)
                fmt_val = "—"
                if val is not None and not (isinstance(val, float) and val != val):
                    if col in ("CSW%","Whiff%"): fmt_val = f"{val:.1f}%"
                    elif col == "xwOBA":         fmt_val = f"{val:.3f}"
                    elif col == "Stuff+":        fmt_val = f"{int(val)}"
                    elif col in ("HB","VAA","HAA"): fmt_val = f"{val:+.1f}"
                    else:                         fmt_val = f"{val:.1f}"
                row_cells += (
                    f"<td style='padding:4px 6px;text-align:center;background:{bg};color:{txt};"
                    f"font-family:JetBrains Mono,monospace;font-size:11px;font-weight:600;"
                    f"border-radius:4px'>{fmt_val}</td>"
                )
        rows_html += f"<tr style='border-bottom:1px solid #0f1820'>{row_cells}</tr>"

    if total_rows > 500:
        rows_html += (
            f"<tr><td colspan='{len(DISPLAY_COLS)}' style='padding:8px;text-align:center;"
            f"font-family:JetBrains Mono,monospace;font-size:10px;color:#7aaac0'>"
            f"Top 500 of {total_rows:,} by {sort_by} — use filters to see more</td></tr>"
        )

    table_html = (
        "<div style='overflow-x:auto;border:1px solid #1a2a40;border-top:none;border-radius:0 0 10px 10px'>"
        "<table style='width:100%;border-collapse:separate;border-spacing:0 1px;background:#0a0e18'>"
        f"<tbody>{rows_html}</tbody></table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Inline heatmap viewer — one dropdown per visible row ─────────────────
    # Build a list of (Pitcher, Year, Pitch) tuples from current display
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#7aaac0;"
        "text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>"
        "📊 Click a row to view its zone heatmaps</div>",
        unsafe_allow_html=True,
    )

    # One selectbox — choose which pitch row to inspect
    row_labels = [f"{r['Pitcher']}  {r['Year']}  ●{r['Pitch']}"
                  for _, r in view_display.iterrows()]
    sel_row = st.selectbox(
        " ",
        options=["— select a pitch —"] + row_labels,
        key="lb_hm_row",
        label_visibility="collapsed",
    )

    if sel_row != "— select a pitch —" and zone_stats_ok:
        # Parse selection
        idx_sel = row_labels.index(sel_row)
        sel_row_data = view_display.iloc[idx_sel]
        sel_pitcher = sel_row_data["Pitcher"]
        sel_year    = int(sel_row_data["Year"])
        sel_pitch   = sel_row_data["Pitch"]

        def _csv_name(dn):
            parts = dn.rsplit(" ", 1)
            return f"{parts[1]}, {parts[0]}" if len(parts) == 2 else dn
        csv_nm = _csv_name(sel_pitcher)

        st.markdown(
            f"<div style='font-family:Inter,sans-serif;font-size:12px;font-weight:700;"
            f"color:{PITCH_COLORS.get(sel_pitch,'#8aadcc')};letter-spacing:1.5px;"
            f"text-transform:uppercase;margin:8px 0 6px 0'>"
            f"● {sel_pitcher}  {sel_year}  —  {sel_pitch}</div>",
            unsafe_allow_html=True,
        )

        # All-batters heatmaps
        hm_data = pitcher_zone_data(csv_nm, sel_year, sel_pitch)
        if not hm_data.empty:
            hc1, hc2, hc3 = st.columns(3)
            with hc1: st.markdown(render_zone_heatmap(hm_data,"csw_pct","csw","CSW% (All)",fmt=".1%"), unsafe_allow_html=True)
            with hc2: st.markdown(render_zone_heatmap(hm_data,"whiff_pct","whiff","Whiff% (All)",fmt=".1%"), unsafe_allow_html=True)
            with hc3: st.markdown(render_zone_heatmap(hm_data,"xwoba_mean","xwoba","xwOBA (All)",fmt=".3f"), unsafe_allow_html=True)

            # Same/opp splits
            has_stand = "stand" in zone_stats.columns
            if has_stand:
                same_data = pitcher_zone_data_by_stand(csv_nm, sel_year, sel_pitch, "same")
                opp_data  = pitcher_zone_data_by_stand(csv_nm, sel_year, sel_pitch, "opp")
                if not same_data.empty:
                    st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:8px 0 4px 0'>vs Same Hand</div>", unsafe_allow_html=True)
                    hs1, hs2, hs3 = st.columns(3)
                    with hs1: st.markdown(render_zone_heatmap(same_data,"csw_pct","csw","CSW%",fmt=".1%"), unsafe_allow_html=True)
                    with hs2: st.markdown(render_zone_heatmap(same_data,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                    with hs3: st.markdown(render_zone_heatmap(same_data,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
                if not opp_data.empty:
                    st.markdown("<div style='font-family:monospace;font-size:9px;color:#7aaac0;text-transform:uppercase;letter-spacing:1px;margin:8px 0 4px 0'>vs Opposite Hand</div>", unsafe_allow_html=True)
                    ho1, ho2, ho3 = st.columns(3)
                    with ho1: st.markdown(render_zone_heatmap(opp_data,"csw_pct","csw","CSW%",fmt=".1%"), unsafe_allow_html=True)
                    with ho2: st.markdown(render_zone_heatmap(opp_data,"whiff_pct","whiff","Whiff%",fmt=".1%"), unsafe_allow_html=True)
                    with ho3: st.markdown(render_zone_heatmap(opp_data,"xwoba_mean","xwoba","xwOBA",fmt=".3f"), unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='font-family:monospace;font-size:10px;color:#7aaac0;"
                "padding:10px;background:#0c1420;border-radius:8px;border:1px solid #1a2a40'>"
                "No zone data for this pitch — rebuild pitch_zone_stats.csv</div>",
                unsafe_allow_html=True,
            )



# SCREEN: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "results":

    results      = st.session_state.results
    snap         = st.session_state.user_snapshot
    user         = snap["user"]
    pitch_inputs = snap["pitch_inputs"]
    result_mode  = snap.get("mode", "arsenal")  # "arsenal" or "single"
    sp_pitch_type = snap.get("sp_pitch_type")   # pitch type filter for single mode

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Header row: back btn + centered profile strip ─────────────────────
    back_col, _ = st.columns([1, 8])
    with back_col:
        st.markdown('<div class="back-btn-wrap">', unsafe_allow_html=True)
        if st.button("← New Search"):
            st.session_state.screen  = "title"
            st.session_state.results = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Profile strip — full width, centered
    parts = []
    if user.get("hand"):       parts.append(f"<b style='color:#e8dcc8'>{snap['hand_label']}</b>")
    if user.get("rel_height"): parts.append(f"HT <b style='color:#e8dcc8'>{user['rel_height']:.2f}'</b>")
    if user.get("rel_side"):   parts.append(f"SIDE <b style='color:#e8dcc8'>{user['rel_side']:.2f}'</b>")
    if user.get("extension"):  parts.append(f"EXT <b style='color:#e8dcc8'>{user['extension']:.2f}'</b>")
    for g, m in pitch_inputs.items():
        subs = []
        if m.get("velo"): subs.append(f"{m['velo']:.1f}")
        if m.get("ivb"):  subs.append(f"iVB {m['ivb']:.1f}\"")
        if m.get("hb"):   subs.append(f"HB {-m['hb']:.1f}\"")
        if subs: parts.append(f"<b style='color:{PITCH_COLORS[g]}'>{g}</b>: {', '.join(subs)}")
    if parts:
        st.markdown(
            "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#a0c0d4;"
            "background:linear-gradient(165deg,#0e1828,#0c1420);padding:10px 20px;"
            "border-radius:10px;border:1px solid #162236;"
            "text-align:center;margin:4px 0 10px 0'>"
            "<span style='color:#d4a848;font-family:Inter,sans-serif;font-weight:700;"
            "letter-spacing:2px;text-transform:uppercase;font-size:10px'>PROFILE </span>"
            + " &nbsp;·&nbsp; ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RESULTS",       len(results))
    m2.metric("SEASONS",       "2017–2024")
    def _full(raw):
        if "," in raw:
            p = raw.split(",", 1)
            return f"{p[1].strip()} {p[0].strip()}"
        return raw
    m3.metric("TOP MATCH", _full(results[0]["Pitcher"]) if results else "—")
    m4.metric("BEST SCORE",    f"{results[0]['Similarity']:.1f}" if results else "—")
    m5.metric("PITCHERS USED", f"{profiles['player_name'].nunique():,}")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # COMP PANEL — aggregate stats + zone heatmaps for the top-N matches
    # ══════════════════════════════════════════════════════════════════════════════

    # Determine which pitch group(s) to show in the comp panel
    if result_mode == "single":
        # Single-pitch mode: collect matched pitch types, sorted by avg similarity
        # so the best-matching pitch type appears first in the comp panel
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
        # Arsenal mode: show comp panel for each pitch the user entered
        # Sort by PITCH_GROUPS canonical order (4-Seam first, etc.)
        pg_order = list(PITCH_GROUPS.keys())
        comp_groups = sorted(
            list(pitch_inputs.keys()) if pitch_inputs else [],
            key=lambda g: pg_order.index(g) if g in pg_order else 99
        )

    def stat_pill(label, val, color, sub=None):
        sub_html = f"<div style='font-size:8px;color:#000000aa;margin-top:2px;font-family:monospace'>{sub}</div>" if sub else ""
        # Mix color with white at 70% for readable bg; always black text
        return (
            f"<div style='text-align:center;background:{color};border-radius:8px;"
            f"padding:8px 6px;flex:1;min-width:60px;max-width:90px;"
            f"filter:brightness(1.3)'>"
            f"<div style='font-family:monospace;font-size:7px;color:#000000bb;"
            f"text-transform:uppercase;letter-spacing:1px;margin-bottom:3px'>{label}</div>"
            f"<div style='font-family:Rajdhani,sans-serif;font-size:18px;"
            f"font-weight:700;color:#000000;line-height:1'>{val}</div>"
            f"{sub_html}</div>"
        )

    def render_comp_section(title_label, title_color, groups_to_show):
        """Render the aggregate comp stats for a list of pitch groups (or overall)."""
        st.markdown(
            f"<div style='font-family:Inter,sans-serif;font-size:13px;font-weight:700;"
            f"color:{title_color};letter-spacing:2px;text-transform:uppercase;"
            f"margin:0 0 10px 0'>● {title_label}</div>",
            unsafe_allow_html=True,
        )

        for grp in groups_to_show:
            color = PITCH_COLORS.get(grp, "#8aadcc")
            agg = comp_aggregate_stats(results, pitch_group=grp)
            zdf = comp_zone_data(results, pitch_group=grp)

            def fv(k, fmt=""):
                if k not in agg: return "—"
                v, _ = agg[k]
                if fmt == "pct":   return f"{v:.1%}"
                if fmt == "f3":    return f"{v:.3f}"
                if fmt == "f1":    return f"{v:.1f}"
                if fmt == "i":     return f"{v:.0f}"
                return f"{v:.1f}"

            def _fv_hb():
                """HB for display: flip sign so positive = arm-side (matches individual cards)."""
                if "hb" not in agg: return "—"
                raw, _ = agg["hb"]
                # Determine dominant hand from comp results for this pitch group
                hands = [r.get("Hand","R") for r in results
                         if not grp or r.get("Matched Pitch") == grp or r.get("Hand")]
                dominant = "R" if hands.count("R") >= hands.count("L") else "L"
                display = -raw if dominant == "R" else raw
                return f"{display:+.1f}"

            sp_val, _ = agg.get("stuff_plus", (None, 0))
            csw_val, _ = agg.get("csw", (None, 0))
            xw_val, _  = agg.get("xwoba", (None, 0))
            vaa_val, _ = agg.get("vaa", (None, 0))
            haa_val, _ = agg.get("haa", (None, 0))
            sp_c    = stuff_color(sp_val) if sp_val else "#4a6a80"
            csw_c   = stat_gradient_color(csw_val, 0.28, 0.04) if csw_val else "#4a6a80"
            xw_c    = stat_gradient_color(xw_val,  0.36, 0.06, invert=True) if xw_val else "#4a6a80"
            whiff_val, _ = agg.get("whiff", (None, 0))
            whiff_c = stat_gradient_color(whiff_val, 0.22, 0.04) if whiff_val else "#4a6a80"

            # VAA/HAA gradient vs pitch-type league average
            _vl = _vaa_haa_league.get(grp, {})
            if _vl and vaa_val is not None:
                vaa_c = stat_gradient_color(vaa_val, _vl["vaa_mu"], _vl["vaa_sd"], invert=True)
            else:
                vaa_c = "#6aacaa"
            if _vl and haa_val is not None:
                _hz = min(abs((haa_val - _vl["haa_mu"]) / max(_vl["haa_sd"], 0.001)), 2.0)
                _ht = _hz / 2.0
                _hr = int(120 + (220 - 120) * _ht)
                _hg = int(130 + (35  - 130) * _ht)
                _hb_c = int(140 + (35  - 140) * _ht)
                haa_c = f"rgb({_hr},{_hg},{_hb_c})"
            else:
                haa_c = "#6aacaa"

            # Group header
            st.markdown(
                f"<div style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
                f"color:{color};letter-spacing:1.5px;text-transform:uppercase;"
                f"margin-bottom:6px;border-left:3px solid {color};padding-left:8px'>{grp}</div>",
                unsafe_allow_html=True,
            )

            # Pills full-width (no columns = no Streamlit label injection)
            pills_html = (
                "<div style='display:grid;grid-template-columns:repeat(4,1fr);"
                "gap:6px;margin-bottom:12px'>"
                + stat_pill("VELO", fv("velo","f1"), color)
                + stat_pill("iVB",  fv("ivb","f1"),  "#8aadcc")
                + stat_pill("HB",   _fv_hb(),        "#8aadcc")
                + stat_pill("VAA",  fv("vaa","f1"),  vaa_c)
                + stat_pill("HAA",  fv("haa","f1"),  haa_c)
                + stat_pill("FG S+", fv("stuff_plus","i"), sp_c if sp_val else "#3a4a5a",
                            stuff_grade_label(sp_val) if sp_val else "rebuild CSV")
                + stat_pill("CSW%",   fv("csw","pct"),   csw_c)
                + stat_pill("Whiff%", fv("whiff","pct"), whiff_c)
                + stat_pill("xwOBA",  fv("xwoba","f3"),  xw_c)
                + "</div>"
            )
            st.markdown(pills_html, unsafe_allow_html=True)

            # Heatmaps in expander — same layout as pitcher cards
            with st.expander(f"📊  {grp} comp zone heatmaps", expanded=False):
                hm_col1, hm_col2, hm_col3 = st.columns(3)
                with hm_col1:
                    svg = render_zone_heatmap(zdf, "csw_pct", "csw", f"CSW% (All)", fmt=".1%")
                    st.markdown(svg, unsafe_allow_html=True)
                with hm_col2:
                    svg = render_zone_heatmap(zdf, "whiff_pct", "whiff", f"Whiff% (All)", fmt=".1%")
                    st.markdown(svg, unsafe_allow_html=True)
                with hm_col3:
                    svg = render_zone_heatmap(zdf, "xwoba_mean", "xwoba", f"xwOBA (All)", fmt=".3f")
                    st.markdown(svg, unsafe_allow_html=True)

                has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                if has_stand_col:
                    zdf_same = comp_zone_data(results, pitch_group=grp, stand="same")
                    zdf_opp  = comp_zone_data(results, pitch_group=grp, stand="opp")
                    if not zdf_same.empty:
                        st.markdown(
                            "<div style='font-family:monospace;font-size:9px;color:#7aaac0;"
                            "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                            "vs Same Hand</div>", unsafe_allow_html=True)
                        hs1, hs2, hs3 = st.columns(3)
                        with hs1:
                            st.markdown(render_zone_heatmap(zdf_same, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                        with hs2:
                            st.markdown(render_zone_heatmap(zdf_same, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                        with hs3:
                            st.markdown(render_zone_heatmap(zdf_same, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)
                    if not zdf_opp.empty:
                        st.markdown(
                            "<div style='font-family:monospace;font-size:9px;color:#7aaac0;"
                            "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                            "vs Opposite Hand</div>", unsafe_allow_html=True)
                        ho1, ho2, ho3 = st.columns(3)
                        with ho1:
                            st.markdown(render_zone_heatmap(zdf_opp, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                        with ho2:
                            st.markdown(render_zone_heatmap(zdf_opp, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                        with ho3:
                            st.markdown(render_zone_heatmap(zdf_opp, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Overall aggregate (all pitches / single-pitch matched type) ───────
    if result_mode == "single" and comp_groups:
        render_comp_section(f"Comp Average — {comp_groups[0]}", "#c9a84c", comp_groups)
    elif comp_groups:
        render_comp_section("Comp Average — Per Pitch", "#c9a84c", comp_groups)
        # Also overall zone heatmap across everything
        overall_zdf = comp_zone_data(results, pitch_group=None)
        if not overall_zdf.empty:
            with st.expander("📊  Overall all-pitch comp zone heatmaps", expanded=False):
                oz1, oz2, oz3 = st.columns(3)
                with oz1:
                    svg = render_zone_heatmap(overall_zdf, "csw_pct", "csw", "CSW% (All)", fmt=".1%")
                    st.markdown(svg, unsafe_allow_html=True)
                with oz2:
                    svg = render_zone_heatmap(overall_zdf, "whiff_pct", "whiff", "Whiff% (All)", fmt=".1%")
                    st.markdown(svg, unsafe_allow_html=True)
                with oz3:
                    svg = render_zone_heatmap(overall_zdf, "xwoba_mean", "xwoba", "xwOBA (All)", fmt=".3f")
                    st.markdown(svg, unsafe_allow_html=True)
                has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                if has_stand_col:
                    ov_same = comp_zone_data(results, pitch_group=None, stand="same")
                    ov_opp  = comp_zone_data(results, pitch_group=None, stand="opp")
                    if not ov_same.empty:
                        st.markdown(
                            "<div style='font-family:monospace;font-size:9px;color:#7aaac0;"
                            "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                            "vs Same Hand</div>", unsafe_allow_html=True)
                        os1, os2, os3 = st.columns(3)
                        with os1:
                            st.markdown(render_zone_heatmap(ov_same, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                        with os2:
                            st.markdown(render_zone_heatmap(ov_same, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                        with os3:
                            st.markdown(render_zone_heatmap(ov_same, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)
                    if not ov_opp.empty:
                        st.markdown(
                            "<div style='font-family:monospace;font-size:9px;color:#7aaac0;"
                            "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                            "vs Opposite Hand</div>", unsafe_allow_html=True)
                        oo1, oo2, oo3 = st.columns(3)
                        with oo1:
                            st.markdown(render_zone_heatmap(ov_opp, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                        with oo2:
                            st.markdown(render_zone_heatmap(ov_opp, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                        with oo3:
                            st.markdown(render_zone_heatmap(ov_opp, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)

    st.markdown("---")

    # ── Pitcher list ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="sec-label">Top {snap["top_n"]} Matches — click any pitcher to expand</div>',
        unsafe_allow_html=True,
    )

    for idx, r in enumerate(results):
        row     = r["_row"]
        sc      = r["Similarity"]
        sc_c    = sim_color(sc)
        # Full name: CSV stores "Last, First" — convert to "First Last"
        raw_name = r["Pitcher"]
        if "," in raw_name:
            parts    = raw_name.split(",", 1)
            full_name = f"{parts[1].strip()} {parts[0].strip()}"
        else:
            full_name = raw_name
        ext_str = f"{r['Extension']:.2f}" if r["Extension"] else "—"
        hand    = r["Hand"]
        rank    = idx + 1

        matched_pitch = r.get("Matched Pitch", "")
        header = (
            f"#{rank}  {full_name}  {r['Year']}  ({hand}HP)"
            + (f"  ·  {matched_pitch}" if result_mode == "single" and matched_pitch else "")
            + f"  ·  SIM {sc:.1f}"
            + f"  ·  HT {r['Rel Height']:.2f}  SIDE {r['Rel Side']:.2f}  EXT {ext_str}"
        )

        with st.expander(header, expanded=(idx == 0)):

            # ── Overall FG Stuff+ from profiles CSV ──────────────────
            fg_overall = row.get("stuff_plus")
            fg_has = fg_overall is not None and not (isinstance(fg_overall, float) and math.isnan(fg_overall))
            fg_overall_str = f"{fg_overall:.0f}" if fg_has else "—"
            fg_overall_col = stuff_color(fg_overall) if fg_has else "#3a6a8a"
            fg_overall_lbl = stuff_grade_label(fg_overall) if fg_has else "—"

            # ── Release + overall Stuff+ summary strip ────────────────────
            strip = (
                "<div style='display:flex;flex-wrap:wrap;gap:16px;align-items:center;"
                "font-family:JetBrains Mono,monospace;font-size:11px;color:#a8c4d8;"
                "background:linear-gradient(165deg,#0e1828,#0c1420);border:1px solid #162236;"
                "border-radius:10px;padding:12px 18px;margin-bottom:16px'>"
                f"<span>HT <b style='color:#8aadcc'>{r['Rel Height']:.2f} ft</b></span>"
                f"<span>SIDE <b style='color:#8aadcc'>{r['Rel Side']:.2f} ft</b></span>"
                f"<span>EXT <b style='color:#8aadcc'>{ext_str} ft</b></span>"
                + (f"<span>P <b style='color:#8aadcc'>{int(row.get('total_pitches',0)):,}</b></span>"
                   if is_real(row.get('total_pitches')) else '') +
                "<span style='color:#4a6880;opacity:0.5'>│</span>"
                f"<span>SIM <b style='color:{sc_c};font-size:14px'>{sc:.1f}</b></span>"
                "<span style='margin-left:auto;display:flex;align-items:center;gap:10px'>"
                "<span style='color:#8aadcc;font-size:9px;text-transform:uppercase;"
                "letter-spacing:1.2px;font-weight:500'>FG Stuff+</span>"
                f"<span style='background:{fg_overall_col}18;border:1px solid {fg_overall_col}30;"
                f"border-radius:8px;padding:3px 10px'>"
                f"<b style='color:{fg_overall_col};font-size:18px'>{fg_overall_str}</b>"
                "</span>"
                f"<span style='color:#7aaac0;font-size:10px'>({fg_overall_lbl})</span>"
                "</span>"
                "</div>"
            )
            st.markdown(strip, unsafe_allow_html=True)

            # ── Pitch cards — 3 per row ───────────────────────────────────
            # HB display logic:
            #   CSV stores arm-side normalized: negative = arm-side for BOTH hands
            #   (because Statcast pfx_x is negative for RHP arm-side)
            #   For display: RHP should show positive arm-side → flip sign
            #   LHP already shows positive arm-side correctly → keep as-is
            def display_hb(raw_hb, pitcher_hand):
                if not is_real(raw_hb):
                    return None
                return -raw_hb if pitcher_hand == "R" else raw_hb

            # Only show pitches with sufficient sample (n >= 50 pitches, >= 1% usage)
            # n_{group} column added by new build_profiles.py
            MIN_DISPLAY_N   = 50
            MIN_DISPLAY_PCT = 0.01
            def pitch_has_data(g):
                if not is_real(row.get(f"velo_{g}")):
                    return False
                n   = row.get(f"n_{g}")
                pct = row.get(f"pct_{g}")
                # If n_ columns don't exist yet (old CSV), fall back to velo presence
                if not is_real(n):
                    return True
                return float(n) >= MIN_DISPLAY_N and float(pct) >= MIN_DISPLAY_PCT
            # Sort pitches by usage % descending (most-used first)
            # Falls back to PITCH_GROUPS order if pct_ columns don't exist yet
            def pitch_pct(g):
                pct = row.get(f"pct_{g}")
                return float(pct) if is_real(pct) else 0.0
            # Sort pitches:
            # - Arsenal mode: by usage % descending
            # - Single-pitch mode: by similarity to user's entered metrics descending
            matched = r.get("Matched Pitch") if result_mode == "single" else None
            snap_sp = st.session_state.get("user_snapshot", {})

            def pitch_similarity_score(g):
                """Score this pitcher's pitch g against user's single-pitch input."""
                if result_mode != "single":
                    return 0.0
                sp_v = snap_sp.get("sp_velo")
                sp_i = snap_sp.get("sp_ivb")
                sp_h = snap_sp.get("sp_hb_csv")
                mv_velo = row.get(f"velo_{g}")
                mv_ivb  = row.get(f"ivb_{g}")
                mv_hb   = row.get(f"hb_{g}")
                if not is_real(mv_velo):
                    return 0.0
                score = 0.0
                n = 0
                sv = velo_sigma(sp_v)
                for uval, mval, sigma in [
                    (sp_v, mv_velo, sv),
                    (sp_i, mv_ivb,  SIGMA["ivb"]),
                    (sp_h, mv_hb,   SIGMA["hb"]),
                ]:
                    if uval is None or not is_real(mval):
                        continue
                    score += gaussian_sim(mval, uval, sigma)
                    n += 1
                return score / max(n, 1)

            def pitch_sort_key(g):
                matched = r.get("Matched Pitch")
                if result_mode == "single":
                    if g == matched:
                        return (0, 0.0)        # matched pitch always first
                    return (1, -pitch_pct(g))  # rest by usage %
                return (0, -pitch_pct(g))      # arsenal: all by usage %

            active = sorted(
                [g for g in PITCH_GROUPS if pitch_has_data(g)],
                key=pitch_sort_key
            )

            if not active:
                st.markdown(
                    "<div style='color:#8aadcc;font-family:JetBrains Mono,monospace;font-size:11px'>"
                    "No pitch data for this season.</div>",
                    unsafe_allow_html=True,
                )
            else:
                def sub_label(val, color_hex="#5a8aaa"):
                    if val:
                        return (
                            f"<div style='font-family:JetBrains Mono,monospace;font-size:9px;"
                            f"color:{color_hex};margin-top:2px'>{val}</div>"
                        )
                    return ""

                # ── Per-pitch cards with inline heatmap expander ──────────
                for group in active:
                    color     = PITCH_COLORS[group]
                    user_m    = pitch_inputs.get(group, {})
                    # In single-pitch mode, inject user metrics for the matched pitch
                    if result_mode == "single" and group == r.get("Matched Pitch") and not user_m:
                        sp_v = snap.get("sp_velo")
                        sp_i = snap.get("sp_ivb")
                        sp_h = snap.get("sp_hb_csv")
                        user_m = {
                            k: v for k, v in
                            [("velo", sp_v), ("ivb", sp_i), ("hb", sp_h)]
                            if v is not None
                        }
                    mv_velo   = row.get(f"velo_{group}")
                    mv_ivb    = row.get(f"ivb_{group}")
                    mv_hb_raw = row.get(f"hb_{group}")
                    mv_hb     = display_hb(mv_hb_raw, hand)
                    vaa_v     = row.get(f"vaa_{group}")
                    haa_v     = row.get(f"haa_{group}")

                    velo_s = f"{mv_velo:.1f}"   if is_real(mv_velo) else "—"
                    ivb_s  = f"{mv_ivb:.1f}\""  if is_real(mv_ivb)  else "—"
                    hb_s   = f"{mv_hb:+.1f}\"" if mv_hb is not None else "—"
                    vaa_s  = f"{vaa_v:.1f}°"   if is_real(vaa_v)   else "—"
                    haa_s  = f"{haa_v:.1f}°"   if is_real(haa_v)   else "—"

                    # VAA/HAA gradient colors vs pitch-type league average
                    _vl = _vaa_haa_league.get(group, {})
                    if _vl and is_real(vaa_v):
                        # More negative VAA = steeper descent = better for most pitches → red
                        vaa_gc = stat_gradient_color(vaa_v, _vl["vaa_mu"], _vl["vaa_sd"], invert=True)
                    else:
                        vaa_gc = "#2a4a5a"
                    if _vl and is_real(haa_v):
                        # HAA: extreme values (either direction) are notable; color by abs z-score
                        _hz = abs((haa_v - _vl["haa_mu"]) / max(_vl["haa_sd"], 0.001))
                        _hz = min(_hz, 2.0)
                        _ht = _hz / 2.0   # 0=grey, 1=red (extreme in either direction)
                        _hr = int(120 + (220 - 120) * _ht)
                        _hg = int(130 + (35  - 130) * _ht)
                        _hb = int(140 + (35  - 140) * _ht)
                        haa_gc = f"rgb({_hr},{_hg},{_hb})"
                    else:
                        haa_gc = "#2a4a5a"

                    u_velo = f"{user_m['velo']:.1f} you" if user_m.get("velo") is not None else ""
                    u_ivb  = f"{user_m['ivb']:.1f}\" you" if user_m.get("ivb") is not None else ""
                    u_hb   = f"{-user_m['hb']:.1f}\" you" if user_m.get("hb") is not None else ""

                    # ── Look up per-pitch FG Stuff+ (falls back to overall) ──────
                    sp_col    = FG_SP_COL.get(group)
                    sp_val    = row.get(sp_col) if sp_col else None
                    # Per-pitch col missing (CSV not yet rebuilt) → fall back to overall
                    if sp_val is None or (isinstance(sp_val, float) and math.isnan(sp_val)):
                        sp_val = row.get("stuff_plus")
                    sp_has    = (sp_val is not None
                                 and not (isinstance(sp_val, float) and math.isnan(sp_val)))
                    sp_str    = f"{sp_val:.0f}" if sp_has else "—"
                    sp_color  = stuff_color(sp_val) if sp_has else "#3a6a8a"
                    sp_lbl    = stuff_grade_label(sp_val) if sp_has else "—"
                    # Usage stats for this pitch (from n_ / pct_ columns)
                    n_col     = f"n_{group}"
                    pct_col   = f"pct_{group}"
                    n_val     = row.get(n_col)
                    pct_val   = row.get(pct_col)
                    n_str     = f"{int(n_val):,}" if is_real(n_val) else "—"
                    pct_str   = f"{pct_val:.0%}" if is_real(pct_val) else "—"

                    # ── Look up zone stats before building card so we can embed totals ──
                    pz_data  = pitcher_zone_data(r["Pitcher"], r["Year"], group)
                    p_csw    = None
                    p_xwoba  = None

                    # Compute overall (all-zone) CSW%, xwOBA for this pitch type
                    if not pz_data.empty and "n_pitches" in pz_data.columns:
                        total_n = pz_data["n_pitches"].sum()
                        if total_n > 0:
                            p_csw   = (pz_data["csw_pct"] * pz_data["n_pitches"]).sum() / total_n
                        xw_vals = pz_data["xwoba_mean"].dropna()
                        if not xw_vals.empty:
                            p_xwoba = float(xw_vals.mean())

                    # Also try zone_stats directly if pz_data is empty (catches missing joins)
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
                            xw_vals = sub["xwoba_mean"].dropna()
                            if not xw_vals.empty:
                                p_xwoba = float(xw_vals.mean())

                    csw_str   = f"{p_csw:.1%}"  if p_csw   is not None else "—"
                    xwoba_str = f"{p_xwoba:.3f}" if p_xwoba is not None else "—"

                    # Gradient colors for CSW% and xwOBA vs league avg for this pitch type
                    if not pitch_grp_league.empty and group in pitch_grp_league.index:
                        pg     = pitch_grp_league.loc[group]
                        csw_gc = stat_gradient_color(p_csw,   pg["csw_mu"], pg["csw_sd"], invert=False)
                        xw_gc  = stat_gradient_color(p_xwoba, pg["xw_mu"],  pg["xw_sd"],  invert=True)
                    else:
                        csw_gc = xw_gc = "#1a3550"

                    card_html = (
                        f"<div style='background:linear-gradient(165deg,#0e1828,#0c1420);"
                        f"border:1px solid #162236;border-left:3px solid {color};"
                        f"border-radius:10px;padding:14px 16px;margin-bottom:4px'>"
                        f"<div style='font-family:Inter,sans-serif;font-size:12px;"
                        f"font-weight:700;color:{color};letter-spacing:1.5px;"
                        f"text-transform:uppercase;margin-bottom:10px;"
                        f"display:flex;justify-content:space-between;align-items:baseline'>"
                        f"<span>● {group}</span>"
                        f"<span style='font-size:10px;color:#7aaac0;font-weight:400;"
                        f"letter-spacing:0.5px'>{pct_str}</span>"
                        f"</div>"
                        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px'>"
                        "<div style='text-align:center'>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>VELO</div>"
                        f"<div style='font-family:Inter,sans-serif;font-size:22px;"
                        f"font-weight:700;color:#e8dcc8;line-height:1'>{velo_s}</div>"
                        + sub_label(u_velo) + "</div>"
                        "<div style='text-align:center'>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>iVB</div>"
                        f"<div style='font-family:Inter,sans-serif;font-size:22px;"
                        f"font-weight:700;color:#e8dcc8;line-height:1'>{ivb_s}</div>"
                        + sub_label(u_ivb) + "</div>"
                        "<div style='text-align:center'>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>HB</div>"
                        f"<div style='font-family:Inter,sans-serif;font-size:22px;"
                        f"font-weight:700;color:#e8dcc8;line-height:1'>{hb_s}</div>"
                        + sub_label(u_hb) + "</div>"
                        "</div>"
                        # Outcome stats row: CSW% | xwOBA | FG S+
                        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                        "gap:4px;margin-bottom:8px;border-top:1px solid #141e2e;padding-top:6px'>"
                        f"<div style='text-align:center;background:{csw_gc}22;border-radius:6px;padding:5px 4px'>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:8px;color:#7aaac0;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:1px'>CSW%</div>"
                        f"<div style='font-family:Inter,sans-serif;font-size:15px;"
                        f"font-weight:700;color:{csw_gc}'>{csw_str}</div></div>"
                        f"<div style='text-align:center;background:{xw_gc}22;border-radius:6px;padding:5px 4px'>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:8px;color:#7aaac0;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:1px'>xwOBA</div>"
                        f"<div style='font-family:Inter,sans-serif;font-size:15px;"
                        f"font-weight:700;color:{xw_gc}'>{xwoba_str}</div></div>"
                        f"<div style='text-align:center;background:{sp_color}22;border-radius:6px;padding:5px 4px'>"
                        "<div style='font-family:JetBrains Mono,monospace;font-size:8px;color:#7aaac0;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:1px'>FG S+</div>"
                        f"<div style='font-family:Inter,sans-serif;font-size:15px;"
                        f"font-weight:700;color:{sp_color}'>{sp_str}</div>"
                        f"<div style='font-family:JetBrains Mono,monospace;font-size:8px;color:#6a90a8'>{sp_lbl}</div>"
                        "</div>"
                        "</div>"
                        # VAA/HAA footer with gradient color
                        "<div style='border-top:1px solid #141e2e;padding-top:6px;margin-top:4px;"
                        "display:flex;justify-content:space-between;align-items:center'>"
                        "<div style='display:flex;gap:8px'>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:9px;"
                        f"background:{vaa_gc}15;border:1px solid {vaa_gc}35;"
                        f"border-radius:6px;padding:3px 8px;"
                        f"color:{vaa_gc}'>VAA {vaa_s}</span>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:9px;"
                        f"background:{haa_gc}15;border:1px solid {haa_gc}35;"
                        f"border-radius:6px;padding:3px 8px;"
                        f"color:{haa_gc}'>HAA {haa_s}</span>"
                        "</div>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:9px;"
                        f"color:#5a7a90'>{n_str} pitches</span>"
                        "</div></div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    # ── Per-pitch heatmaps (click to expand) ─────────────
                    if not pz_data.empty:
                        with st.expander(
                            f"📊  {group} zone heatmaps",
                            expanded=False,
                        ):
                            # All batters
                            hm1, hm2, hm3 = st.columns(3)
                            with hm1:
                                svg = render_zone_heatmap(pz_data, "csw_pct", "csw",
                                                          "CSW% — All", fmt=".1%")
                                st.markdown(svg, unsafe_allow_html=True)
                            with hm2:
                                svg = render_zone_heatmap(pz_data, "whiff_pct", "whiff",
                                                          "Whiff% — All", fmt=".1%")
                                st.markdown(svg, unsafe_allow_html=True)
                            with hm3:
                                svg = render_zone_heatmap(pz_data, "xwoba_mean", "xwoba",
                                                          "xwOBA — All", fmt=".3f")
                                st.markdown(svg, unsafe_allow_html=True)
                            # Same/Opp hand splits (post-rebuild only)
                            has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                            if has_stand_col:
                                pz_same = pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], group, "same")
                                pz_opp  = pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], group, "opp")
                                if not pz_same.empty:
                                    st.markdown(
                                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;"
                                        "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                                        "vs Same Hand</div>", unsafe_allow_html=True)
                                    ps1, ps2, ps3 = st.columns(3)
                                    with ps1:
                                        st.markdown(render_zone_heatmap(pz_same, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                                    with ps2:
                                        st.markdown(render_zone_heatmap(pz_same, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                                    with ps3:
                                        st.markdown(render_zone_heatmap(pz_same, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)
                                if not pz_opp.empty:
                                    st.markdown(
                                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;"
                                        "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                                        "vs Opposite Hand</div>", unsafe_allow_html=True)
                                    po1, po2, po3 = st.columns(3)
                                    with po1:
                                        st.markdown(render_zone_heatmap(pz_opp, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                                    with po2:
                                        st.markdown(render_zone_heatmap(pz_opp, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                                    with po3:
                                        st.markdown(render_zone_heatmap(pz_opp, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)

                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Overall pitcher heatmaps (all pitches combined) ───────
                overall_data = overall_pitcher_zone_data(r["Pitcher"], r["Year"])
                has_stand_col = zone_stats_ok and not zone_stats.empty and "stand" in zone_stats.columns
                if not overall_data.empty or not zone_stats_ok:
                    with st.expander("📊  Overall zone profile — all pitches", expanded=False):
                        if not overall_data.empty:
                            oh1, oh2, oh3 = st.columns(3)
                            with oh1:
                                svg = render_zone_heatmap(overall_data, "csw_pct", "csw",
                                                          "CSW% — All", fmt=".1%")
                                st.markdown(svg, unsafe_allow_html=True)
                            with oh2:
                                svg = render_zone_heatmap(overall_data, "whiff_pct", "whiff",
                                                          "Whiff% — All", fmt=".1%")
                                st.markdown(svg, unsafe_allow_html=True)
                            with oh3:
                                svg = render_zone_heatmap(overall_data, "xwoba_mean", "xwoba",
                                                          "xwOBA — All", fmt=".3f")
                                st.markdown(svg, unsafe_allow_html=True)
                            # ── Same/Opp hand splits ──────────────────────
                            if has_stand_col:
                                ov_same = overall_pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], "same")
                                ov_opp  = overall_pitcher_zone_data_by_stand(r["Pitcher"], r["Year"], "opp")
                                if not ov_same.empty:
                                    st.markdown(
                                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;"
                                        "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                                        "vs Same Hand</div>", unsafe_allow_html=True)
                                    s1, s2, s3 = st.columns(3)
                                    with s1:
                                        st.markdown(render_zone_heatmap(ov_same, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                                    with s2:
                                        st.markdown(render_zone_heatmap(ov_same, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                                    with s3:
                                        st.markdown(render_zone_heatmap(ov_same, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)
                                if not ov_opp.empty:
                                    st.markdown(
                                        "<div style='font-family:JetBrains Mono,monospace;font-size:9px;color:#7aaac0;"
                                        "text-transform:uppercase;letter-spacing:1px;margin:10px 0 4px 0'>"
                                        "vs Opposite Hand</div>", unsafe_allow_html=True)
                                    o1, o2, o3 = st.columns(3)
                                    with o1:
                                        st.markdown(render_zone_heatmap(ov_opp, "csw_pct", "csw", "CSW%", fmt=".1%"), unsafe_allow_html=True)
                                    with o2:
                                        st.markdown(render_zone_heatmap(ov_opp, "whiff_pct", "whiff", "Whiff%", fmt=".1%"), unsafe_allow_html=True)
                                    with o3:
                                        st.markdown(render_zone_heatmap(ov_opp, "xwoba_mean", "xwoba", "xwOBA", fmt=".3f"), unsafe_allow_html=True)
                        else:
                            st.markdown(
                                "<div style='font-family:JetBrains Mono,monospace;font-size:10px;color:#6a90a8;"
                                "background:#0a0e18;border:1px solid #162236;border-radius:8px;"
                                "padding:8px 12px'>"
                                "⚠ Zone heatmaps require rebuilding pitch_zone_stats.csv</div>",
                                unsafe_allow_html=True,
                            )

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────────────────
    export = [{k: v for k, v in r.items() if k != "_row"} for r in results]
    csv    = pd.DataFrame(export).to_csv(index=False).encode("utf-8")
    _, dl_col, _ = st.columns([2, 3, 2])
    with dl_col:
        st.download_button("⬇  Download Results CSV", data=csv,
                           file_name="pitcher_similarity_results.csv",
                           mime="text/csv", use_container_width=True)
