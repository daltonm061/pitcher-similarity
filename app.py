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
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
.stApp { background: #06101c; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── APP BAR ── */
.app-bar {
    background: linear-gradient(90deg, #050d18 0%, #0c1e30 60%, #050d18 100%);
    border-bottom: 2px solid #c9a84c;
    padding: 16px 40px; display: flex; align-items: center; gap: 14px;
}
.app-bar-title {
    font-family: 'Rajdhani', sans-serif; font-size: 24px; font-weight: 700;
    color: #c9a84c; letter-spacing: 3px; text-transform: uppercase; margin: 0; line-height: 1;
}
.app-bar-sub { font-size: 10px; color: #2a5a7a; letter-spacing: 1.5px; margin-top: 3px; font-family: 'IBM Plex Mono', monospace; }
.status-bar {
    background: #080f1a; border-bottom: 1px solid #0f2030;
    padding: 5px 40px; font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    color: #1e4a6a; display: flex; gap: 20px; flex-wrap: wrap;
}

/* ── SECTION LABELS ── */
.sec-label {
    font-family: 'Rajdhani', sans-serif; font-size: 11px; font-weight: 700;
    color: #c9a84c; letter-spacing: 2.5px; text-transform: uppercase;
    border-bottom: 1px solid #0f2030; padding-bottom: 5px; margin-bottom: 12px;
}

/* ── PITCH CARDS ── */
.pitch-card {
    background: #0a1828; border: 1px solid #0f2030;
    border-radius: 6px; padding: 12px 14px; margin-bottom: 8px;
}
.pitch-card-title {
    font-family: 'Rajdhani', sans-serif; font-size: 13px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 10px;
}
.field-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #3a6a8a;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 2px;
}

/* ── NUMBER INPUTS — fix Enter key overlap ── */
.stNumberInput { margin-bottom: 4px !important; }
.stNumberInput > label { display: none !important; }
.stNumberInput > div > div > input {
    background: #06101c !important; color: #e8dcc8 !important;
    border: 1px solid #0f2030 !important; border-radius: 4px !important;
    font-size: 14px !important; font-family: 'IBM Plex Mono', monospace !important;
    padding: 7px 10px !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #c9a84c !important; box-shadow: 0 0 0 1px #c9a84c30 !important;
}
/* Hide "Press Enter to apply" — Streamlit injects this as data-testid="InputInstructions" */
.stNumberInput > div > div > input::placeholder { color: transparent !important; opacity: 0 !important; }
.stNumberInput > div > div > input::-webkit-input-placeholder { color: transparent !important; }
.stNumberInput > div > div > input::-moz-placeholder { color: transparent !important; }
[data-testid="InputInstructions"] { display: none !important; }
/* Hide "streamlitApp" tooltip that appears from collapsed labels */
.stNumberInput > label { display: none !important; pointer-events: none !important; }
[data-baseweb="tooltip"] { display: none !important; }
[role="tooltip"] { display: none !important; }
/* stepper buttons */
.stNumberInput button { background: #0a1828 !important; border-color: #0f2030 !important; }

/* ── RADIO ── */
.stRadio > label { display: none !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #c8d8e8 !important; font-size: 14px !important; }

/* ── SLIDER ── */
.stSlider > label { color: #3a6a8a !important; font-size: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 1px; }

/* ── RUN BUTTON — scoped, won't bleed into back btn ── */
.run-btn-wrap > div > button {
    background: linear-gradient(135deg, #c9a84c, #e8c96a) !important;
    color: #06101c !important; font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; font-size: 17px !important;
    letter-spacing: 3px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 6px !important;
    padding: 13px 40px !important; width: 100% !important;
    white-space: nowrap !important;
    transition: all 0.2s !important;
}
.run-btn-wrap > div > button:hover {
    transform: translateY(-1px); box-shadow: 0 4px 20px #c9a84c40 !important;
}

/* ── BACK BUTTON ── */
.back-btn-wrap > div > button {
    background: transparent !important; color: #c9a84c !important;
    border: 1px solid #c9a84c40 !important; font-size: 12px !important;
    padding: 5px 16px !important;
    font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1.5px !important;
    width: auto !important; white-space: nowrap !important;
}

/* ── METRICS ── */
[data-testid="metric-container"] {
    background: #0a1828; border: 1px solid #0f2030; border-radius: 6px; padding: 10px 14px;
}
[data-testid="metric-container"] label {
    color: #2a5a7a !important; font-size: 9px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c9a84c !important; font-family: 'Rajdhani', sans-serif !important; font-size: 22px !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; font-family: 'IBM Plex Mono', monospace !important; }

hr { border-color: #0f2030 !important; margin: 20px 0 !important; }

/* ── EXPANDERS ── */
.streamlit-expanderHeader {
    background: #0a1828 !important; color: #6a9ab8 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 1px; font-size: 11px !important;
    border: 1px solid #0f2030 !important; border-radius: 6px !important;
}
.streamlit-expanderContent { background: #06101c !important; border: 1px solid #0f2030 !important; }

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid #0f2030 !important; border-radius: 6px !important; }

/* ── TRACKMAN CARD ── */
.tm-card {
    background: #0a1828; border: 1px solid #0f2030;
    border-top: 2px solid #c9a84c40; border-radius: 6px; padding: 16px 18px;
    margin-bottom: 12px;
}

/* ── SIMILARITY BARS ── */
.sim-bar-bg { background: #0f2030; border-radius: 3px; height: 6px; width: 100%; margin-top: 4px; }
.sim-bar-fill { border-radius: 3px; height: 6px; transition: width 0.3s; }

/* ── METRIC COMPARE ROW ── */
.metric-row {
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid #0f2030; padding: 7px 0;
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
}
.metric-label { color: #3a6a8a; width: 80px; flex-shrink: 0; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-mlb   { width: 70px; text-align: right; font-weight: 500; }
.metric-you   { width: 60px; text-align: right; color: #5a8aaa; }
.metric-bar-wrap { flex: 1; position: relative; height: 16px; }
.metric-bar-center { position: absolute; left: 50%; top: 50%; width: 1px; height: 12px;
    background: #1a3550; transform: translateY(-50%); }
.metric-bar-fill { position: absolute; top: 50%; height: 6px; border-radius: 2px;
    transform: translateY(-50%); }

/* hide streamlit upload text overflow */
.stFileUploader > label { color: #3a6a8a !important; font-size: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important; text-transform: uppercase; }
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
    velo       = 1.5,    # ±1.5 mph velocity
    ivb        = 2.5,    # ±2.5" induced vertical break
    hb         = 2.5,    # ±2.5" horizontal break
    extension  = 0.50,   # ±0.50 ft extension (least important)
)

# Weights control how much each dimension pulls in the geometric mean exponent.
# Higher weight = that dimension dominates more when it's an outlier.
WEIGHTS = dict(
    rel_height = 3.0,   # slot height — critical for arm-slot matching
    rel_side   = 2.5,   # slot side — important
    velo       = 3.5,   # velocity — most important single metric
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
    "fastball": "4-Seam", "four-seam": "4-Seam", "four seam": "4-Seam", "4-seam fastball": "4-Seam",
    "sinker": "2-Seam/Sinker", "two-seam": "2-Seam/Sinker", "two seam": "2-Seam/Sinker",
    "cutter": "Cutter", "cut fastball": "Cutter",
    "slider": "Slider",
    "sweeper": "Sweeper",
    "curveball": "Curveball", "curve": "Curveball", "knuckle curve": "Curveball", "knucklecurve": "Curveball",
    "splitter": "Splitter", "split-finger": "Splitter", "splitfinger": "Splitter",
    "changeup": "Changeup", "change-up": "Changeup", "change up": "Changeup",
    "knuckleball": "Knuckleball",
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("screen","input"), ("results",None), ("user_snapshot",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── APP BAR — render immediately so health check passes ──────────────────────
st.markdown("""
<div class="app-bar">
  <span style="font-size:34px;line-height:1">⚾</span>
  <div>
    <div class="app-bar-title">Pitcher Similarity Engine</div>
    <div class="app-bar-sub">STATCAST 2017–2024 · ARM-SIDE NORMALIZED · WEIGHTED SCORING</div>
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
    return pd.read_csv("pitch_zone_stats.csv")

try:
    zone_stats = load_zone_stats()
    zone_stats["zone"] = zone_stats["zone"].astype(int)
    # Pre-compute league-wide means and stds per stat for z-score coloring
    league_csw  = (zone_stats.groupby("zone")["csw_pct"].agg(["mean","std"])
                   .rename(columns={"mean":"csw_mu","std":"csw_sd"}))
    league_xwoba= (zone_stats.groupby("zone")["xwoba_mean"].agg(["mean","std"])
                   .rename(columns={"mean":"xw_mu","std":"xw_sd"}))
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
except FileNotFoundError:
    zone_stats        = pd.DataFrame()
    zone_league       = pd.DataFrame()
    pitch_grp_league  = pd.DataFrame()
    zone_stats_ok     = False


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
INSIDE_ZONES = [
    (1,0,0),(2,0,1),(3,0,2),
    (4,1,0),(5,1,1),(6,1,2),
    (7,2,0),(8,2,1),(9,2,2),
]

def _lerp_color(z_score, stat_type):
    z = max(-2.5, min(2.5, z_score if z_score == z_score else 0))
    if stat_type == "csw":
        t = (z + 2.5) / 5.0        # higher CSW = better = red
    else:
        t = (-z + 2.5) / 5.0       # higher xwOBA/HH = worse = red (inverted)
    if t < 0.5:
        r = int(40  + (215-40)  * (t*2))
        g = int(90  + (215-90)  * (t*2))
        b = int(200 + (215-200) * (t*2))
    else:
        r = int(215 + (30-215)  * ((t-0.5)*2))   # was wrong, redo
        g = int(215 - (215-60)  * ((t-0.5)*2))
        b = int(215 - (215-50)  * ((t-0.5)*2))
    # Clamp
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return f"rgb({r},{g},{b})"


def stat_gradient_color(val, mu, sd, invert=False):
    """Blue→white→red gradient based on z-score vs league mean/sd."""
    if val is None or (isinstance(val, float) and val != val) or sd == 0:
        return "#1a3550"
    z = max(-2.0, min(2.0, (val - mu) / max(sd, 0.001)))
    if invert:
        z = -z
    t = (z + 2.0) / 4.0
    if t < 0.5:
        s = t * 2
        r = int(40  + (200-40)  * s)
        g = int(80  + (210-80)  * s)
        b = int(160 + (220-160) * s)
    else:
        s = (t - 0.5) * 2
        r = int(200 + (192-200) * s)
        g = int(210 - (210-57)  * s)
        b = int(220 - (220-43)  * s)
    return f"rgb({max(0,min(255,r))},{max(0,min(255,g))},{max(0,min(255,b))})"


def render_zone_heatmap(pitcher_zone_df, stat_col, stat_type, title, fmt=".1%"):
    """
    Render a 3×3 inside-only strike zone heatmap as SVG.
    Each cell shows the stat value and is colored by z-score vs league mean.
    """
    CW, CH = 52, 44       # cell width, height
    PAD_TOP = 16          # space for title
    TOTAL_W = CW * 3 + 2
    TOTAL_H = CH * 3 + PAD_TOP + 2

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
        f"style='width:100%;max-width:200px;display:block;margin:0 auto'>"
        f"<rect width='{TOTAL_W}' height='{TOTAL_H}' fill='#080f1a' rx='4'/>"
        f"<text x='{TOTAL_W//2}' y='11' text-anchor='middle' "
        f"font-family='monospace' font-size='8' fill='#3a6a8a' "
        f"letter-spacing='0.5'>{title}</text>"
    )

    for (zone_id, gr, gc) in INSIDE_ZONES:
        x = gc * CW + 1
        y = gr * CH + PAD_TOP

        row_data = pdata.get(zone_id)
        val = None
        z   = 0.0
        if row_data is not None:
            raw = row_data.get(stat_col)
            if raw is not None and raw == raw:   # not NaN
                val = float(raw)
                mu_col = {"csw_pct": "csw_mu", "xwoba_mean": "xw_mu", "hard_hit_pct": "hh_mu"}.get(stat_col)
                sd_col = {"csw_pct": "csw_sd", "xwoba_mean": "xw_sd", "hard_hit_pct": "hh_sd"}.get(stat_col)
                if (mu_col and sd_col and not zone_league.empty
                        and zone_id in zone_league.index):
                    mu = zone_league.loc[zone_id, mu_col]
                    sd = zone_league.loc[zone_id, sd_col]
                    z  = (val - mu) / max(sd, 0.001) if (sd == sd and sd > 0) else 0.0

        fill     = _lerp_color(z, stat_type) if val is not None else "#0d1e30"
        txt_fill = "#e8dcc8" if val is not None else "#2a5a7a"

        if val is not None:
            display = f"{val:.0%}" if fmt == ".1%" else f"{val:.3f}"
        else:
            display = "—"

        svg += (
            f"<rect x='{x}' y='{y}' width='{CW}' height='{CH}' "
            f"fill='{fill}' stroke='#0f2030' stroke-width='1'/>"
            f"<text x='{x + CW//2}' y='{y + CH//2 + 5}' "
            f"text-anchor='middle' font-family='Rajdhani,sans-serif' "
            f"font-size='12' font-weight='700' fill='{txt_fill}'>{display}</text>"
        )

    # Strike zone border
    svg += (
        f"<rect x='1' y='{PAD_TOP}' width='{CW*3}' height='{CH*3}' "
        f"fill='none' stroke='#c9a84c' stroke-width='1.5'/>"
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
    Gradient color for FG Stuff+ display.
    Scale: 100 = MLB avg, 15 pts = 1 standard deviation.
    ±1.5 SD (77.5 / 122.5) clamps to deep blue / deep red.
    Blue  = below average (easier to hit)
    Red   = above average (nastier pitch)
    """
    if s is None or (isinstance(s, float) and s != s):
        return "#1e3a50"
    # z-score: clamp to [-1.5, 1.5] SD
    z = max(-1.5, min(1.5, (s - 100.0) / 15.0))
    # Map z → t in [0, 1]: 0 = deep blue, 0.5 = white, 1 = deep red
    t = (z + 1.5) / 3.0
    if t < 0.5:
        # Deep blue (#1a3a8a) → white (#c8d8e8)
        s2 = t * 2
        r = int(26  + (200 - 26)  * s2)
        g = int(58  + (216 - 58)  * s2)
        b = int(138 + (232 - 138) * s2)
    else:
        # White (#c8d8e8) → deep red (#c0192b)
        s2 = (t - 0.5) * 2
        r = int(200 + (192 - 200) * s2)
        g = int(216 - (216 - 25)  * s2)
        b = int(232 - (232 - 43)  * s2)
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
        has_match = any(
            is_real(row.get(f"velo_{group}"))
            for group in pitch_inputs
        )
        if not has_match:
            return 0.0

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

        for metric, sigma in [("ivb",  SIGMA["ivb"]),
                               ("hb",   SIGMA["hb"]),
                               ("velo", sv)]:
            val = metrics.get(metric)
            if val is None:
                continue
            if not has_pitch:
                # Pitcher doesn't throw this pitch type at all — heavy penalty.
                # 0.02 in geometric mean space crushes the overall score when
                # weighted at 3.0+, forcing pitch-type coverage to matter.
                sim = 0.05
            else:
                mv  = row.get(f"{metric}_{group}")
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

    # ── PDF (basic text extraction) ───────────────────────────────────────────
    elif filename.lower().endswith(".pdf"):
        try:
            import re
            # Try pdfplumber first, fall back to pypdf
            try:
                import pdfplumber
                text_lines = []
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_lines.extend(t.split("\n"))
            except ImportError:
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(io.BytesIO(file_bytes))
                    text_lines = []
                    for page in reader.pages:
                        t = page.extract_text()
                        if t:
                            text_lines.extend(t.split("\n"))
                except ImportError:
                    return {"_error": "PDF parsing requires pdfplumber or pypdf. Add to requirements.txt."}

            # Very basic: look for lines with pitch type + numbers
            # TrackMan summary PDFs usually have one row per pitch type
            num_re = re.compile(r"-?\d+\.?\d*")
            for line in text_lines:
                ll = line.lower()
                group = None
                for key, grp in TM_PITCH_MAP.items():
                    if key in ll:
                        group = grp
                        break
                if group:
                    nums = num_re.findall(line)
                    if len(nums) >= 3:
                        # Assume order: Velo, iVB, HB (common TrackMan PDF layout)
                        try:
                            results[group] = {
                                "velo": float(nums[0]),
                                "ivb":  float(nums[1]),
                                "hb":   float(nums[2]),
                            }
                        except:
                            pass
        except Exception as e:
            return {"_error": f"PDF parse error: {e}"}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: INPUT
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "input":

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;max-width:680px;margin:0 auto 28px auto;padding:0 20px">
      <div style="font-family:'Rajdhani',sans-serif;font-size:20px;color:#c9a84c;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">
        Enter Your Metrics
      </div>
      <div style="font-size:12px;color:#2a5a7a;line-height:1.8;font-family:'IBM Plex Mono',monospace">
        Leave any field blank = open filter &nbsp;·&nbsp;
        Priority: Hand → Rel Ht → Rel Side → iVB &amp; HB → Velo
      </div>
    </div>
    """, unsafe_allow_html=True)

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
            rel_height_v = st.number_input("_rh", min_value=3.0, max_value=8.0,
                                            value=None, step=0.01, format="%.2f",
                                            placeholder="e.g. 5.00", key="rh",
                                            label_visibility="collapsed")
        with rp3:
            st.markdown("<div class='field-label'>Rel Side (ft)</div>", unsafe_allow_html=True)
            rel_side_v = st.number_input("_rs", min_value=-5.0, max_value=5.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder="e.g. 2.80", key="rs",
                                          label_visibility="collapsed")
        with rp4:
            st.markdown("<div class='field-label'>Extension (ft)</div>", unsafe_allow_html=True)
            extension_v = st.number_input("_ext", min_value=4.0, max_value=8.0,
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
            tm_data = parse_trackman(tm_file.read(), tm_file.name)
            if "_error" in tm_data:
                st.warning(f"TrackMan parse issue: {tm_data['_error']}")
                tm_data = {}
            elif tm_data:
                found = ", ".join(f"**{g}**" for g in tm_data)
                st.success(f"Parsed: {found} — metrics pre-filled below. Edit any value as needed.")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── PITCH ARSENAL ──────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Pitch Arsenal — fill only the pitches you throw</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:10px;color:#2a5a7a;"
            "background:#0a1828;border:1px solid #0f2030;border-left:2px solid #c9a84c30;"
            "border-radius:4px;padding:7px 12px;margin-bottom:12px'>"
            "HB: positive = arm-side, negative = glove-side &nbsp;·&nbsp; "
            "iVB: positive = rise (backspin), negative = drop &nbsp;·&nbsp; "
            "Enter raw Trackman/Rapsodo values as-is"
            "</div>",
            unsafe_allow_html=True,
        )

        pitch_inputs_raw = {}
        all_groups = list(PITCH_GROUPS.keys())

        for row_groups in [all_groups[:4], all_groups[4:7], all_groups[7:]]:
            if not row_groups:
                continue
            cols = st.columns(len(row_groups))
            for col, group in zip(cols, row_groups):
                color   = PITCH_COLORS[group]
                tm_vals = tm_data.get(group, {})
                with col:
                    st.markdown(
                        f"<div class='pitch-card'>"
                        f"<div class='pitch-card-title' style='color:{color}'>● {group}</div>",
                        unsafe_allow_html=True,
                    )
                    # Velo
                    st.markdown("<div class='field-label'>Velocity (mph)</div>", unsafe_allow_html=True)
                    velo_def = float(tm_vals["velo"]) if tm_vals.get("velo") is not None else None
                    velo = st.number_input(f"_velo_{group}", min_value=60.0, max_value=105.0,
                                           value=velo_def, step=0.1, format="%.1f",
                                           placeholder="e.g. 93.5", key=f"velo_{group}",
                                           label_visibility="collapsed")
                    # iVB
                    st.markdown("<div class='field-label'>iVB (in)</div>", unsafe_allow_html=True)
                    ivb_def = float(tm_vals["ivb"]) if tm_vals.get("ivb") is not None else None
                    ivb = st.number_input(f"_ivb_{group}", min_value=-30.0, max_value=30.0,
                                          value=ivb_def, step=0.1, format="%.1f",
                                          placeholder="e.g. 18.0", key=f"ivb_{group}",
                                          label_visibility="collapsed")
                    # HB
                    st.markdown("<div class='field-label'>HB (in)</div>", unsafe_allow_html=True)
                    hb_def = float(tm_vals["hb"]) if tm_vals.get("hb") is not None else None
                    hb = st.number_input(f"_hb_{group}", min_value=-30.0, max_value=30.0,
                                         value=hb_def, step=0.1, format="%.1f",
                                         placeholder="e.g. +14", key=f"hb_{group}",
                                         label_visibility="collapsed")
                    st.markdown("</div>", unsafe_allow_html=True)

                    v, i, h = vn(velo), vn(ivb), vn(hb)
                    # Negate HB: user enters positive=arm-side convention,
                    # but CSV stores negative=arm-side (raw Statcast pfx_x norm).
                    # Flip here once so score_row compares apples-to-apples.
                    h_csv = (-h) if h is not None else None
                    if any(x is not None for x in [v, i, h]):
                        pitch_inputs_raw[group] = {"velo": v, "ivb": i, "hb": h_csv}

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        # ── RUN BUTTON ─────────────────────────────────────────────────────
        _, btn_col, _ = st.columns([3, 4, 3])
        with btn_col:
            st.markdown('<div class="run-btn-wrap">', unsafe_allow_html=True)
            run = st.button("⚾  Find My MLB Comps", key="run_btn")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        if run:
            user = {
                "hand":       None if hand_choice == "Any" else hand_choice[0],
                "rel_height": vn(rel_height_v),
                "rel_side":   vn(rel_side_v),
                "extension":  vn(extension_v),
            }
            if not any(v is not None for v in user.values()) and not pitch_inputs_raw:
                st.error("Enter at least one metric to search.")
            else:
                st.session_state.user_snapshot = {
                    "user": user, "pitch_inputs": pitch_inputs_raw,
                    "top_n": top_n, "hand_label": hand_choice,
                }
                with st.spinner("Scoring…"):
                    results = run_search(user, pitch_inputs_raw, top_n)
                st.session_state.results = results
                st.session_state.screen  = "results"
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "results":

    results      = st.session_state.results
    snap         = st.session_state.user_snapshot
    user         = snap["user"]
    pitch_inputs = snap["pitch_inputs"]

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Header row: back btn + profile strip ──────────────────────────────
    hdr_l, hdr_r = st.columns([1, 8])
    with hdr_l:
        st.markdown('<div class="back-btn-wrap">', unsafe_allow_html=True)
        if st.button("← New Search"):
            st.session_state.screen  = "input"
            st.session_state.results = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with hdr_r:
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
        st.markdown(
            "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:10px;color:#2a5a7a;"
            "background:#0a1828;padding:8px 16px;border-radius:4px;border:1px solid #0f2030'>"
            "<span style='color:#c9a84c;font-family:Rajdhani,sans-serif;font-weight:700;"
            "letter-spacing:1.5px;text-transform:uppercase;font-size:10px'>PROFILE: </span>"
            + " &nbsp;·&nbsp; ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Summary metrics ───────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RESULTS",       len(results))
    m2.metric("SEASONS",       "2017–2024")
    m3.metric("TOP MATCH",     results[0]["Pitcher"].split(",")[0] if results else "—")
    m4.metric("BEST SCORE",    f"{results[0]['Similarity']:.1f}" if results else "—")
    m5.metric("PITCHERS USED", f"{profiles['player_name'].nunique():,}")

    st.markdown("---")

    # ── Main layout: table left, detail right ──────────────────────────────
    # ── Single-column layout: ranked list with inline expandable pitch profiles ──
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

        header = (
            f"#{rank}  {full_name}  {r['Year']}  ({hand}HP)"
            f"  ·  SIM {sc:.1f}"
            f"  ·  HT {r['Rel Height']:.2f}  SIDE {r['Rel Side']:.2f}  EXT {ext_str}"
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
                "font-family:monospace;font-size:11px;color:#3a6a8a;"
                "background:#080f1a;border:1px solid #0f2030;border-radius:6px;"
                "padding:10px 16px;margin-bottom:14px'>"
                f"<span>HT <b style='color:#8aadcc'>{r['Rel Height']:.2f} ft</b></span>"
                f"<span>SIDE <b style='color:#8aadcc'>{r['Rel Side']:.2f} ft</b></span>"
                f"<span>EXT <b style='color:#8aadcc'>{ext_str} ft</b></span>"
                + (f"<span>P <b style='color:#8aadcc'>{int(row.get('total_pitches',0)):,}</b></span>"
                   if is_real(row.get('total_pitches')) else '') +
                "<span style='color:#1a3550'>|</span>"
                f"<span>SIM <b style='color:{sc_c};font-size:13px'>{sc:.1f}</b></span>"
                "<span style='margin-left:auto;display:flex;align-items:center;gap:8px'>"
                "<span style='color:#3a6a8a;font-size:9px;text-transform:uppercase;"
                "letter-spacing:1px'>FG Stuff+</span>"
                f"<span style='background:{fg_overall_col}22;border-radius:4px;"
                f"padding:2px 8px'>"
                f"<b style='color:{fg_overall_col};font-size:18px'>{fg_overall_str}</b>"
                "</span>"
                f"<span style='color:#2a5a7a;font-size:10px'>({fg_overall_lbl})</span>"
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
            active = [g for g in PITCH_GROUPS if pitch_has_data(g)]

            if not active:
                st.markdown(
                    "<div style='color:#2a5a7a;font-family:monospace;font-size:11px'>"
                    "No pitch data for this season.</div>",
                    unsafe_allow_html=True,
                )
            else:
                def sub_label(val, color_hex="#5a8aaa"):
                    if val:
                        return (
                            f"<div style='font-family:monospace;font-size:9px;"
                            f"color:{color_hex};margin-top:2px'>{val}</div>"
                        )
                    return ""

                # ── Per-pitch cards with inline heatmap expander ──────────
                for group in active:
                    color     = PITCH_COLORS[group]
                    user_m    = pitch_inputs.get(group, {})
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

                    u_velo = f"{user_m['velo']:.1f} you" if user_m.get("velo") is not None else ""
                    u_ivb  = f"{user_m['ivb']:.1f}\" you" if user_m.get("ivb") is not None else ""
                    u_hb   = f"{-user_m['hb']:.1f}\" you" if user_m.get("hb") is not None else ""

                    # ── Look up per-pitch FG Stuff+ and usage count ──────────
                    sp_col    = FG_SP_COL.get(group)
                    sp_val    = row.get(sp_col) if sp_col else None
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
                    pz_data = pitcher_zone_data(r["Pitcher"], r["Year"], group)

                    # Compute overall (all-zone) CSW%, xwOBA, HardHit% for this pitch
                    if not pz_data.empty and "n_pitches" in pz_data.columns:
                        total_n  = pz_data["n_pitches"].sum()
                        p_csw    = (pz_data["csw_pct"]      * pz_data["n_pitches"]).sum() / max(total_n, 1)
                        p_xwoba  = pz_data["xwoba_mean"].mean()
                        csw_str  = f"{p_csw:.1%}"
                        xwoba_str= f"{p_xwoba:.3f}" if p_xwoba == p_xwoba else "—"
                    else:
                        csw_str = xwoba_str = "—"

                    # Gradient colors for CSW% and xwOBA vs league avg for this pitch type
                    if not pitch_grp_league.empty and group in pitch_grp_league.index:
                        pg     = pitch_grp_league.loc[group]
                        p_csw_v  = p_csw  if csw_str  != "—" else None
                        p_xw_v   = p_xwoba if xwoba_str != "—" else None
                        csw_gc = stat_gradient_color(p_csw_v, pg["csw_mu"], pg["csw_sd"], invert=False)
                        xw_gc  = stat_gradient_color(p_xw_v,  pg["xw_mu"],  pg["xw_sd"],  invert=True)
                    else:
                        csw_gc = xw_gc = "#1a3550"

                    card_html = (
                        f"<div style='background:#0a1828;border:1px solid #0f2030;"
                        f"border-top:2px solid {color};border-radius:6px;padding:12px 14px;margin-bottom:2px'>"
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:12px;"
                        f"font-weight:700;color:{color};letter-spacing:1.5px;"
                        f"text-transform:uppercase;margin-bottom:10px'>● {group}</div>"
                        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-bottom:8px'>"
                        "<div style='text-align:center'>"
                        "<div style='font-family:monospace;font-size:9px;color:#3a6a8a;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>VELO</div>"
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:22px;"
                        f"font-weight:700;color:#e8dcc8;line-height:1'>{velo_s}</div>"
                        + sub_label(u_velo) + "</div>"
                        "<div style='text-align:center'>"
                        "<div style='font-family:monospace;font-size:9px;color:#3a6a8a;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>iVB</div>"
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:22px;"
                        f"font-weight:700;color:#e8dcc8;line-height:1'>{ivb_s}</div>"
                        + sub_label(u_ivb) + "</div>"
                        "<div style='text-align:center'>"
                        "<div style='font-family:monospace;font-size:9px;color:#3a6a8a;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:2px'>HB</div>"
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:22px;"
                        f"font-weight:700;color:#e8dcc8;line-height:1'>{hb_s}</div>"
                        + sub_label(u_hb) + "</div>"
                        "</div>"
                        # Outcome stats row: CSW% | xwOBA | FG S+
                        "<div style='display:grid;grid-template-columns:1fr 1fr 1fr;"
                        "gap:4px;margin-bottom:8px;border-top:1px solid #0f2030;padding-top:6px'>"
                        f"<div style='text-align:center;background:{csw_gc}22;border-radius:4px;padding:4px 2px'>"
                        "<div style='font-family:monospace;font-size:8px;color:#3a6a8a;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:1px'>CSW%</div>"
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:15px;"
                        f"font-weight:700;color:{csw_gc}'>{csw_str}</div></div>"
                        f"<div style='text-align:center;background:{xw_gc}22;border-radius:4px;padding:4px 2px'>"
                        "<div style='font-family:monospace;font-size:8px;color:#3a6a8a;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:1px'>xwOBA</div>"
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:15px;"
                        f"font-weight:700;color:{xw_gc}'>{xwoba_str}</div></div>"
                        f"<div style='text-align:center;background:{sp_color}22;border-radius:4px;padding:4px 2px'>"
                        "<div style='font-family:monospace;font-size:8px;color:#3a6a8a;"
                        "text-transform:uppercase;letter-spacing:1px;margin-bottom:1px'>FG S+</div>"
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:15px;"
                        f"font-weight:700;color:{sp_color}'>{sp_str}</div>"
                        f"<div style='font-family:monospace;font-size:8px;color:#2a5a7a'>{sp_lbl}</div>"
                        "</div>"
                        "</div>"
                        # VAA/HAA + Stuff+ footer
                        "<div style='font-family:monospace;font-size:9px;color:#2a5a7a;"
                        "border-top:1px solid #0f2030;padding-top:6px;margin-top:4px;"
                        "display:flex;justify-content:space-between'>"
                        f"<span>VAA {vaa_s} · HAA {haa_s}</span>"
                        f"<span style='color:#1e3a50'>{n_str} pitches · {pct_str}</span>"
                        "</div></div>"
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    # ── Per-pitch heatmaps (click to expand) ─────────────
                    if not pz_data.empty:
                        with st.expander(
                            f"📊  {group} zone heatmaps",
                            expanded=False,
                        ):
                            hm1, hm2 = st.columns(2)
                            with hm1:
                                svg = render_zone_heatmap(pz_data, "csw_pct", "csw",
                                                          "CSW%", fmt=".1%")
                                st.markdown(svg, unsafe_allow_html=True)
                            with hm2:
                                svg = render_zone_heatmap(pz_data, "xwoba_mean", "xwoba",
                                                          "xwOBA", fmt=".3f")
                                st.markdown(svg, unsafe_allow_html=True)

                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                # ── Overall pitcher heatmaps (all pitches combined) ───────
                overall_data = overall_pitcher_zone_data(r["Pitcher"], r["Year"])
                if not overall_data.empty:
                    st.markdown(
                        "<div style='font-family:Rajdhani,sans-serif;font-size:11px;"
                        "font-weight:700;color:#c9a84c;letter-spacing:2px;"
                        "text-transform:uppercase;margin:16px 0 8px 0'>"
                        "● OVERALL ZONE PROFILE</div>",
                        unsafe_allow_html=True,
                    )
                    oh1, oh2 = st.columns(2)
                    with oh1:
                        svg = render_zone_heatmap(overall_data, "csw_pct", "csw",
                                                  "CSW% — All Pitches", fmt=".1%")
                        st.markdown(svg, unsafe_allow_html=True)
                    with oh2:
                        svg = render_zone_heatmap(overall_data, "xwoba_mean", "xwoba",
                                                  "xwOBA — All Pitches", fmt=".3f")
                        st.markdown(svg, unsafe_allow_html=True)
                elif not zone_stats_ok:
                    st.markdown(
                        "<div style='font-family:monospace;font-size:10px;color:#2a5a7a;"
                        "background:#080f1a;border:1px solid #0f2030;border-radius:4px;"
                        "padding:8px 12px;margin-top:12px'>"
                        "⚠ Zone heatmaps require rebuilding pitcher_profiles.csv — "
                        "run build_profiles.py locally and commit pitch_zone_stats.csv</div>",
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
