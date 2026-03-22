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
W_BASE = dict(hand=1000, rel_height=38, rel_side=30, extension=10, ivb=25, hb=25, velo=12)
NORM   = dict(rel_height=1.5, rel_side=2.0, extension=1.5, ivb=20.0, hb=20.0, velo=15.0)

# Velo thresholds for dynamic weighting (#5)
VELO_BOOST_THRESHOLD = 95.0   # above this, velo weight scales up
VELO_BOOST_MAX       = 2.5    # max multiplier at 100+ mph

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

# ── Load profiles ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_profiles() -> pd.DataFrame:
    return pd.read_csv("pitcher_profiles.csv")

try:
    profiles = load_profiles()
    data_ok  = True
except FileNotFoundError:
    data_ok  = False
    profiles = None

# ── APP BAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-bar">
  <span style="font-size:34px;line-height:1">⚾</span>
  <div>
    <div class="app-bar-title">Pitcher Similarity Engine</div>
    <div class="app-bar-sub">STATCAST 2017–2024 · ARM-SIDE NORMALIZED · WEIGHTED SCORING</div>
  </div>
</div>
""", unsafe_allow_html=True)

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

def norm_diff(a, b, scale):
    return min(abs(a - b) / scale, 1.0)

def sim_color(s):
    if s >= 82: return "#06d6a0"
    if s >= 68: return "#c9a84c"
    if s >= 52: return "#f4a261"
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
    if s is None:   return "#3a6a8a"
    if s >= 120:    return "#06d6a0"
    if s >= 110:    return "#c9a84c"
    if s >= 100:    return "#8aadcc"
    if s >= 90:     return "#f4a261"
    return "#e06060"

def stuff_grade_label(s):
    """Descriptive label for Stuff+ display."""
    if s is None:   return "—"
    if s >= 130:    return "Elite"
    if s >= 115:    return "Plus"
    if s >= 105:    return "Avg+"
    if s >= 95:     return "Avg"
    if s >= 85:     return "Below"
    return "Poor"


def velo_weight(user_velo):
    """Scale velocity weight upward for harder throwers."""
    if user_velo is None or user_velo < VELO_BOOST_THRESHOLD:
        return W_BASE["velo"]
    boost = 1.0 + (min(user_velo, 102) - VELO_BOOST_THRESHOLD) / (102 - VELO_BOOST_THRESHOLD) * (VELO_BOOST_MAX - 1.0)
    return W_BASE["velo"] * boost


# ── Similarity scoring ────────────────────────────────────────────────────────
def score_row(user, pitch_inputs, row):
    tw = tp = 0.0

    if user.get("hand"):
        tw += W_BASE["hand"]
        tp += W_BASE["hand"] * (0.0 if row["hand"] == user["hand"] else 1.0)

    for key in ("rel_height", "rel_side", "extension"):
        if user.get(key) is None:
            continue
        mv = row.get(key)
        if not is_real(mv):
            pen = 0.5
        else:
            ref = abs(user[key]) if key == "rel_side" else user[key]
            cmp = abs(mv)        if key == "rel_side" else mv
            pen = norm_diff(cmp, ref, NORM[key])
        tw += W_BASE[key]; tp += W_BASE[key] * pen

    for group, metrics in pitch_inputs.items():
        # Dynamic velo weight for this pitch group
        dyn_velo_w = velo_weight(metrics.get("velo"))

        for metric, weight in [("ivb", W_BASE["ivb"]), ("hb", W_BASE["hb"]), ("velo", dyn_velo_w)]:
            val = metrics.get(metric)
            if val is None:
                continue
            mv  = row.get(f"{metric}_{group}")
            pen = norm_diff(mv, val, NORM[metric]) if is_real(mv) else 0.6
            tw += weight; tp += weight * pen

    return round(max(1.0 - tp / tw, 0) * 100, 1) if tw > 0 else 0.0


def run_search(user, pitch_inputs, top_n):
    rows = []
    for _, r in profiles.iterrows():
        s = score_row(user, pitch_inputs, r)
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
                    if any(x is not None for x in [v, i, h]):
                        pitch_inputs_raw[group] = {"velo": v, "ivb": i, "hb": h}

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
            if m.get("hb"):   subs.append(f"HB {m['hb']:.1f}\"")
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
    left_col, right_col = st.columns([5, 7])

    with left_col:
        st.markdown(
            f'<div class="sec-label">Top {snap["top_n"]} Matches</div>',
            unsafe_allow_html=True,
        )

        # Results table — use st.dataframe for reliable rendering
        display_rows = []
        for rank, r in enumerate(results, 1):
            ext_str = f"{r['Extension']:.2f}" if r["Extension"] else "—"
            display_rows.append({
                "#":          rank,
                "Pitcher":    last_name(r["Pitcher"]),
                "Year":       r["Year"],
                "Hand":       r["Hand"],
                "Similarity": r["Similarity"],
                "Rel Ht":     round(r["Rel Height"], 2),
                "Rel Side":   round(r["Rel Side"], 2),
                "Ext":        ext_str,
            })
        df_results = pd.DataFrame(display_rows)

        def color_sim(val):
            try:
                return f"color: {sim_color(float(val))}; font-weight: bold; font-family: monospace"
            except:
                return ""

        styled = (
            df_results.style
            .applymap(color_sim, subset=["Similarity"])
            .format({"Similarity": "{:.1f}", "Rel Ht": "{:.2f}", "Rel Side": "{:.2f}"})
            .set_properties(**{
                "background-color": "#0a1828",
                "color":            "#c8d8e8",
                "border":           "1px solid #0f2030",
                "font-family":      "monospace",
                "font-size":        "12px",
            })
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("background-color", "#06101c"),
                    ("color",            "#c9a84c"),
                    ("font-family",      "monospace"),
                    ("font-size",        "10px"),
                    ("text-transform",   "uppercase"),
                    ("letter-spacing",   "1px"),
                    ("border",           "1px solid #0f2030"),
                ],
            }, {
                "selector": "tr:hover td",
                "props": [("background-color", "#0f2236")],
            }])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)


    with right_col:
        st.markdown('<div class="sec-label">Pitch Breakdown — Top 5</div>', unsafe_allow_html=True)

        if pitch_inputs:
            for idx, r in enumerate(results[:5]):
                row  = r["_row"]
                sc   = r["Similarity"]
                sc_c = sim_color(sc)
                ln   = last_name(r["Pitcher"])

                with st.expander(
                    f"#{idx+1}  {ln}  {r['Year']}  ({r['Hand']}HP)  ·  {sc:.1f}",
                    expanded=(idx == 0),
                ):
                    # Release point summary
                    ext_str = f"{r['Extension']:.2f}" if r['Extension'] else '—'
                    rel_html = (
                        f"<div style='display:flex;gap:20px;font-family:\"IBM Plex Mono\",monospace;"
                        f"font-size:10px;color:#3a6a8a;margin-bottom:12px;padding-bottom:8px;"
                        f"border-bottom:1px solid #0f2030'>"
                        f"<span>HT <b style='color:#8aadcc'>{r['Rel Height']:.2f} ft</b></span>"
                        f"<span>SIDE <b style='color:#8aadcc'>{r['Rel Side']:.2f} ft</b></span>"
                        f"<span>EXT <b style='color:#8aadcc'>{ext_str} ft</b></span>"
                        f"</div>"
                    )
                    st.markdown(rel_html, unsafe_allow_html=True)

                    for group in PITCH_GROUPS:
                        color   = PITCH_COLORS[group]
                        user_m  = pitch_inputs.get(group, {})
                        mv_velo = row.get(f"velo_{group}")
                        mv_ivb  = row.get(f"ivb_{group}")
                        mv_hb   = row.get(f"hb_{group}")

                        mlb_has  = is_real(mv_velo)
                        user_has = any(v is not None for v in user_m.values())
                        if not mlb_has and not user_has:
                            continue

                        st.markdown(
                            f"<div style='font-family:Rajdhani,sans-serif;font-size:12px;"
                            f"font-weight:700;color:{color};letter-spacing:1.5px;"
                            f"text-transform:uppercase;margin:10px 0 4px 0'>● {group}</div>",
                            unsafe_allow_html=True,
                        )

                        # Build metric compare rows
                        metric_rows = [
                            ("VELO", mv_velo, user_m.get("velo"), " mph", 15.0, False),
                            ("iVB",  mv_ivb,  user_m.get("ivb"),  '"',    20.0, True),
                            ("HB",   mv_hb,   user_m.get("hb"),   '"',    20.0, True),
                        ]

                        gc1, gc2, gc3 = st.columns(3)
                        for col_w, (label, mlb_val, user_val, unit, scale, is_movement) in zip(
                            [gc1, gc2, gc3], metric_rows
                        ):
                            mlb_str  = f"{mlb_val:.1f}{unit}" if is_real(mlb_val) else "—"
                            user_str = f"{user_val:.1f}{unit}" if user_val is not None else "—"

                            # #6: color logic — red = you're better, blue = you're worse
                            delta_color = "off"
                            delta_str   = None
                            if is_real(mlb_val) and user_val is not None:
                                diff = round(mlb_val - user_val, 1)
                                delta_str = f"{diff:+.1f}{unit}"
                                # For movement: abs value matters (more movement = better)
                                if is_movement:
                                    you_better = abs(user_val) > abs(mlb_val)
                                else:
                                    you_better = user_val > mlb_val  # higher velo = better
                                delta_color = "inverse" if you_better else "normal"

                            col_w.metric(
                                label=f"{label}  {ln} vs You",
                                value=mlb_str,
                                delta=delta_str if delta_str else f"You: {user_str}",
                                delta_color=delta_color,
                            )

                        # VAA / HAA display (computed, not user-input — #8)
                        vaa_col = row.get(f"vaa_{group}")
                        haa_col = row.get(f"haa_{group}")
                        if is_real(vaa_col) or is_real(haa_col):
                            vaa_str = f"{vaa_col:.1f}°" if is_real(vaa_col) else "—"
                            haa_str = f"{haa_col:.1f}°" if is_real(haa_col) else "—"
                            st.markdown(
                                f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
                                f"color:#2a5a7a;margin-top:4px'>"
                                f"VAA {vaa_str} &nbsp;·&nbsp; HAA {haa_str}</div>",
                                unsafe_allow_html=True,
                            )

                        # FanGraphs Stuff+ — pitcher-season level from profiles CSV
                        fg_stuff = row.get("stuff_plus")
                        if fg_stuff is not None and not (isinstance(fg_stuff, float) and math.isnan(fg_stuff)):
                            sc_v  = stuff_color(fg_stuff)
                            label = stuff_grade_label(fg_stuff)
                            st.markdown(
                                f"<div style='display:flex;align-items:center;gap:12px;"
                                f"font-family:monospace;font-size:11px;"
                                f"background:#080f1a;border:1px solid #0f2030;border-radius:4px;"
                                f"padding:6px 12px;margin-top:6px'>"
                                f"<span style='color:#3a6a8a;font-size:10px;text-transform:uppercase;"
                                f"letter-spacing:1px;min-width:70px'>FG Stuff+</span>"
                                f"<span style='color:{sc_v};font-weight:700;font-size:16px'>"
                                f"{fg_stuff:.0f}</span>"
                                f"<span style='color:#2a5a7a;font-size:10px'>({label})</span>"
                                f"<span style='color:#2a5a7a;font-size:9px;margin-left:auto'>"
                                f"overall pitcher · FanGraphs</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
        else:
            st.markdown(
                "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:11px;color:#2a5a7a;"
                "padding:20px;text-align:center'>No pitch metrics entered — only release profile matched.</div>",
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
