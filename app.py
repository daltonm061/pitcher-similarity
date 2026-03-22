import warnings
import math
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
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Source+Serif+4:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Serif 4', Georgia, serif; }
.stApp { background: #07111e; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.app-bar {
    background: linear-gradient(90deg, #060e1c 0%, #0f2236 60%, #060e1c 100%);
    border-bottom: 2px solid #c9a84c;
    padding: 18px 48px; display: flex; align-items: center; gap: 14px;
}
.app-bar-title {
    font-family: 'Rajdhani', sans-serif; font-size: 26px; font-weight: 700;
    color: #c9a84c; letter-spacing: 3px; text-transform: uppercase; margin: 0; line-height: 1;
}
.app-bar-sub { font-size: 11px; color: #3a6a8a; letter-spacing: 1.5px; margin-top: 3px; }

.status-bar {
    background: #0a1828; border-bottom: 1px solid #1a3550;
    padding: 5px 48px; font-family: monospace; font-size: 11px;
    color: #2a5a7a; display: flex; gap: 20px;
}

.sec-label {
    font-family: 'Rajdhani', sans-serif; font-size: 12px; font-weight: 700;
    color: #c9a84c; letter-spacing: 2.5px; text-transform: uppercase;
    border-bottom: 1px solid #1a3550; padding-bottom: 6px; margin-bottom: 14px;
}
.pitch-card {
    background: #0b1a2e; border: 1px solid #1a3550;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
}
.pitch-card-title {
    font-family: 'Rajdhani', sans-serif; font-size: 14px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 12px;
}
.hb-note {
    font-size: 11px; color: #3a6a8a; background: #0b1a2e;
    border: 1px solid #1a3550; border-left: 3px solid #c9a84c40;
    border-radius: 4px; padding: 8px 14px; margin-bottom: 14px; line-height: 1.6;
}
.stNumberInput > label {
    color: #6a9ab8 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px;
    white-space: nowrap !important; overflow: visible !important;
}
.stNumberInput input {
    background: #07111e !important; color: #e8dcc8 !important;
    border: 1px solid #1a3550 !important; border-radius: 4px !important;
    font-size: 15px !important;
}
.stNumberInput input:focus { border-color: #c9a84c !important; }
.stRadio > label { color: #6a9ab8 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #c8d8e8 !important; font-size: 15px !important; }
.stSlider > label { color: #6a9ab8 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 1px; }
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c9a84c, #e8c96a) !important;
    color: #07111e !important; font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important; font-size: 18px !important;
    letter-spacing: 3px !important; text-transform: uppercase !important;
    border: none !important; border-radius: 6px !important;
    padding: 14px 48px !important; width: 100%; transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px); box-shadow: 0 6px 24px #c9a84c50 !important;
}
[data-testid="metric-container"] {
    background: #0b1a2e; border: 1px solid #1a3550; border-radius: 8px; padding: 12px 16px;
}
[data-testid="metric-container"] label { color: #4a7a9a !important;
    font-size: 10px !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c9a84c !important; font-family: 'Rajdhani', sans-serif !important; font-size: 26px !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }
hr { border-color: #1a3550 !important; margin: 24px 0 !important; }
.streamlit-expanderHeader {
    background: #0b1a2e !important; color: #8aadcc !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1.5px; text-transform: uppercase; font-size: 13px !important;
    border: 1px solid #1a3550 !important; border-radius: 6px !important;
}
.streamlit-expanderContent { background: #07111e !important; border: 1px solid #1a3550 !important; }
.stDataFrame { border: 1px solid #1a3550 !important; border-radius: 8px !important; }
.back-btn > button {
    background: transparent !important; color: #c9a84c !important;
    border: 1px solid #c9a84c50 !important; font-size: 13px !important;
    padding: 6px 18px !important; width: auto !important;
    font-family: 'Rajdhani', sans-serif !important; letter-spacing: 1.5px !important;
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
PITCH_COLORS = {
    "4-Seam":        "#e63946",
    "2-Seam/Sinker": "#f4a261",
    "Cutter":        "#2a9d8f",
    "Slider":        "#457b9d",
    "Sweeper":       "#a855f7",
    "Curveball":     "#00b4d8",
    "Splitter":      "#e9c46a",
    "Changeup":      "#06d6a0",
    "Knuckleball":   "#ffffff",
}
W    = dict(hand=1000, rel_height=50, rel_side=30, extension=10, ivb=25, hb=25, velo=10)
NORM = dict(rel_height=1.5, rel_side=2.0, extension=1.5, ivb=20.0, hb=20.0, velo=15.0)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("screen","input"), ("results",None), ("user_snapshot",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Load pre-built profiles CSV (instant — just reading a file) ───────────────
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
  <span style="font-size:38px;line-height:1">⚾</span>
  <div>
    <div class="app-bar-title">MLB Pitcher Similarity Finder</div>
    <div class="app-bar-sub">STATCAST 2017–2024 · ARM-SIDE NORMALIZED MOVEMENT · WEIGHTED SIMILARITY SCORING</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error(
        "**`pitcher_profiles.csv` not found.** "
        "Run `build_profiles.py` locally first, then commit the output CSV to your repo alongside `app.py`."
    )
    st.stop()

# Status bar
yr_min = int(profiles["year"].min())
yr_max = int(profiles["year"].max())
st.markdown(
    f"<div class='status-bar'>"
    f"<span>✓ <b style='color:#2a8a5a'>{profiles['year'].nunique()} seasons</b> loaded</span>"
    f"<span>·</span>"
    f"<span><b style='color:#2a8a5a'>{len(profiles):,}</b> pitcher-seasons indexed</span>"
    f"<span>·</span>"
    f"<span><b style='color:#2a8a5a'>{profiles['player_name'].nunique():,}</b> unique pitchers</span>"
    f"<span>·</span>"
    f"<span>⚡ searches are instant</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Similarity engine ─────────────────────────────────────────────────────────
def norm_diff(a, b, scale):
    return min(abs(a - b) / scale, 1.0)

def is_real(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))

def score_row(user, pitch_inputs, row):
    tw = tp = 0.0
    if user.get("hand"):
        tw += W["hand"]
        tp += W["hand"] * (0.0 if row["hand"] == user["hand"] else 1.0)
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
        tw += W[key]; tp += W[key] * pen
    for group, metrics in pitch_inputs.items():
        for metric in ("ivb", "hb", "velo"):
            val = metrics.get(metric)
            if val is None:
                continue
            mv  = row.get(f"{metric}_{group}")
            pen = norm_diff(mv, val, NORM[metric]) if is_real(mv) else 0.6
            tw += W[metric]; tp += W[metric] * pen
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


def vn(v):
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else v

def sim_color(s):
    if s >= 80: return "#06d6a0"
    if s >= 65: return "#c9a84c"
    if s >= 50: return "#f4a261"
    return "#e63946"


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN: INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "input":

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;max-width:720px;margin:0 auto 32px auto;padding:0 24px">
      <div style="font-family:'Rajdhani',sans-serif;font-size:22px;color:#c9a84c;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">
        Enter Your Release &amp; Pitch Metrics
      </div>
      <div style="font-size:14px;color:#3a6a8a;line-height:1.8">
        Fill in whatever you know — leave any field blank to treat it as an open filter.<br>
        <strong style="color:#6a9ab8">Handedness → Rel Height → Rel Side → iVB &amp; HB → Velocity</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, main_col, _ = st.columns([0.5, 11, 0.5])
    with main_col:

        st.markdown('<div class="sec-label">Release Profile</div>', unsafe_allow_html=True)
        rp1, rp2, rp3, rp4, rp5 = st.columns([2, 2, 2, 2, 2])
        with rp1:
            hand_choice = st.radio("Throwing Hand", ["Any","RHP","LHP"],
                                   horizontal=True, index=0, key="hand_r")
        with rp2:
            rel_height_v = st.number_input("Rel Height (ft)", min_value=3.0, max_value=8.0,
                                            value=None, step=0.01, format="%.2f",
                                            placeholder="e.g. 5.00", key="rh")
        with rp3:
            rel_side_v   = st.number_input("Rel Side (ft)",   min_value=0.0, max_value=5.0,
                                            value=None, step=0.01, format="%.2f",
                                            placeholder="e.g. 2.80", key="rs")
        with rp4:
            extension_v  = st.number_input("Extension (ft)",  min_value=4.0, max_value=8.0,
                                            value=None, step=0.01, format="%.2f",
                                            placeholder="e.g. 6.20", key="ext")
        with rp5:
            top_n = st.slider("Top N Results", 5, 50, 20, 5, key="topn")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="sec-label">Pitch Arsenal — only fill the pitches you throw</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "<div class='hb-note'>"
            "<b style='color:#8aadcc'>HB (in):</b> Positive = arm-side run, Negative = glove-side. "
            "Examples: RHP 4-seam <b style='color:#e8dcc8'>+14\"</b>, "
            "RHP slider <b style='color:#e8dcc8'>−4\"</b>, "
            "LHP changeup <b style='color:#e8dcc8'>+12\"</b>"
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
                color = PITCH_COLORS[group]
                with col:
                    st.markdown(
                        f"<div class='pitch-card'>"
                        f"<div class='pitch-card-title' style='color:{color}'>● {group}</div>",
                        unsafe_allow_html=True,
                    )
                    velo = st.number_input("Velocity (mph)", min_value=60.0, max_value=105.0,
                                           value=None, step=0.1, format="%.1f",
                                           placeholder="e.g. 93.5", key=f"velo_{group}")
                    ivb  = st.number_input("iVB (in)",       min_value=-30.0, max_value=30.0,
                                           value=None, step=0.1, format="%.1f",
                                           placeholder="e.g. 18.0", key=f"ivb_{group}")
                    hb   = st.number_input("HB (in)",        min_value=-30.0, max_value=30.0,
                                           value=None, step=0.1, format="%.1f",
                                           placeholder="e.g. +14", key=f"hb_{group}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    v, i, h = vn(velo), vn(ivb), vn(hb)
                    if any(x is not None for x in [v, i, h]):
                        pitch_inputs_raw[group] = {"velo": v, "ivb": i, "hb": h}

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([3, 4, 3])
        with btn_col:
            run = st.button("⚾  Find My MLB Comps  →", key="run_btn")
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
                with st.spinner("Scoring similarity across all pitcher-seasons…"):
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

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    hdr_l, hdr_r = st.columns([1, 6])
    with hdr_l:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← New Search"):
            st.session_state.screen  = "input"
            st.session_state.results = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with hdr_r:
        parts = []
        if user.get("hand"):       parts.append(f"<b style='color:#e8dcc8'>{snap['hand_label']}</b>")
        if user.get("rel_height"): parts.append(f"Ht <b style='color:#e8dcc8'>{user['rel_height']:.2f} ft</b>")
        if user.get("rel_side"):   parts.append(f"Side <b style='color:#e8dcc8'>{user['rel_side']:.2f} ft</b>")
        if user.get("extension"):  parts.append(f"Ext <b style='color:#e8dcc8'>{user['extension']:.2f} ft</b>")
        for g, m in pitch_inputs.items():
            subs = []
            if m.get("velo"): subs.append(f"{m['velo']:.1f} mph")
            if m.get("ivb"):  subs.append(f"iVB {m['ivb']:.1f}\"")
            if m.get("hb"):   subs.append(f"HB {m['hb']:.1f}\"")
            if subs: parts.append(f"<b style='color:{PITCH_COLORS[g]}'>{g}</b>: {', '.join(subs)}")
        st.markdown(
            "<div style='font-family:Georgia,serif;font-size:12px;color:#4a7a9a;"
            "background:#0b1a2e;padding:10px 18px;border-radius:6px;border:1px solid #1a3550'>"
            "<span style='color:#c9a84c;font-family:Rajdhani,sans-serif;font-weight:700;"
            "letter-spacing:1.5px;text-transform:uppercase;font-size:11px'>Your Profile: </span>"
            + " &nbsp;·&nbsp; ".join(parts) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Results Found",   len(results))
    m2.metric("Years Searched",  "2017–2024")
    m3.metric("Top Match",       f"{results[0]['Pitcher']} ({results[0]['Year']})" if results else "—")
    m4.metric("Best Similarity", f"{results[0]['Similarity']:.1f}" if results else "—")

    st.markdown("---")
    st.markdown(
        f'<div class="sec-label">Top {snap["top_n"]} Similar MLB Pitcher-Seasons (2017–2024)</div>',
        unsafe_allow_html=True,
    )

    display_rows = []
    for rank, r in enumerate(results, 1):
        display_rows.append({
            "#":             rank,
            "Pitcher":       r["Pitcher"],
            "Year":          r["Year"],
            "Hand":          r["Hand"],
            "Similarity":    r["Similarity"],
            "Rel Height":    r["Rel Height"],
            "Rel Side":      r["Rel Side"],
            "Extension":     r["Extension"] if r["Extension"] else "—",
            "Total Pitches": f"{r['Total Pitches']:,}",
        })

    df_disp = pd.DataFrame(display_rows)

    def color_sim_cell(val):
        try: return f"color: {sim_color(float(val))}; font-weight: bold"
        except: return ""

    styled = (
        df_disp.style
        .applymap(color_sim_cell, subset=["Similarity"])
        .format({"Similarity": "{:.1f}", "Rel Height": "{:.2f}", "Rel Side": "{:.2f}"})
        .set_properties(**{"background-color":"#0b1a2e","color":"#c8d8e8",
                           "border":"1px solid #1a3550","font-family":"Georgia, serif","font-size":"13px"})
        .set_table_styles([{"selector":"th","props":[
            ("background-color","#07111e"),("color","#c9a84c"),
            ("font-family","Rajdhani,sans-serif"),("font-size","11px"),
            ("text-transform","uppercase"),("letter-spacing","1.5px"),
            ("border","1px solid #1a3550"),
        ]}])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("---")

    if pitch_inputs:
        st.markdown('<div class="sec-label">Pitch Metric Breakdown — Top 5 Matches</div>',
                    unsafe_allow_html=True)
        for idx, r in enumerate(results[:5]):
            row = r["_row"]
            with st.expander(
                f"#{idx+1}  {r['Pitcher']}  {r['Year']}  ({r['Hand']}HP)  ·  "
                f"Similarity {r['Similarity']:.1f}  ·  "
                f"Rel Ht {r['Rel Height']:.2f} ft  ·  Rel Side {r['Rel Side']:.2f} ft",
                expanded=(idx == 0),
            ):
                for group in PITCH_GROUPS:
                    color   = PITCH_COLORS[group]
                    user_m  = pitch_inputs.get(group, {})
                    mv_velo = row.get(f"velo_{group}")
                    mv_ivb  = row.get(f"ivb_{group}")
                    mv_hb   = row.get(f"hb_{group}")
                    if not is_real(mv_velo) and not any(v is not None for v in user_m.values()):
                        continue
                    st.markdown(
                        f"<div style='font-family:Rajdhani,sans-serif;font-size:13px;"
                        f"font-weight:700;color:{color};letter-spacing:1.5px;"
                        f"text-transform:uppercase;margin:12px 0 6px 0'>● {group}</div>",
                        unsafe_allow_html=True,
                    )
                    gc1, gc2, gc3 = st.columns(3)
                    for col, label, mlb_val, user_val, unit in [
                        (gc1, "Velocity", mv_velo, user_m.get("velo"), " mph"),
                        (gc2, "iVB",      mv_ivb,  user_m.get("ivb"),  '"'),
                        (gc3, "HB",       mv_hb,   user_m.get("hb"),   '"'),
                    ]:
                        mlb_str  = f"{mlb_val:.1f}{unit}" if is_real(mlb_val) else "—"
                        user_str = f"{user_val:.1f}{unit}" if user_val is not None else "—"
                        delta    = round(mlb_val - user_val, 1) if (is_real(mlb_val) and user_val is not None) else None
                        col.metric(
                            label=f"{label}  (MLB · You)",
                            value=mlb_str,
                            delta=f"{delta:+.1f}{unit} vs {user_str}" if delta is not None else f"You: {user_str}",
                            delta_color="off",
                        )

    st.markdown("---")
    export = [{k: v for k, v in r.items() if k != "_row"} for r in results]
    csv    = pd.DataFrame(export).to_csv(index=False).encode("utf-8")
    _, dl_col, _ = st.columns([2, 3, 2])
    with dl_col:
        st.download_button("⬇  Download Results CSV", data=csv,
                           file_name="pitcher_similarity_results.csv",
                           mime="text/csv", use_container_width=True)
