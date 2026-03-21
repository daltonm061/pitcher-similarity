import warnings
import math
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Pitcher Similarity Finder",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
}

.stApp {
    background: #080f1a;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0c1828 !important;
    border-right: 1px solid #1e3a55;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c9a84c;
    font-family: 'Rajdhani', sans-serif;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0c1828 0%, #122040 100%);
    border-bottom: 2px solid #c9a84c;
    padding: 24px 32px;
    margin: -1rem -1rem 2rem -1rem;
    display: flex;
    align-items: center;
    gap: 16px;
}
.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 32px;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}
.main-subtitle {
    font-size: 13px;
    color: #5a8aaa;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Pitch section cards */
.pitch-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3a55;
    padding-bottom: 6px;
    margin-bottom: 4px;
}

/* Metric inputs */
.stNumberInput > label, .stSelectbox > label, .stRadio > label {
    color: #8aadcc !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stNumberInput input {
    background: #0a1628 !important;
    color: #e8dcc8 !important;
    border: 1px solid #1e3a55 !important;
    border-radius: 4px !important;
}
.stNumberInput input:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 1px #c9a84c40 !important;
}

/* Radio buttons */
.stRadio [data-testid="stMarkdownContainer"] p { color: #8aadcc !important; }

/* Run button */
.stButton > button {
    background: linear-gradient(135deg, #c9a84c, #e8c96a) !important;
    color: #080f1a !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 12px 32px !important;
    width: 100%;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #e8c96a, #c9a84c) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px #c9a84c40 !important;
}

/* Divider */
hr { border-color: #1e3a55 !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: #0c1828;
    border: 1px solid #1e3a55;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label {
    color: #5a8aaa !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c9a84c !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 28px !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #1e3a55 !important;
    border-radius: 8px !important;
}

/* Section labels */
.section-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #c9a84c;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 8px 0;
}

/* Similarity bar */
.sim-bar-wrap { background: #0c1828; border-radius: 6px; height: 8px; width: 100%; }
.sim-bar-fill { border-radius: 6px; height: 8px; }

/* Spinner text */
.stSpinner > div { color: #c9a84c !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #0c1828 !important;
    color: #8aadcc !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 13px !important;
}
.streamlit-expanderContent {
    background: #080f1a !important;
    border: 1px solid #1e3a55 !important;
}

/* Warning / info */
.stAlert { background: #0c1828 !important; border-color: #1e3a55 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
YEAR = 2024
MIN_PITCHER_PITCHES = 100
MIN_PITCH_TYPE_N = 20

PITCH_GROUPS = {
    "4-Seam":        ["FF"],
    "2-Seam/Sinker": ["FT", "SI"],
    "Cutter":        ["FC"],
    "Slider":        ["SL"],
    "Sweeper":       ["ST"],
    "Splitter":      ["FS"],
    "Changeup":      ["CH"],
}

PITCH_COLORS = {
    "4-Seam":        "#e63946",
    "2-Seam/Sinker": "#f4a261",
    "Cutter":        "#2a9d8f",
    "Slider":        "#457b9d",
    "Sweeper":       "#a855f7",
    "Splitter":      "#e9c46a",
    "Changeup":      "#06d6a0",
}

W_HAND       = 1000
W_REL_HEIGHT = 50
W_REL_SIDE   = 30
W_EXTENSION  = 10
W_IVB        = 25
W_HB         = 25
W_VELO       = 10

NORM = {
    "rel_height": 1.5,
    "rel_side":   2.0,
    "extension":  1.5,
    "ivb":        20.0,
    "hb":         20.0,
    "velo":       15.0,
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_build_profiles(year: int):
    from pybaseball import statcast, cache as pb_cache
    pb_cache.enable()

    df = statcast(start_dt=f"{year}-03-20", end_dt=f"{year}-11-01")

    keep = ["player_name", "pitch_type", "p_throws",
            "release_speed", "release_spin_rate",
            "pfx_x", "pfx_z", "release_extension",
            "release_pos_x", "release_pos_z"]
    df = df[keep].dropna(subset=["release_pos_x", "release_pos_z", "pitch_type", "p_throws"]).copy()

    # Arm-side normalize HB
    df["pfx_x_norm"] = np.where(df["p_throws"] == "L", -df["pfx_x"], df["pfx_x"])
    df["ivb_in"]     = df["pfx_z"]      * 12
    df["hb_in"]      = df["pfx_x_norm"] * 12

    code_to_group = {c: g for g, codes in PITCH_GROUPS.items() for c in codes}
    df["pitch_group"] = df["pitch_type"].map(code_to_group)

    rel_profile = (
        df.groupby("player_name")
        .agg(
            hand          = ("p_throws",         "first"),
            rel_height    = ("release_pos_z",    "mean"),
            rel_side      = ("release_pos_x",    "mean"),
            extension     = ("release_extension","mean"),
            total_pitches = ("pitch_type",       "count"),
        )
        .reset_index()
    )
    rel_profile = rel_profile[rel_profile["total_pitches"] >= MIN_PITCHER_PITCHES]

    pitch_profile = (
        df[df["pitch_group"].notna()]
        .groupby(["player_name", "pitch_group"])
        .agg(velo=("release_speed","mean"), ivb=("ivb_in","mean"),
             hb=("hb_in","mean"), count=("pitch_type","count"))
        .reset_index()
    )
    pitch_profile = pitch_profile[pitch_profile["count"] >= MIN_PITCH_TYPE_N]

    pitch_wide = pitch_profile.pivot(index="player_name", columns="pitch_group",
                                     values=["velo","ivb","hb"])
    pitch_wide.columns = [f"{m}_{g}" for m, g in pitch_wide.columns]
    pitch_wide = pitch_wide.reset_index()

    return rel_profile.merge(pitch_wide, on="player_name", how="left")


# ── Similarity engine ─────────────────────────────────────────────────────────
def norm_diff(val, ref, scale):
    return min(abs(val - ref) / scale, 1.0)


def compute_similarity(user, pitch_inputs, mlb_row):
    total_weight = 0
    total_penalty = 0

    if user.get("hand"):
        w = W_HAND
        penalty = 0.0 if mlb_row["hand"] == user["hand"] else 1.0
        total_weight  += w
        total_penalty += w * penalty

    for key, weight, scale in [
        ("rel_height", W_REL_HEIGHT, NORM["rel_height"]),
        ("rel_side",   W_REL_SIDE,   NORM["rel_side"]),
        ("extension",  W_EXTENSION,  NORM["extension"]),
    ]:
        if user.get(key) is not None:
            mlb_val = mlb_row.get(key)
            if key == "rel_side":
                penalty = norm_diff(abs(mlb_val or 0), abs(user[key]), scale)
            else:
                penalty = norm_diff(mlb_val or 0, user[key], scale) if pd.notna(mlb_val) else 0.5
            total_weight  += weight
            total_penalty += weight * penalty

    for group, metrics in pitch_inputs.items():
        for metric, weight, scale in [
            ("ivb",  W_IVB,  NORM["ivb"]),
            ("hb",   W_HB,   NORM["hb"]),
            ("velo", W_VELO, NORM["velo"]),
        ]:
            val = metrics.get(metric)
            if val is None:
                continue
            col = f"{metric}_{group}"
            mlb_val = mlb_row.get(col)
            if mlb_val is None or (isinstance(mlb_val, float) and math.isnan(mlb_val)):
                penalty = 0.6
            else:
                penalty = norm_diff(mlb_val, val, scale)
            total_weight  += weight
            total_penalty += weight * penalty

    if total_weight == 0:
        return 0.0
    return round(max(1.0 - total_penalty / total_weight, 0) * 100, 1)


def find_similar(user, pitch_inputs, profiles, top_n):
    rows = []
    for _, r in profiles.iterrows():
        s = compute_similarity(user, pitch_inputs, r)
        rows.append({
            "Pitcher":       r["player_name"],
            "Hand":          r["hand"],
            "Similarity":    s,
            "Rel Height":    round(r["rel_height"], 2),
            "Rel Side":      round(r["rel_side"],   2),
            "Extension":     round(r["extension"],  2),
            "Total Pitches": int(r["total_pitches"]),
            "_row":          r,
        })
    return sorted(rows, key=lambda x: -x["Similarity"])[:top_n]


# ── Helper ────────────────────────────────────────────────────────────────────
def val_or_none(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def sim_color(score):
    if score >= 80: return "#06d6a0"
    if score >= 65: return "#c9a84c"
    if score >= 50: return "#f4a261"
    return "#e63946"


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
  <span style="font-size:44px;line-height:1">⚾</span>
  <div>
    <div class="main-title">MLB Pitcher Similarity Finder</div>
    <div class="main-subtitle">
      Enter your release metrics &amp; pitch characteristics to find your MLB comps
      &nbsp;·&nbsp; {YEAR} Statcast &nbsp;·&nbsp; Arm-side normalized movement
    </div>
  </div>
</div>
""".replace("{YEAR}", str(YEAR)), unsafe_allow_html=True)

# ── Sidebar — inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Your Profile")
    st.caption("Leave any field blank to treat it as an open filter.")

    st.markdown("---")
    st.markdown('<div class="section-label">Release Profile</div>', unsafe_allow_html=True)

    hand_choice = st.radio(
        "Throwing Hand",
        options=["Any", "RHP", "LHP"],
        horizontal=True,
        index=0,
    )
    hand = None if hand_choice == "Any" else hand_choice[0]

    c1, c2 = st.columns(2)
    with c1:
        rel_height_raw = st.number_input("Rel Height (ft)", min_value=3.0, max_value=8.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder="e.g. 5.00")
        extension_raw  = st.number_input("Extension (ft)",  min_value=4.0, max_value=8.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder="e.g. 6.2")
    with c2:
        rel_side_raw   = st.number_input("Rel Side (ft)",   min_value=0.0, max_value=5.0,
                                          value=None, step=0.01, format="%.2f",
                                          placeholder="e.g. 2.80")

    st.markdown("---")
    st.markdown('<div class="section-label">Pitch Arsenal</div>', unsafe_allow_html=True)
    st.caption("Fill in only the pitch types you throw.")

    pitch_inputs_raw = {}
    for group, color in PITCH_COLORS.items():
        dot = f"<span style='color:{color};font-size:14px'>●</span>"
        with st.expander(f"{group}", expanded=False):
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                velo = st.number_input(f"Velo (mph)", key=f"velo_{group}",
                                       min_value=60.0, max_value=105.0,
                                       value=None, step=0.1, format="%.1f",
                                       placeholder="e.g. 93.5")
            with pc2:
                ivb  = st.number_input(f"iVB (in)", key=f"ivb_{group}",
                                       min_value=-30.0, max_value=30.0,
                                       value=None, step=0.1, format="%.1f",
                                       placeholder="e.g. 18.0")
            with pc3:
                hb   = st.number_input(f"HBreak arm+ (in)", key=f"hb_{group}",
                                       min_value=-30.0, max_value=30.0,
                                       value=None, step=0.1, format="%.1f",
                                       placeholder="e.g. 14.0")
            v = val_or_none(velo)
            i = val_or_none(ivb)
            h = val_or_none(hb)
            if any(x is not None for x in [v, i, h]):
                pitch_inputs_raw[group] = {"velo": v, "ivb": i, "hb": h}

    st.markdown("---")
    st.markdown('<div class="section-label">Options</div>', unsafe_allow_html=True)
    top_n   = st.slider("Top N results", 5, 50, 15, 5)
    run_btn = st.button("🔍  Find My MLB Comps", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────

# Load data
with st.spinner("Loading 2024 Statcast data… (first load ~45s, then cached instantly)"):
    try:
        profiles = load_and_build_profiles(YEAR)
        data_ok = True
    except Exception as e:
        st.error(f"Failed to load Statcast data: {e}")
        data_ok = False

if data_ok:
    st.markdown(
        f"<div style='font-size:12px;color:#2a5070;font-family:monospace;margin-bottom:16px'>"
        f"✓ {len(profiles):,} MLB pitcher profiles loaded from {YEAR} Statcast</div>",
        unsafe_allow_html=True,
    )

if run_btn and data_ok:
    user = {
        "hand":       hand,
        "rel_height": val_or_none(rel_height_raw),
        "rel_side":   val_or_none(rel_side_raw),
        "extension":  val_or_none(extension_raw),
    }

    has_any_input = (
        any(v is not None for v in user.values()) or
        bool(pitch_inputs_raw)
    )

    if not has_any_input:
        st.warning("Enter at least one metric in the sidebar to find similar pitchers.")
    else:
        with st.spinner("Computing similarity scores..."):
            results = find_similar(user, pitch_inputs_raw, profiles, top_n)

        # ── Your profile summary bar ───────────────────────────────
        st.markdown('<div class="section-label">Your Input Profile</div>', unsafe_allow_html=True)
        prof_cols = st.columns(6)
        labels = [
            ("Hand",       hand_choice),
            ("Rel Height", f"{rel_height_raw:.2f} ft" if val_or_none(rel_height_raw) else "—"),
            ("Rel Side",   f"{rel_side_raw:.2f} ft"   if val_or_none(rel_side_raw)   else "—"),
            ("Extension",  f"{extension_raw:.2f} ft"  if val_or_none(extension_raw)  else "—"),
            ("Pitches In", str(len(pitch_inputs_raw))),
            ("Top N",      str(top_n)),
        ]
        for col, (label, val) in zip(prof_cols, labels):
            col.metric(label, val)

        st.markdown("---")

        # ── Results table ──────────────────────────────────────────
        st.markdown(
            f'<div class="section-label">Top {top_n} Similar MLB Pitchers — {YEAR}</div>',
            unsafe_allow_html=True,
        )

        # Build display df
        display_rows = []
        for rank, r in enumerate(results, 1):
            sc = r["Similarity"]
            bar_html = (
                f"<div class='sim-bar-wrap'>"
                f"<div class='sim-bar-fill' style='width:{sc}%;background:{sim_color(sc)}'></div>"
                f"</div>"
            )
            display_rows.append({
                "#":             rank,
                "Pitcher":       r["Pitcher"],
                "Hand":          r["Hand"],
                "Similarity":    f"{sc:.1f}",
                "Rel Height":    f"{r['Rel Height']:.2f}",
                "Rel Side":      f"{r['Rel Side']:.2f}",
                "Extension":     f"{r['Extension']:.2f}",
                "Total Pitches": f"{r['Total Pitches']:,}",
            })

        df_display = pd.DataFrame(display_rows)

        # Color-code similarity column
        def color_sim(val):
            try:
                v = float(val)
                c = sim_color(v)
                return f"color: {c}; font-weight: bold; font-family: 'Rajdhani', sans-serif; font-size: 16px"
            except:
                return ""

        styled = (
            df_display.style
            .applymap(color_sim, subset=["Similarity"])
            .set_properties(**{
                "background-color": "#0c1828",
                "color": "#e8dcc8",
                "border": "1px solid #1e3a55",
                "font-family": "Georgia, serif",
                "font-size": "13px",
            })
            .set_table_styles([
                {"selector": "th", "props": [
                    ("background-color", "#080f1a"),
                    ("color", "#c9a84c"),
                    ("font-family", "Rajdhani, sans-serif"),
                    ("font-size", "11px"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "1.5px"),
                    ("border", "1px solid #1e3a55"),
                ]},
                {"selector": "tr:hover td", "props": [
                    ("background-color", "#122040"),
                ]},
            ])
        )
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Top 5 pitch detail breakdown ───────────────────────────
        if pitch_inputs_raw:
            st.markdown('<div class="section-label">Pitch Metric Breakdown — Top 5 Matches</div>',
                        unsafe_allow_html=True)

            for r in results[:5]:
                row      = r["_row"]
                sim_val  = r["Similarity"]
                pitcher  = r["Pitcher"]
                hand_str = r["Hand"]
                sc       = sim_color(sim_val)

                with st.expander(
                    f"#{results.index(r)+1}  {pitcher}  ({hand_str}HP)  —  "
                    f"Similarity: {sim_val:.1f}  |  "
                    f"Rel Height: {r['Rel Height']:.2f} ft  |  "
                    f"Rel Side: {r['Rel Side']:.2f} ft",
                    expanded=(results.index(r) == 0),
                ):
                    for group in PITCH_GROUPS:
                        color    = PITCH_COLORS[group]
                        user_m   = pitch_inputs_raw.get(group, {})
                        mlb_velo = row.get(f"velo_{group}")
                        mlb_ivb  = row.get(f"ivb_{group}")
                        mlb_hb   = row.get(f"hb_{group}")

                        mlb_has  = not (mlb_velo is None or (isinstance(mlb_velo, float) and math.isnan(mlb_velo)))
                        user_has = any(v is not None for v in user_m.values())

                        if not mlb_has and not user_has:
                            continue

                        st.markdown(
                            f"<div style='font-family:Rajdhani,sans-serif;font-size:13px;"
                            f"font-weight:700;color:{color};letter-spacing:1.5px;"
                            f"text-transform:uppercase;margin:10px 0 4px 0'>"
                            f"● {group}</div>",
                            unsafe_allow_html=True,
                        )

                        gc1, gc2, gc3 = st.columns(3)
                        for col, metric, mlb_val, user_val, unit in [
                            (gc1, "Velocity",   mlb_velo, user_m.get("velo"), " mph"),
                            (gc2, "iVB",        mlb_ivb,  user_m.get("ivb"),  '"'),
                            (gc3, "HBreak arm+",mlb_hb,   user_m.get("hb"),   '"'),
                        ]:
                            mlb_str  = f"{mlb_val:.1f}{unit}"  if (mlb_val  is not None and not (isinstance(mlb_val,  float) and math.isnan(mlb_val)))  else "—"
                            user_str = f"{user_val:.1f}{unit}" if user_val is not None else "—"

                            delta = None
                            if mlb_val is not None and user_val is not None:
                                if not (isinstance(mlb_val, float) and math.isnan(mlb_val)):
                                    delta = round(mlb_val - user_val, 1)

                            col.metric(
                                label=f"{metric}  (MLB vs You)",
                                value=mlb_str,
                                delta=f"{delta:+.1f}{unit} vs yours ({user_str})" if delta is not None else f"You: {user_str}",
                                delta_color="off",
                            )

        # ── Download ───────────────────────────────────────────────
        st.markdown("---")
        export_rows = [{k: v for k, v in r.items() if k != "_row"} for r in results]
        csv = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download Results CSV",
            data=csv,
            file_name=f"pitcher_similarity_{YEAR}.csv",
            mime="text/csv",
            use_container_width=False,
        )

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#2a5070">
      <div style="font-size:64px;opacity:0.3;margin-bottom:20px">⚾</div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:20px;color:#5a8aaa;
                  letter-spacing:2px;text-transform:uppercase;margin-bottom:12px">
        Ready to Find Your MLB Comps
      </div>
      <div style="font-size:14px;color:#2a5070;max-width:480px;margin:0 auto;line-height:1.8">
        Fill in your release metrics and pitch characteristics in the sidebar,
        then click <strong style="color:#c9a84c">Find My MLB Comps</strong>.<br><br>
        Leave any field blank — it acts as an open filter.
        The similarity score weights handedness first, then release height,
        release side, movement, and velocity.
      </div>
    </div>
    """, unsafe_allow_html=True)
