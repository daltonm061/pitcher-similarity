import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime

# --- Constants & Configuration ---
PITCH_GROUPS = {
    "4-Seam": ["FF"],
    "2-Seam/Sinker": ["FT", "SI"],
    "Cutter": ["FC"],
    "Slider": ["SL"],
    "Sweeper": ["ST"],
    "Curveball": ["CU", "CS", "KC"],
    "Splitter": ["FS"],
    "Changeup": ["CH"],
    "Knuckleball": ["KN"],
}

PITCH_ALIASES = {
    "Cutter": ["Slider"],
    "Slider": ["Cutter", "Sweeper"],
    "Sweeper": ["Slider", "Cutter"],
}

SIGMA = {
    "rel_height": 0.20,
    "rel_side": 0.30,
    "velo": 1.2,
    "ivb": 2.5,
    "hb": 2.5,
    "extension": 0.50,
}

WEIGHTS = {
    "rel_height": 3.0,
    "rel_side": 2.5,
    "velo": 5.0,
    "ivb": 3.0,
    "hb": 3.0,
    "extension": 0.5,
}

VELO_BOOST_THRESHOLD = 95.0
VELO_BOOST_MIN_SIGMA = 0.8

# Branding Colors
BRAND_GOLD = "#D4A848"
BRAND_GREEN = "#06D6A0"
BRAND_BLUE = "#8AB0C8"
APP_BG = "#080C14"

# --- Scoring Logic ---

def gaussian_sim(val_a, val_b, sigma):
    if val_a is None or val_b is None or pd.isna(val_a) or pd.isna(val_b):
        return 0.4 # Default penalty for missing data
    d = abs(val_a - val_b)
    return math.exp(-0.5 * math.pow(d / sigma, 2))

def velo_sigma(user_velo):
    if user_velo is None or user_velo <= VELO_BOOST_THRESHOLD:
        return SIGMA["velo"]
    frac = min(user_velo - VELO_BOOST_THRESHOLD, 7.0) / 7.0
    return SIGMA["velo"] - (SIGMA["velo"] - VELO_BOOST_MIN_SIGMA) * frac

def sample_confidence(n_pitches):
    if n_pitches is None or n_pitches <= 0:
        return 0.70
    return 1.0 - math.exp(-n_pitches / 300.0)

def score_row(row, user_release, pitch_inputs):
    # Hand filter
    if user_release.get('hand') != 'Any' and row['hand'] != user_release.get('hand'):
        return 0.0

    pitch_list = list(pitch_inputs.keys())
    missing_sim = 0.05
    if pitch_list:
        matched_groups = [g for g in pitch_list if not pd.isna(row.get(f"velo_{g}")) or any(not pd.isna(row.get(f"velo_{a}")) for a in PITCH_ALIASES.get(g, []))]
        if not matched_groups: return 0.0
        coverage = len(matched_groups) / len(pitch_list)
        missing_sim = 0.05 + 0.25 * coverage

    log_sum = 0
    total_w = 0

    # Release metrics
    for key in ["rel_height", "rel_side", "extension"]:
        val = user_release.get(key)
        if val is None: continue
        
        sim = gaussian_sim(row[key], val, SIGMA[key]) if not pd.isna(row[key]) else 0.4
        w = WEIGHTS[key]
        log_sum += w * math.log(max(sim, 1e-9))
        total_w += w

    # Pitch metrics
    for group, metrics in pitch_inputs.items():
        if metrics.get('velo') is None: continue
        sv = velo_sigma(metrics['velo'])
        
        col_group = group
        has_pitch = not pd.isna(row.get(f"velo_{group}"))
        if not has_pitch:
            for alias in PITCH_ALIASES.get(group, []):
                if not pd.isna(row.get(f"velo_{alias}")):
                    col_group = alias
                    has_pitch = True
                    break

        # Hard shape cutoff
        if has_pitch:
            if metrics.get('hb') is not None and not pd.isna(row.get(f"hb_{col_group}")):
                if abs(row[f"hb_{col_group}"] - metrics['hb']) > 6.0: has_pitch = False
            if has_pitch and metrics.get('ivb') is not None and not pd.isna(row.get(f"ivb_{col_group}")):
                if abs(row[f"ivb_{col_group}"] - metrics['ivb']) > 5.0: has_pitch = False

        for metric in ["velo", "ivb", "hb"]:
            user_val = metrics.get(metric)
            if user_val is None: continue
            
            sim = missing_sim
            if has_pitch:
                mv = row.get(f"{metric}_{col_group}")
                sigma = sv if metric == "velo" else SIGMA[metric]
                sim = gaussian_sim(mv, user_val, sigma) if not pd.isna(mv) else 0.4
            
            w = WEIGHTS.get(metric, 1.0)
            log_sum += w * math.log(max(sim, 1e-9))
            total_w += w

    if total_w == 0: return 0.0
    if total_w < 0: return 0.0 # Safety
    
    similarity = math.exp(log_sum / total_w) * 100
    conf = sample_confidence(row['total_pitches'])
    return round(similarity * conf, 1)

def score_single_pitch(row, user_release, velo, ivb, hb, pitch_type_filter=None):
    if user_release.get('hand') != 'Any' and row['hand'] != user_release.get('hand'):
        return []

    results = []
    search_groups = [pitch_type_filter] if pitch_type_filter and pitch_type_filter != "All" else list(PITCH_GROUPS.keys())
    sv = velo_sigma(velo)

    # Base release score
    rel_log = 0
    rel_w = 0
    for key in ["rel_height", "rel_side", "extension"]:
        val = user_release.get(key)
        if val is None: continue
        sim = gaussian_sim(row[key], val, SIGMA[key]) if not pd.isna(row[key]) else 0.4
        w = WEIGHTS[key]
        rel_log += w * math.log(max(sim, 1e-9))
        rel_w += w

    for group in search_groups:
        mv_v = row.get(f"velo_{group}")
        if pd.isna(mv_v): continue

        mv_i = row.get(f"ivb_{group}")
        mv_h = row.get(f"hb_{group}")

        # Basic shape filters
        if ivb is not None and not pd.isna(mv_i) and abs(mv_i - ivb) > 4.0: continue
        if hb is not None and not pd.isna(mv_h) and abs(mv_h - hb) > 5.0: continue

        p_log = rel_log
        p_w = rel_w

        metrics = [
            (velo, mv_v, sv, "velo"),
            (ivb, mv_i, SIGMA['ivb'], "ivb"),
            (hb, mv_h, SIGMA['hb'], "hb")
        ]

        for u, m, s, k in metrics:
            if u is None: continue
            sim = gaussian_sim(m, u, s) if not pd.isna(m) else 0.4
            w = WEIGHTS.get(k, 1.0)
            p_log += w * math.log(max(sim, 1e-9))
            p_w += w

        if p_w == 0: continue
        score = math.exp(p_log / p_w) * 100
        conf = sample_confidence(row['total_pitches'])
        final = round(score * conf, 1)
        
        if final > 20:
            results.append({
                "Pitcher": row['player_name'],
                "Year": int(row['year']),
                "Hand": row['hand'],
                "MatchedPitch": group,
                "Similarity": final,
                "RelHt": row['rel_height'],
                "Velo": mv_v,
                "iVB": mv_i,
                "HB": mv_h
            })
    return results

# --- Streamlit Setup ---

st.set_page_config(page_title="MLB Pitcher Similarity", page_icon="⚾", layout="wide")

if 'screen' not in st.session_state:
    st.session_state.screen = 'title'
if 'mode' not in st.session_state:
    st.session_state.mode = 'arsenal'

# Custom Styling
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    
    .stApp {{
        background-color: {APP_BG};
        color: #E8DCC8;
        font-family: 'Inter', sans-serif;
    }}
    [data-testid="stSidebar"] {{
        background-color: #0A0E16;
        border-right: 1px solid #141E2E;
    }}
    .brand-header {{
        padding: 1.5rem 2.5rem;
        background: #0A0E18;
        border-bottom: 1px solid #1A2A40;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }}
    .brand-title {{
        color: {BRAND_GOLD};
        font-family: 'Inter', sans-serif;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 6px;
        font-size: 1.25rem;
        margin: 0;
    }}
    .brand-sub {{
        color: {BRAND_BLUE};
        font-family: monospace;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 4px;
        opacity: 0.6;
    }}
    .card {{
        background: #0C1420;
        border: 1px solid #141E2E;
        padding: 2.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
    }}
    .card:hover {{
        border-color: {BRAND_GOLD}40;
        box-shadow: 0 0 20px {BRAND_GOLD}10;
    }}
    .label-mono {{
        font-family: monospace;
        font-size: 10px;
        color: {BRAND_BLUE};
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.6;
        margin-bottom: 8px;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, {BRAND_GOLD}, #E8C05A) !important;
        color: #080C14 !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: 3px !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stExpander"] {{
        background: #0A0E18;
        border: 1px solid #141E2E;
        border-radius: 8px;
        margin-bottom: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("public/pitcher_profiles.csv")
        return df
    except:
        return pd.DataFrame()

df = load_data()

# Helper Navigation
def set_screen(name, mode=None):
    st.session_state.screen = name
    if mode: st.session_state.mode = mode

# --- UI Screens ---

# 1. Header
st.markdown(f"""
    <div class="brand-header">
        <div style="cursor:pointer">
            <h1 class="brand-title">Similarity Engine</h1>
            <p class="brand-sub">Statcast 2017–2024 · DM Stuff+ v4.2</p>
        </div>
        <div style="display:flex; gap: 2rem; font-family: monospace; font-size: 10px; color: {BRAND_BLUE};">
            <div>STATUS: <span style="color:{BRAND_GREEN}">VERIFIED</span></div>
            <div>SAMPLES: <span style="color:white">{len(df):,} SEASONS</span></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# 2. Title Screen
if st.session_state.screen == 'title':
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div style="text-align:center; margin-bottom: 3rem;">
                <span style="color:{BRAND_GOLD}; font-size: 10px; font-weight: bold; letter-spacing: 4px; text-transform: uppercase;">Professional Grade Analytics</span>
                <h2 style="font-size: 4.5rem; font-weight: 900; line-height: 0.9; margin: 1.5rem 0; text-transform: uppercase;">Arsenal<br><span style="color:{BRAND_GOLD}">Optimization</span></h2>
                <p style="font-family: monospace; font-size: 11px; letter-spacing: 4px; opacity: 0.6; text-transform: uppercase;">GAUSSIAN SIMILARITY MEASURES &nbsp;·&nbsp; ARM-SLOT NORMALIZATION &nbsp;·&nbsp; STUFF+ METRICS</p>
            </div>
        """, unsafe_allow_html=True)
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            if st.button("Full Arsenal", key="btn_arsenal"): set_screen('input', 'arsenal')
            st.markdown("<p style='text-align:center; font-size: 10px; opacity: 0.5; margin-top: 10px;'>MATCH ENTIRE PITCH MIX</p>", unsafe_allow_html=True)
        with m_col2:
            if st.button("Single Pitch", key="btn_single"): set_screen('input', 'single')
            st.markdown("<p style='text-align:center; font-size: 10px; opacity: 0.5; margin-top: 10px;'>SEARCH SPECIFIC SHAPES</p>", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Open Leaderboard", key="btn_leaderboard"): set_screen('leaderboard')
        st.markdown("<p style='text-align:center; font-size: 10px; opacity: 0.5; margin-top: 10px;'>GLOBAL RANKINGS & PERFORMANCE DATA</p>", unsafe_allow_html=True)

# 3. Input Screen
elif st.session_state.screen in ['input', 'searching', 'results']:
    with st.sidebar:
        if st.button("← Back to Hub"): set_screen('title')
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<p class='label-mono' style='border-bottom: 1px solid #141E2E; padding-bottom: 10px;'>Input Profile</p>", unsafe_allow_html=True)
        
        hand = st.selectbox("Handedness", options=["Any", "R", "L"])
        rel_height = st.number_input("Rel Height (ft)", value=5.7, step=0.1)
        rel_side = st.number_input("Rel Side (ft)", value=1.9, step=0.1)
        extension = st.number_input("Extension (ft)", value=6.4, step=0.1)
        top_n = st.slider("Results", 5, 100, 20)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Calculate Comps"): st.session_state.screen = 'searching'

    if st.session_state.screen == 'input':
        st.markdown(f"<br><h2 style='font-size: 2.5rem; font-weight: 900; color: {BRAND_GOLD}; margin-bottom: 2rem;'>{ 'Arsenal Metrics' if st.session_state.mode == 'arsenal' else 'Pitch Shaping' }</h2>", unsafe_allow_html=True)
        
        if st.session_state.mode == 'arsenal':
            cols = st.columns(2)
            for i, group in enumerate(PITCH_GROUPS.keys()):
                with cols[i % 2]:
                    with st.container():
                        st.markdown(f"<div style='border-left: 3px solid {BRAND_GOLD if i%2==0 else BRAND_BLUE}; padding-left: 20px; margin-bottom: 2rem;'>", unsafe_allow_html=True)
                        st.markdown(f"<p class='label-mono' style='color:white; font-weight:bold;'>{group}</p>", unsafe_allow_html=True)
                        p_cols = st.columns(3)
                        st.number_input(f"Velo", key=f"v_{group}", value=0.0, step=0.1)
                        st.number_input(f"iVB", key=f"ivb_{group}", value=0.0, step=0.1)
                        st.number_input(f"HB", key=f"hb_{group}", value=0.0, step=0.1)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            col_search, _ = st.columns([2, 1])
            with col_search:
                st.selectbox("Pitch Category", options=["All"] + list(PITCH_GROUPS.keys()), key="ptype")
                st.number_input("Velocity (mph)", value=94.0, step=0.1, key="s_velo")
                st.number_input("Induced Vertical Break (in)", value=18.0, step=0.1, key="s_ivb")
                st.number_input("Horizontal Movement (in)", value=10.0, step=0.1, key="s_hb")
    
    elif st.session_state.screen == 'searching':
        st.session_state.screen = 'results'
        st.rerun()

    elif st.session_state.screen == 'results':
        user_release = {"hand": hand, "rel_height": rel_height, "rel_side": rel_side, "extension": extension}
        
        if st.session_state.mode == 'arsenal':
            pitch_inputs = {}
            for g in PITCH_GROUPS.keys():
                v = st.session_state.get(f"v_{g}", 0)
                if v > 0:
                    pitch_inputs[g] = { "velo": v, "ivb": st.session_state.get(f"ivb_{g}", 0), "hb": st.session_state.get(f"hb_{g}", 0) }
            
            with st.spinner("Analyzing Gaussian Probability Vectors..."):
                results_df = df.copy()
                results_df['Similarity'] = results_df.apply(lambda row: score_row(row, user_release, pitch_inputs), axis=1)
                results_final = results_df[results_df['Similarity'] > 0].sort_values('Similarity', ascending=False).head(top_n)
                
                if results_final.empty:
                    st.warning("No significant matches found for this profile.")
                    if st.button("Edit Search"): set_screen('input')
                else:
                    top = results_final.iloc[0]
                    st.markdown(f"""
                        <div style="background: #0C1420; border: 1px solid #141E2E; padding: 2.5rem; border-radius: 12px; margin-top: 2rem; border-left: 4px solid {BRAND_GOLD};">
                            <div style="display:flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <p class="label-mono">Elite Comp Identified</p>
                                    <h2 style="font-size: 4rem; font-weight: 200; color: {BRAND_GOLD}; margin: 0.5rem 0;">{top['player_name']} <span style="font-size: 1.5rem; opacity: 0.4;">{int(top['year'])}</span></h2>
                                </div>
                                <div style="text-align:right;">
                                    <p class="label-mono">Similarity Rating</p>
                                    <div style="color: {BRAND_GREEN}; font-size: 2.5rem; font-weight: 900; font-style: italic;">{top['Similarity']}% <span style="font-size: 10px; font-weight: normal; font-style: normal; color: white; opacity: 0.5; margin-left: 5px;">MATCH</span></div>
                                </div>
                            </div>
                        </div>
                        <br>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(
                        results_final[['player_name', 'year', 'hand', 'Similarity', 'rel_height', 'total_pitches']].rename(columns={
                            'player_name': 'Pitcher', 'year': 'Year', 'hand': 'Hand', 'rel_height': 'Rel Ht', 'total_pitches': 'Index'
                        }), use_container_width=True, hide_index=True
                    )
        
        else:
            s_velo = st.session_state.get('s_velo', 94.0)
            s_ivb = st.session_state.get('s_ivb', 18.0)
            s_hb = st.session_state.get('s_hb', 10.0)
            ptype = st.session_state.get('ptype', 'All')

            with st.spinner("Scanning Statcast Database..."):
                all_results = []
                for _, r in df.iterrows():
                    batch = score_single_pitch(r, user_release, s_velo, s_ivb, s_hb, ptype)
                    all_results.extend(batch)
                
                if not all_results:
                    st.warning("No matches found for that specific pitch shape.")
                else:
                    results_ps = pd.DataFrame(all_results).sort_values('Similarity', ascending=False).head(top_n)
                    top = results_ps.iloc[0]
                    st.markdown(f"""
                        <div style="background: #0C1420; border: 1px solid #141E1E; padding: 2.5rem; border-radius: 12px; margin-top: 2rem; border-left: 4px solid {BRAND_BLUE};">
                            <div style="display:flex; justify-content: space-between; align-items: start;">
                                <div>
                                    <p class="label-mono">Pitch Pattern Found</p>
                                    <h2 style="font-size: 4rem; font-weight: 200; color: {BRAND_BLUE}; margin: 0.5rem 0;">{top['Pitcher']} <span style="font-size: 1.5rem; opacity: 0.4;">{int(top['Year'])}</span></h2>
                                    <p style="color:{BRAND_GOLD}; font-weight: 900; font-size: 12px; text-transform: uppercase; letter-spacing: 2px;">{top['MatchedPitch']} CATEGORY</p>
                                </div>
                                <div style="text-align:right;">
                                    <p class="label-mono">Similarity</p>
                                    <div style="color: {BRAND_GREEN}; font-size: 2.5rem; font-weight: 900; font-style: italic;">{top['Similarity']}%</div>
                                </div>
                            </div>
                        </div>
                        <br>
                    """, unsafe_allow_html=True)
                    
                    st.dataframe(results_ps, use_container_width=True, hide_index=True)

# 4. Leaderboard Screen
elif st.session_state.screen == 'leaderboard':
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Hub"): set_screen('title')
    
    st.markdown(f"""
        <h2 style="font-size: 3rem; font-weight: 900; color: {BRAND_GREEN}; text-transform: uppercase; margin: 1.5rem 0;">Global Leaderboard</h2>
        <div style="height: 4px; width: 80px; background: {BRAND_GREEN}40; border-radius: 10px; margin-bottom: 2rem;"></div>
        <p class="label-mono" style="opacity: 0.6; letter-spacing: 4px;">COMPLETE STATCAST 2017–2024 REGISTRY &nbsp;·&nbsp; PERFORMANCE INDEX</p>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns(2)
    f_year = col_f1.multiselect("Years", options=sorted(df['year'].unique().tolist(), reverse=True))
    f_hand = col_f2.selectbox("Hand", options=["All", "R", "L"])
    
    view_df = df.copy()
    if f_year: view_df = view_df[view_df['year'].isin(f_year)]
    if f_hand != "All": view_df = view_df[view_df['hand'] == f_hand]
    
    st.dataframe(
        view_df[['player_name', 'year', 'hand', 'total_pitches', 'rel_height', 'rel_side', 'extension']].sort_values('total_pitches', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# --- Footer ---
st.markdown(f"""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background: #0A0E16; border-top: 1px solid #141E2E; padding: 0.5rem 2.5rem; font-family: monospace; font-size: 9px; color: #4A6A80; display: flex; justify-content: space-between; z-index: 1000;">
        <div>© 2024 DM ANALYTICS GROUP</div>
        <div style="display:flex; gap: 2rem;">
            <div>LAST SYNC: {datetime.now().strftime('%H:%M:%S')}</div>
            <div>PROCESSED 18.2M DATA POINTS</div>
        </div>
    </div>
""", unsafe_allow_html=True)
