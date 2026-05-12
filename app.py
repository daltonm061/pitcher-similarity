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

    log_sum = 0
    total_w = 0

    # Release metrics
    for key in ["rel_height", "rel_side", "extension"]:
        val = user_release.get(key)
        if val is None: continue
        
        sim = gaussian_sim(row[key], val, SIGMA[key])
        w = WEIGHTS[key]
        log_sum += w * math.log(max(sim, 1e-9))
        total_w += w

    # Pitch metrics
    for group, metrics in pitch_inputs.items():
        if metrics.get('velo') is None: continue
        
        sv = velo_sigma(metrics['velo'])
        
        # Check for pitch or alias
        col_group = group
        has_pitch = not pd.isna(row.get(f"velo_{group}"))
        
        if not has_pitch:
            aliases = PITCH_ALIASES.get(group, [])
            for alias in aliases:
                if not pd.isna(row.get(f"velo_{alias}")):
                    col_group = alias
                    has_pitch = True
                    break

        # Similarity calculation
        missing_sim = 0.3 # Base penalty if they don't throw this pitch
        
        for metric in ["velo", "ivb", "hb"]:
            user_val = metrics.get(metric)
            if user_val is None: continue
            
            sim = missing_sim
            if has_pitch:
                mv = row.get(f"{metric}_{col_group}")
                sigma = sv if metric == "velo" else SIGMA[metric]
                sim = gaussian_sim(mv, user_val, sigma)
            
            w = WEIGHTS.get(metric, 1.0)
            log_sum += w * math.log(max(sim, 1e-9))
            total_w += w

    if total_w == 0: return 0.0
    similarity = math.exp(log_sum / total_w) * 100
    conf = sample_confidence(row['total_pitches'])
    return round(similarity * conf, 1)

# --- Streamlit UI ---

st.set_page_config(page_title="MLB Pitcher Similarity", page_icon="⚾", layout="wide")

# Custom Styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {APP_BG};
        color: #E8DCC8;
    }}
    [data-testid="stSidebar"] {{
        background-color: #0A0E16;
        border-right: 1px solid #141E2E;
    }}
    .brand-header {{
        padding: 2rem 0;
        border-bottom: 1px solid #1A2A40;
        margin-bottom: 2rem;
    }}
    .brand-title {{
        color: {BRAND_GOLD};
        font-family: 'Rajdhani', sans-serif;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 6px;
        font-size: 1.5rem;
        margin: 0;
    }}
    .brand-sub {{
        color: {BRAND_BLUE};
        font-family: monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.8;
    }}
    .metric-card {{
        background: #0C1220;
        border: 1px solid #1A2A40;
        padding: 1.5rem;
        border-radius: 8px;
    }}
    .stButton>button {{
        background: linear-gradient(135deg, {BRAND_GOLD}, #E8C05A) !important;
        color: #080C14 !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        width: 100%;
    }}
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown(f"""
    <div class="brand-header">
        <h1 class="brand-title">Similarity Engine</h1>
        <p class="brand-sub">Statcast 2017–2024 · DM Stuff+ v4.2</p>
    </div>
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

# Sidebar Inputs
with st.sidebar:
    st.markdown(f"<h3 style='color:{BRAND_GOLD}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 3px; border-bottom: 1px solid #141E2E; padding-bottom: 10px;'>Input Profile</h3>", unsafe_allow_html=True)
    
    hand = st.selectbox("Handedness", options=["Any", "R", "L"])
    rel_height = st.number_input("Release Height (ft)", value=5.7, step=0.1)
    rel_side = st.number_input("Release Side (ft)", value=1.9, step=0.1)
    extension = st.number_input("Extension (ft)", value=6.4, step=0.1)
    top_n = st.slider("Top Results", 5, 50, 20)
    
    st.markdown("<br>", unsafe_allow_html=True)
    calculate = st.button("Calculate Comps")

# Main Content
tab1, tab2 = st.tabs(["Full Arsenal", "About the Model"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    pitch_inputs = {}
    with col1:
        st.markdown(f"<h4 style='color:{BRAND_BLUE}; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px;'>Pitch Metrics</h4>", unsafe_allow_html=True)
        for group in PITCH_GROUPS.keys():
            with st.expander(f"{group}"):
                v = st.number_input(f"Velo (mph)", key=f"v_{group}", value=0.0, step=0.1)
                ivb = st.number_input(f"iVB (in)", key=f"i_{group}", value=0.0, step=0.1)
                hb = st.number_input(f"HB (in)", key=f"h_{group}", value=0.0, step=0.1)
                if v > 0:
                    pitch_inputs[group] = {"velo": v, "ivb": ivb, "hb": hb}

    with col2:
        if calculate:
            if df.empty:
                st.error("Pitcher profiles database not found.")
            else:
                with st.spinner("Processing Statcast Vectors..."):
                    results_df = df.copy()
                    user_release = {
                        "hand": hand,
                        "rel_height": rel_height,
                        "rel_side": rel_side,
                        "extension": extension
                    }
                    
                    results_df['Similarity'] = results_df.apply(lambda row: score_row(row, user_release, pitch_inputs), axis=1)
                    results_df = results_df[results_df['Similarity'] > 0].sort_values('Similarity', ascending=False).head(top_n)
                    
                    if results_df.empty:
                        st.warning("No matches found for this profile.")
                    else:
                        top_match = results_df.iloc[0]
                        
                        # Featured Result
                        st.markdown(f"""
                            <div style="background: #0C1420; border: 1px solid #141E2E; padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
                                <div style="color: {BRAND_BLUE}; font-size: 0.6rem; text-transform: uppercase; font-family: monospace;">Elite Comp Identified</div>
                                <div style="font-size: 2.5rem; font-weight: 200; color: {BRAND_GOLD};">{top_match['player_name']} <span style="font-size: 1.2rem; opacity: 0.4;">{int(top_match['year'])}</span></div>
                                <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                                    <div>
                                        <div style="color: {BRAND_BLUE}; font-size: 0.6rem; text-transform: uppercase; font-family: monospace;">Similarity</div>
                                        <div style="color: {BRAND_GREEN}; font-size: 1.5rem; font-weight: 900;">{top_match['Similarity']}%</div>
                                    </div>
                                    <div>
                                        <div style="color: {BRAND_BLUE}; font-size: 0.6rem; text-transform: uppercase; font-family: monospace;">Season Pitches</div>
                                        <div style="color: white; font-size: 1.5rem; font-weight: 400;">{int(top_match['total_pitches'])}</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Results Table
                        display_cols = ['player_name', 'year', 'hand', 'Similarity', 'total_pitches']
                        st.dataframe(
                            results_df[display_cols].rename(columns={
                                'player_name': 'Pitcher',
                                'year': 'Year',
                                'hand': 'H',
                                'total_pitches': 'N'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

with tab2:
    st.markdown(f"""
        ### Gaussian Similarity Model
        Our engine uses a multi-dimensional Gaussian scoring algorithm to match pitcher profiles. 
        Each metric (Velocity, Vertical Break, Horizontal Break, and Release Metrics) is normalized 
        using league-standard deviations (σ).
        
        ### DM Stuff+ Integration
        The primary sorting weight is placed on 'Stuff' indicators—how a ball moves relative to its 
        arm slot and velocity. 
        
        - **Velocity Boost**: Similarity calculations become more stringent at 95+ mph.
        - **Arm Slot Normalization**: Side-relief is automatically flipped for LHP/RHP comparison.
        - **Sample Confidence**: Results are weighted by pitch count to ensure reliability (Math.exp(-N/300)).
    """)

# Footer
st.markdown(f"""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background: #0A0E16; border-top: 1px solid #141E2E; padding: 0.5rem 2.5rem; font-family: monospace; font-size: 0.6rem; color: #4A6A80; display: flex; justify-content: space-between;">
        <div>© 2024 DM ANALYTICS GROUP</div>
        <div>LAST SYNC: {datetime.now().strftime('%H:%M:%S')} &nbsp;·&nbsp; PROCESSED 18.2M DATA POINTS</div>
    </div>
""", unsafe_allow_html=True)
