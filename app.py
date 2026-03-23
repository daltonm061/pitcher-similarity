"""
Run this ONCE locally to build the pitcher profiles and zone stats CSVs.
Commit the 'zone_stats/' folder and 'pitcher_profiles.csv' to your GitHub repo.

    pip install pybaseball pandas numpy requests
    python build_profiles.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import requests
import time
from pybaseball import statcast, cache as pb_cache

warnings.filterwarnings("ignore")
pb_cache.enable()

# --- Configuration ---
STATCAST_YEARS = list(range(2017, 2025))
FG_YEARS = [yr for yr in STATCAST_YEARS if yr >= 2020]

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

MIN_PITCHER_PITCHES = 100
MIN_PITCH_TYPE_N    = 50
code_to_group = {c: g for g, codes in PITCH_GROUPS.items() for c in codes}

# --- Physics Constants for VAA/HAA ---
Y0 = 50.0          # Statcast reference y (ft)
YF = 17.0 / 12.0   # Front edge of home plate (ft)

# --- Zone Definitions ---
SZ_LEFT, SZ_RIGHT = -0.83, 0.83
SZ_BOT, SZ_TOP    = 1.5, 3.5
X1 = SZ_LEFT  + (SZ_RIGHT - SZ_LEFT) / 3
X2 = SZ_LEFT  + (SZ_RIGHT - SZ_LEFT) * 2/3
Z1 = SZ_BOT + (SZ_TOP - SZ_BOT) / 3
Z2 = SZ_BOT + (SZ_TOP - SZ_BOT) * 2/3

CSW_SET = {"called_strike", "swinging_strike", "swinging_strike_blocked"}
WHIFF_SET = {"swinging_strike", "swinging_strike_blocked"}

def assign_zone(px, pz):
    CELL_W = (SZ_RIGHT - SZ_LEFT) / 3
    CELL_H = (SZ_TOP   - SZ_BOT)  / 3
    OUTER_LEFT, OUTER_RIGHT = SZ_LEFT - CELL_W, SZ_RIGHT + CELL_W
    OUTER_BOT, OUTER_TOP    = SZ_BOT - CELL_H, SZ_TOP + CELL_H

    if not (OUTER_LEFT <= px <= OUTER_RIGHT and OUTER_BOT <= pz <= OUTER_TOP):
        return None

    gc = 0 if px < SZ_LEFT else 1 if px < X1 else 2 if px < X2 else 3 if px < SZ_RIGHT else 4
    gr = 0 if pz >= SZ_TOP else 1 if pz >= Z2 else 2 if pz >= Z1 else 3 if pz >= SZ_BOT else 4

    if 1 <= gc <= 3 and 1 <= gr <= 3:
        return (gr - 1) * 3 + (gc - 1) + 1

    OUTER_MAP = {
        (0,0):11, (0,1):12, (0,2):13, (0,3):14, (0,4):15,
        (1,0):16, (2,0):17, (3,0):18, (1,4):19, (2,4):20, (3,4):21,
        (4,0):22, (4,1):23, (4,2):24, (4,3):25, (4,4):26,
    }
    return OUTER_MAP.get((gr, gc))

def normalize_name(n):
    if "," in str(n):
        parts = str(n).split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return str(n).strip()

def optimize_df(df):
    """Downcasts and rounds to reduce CSV footprint."""
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].round(3).astype(np.float32)
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# --- Main Logic ---
all_profiles = []
zone_frames_all = []

for year in STATCAST_YEARS:
    print(f"\nProcessing {year}...")
    raw = statcast(start_dt=f"{year}-03-20", end_dt=f"{year}-11-01")
    
    # 1. Physics Calculations (VAA/HAA)
    df = raw.dropna(subset=["release_pos_x", "release_pos_z", "pitch_type", "p_throws"]).copy()
    df["pfx_x_norm"] = np.where(df["p_throws"] == "L", -df["pfx_x"], df["pfx_x"])
    df["ivb_in"], df["hb_in"] = df["pfx_z"] * 12, df["pfx_x_norm"] * 12
    df["year"], df["pitch_group"] = year, df["pitch_type"].map(code_to_group)

    if all(c in df.columns for c in ["vx0","vy0","vz0","ax","ay","az"]):
        vy_f2 = df["vy0"]**2 - 2 * df["ay"] * (Y0 - YF)
        vy_f = -np.sqrt(vy_f2.clip(lower=0.01))
        t = (vy_f - df["vy0"]) / df["ay"].replace(0, np.nan).fillna(0.4)
        df["vaa"] = -np.degrees(np.arctan((df["vz0"] + df["az"] * t) / vy_f.abs()))
        df["haa"] = -np.degrees(np.arctan((df["vx0"] + df["ax"] * t) / vy_f.abs()))

    # 2. Profiles Aggregation
    rel = df.groupby(["player_name", "year"]).agg(
        hand=("p_throws", "first"), rel_height=("release_pos_z", "mean"),
        rel_side=("release_pos_x", "mean"), extension=("release_extension", "mean"),
        total_pitches=("pitch_type", "count")
    ).reset_index()
    rel = rel[rel["total_pitches"] >= MIN_PITCHER_PITCHES]

    pp = df[df["pitch_group"].notna()].groupby(["player_name", "year", "pitch_group"]).agg(
        velo=("release_speed", "mean"), ivb=("ivb_in", "mean"), hb=("hb_in", "mean"),
        vaa=("vaa", "mean"), haa=("haa", "mean"), n=("pitch_type", "count")
    ).reset_index()
    
    total_typed = df[df["pitch_group"].notna()].groupby(["player_name", "year"])["pitch_type"].count()
    pp = pp.join(total_typed.rename("total_typed"), on=["player_name", "year"])
    pp["pct"] = pp["n"] / pp["total_typed"].clip(lower=1)
    pp = pp[(pp["n"] >= MIN_PITCH_TYPE_N) & (pp["pct"] >= 0.01)]

    wide = pp.pivot_table(index=["player_name", "year"], columns="pitch_group", 
                          values=["velo", "ivb", "hb", "vaa", "haa", "n", "pct"])
    wide.columns = [f"{m}_{g}" for m, g in wide.columns]
    all_profiles.append(rel.merge(wide.reset_index(), on=["player_name", "year"], how="left"))

    # 3. Zone Stats Aggregation
    zdf = df[df["pitch_group"].notna()].copy()
    zdf["is_csw"] = zdf["description"].isin(CSW_SET).astype(int)
    zdf["is_whiff"] = zdf["description"].isin(WHIFF_SET).astype(int)
    zdf["xwoba"] = pd.to_numeric(zdf.get("estimated_woba_using_speedangle", np.nan), errors="coerce")
    zdf.loc[zdf["type"] != "X", "xwoba"] = np.nan
    zdf["zone"] = [assign_zone(px, pz) for px, pz in zip(zdf["plate_x"], zdf["plate_z"])]
    zdf = zdf[zdf["zone"].notna()]

    def get_zone_grp(data, label):
        g = data.groupby(["player_name", "year", "pitch_group", "zone"]).agg(
            n_pitches=("is_csw", "count"), csw_count=("is_csw", "sum"),
            whiff_count=("is_whiff", "sum"), xwoba_mean=("xwoba", "mean")
        ).reset_index()
        g["csw_pct"], g["whiff_pct"], g["stand"] = g["csw_count"]/g["n_pitches"], g["whiff_count"]/g["n_pitches"], label
        return g[g["n_pitches"] >= 3]

    zone_frames_all.append(get_zone_grp(zdf, "all"))
    zone_frames_all.append(get_zone_grp(zdf[zdf["stand"] == zdf["p_throws"]], "same"))
    zone_frames_all.append(get_zone_grp(zdf[zdf["stand"] != zdf["p_throws"]], "opp"))
    del raw, df, zdf

# --- Final Merge and Savings ---
final = pd.concat(all_profiles, ignore_index=True)
zone_stats = pd.concat(zone_frames_all, ignore_index=True)

# Save split zone stats
os.makedirs("zone_stats", exist_ok=True)
for yr in STATCAST_YEARS:
    yr_data = optimize_df(zone_stats[zone_stats["year"] == yr])
    if not yr_data.empty:
        yr_data.to_csv(f"zone_stats/pitch_zone_stats_{yr}.csv", index=False)

# Save main profile
optimize_df(final).to_csv("pitcher_profiles.csv", index=False)
print("\n✓ Profiles saved. Upload 'zone_stats/' and 'pitcher_profiles.csv' to GitHub.")
