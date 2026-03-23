"""
Run this ONCE locally to build the pitcher profiles CSV.
Commit the output file to your GitHub repo alongside app.py.

    pip install pybaseball pandas numpy
    python build_profiles.py

Output: pitcher_profiles.csv
"""

import warnings
import numpy as np
import pandas as pd
from pybaseball import statcast, cache as pb_cache

warnings.filterwarnings("ignore")
pb_cache.enable()

STATCAST_YEARS = list(range(2017, 2025))

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

all_profiles = []

for year in STATCAST_YEARS:
    print(f"\n{'='*50}\n  Fetching {year}...\n{'='*50}")

    raw = statcast(start_dt=f"{year}-03-20", end_dt=f"{year}-11-01")
    print(f"  Raw rows: {len(raw):,}")

    keep = ["player_name", "pitch_type", "p_throws",
            "release_speed", "release_spin_rate",
            "pfx_x", "pfx_z", "release_extension",
            "release_pos_x", "release_pos_z",
            # VAA / HAA source columns
            "vx0", "vy0", "vz0", "ax", "ay", "az",
            "plate_x", "plate_z",
            # Outcome columns for zone heatmaps
            "description", "type",
            "estimated_woba_using_speedangle",
            "launch_speed"]

    # Only keep columns that exist in this year's data
    keep = [c for c in keep if c in raw.columns]
    df = raw[keep].dropna(
        subset=["release_pos_x", "release_pos_z", "pitch_type", "p_throws"]
    ).copy()
    del raw

    df["pfx_x_norm"] = np.where(df["p_throws"] == "L", -df["pfx_x"], df["pfx_x"])
    df["ivb_in"]     = df["pfx_z"]      * 12
    df["hb_in"]      = df["pfx_x_norm"] * 12
    df["year"]       = year
    df["pitch_group"] = df["pitch_type"].map(code_to_group)

        # ── Compute VAA and HAA using correct physics (Pavlidis / Chamberlain) ──
    # Standard formula: propagate kinematics from y0=50 ft to yf=17/12 ft
    # (front edge of home plate, 17 inches from back).
    # vy_f = -sqrt(vy0² - 2*ay*(y0-yf))
    # t    = (vy_f - vy0) / ay
    # vz_f = vz0 + az*t       → VAA = -arctan(vz_f / vy_f) * 180/π
    # vx_f = vx0 + ax*t       → HAA = -arctan(vx_f / vy_f) * 180/π
    # The negative sign in both formulas is because vy_f is negative
    # (ball moving toward plate) so -arctan gives the conventional sign.
    # VAA for a typical 4-seam: ~-4° to -6° (negative = downward approach)
    # HAA for RHP 4-seam: slightly positive (moving glove-side from catcher view)

    Y0 = 50.0          # Statcast reference y (feet from back of plate)
    YF = 17.0 / 12.0   # front edge of home plate (17 inches = 1.417 ft)

    if all(c in df.columns for c in ["vx0","vy0","vz0","ax","ay","az"]):
        vy0 = df["vy0"]
        ay  = df["ay"]
        az  = df["az"]
        ax  = df["ax"]

        # vy at plate front using kinematic equation (always negative — toward plate)
        vy_f2 = vy0**2 - 2 * ay * (Y0 - YF)
        vy_f2 = vy_f2.clip(lower=0.01)   # guard against floating point negatives
        vy_f  = -np.sqrt(vy_f2)           # negative: ball moving in -y direction

        # Time from y=50 to front of plate
        t = (vy_f - vy0) / ay.replace(0, np.nan)
        t = t.fillna(0.4)                  # fallback for zero acceleration

        # Velocity components at plate
        vz_f = df["vz0"] + az * t
        vx_f = df["vx0"] + ax * t

        # VAA and HAA (degrees) — negative means downward/glove-side approach
        df["vaa"] = -np.degrees(np.arctan(vz_f / vy_f.abs()))
        df["haa"] = -np.degrees(np.arctan(vx_f / vy_f.abs()))
        has_angles = True
    else:
        df["vaa"] = np.nan
        df["haa"] = np.nan
        has_angles = False

    print(f"  VAA/HAA computed: {has_angles}")

    # ── Release profile ─────────────────────────────────────────────────
    rel = (
        df.groupby(["player_name", "year"])
        .agg(
            hand          = ("p_throws",          "first"),
            rel_height    = ("release_pos_z",     "mean"),
            rel_side      = ("release_pos_x",     "mean"),
            extension     = ("release_extension", "mean"),
            total_pitches = ("pitch_type",        "count"),
        )
        .reset_index()
    )
    rel = rel[rel["total_pitches"] >= MIN_PITCHER_PITCHES]

    # ── Per-pitch-type metrics ──────────────────────────────────────────
    total_by_pitcher = (
        df[df["pitch_group"].notna()]
        .groupby(["player_name", "year"])["pitch_type"]
        .count()
        .rename("total_typed")
    )
    pp = (
        df[df["pitch_group"].notna()]
        .groupby(["player_name", "year", "pitch_group"])
        .agg(
            velo  = ("release_speed", "mean"),
            ivb   = ("ivb_in",        "mean"),
            hb    = ("hb_in",         "mean"),
            vaa   = ("vaa",           "mean"),
            haa   = ("haa",           "mean"),
            n     = ("pitch_type",    "count"),
        )
        .reset_index()
    )
    pp = pp.join(total_by_pitcher, on=["player_name", "year"])
    pp["pct"] = pp["n"] / pp["total_typed"].clip(lower=1)
    # Minimum 50 pitches AND at least 1% usage to be included
    pp = pp[(pp["n"] >= MIN_PITCH_TYPE_N) & (pp["pct"] >= 0.01)]

    wide = pp.pivot_table(
        index=["player_name", "year"],
        columns="pitch_group",
        values=["velo", "ivb", "hb", "vaa", "haa", "n", "pct"],
    )
    wide.columns = [f"{m}_{g}" for m, g in wide.columns]
    wide = wide.reset_index()

    profiles = rel.merge(wide, on=["player_name", "year"], how="left")
    all_profiles.append(profiles)
    del df, pp, wide

    print(f"  Profiles: {len(profiles)} pitcher-seasons")

print("\nCombining all years...")
final = pd.concat(all_profiles, ignore_index=True)
print(f"Total pitcher-seasons: {len(final):,}")
print(f"Unique pitchers:       {final['player_name'].nunique():,}")


# ══════════════════════════════════════════════════════════════════════════════
# ZONE STATS — CSW%, xwOBA per pitcher-season-pitch_group-zone (inside 9 only)
#
# CSW% = (Called Strikes + Whiffs) / Total Pitches
#   Called strike:  description == "called_strike"
#   Whiff:          description in {"swinging_strike","swinging_strike_blocked",
#                                   "foul_tip"}
#   Foul tip counts as a whiff (bat touched ball but catcher held it = strike 3)
#   Foul balls do NOT count — the batter made contact
#
# xwOBA = estimated_woba_using_speedangle
#   Only defined on batted ball events (type == "X")
#   For non-contact outcomes this is NaN — we take mean over batted balls only
#   This is BABIP-neutral (uses Statcast xwOBA, not actual wOBA)
#
# Zone grid — 9 inside zones only (catcher's perspective, reading order):
#   1=up-in  2=up-mid  3=up-out   (from catcher: left=glove side for RHP)
#   4=mid-in 5=center  6=mid-out
#   7=dn-in  8=dn-mid  9=dn-out
#   Strike zone boundaries: plate_x in [-0.83, 0.83] ft
#                           plate_z in [1.5, 3.5] ft (average sz; batter-invariant)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding zone stats (CSW%, xwOBA by zone)...")

SZ_LEFT  = -0.83   # ft  (left edge from catcher view = RHP glove side)
SZ_RIGHT =  0.83   # ft
SZ_BOT   =  1.5    # ft
SZ_TOP   =  3.5    # ft

# x thirds (left→right from catcher)
X1 = SZ_LEFT  + (SZ_RIGHT - SZ_LEFT) / 3    # -0.277
X2 = SZ_LEFT  + (SZ_RIGHT - SZ_LEFT) * 2/3  #  0.277

# z thirds (bottom→top)
Z1 = SZ_BOT + (SZ_TOP - SZ_BOT) / 3         #  2.167
Z2 = SZ_BOT + (SZ_TOP - SZ_BOT) * 2/3       #  2.833

# CSW descriptions (called strike or whiff)
CSW_SET = {"called_strike", "swinging_strike",
           "swinging_strike_blocked", "foul_tip"}

def assign_zone(px, pz):
    """
    Return zone 1-26 for a pitch location.
    Zones 1-9:  inside 3×3 strike zone grid
    Zones 11-26: surrounding 16-cell outer grid (5×5 minus 3×3 center)
    Returns None only for extreme outliers beyond the outer grid boundary.
    """
    # Outer grid extends 1 cell-width beyond strike zone on each side
    # Cell widths in each dimension:
    #   Inner zone: 1.66/3 ft wide (~0.553 ft), 2.0/3 ft tall (~0.667 ft)
    # Outer cells use same dimensions
    CELL_W = (SZ_RIGHT - SZ_LEFT) / 3      # ~0.553 ft
    CELL_H = (SZ_TOP   - SZ_BOT)  / 3      # ~0.667 ft

    OUTER_LEFT  = SZ_LEFT  - CELL_W
    OUTER_RIGHT = SZ_RIGHT + CELL_W
    OUTER_BOT   = SZ_BOT   - CELL_H
    OUTER_TOP   = SZ_TOP   + CELL_H

    if not (OUTER_LEFT <= px <= OUTER_RIGHT and OUTER_BOT <= pz <= OUTER_TOP):
        return None  # completely outside tracking area

    # Determine column (0-4) and row (0-4) in 5×5 grid
    # col: 0=far-left, 1=inner-left, 2=inner-mid, 3=inner-right, 4=far-right
    if   px < SZ_LEFT:   gc = 0
    elif px < X1:        gc = 1
    elif px < X2:        gc = 2
    elif px < SZ_RIGHT:  gc = 3
    else:                gc = 4

    # row: 0=top-out, 1=inner-top, 2=inner-mid, 3=inner-bot, 4=bot-out
    # (SVG row 0 = top, so high pz = low row)
    if   pz >= SZ_TOP:   gr = 0
    elif pz >= Z2:       gr = 1
    elif pz >= Z1:       gr = 2
    elif pz >= SZ_BOT:   gr = 3
    else:                gr = 4

    # Inner 3×3 → zones 1-9
    if 1 <= gc <= 3 and 1 <= gr <= 3:
        inner_col = gc - 1   # 0,1,2
        inner_row = gr - 1   # 0,1,2
        return inner_row * 3 + inner_col + 1  # 1-9

    # Outer cells → zones 11-26
    # Map (gr, gc) to zone id using lookup table
    OUTER_MAP = {
        # top row (gr=0)
        (0,0):11, (0,1):12, (0,2):13, (0,3):14, (0,4):15,
        # mid rows: left col (gc=0) and right col (gc=4)
        (1,0):16, (2,0):17, (3,0):18,
        (1,4):19, (2,4):20, (3,4):21,
        # bottom row (gr=4)
        (4,0):22, (4,1):23, (4,2):24, (4,3):25, (4,4):26,
    }
    return OUTER_MAP.get((gr, gc))

zone_frames_all = []

for year in STATCAST_YEARS:
    print(f"  Zone stats {year}...")
    raw = statcast(start_dt=f"{year}-03-20", end_dt=f"{year}-11-01")

    needed = ["player_name", "pitch_type", "p_throws",
              "plate_x", "plate_z",
              "description", "type",
              "estimated_woba_using_speedangle"]
    needed = [c for c in needed if c in raw.columns]
    zdf = raw[needed].dropna(subset=["plate_x","plate_z","pitch_type","p_throws"]).copy()
    del raw

    zdf["year"] = year
    zdf["pitch_group"] = zdf["pitch_type"].map(code_to_group)
    zdf = zdf[zdf["pitch_group"].notna()]

    # ── CSW flag (every pitch gets 0 or 1) ──────────────────────────────
    zdf["is_csw"] = zdf["description"].isin(CSW_SET).astype(int)

    # ── xwOBA (only on batted balls; NaN otherwise) ──────────────────────
    zdf["xwoba"] = pd.to_numeric(
        zdf.get("estimated_woba_using_speedangle", pd.Series(dtype=float)),
        errors="coerce"
    )
    # Ensure xwoba is NaN for non-batted-ball pitches
    if "type" in zdf.columns:
        not_bip = zdf["type"].astype(str) != "X"
        zdf.loc[not_bip, "xwoba"] = np.nan

    # ── Assign zones (1-9 inside, 11-26 outside) ────────────────────────
    zdf["zone"] = [assign_zone(px, pz)
                   for px, pz in zip(zdf["plate_x"], zdf["plate_z"])]
    zdf = zdf[zdf["zone"].notna()]
    zdf["zone"] = zdf["zone"].astype(int)

    # ── Aggregate per pitcher-year-pitch_group-zone ──────────────────────
    grp = zdf.groupby(["player_name","year","pitch_group","zone"]).agg(
        n_pitches  = ("is_csw",   "count"),   # total pitches in zone
        csw_count  = ("is_csw",   "sum"),     # CSW count
        xwoba_mean = ("xwoba",    "mean"),    # mean xwOBA on contact
    ).reset_index()

    grp["csw_pct"] = grp["csw_count"] / grp["n_pitches"].clip(lower=1)

    # Minimum 10 pitches per zone for reliability
    grp = grp[grp["n_pitches"] >= 10]
    zone_frames_all.append(grp[["player_name","year","pitch_group","zone",
                                 "n_pitches","csw_pct","xwoba_mean"]])
    print(f"    {len(grp):,} pitcher-pitch-zone rows")

zone_stats = pd.concat(zone_frames_all, ignore_index=True)
zone_stats["zone"] = zone_stats["zone"].astype(int)
zone_stats.to_csv("pitch_zone_stats.csv", index=False)
print(f"\n✓ Saved pitch_zone_stats.csv — {len(zone_stats):,} rows")
print(f"  CSW% range: {zone_stats['csw_pct'].min():.3f} – {zone_stats['csw_pct'].max():.3f}")
print(f"  xwOBA range: {zone_stats['xwoba_mean'].dropna().min():.3f} – {zone_stats['xwoba_mean'].dropna().max():.3f}")


# ── Fetch FanGraphs Stuff+ via direct REST API (type=36 pitch-specific leaderboard) ──
# FanGraphs hosts Stuff+ at the pitch-specific leaderboard endpoint, NOT in pitching_stats().
# Available 2020+ (model backdated to 2020 when launched in 2023).
#
# API endpoint:
#   https://www.fangraphs.com/api/leaders/pitch-specific/data
#   ?pos=all&stats=pit&lg=all&qual=1&type=36&season={yr}&season1={yr}&ind=0&rost=0&team=0
#
# Key columns returned (inspect printed output if names change):
#   "PlayerName" or "Name"  — pitcher name
#   "Stf+"                  — overall Stuff+ (100=avg, >100=better)
#   "ff_Stf+"               — 4-Seam Stuff+
#   "si_Stf+"               — 2-Seam/Sinker Stuff+
#   "fc_Stf+"               — Cutter Stuff+
#   "sl_Stf+"               — Slider Stuff+
#   "st_Stf+"               — Sweeper Stuff+
#   "cu_Stf+"               — Curveball Stuff+
#   "fs_Stf+"               — Splitter Stuff+
#   "ch_Stf+"               — Changeup Stuff+

import requests, time

print("\nFetching FanGraphs Stuff+ via REST API (type=36)...")

def normalize_name(n):
    """Convert 'Last, First' → 'First Last' for matching with Statcast names."""
    if "," in str(n):
        parts = str(n).split(",", 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return str(n).strip()

FG_API_URL = (
    "https://www.fangraphs.com/api/leaders/pitch-specific/data"
    "?pos=all&stats=pit&lg=all&qual=1&type=36"
    "&season={yr}&season1={yr}&ind=0&rost=0&team=0&pageitems=10000&pagenum=1"
)
FG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.fangraphs.com/leaders/pitch-specific",
}

# Column name → our pitch group label
FG_PITCH_COLS = {
    "ff_Stf+":  "4-Seam",
    "si_Stf+":  "2-Seam/Sinker",
    "fc_Stf+":  "Cutter",
    "sl_Stf+":  "Slider",
    "st_Stf+":  "Sweeper",
    "cu_Stf+":  "Curveball",
    "fs_Stf+":  "Splitter",
    "ch_Stf+":  "Changeup",
}
# Possible overall Stuff+ col names (FG has changed this over time)
FG_OVERALL_NAMES = ["Stf+", "Stuff+", "sp_Stf", "spStf", "Overall Stf+"]

FG_YEARS = [yr for yr in STATCAST_YEARS if yr >= 2020]

fg_frames = []
for yr in FG_YEARS:
    try:
        url = FG_API_URL.format(yr=yr)
        resp = requests.get(url, headers=FG_HEADERS, timeout=20)
        resp.raise_for_status()
        payload = resp.json()

        # API returns {"data": [...]} or bare list
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not rows:
            print(f"  {yr}: empty response")
            continue

        fg = pd.DataFrame(rows)
        print(f"  {yr}: {len(fg)} rows, cols: {list(fg.columns)}")

        # Find pitcher name column
        name_col = next((c for c in fg.columns if str(c).lower() in
                         ("playername", "name", "pitcher", "player")), None)
        if name_col is None:
            print(f"  {yr}: no name column found, skipping")
            continue

        # Find overall Stuff+ column
        overall_col = next((c for c in fg.columns if str(c) in FG_OVERALL_NAMES), None)
        if overall_col is None:
            # Try case-insensitive
            overall_col = next((c for c in fg.columns
                                if "stf" in str(c).lower() and "+" in str(c)), None)

        # Find per-pitch Stuff+ columns
        per_pitch = {c: grp for c, grp in FG_PITCH_COLS.items() if c in fg.columns}

        if overall_col is None and not per_pitch:
            print(f"  {yr}: no Stuff+ columns found. All cols: {list(fg.columns)}")
            continue

        keep = [name_col]
        rename = {name_col: "fg_name"}
        if overall_col:
            keep.append(overall_col)
            rename[overall_col] = "stuff_plus"
        for fg_col, grp in per_pitch.items():
            keep.append(fg_col)
            rename[fg_col] = f"sp_{grp}"

        sub = fg[keep].copy().rename(columns=rename)
        sub["year"] = yr
        fg_frames.append(sub)
        print(f"  {yr}: overall={'yes' if overall_col else 'no'}, "
              f"per-pitch={list(per_pitch.keys())}")

        time.sleep(0.5)   # be polite to FG servers

    except Exception as e:
        print(f"  {yr}: failed — {e}")

if fg_frames:
    fg_all = pd.concat(fg_frames, ignore_index=True).fillna(np.nan)
    fg_all["name_norm"] = fg_all["fg_name"].apply(normalize_name)
    fg_all["year"]      = fg_all["year"].astype(int)
    final["name_norm"]  = final["player_name"].apply(normalize_name)
    final["year"]       = final["year"].astype(int)

    merge_cols = ["name_norm", "year"]
    if "stuff_plus" in fg_all.columns:
        merge_cols.append("stuff_plus")
    sp_cols = [c for c in fg_all.columns if c.startswith("sp_")]
    merge_cols += sp_cols

    final = final.merge(fg_all[merge_cols], on=["name_norm", "year"], how="left")
    final.drop(columns=["name_norm"], inplace=True)

    matched = final["stuff_plus"].notna().sum() if "stuff_plus" in final.columns else 0
    print(f"  Matched overall Stuff+ for {matched:,} / {len(final):,} pitcher-seasons")
    print(f"  Per-pitch cols added: {sp_cols}")
else:
    final["stuff_plus"] = float("nan")
    for grp in ["4-Seam","2-Seam/Sinker","Cutter","Slider","Sweeper","Curveball","Splitter","Changeup"]:
        final[f"sp_{grp}"] = float("nan")
    print("  No FanGraphs data fetched — all Stuff+ set to NaN")
    print("  → Check internet connection and re-run build_profiles.py")


out = "pitcher_profiles.csv"
final.to_csv(out, index=False)
mb = final.memory_usage(deep=True).sum() / 1e6
print(f"\n✓ Saved → {out}  ({mb:.1f} MB in-memory)")
print(f"\nCommit {out} to your GitHub repo alongside app.py")
