"""
score_zone_stuff.py — Inference module for zone-conditional Stuff+
=========================================================================
Given a pitch's shape features and arsenal context, returns the predicted
Stuff+ for every (zone, platoon) combination — 26 zones × 2 platoons = 52
predictions per pitch.

Usage from app.py:
    from score_zone_stuff import load_zone_model, score_zone_grid
    bundle = load_zone_model("models/zone_stuff_v1.joblib")
    grid = score_zone_grid(bundle, shape_row, pitcher_hand="R")
    # grid is dict with keys 'vs_rhb' and 'vs_lhb', each a dict
    # mapping zone_int (1..26) → stuff_plus_value.

The `shape_row` is a dict containing the 27 v5e shape features. Arsenal
features can be NaN (LightGBM handles them).
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


def load_zone_model(model_path="models/zone_stuff_v1.joblib"):
    """Load and validate the zone-Stuff+ bundle."""
    p = Path(model_path)
    if not p.exists():
        return None
    try:
        b = joblib.load(p)
    except Exception as e:
        print(f"  ! Failed to load zone-Stuff+ bundle: {e}")
        return None
    # Validate the bundle has what we need
    required = ["model", "features", "cat_features", "group_to_int",
                 "norms", "per_type_scalers", "fallback_scaler"]
    missing = [k for k in required if k not in b]
    if missing:
        print(f"  ! Zone-Stuff+ bundle missing keys: {missing}")
        return None
    return b


# Zone list: 1-26 (1-9 inner, 11-26 outer, no zone 10)
ALL_ZONES = list(range(1, 10)) + list(range(11, 27))


def _apply_per_type_scaling(X_df, pt_int_arr, scalers, fallback_scaler,
                              features, cat_features):
    """Apply per-pitch-type RobustScaler to continuous features."""
    cont_idx = [i for i, c in enumerate(features) if c not in cat_features]
    X = X_df.values.astype(np.float64)
    X_out = X.copy()
    for grp_int in np.unique(pt_int_arr):
        mask = pt_int_arr == grp_int
        scaler = scalers.get(int(grp_int), fallback_scaler)
        sub_cont = X[mask][:, cont_idx]
        sub_scaled = scaler.transform(sub_cont)
        rows = np.where(mask)[0]
        for i, col in enumerate(cont_idx):
            X_out[rows, col] = sub_scaled[:, i]
    return X_out


def score_zone_grid(bundle, shape_row, pitcher_hand="R"):
    """Predict Stuff+ for every (zone, platoon) combination for a given pitch.

    Parameters
    ----------
    bundle : dict
        Output of load_zone_model().
    shape_row : dict
        Per-pitch features. Must contain the 27 v5e shape features at
        minimum. Arsenal-context features can be NaN.
    pitcher_hand : str
        "R" or "L" — the pitcher's throwing hand. Used to determine
        is_same_hand for each batter platoon.

    Returns
    -------
    dict
        {
            "vs_rhb": {1: 102.3, 2: 95.4, ..., 26: 88.1},  # 26 zones
            "vs_lhb": {1: 110.1, 2: 89.5, ..., 26: 92.4},
        }
        Values are on the Stuff+ scale (mean 100, sd 10).
    """
    if bundle is None:
        return None

    FEATURES     = bundle["features"]
    CAT_FEATURES = bundle["cat_features"]
    NORMS        = bundle["norms"]
    model        = bundle["model"]
    per_type     = bundle["per_type_scalers"]
    fallback     = bundle["fallback_scaler"]

    # Build a row for every (zone, is_same_hand) combination — 52 rows total
    rows = []
    keys = []  # list of (platoon_key, zone_int)
    for platoon_key, batter_hand in (("vs_rhb", "R"), ("vs_lhb", "L")):
        is_same_hand = 1 if pitcher_hand == batter_hand else 0
        for zone in ALL_ZONES:
            r = dict(shape_row)  # shallow copy of all shape features
            r["zone_int"] = zone
            r["is_same_hand"] = is_same_hand
            # Compute interaction features
            r["ivb_x_zone"]        = r.get("ivb_in", 0)      * zone
            r["hb_x_zone"]         = r.get("hb_arm_in", 0)   * zone
            r["velo_x_zone"]       = r.get("start_speed", 0) * zone
            r["rel_height_x_zone"] = r.get("rel_height", 0)  * zone
            rows.append(r)
            keys.append((platoon_key, zone))

    df = pd.DataFrame(rows)
    # Ensure every feature column exists in df (NaN if not provided)
    for f in FEATURES:
        if f not in df.columns:
            df[f] = np.nan
    X = df[FEATURES]
    pt_int = df["pitch_type_int"].to_numpy()

    X_scaled = _apply_per_type_scaling(X, pt_int, per_type, fallback,
                                          FEATURES, CAT_FEATURES)
    raw = model.predict(X_scaled)

    # Standardize per pitch type to Stuff+ scale (100±10).
    # Use the same pitch_type as the shape_row provides.
    pt = int(shape_row.get("pitch_type_int", -1))
    # Look up the group name from pt_int
    group_name = None
    for gname, gint in bundle["group_to_int"].items():
        if gint == pt:
            group_name = gname
            break
    if group_name and group_name in NORMS.get("by_type", {}):
        params = NORMS["by_type"][group_name]
    else:
        params = NORMS["overall"]
    m_, s_ = params["mean"], max(params["sd"], 1e-6)
    stuff_plus = 100.0 + ((raw - m_) / s_) * 10.0

    # Reshape into the output dict
    out = {"vs_rhb": {}, "vs_lhb": {}}
    for (plat_key, zone), val in zip(keys, stuff_plus):
        out[plat_key][zone] = round(float(val), 1)
    return out
