"""
score_dm_stuff_plus.py - Inference for DM Stuff+ v5c
=========================================================================
Bundle file: dm_stuff_plus_v5.joblib  (version "dm_stuff_plus_v5c")

v5c inference changes from v5a:
  - vaa/haa replaced with vaa_aa/haa_aa (residuals after release-geometry
    baseline). Bundle stores baselines dict; inference applies the same
    linear regression to derive residuals.
  - VAA sign convention fixed (descending ball -> negative VAA).
  - vaa_x_velo -> vaa_aa_x_velo (uses residual).
  - Slider/Sweeper reclassification applied at scoring time so pitches
    are scored using the correct per-type scaler.

Inherits from v5a:
  - 21 features
  - Per-pitch-type RobustScaler
  - No sign flip on predictions (target was negated in training)
  - No clipping
  - Single mean model
  - Production norms recalibration via dm_stuff_plus_v5_norms.json
"""

from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

_BUNDLE = None
_FEATURES = None
_CAT_FEATURES = None
_GROUP_TO_INT = None
_NORMS = None
_PER_TYPE_SCALERS = None
_FALLBACK_SCALER = None
_VAA_HAA_BASELINES = None   # v5c: dict for residualization


def _load(model_dir="models"):
    global _BUNDLE, _FEATURES, _CAT_FEATURES, _GROUP_TO_INT, _NORMS
    global _PER_TYPE_SCALERS, _FALLBACK_SCALER, _VAA_HAA_BASELINES
    p = Path(model_dir) / "dm_stuff_plus_v5.joblib"
    try:
        b = joblib.load(p)
        _BUNDLE       = b
        _FEATURES     = b["features"]
        _CAT_FEATURES = b["cat_features"]
        _GROUP_TO_INT = b["group_to_int"]
        _NORMS        = b["norms"]
        _PER_TYPE_SCALERS = b.get("per_type_scalers", {})
        _VAA_HAA_BASELINES = b.get("vaa_haa_baselines", {})
        _FALLBACK_SCALER  = b.get("fallback_scaler") or b.get("scaler")

        # Production-recalibrated norms override (saved by batch path)
        norms_override = Path(model_dir) / "dm_stuff_plus_v5_norms.json"
        if norms_override.exists():
            try:
                _NORMS = json.loads(norms_override.read_text())
                print(f"DM Stuff+ v5: using recalibrated norms from {norms_override.name}")
            except Exception as e:
                print(f"DM Stuff+ v5: norms override failed ({e}); using bundle norms")
        return True
    except Exception as e:
        print(f"DM Stuff+ v5 load failed: {e}")
        return False


# Median values used when a live single-pitch call is missing a feature.
# v5c: vaa/haa values are now in correct (negative-for-descending) convention.
MEDIANS = {
    "start_speed": 91.0, "spin_rate": 2300.0, "extension": 6.4,
    "ivb_in": 12.0, "hb_arm_in": 8.0,
    "vaa": -5.0, "haa": 0.0,
    "vaa_aa": 0.0, "haa_aa": 0.0,        # residuals — median is ~0 by construction
    "rel_height": 5.8, "rel_side_arm": -1.7,
    "velo_diff": -2.0, "ivb_diff": -3.0, "hb_diff": 4.0,
    "ssw_magnitude": 0.0,
    "vaa_aa_x_velo": 0.0,
    "rel_height_x_velo": 528.0, "rel_side_x_typeint": -5.1,
    "active_spin_rate": 2200.0, "rel_quadrant": -9.9,
}


def _v(val, key):
    if val is None or val is pd.NA:
        return MEDIANS.get(key, 0.0)
    try:
        f = float(val)
        if math.isnan(f): return MEDIANS.get(key, 0.0)
        return f
    except (TypeError, ValueError):
        return MEDIANS.get(key, 0.0)


def _compute_vaa_haa(velo, az, ax, vy0=None, vz0=None, vx0=None, hand="R"):
    """v5c: sign fixed so descending ball -> negative VAA (matches physics)."""
    if vy0 is None: vy0 = -velo * 1.467 * 0.985
    if vz0 is None: vz0 = -3.0
    if vx0 is None: vx0 = 4.0 if hand == "R" else -4.0
    Y0, YF = 50.0, 17.0/12.0
    vy_f = -math.sqrt(max(vy0**2 - 2*(-25.0)*(Y0-YF), 0.01))
    t = (vy_f - vy0) / -25.0
    vz_f = vz0 + az * t
    vx_f = vx0 + ax * t
    # v5c FIX: no negation. Descending ball (vz_f < 0) -> negative VAA.
    vaa = math.degrees(math.atan(vz_f / abs(vy_f)))
    haa = math.degrees(math.atan(vx_f / abs(vy_f)))
    if hand == "L": haa = -haa
    return vaa, haa


def _apply_vaa_haa_residual(vaa, haa, rel_height, rel_side_arm, pitch_type_int, is_lefty):
    """Apply v5c baselines to get vaa_aa and haa_aa residuals.
    Falls back to raw vaa/haa if no baseline available for this cell.
    """
    if not _VAA_HAA_BASELINES:
        return vaa, haa
    bl_v = _VAA_HAA_BASELINES.get(("vaa", int(pitch_type_int), int(is_lefty)))
    bl_h = _VAA_HAA_BASELINES.get(("haa", int(pitch_type_int), int(is_lefty)))
    vaa_aa = (vaa - (bl_v[0] + bl_v[1] * rel_height)) if bl_v else vaa
    haa_aa = (haa - (bl_h[0] + bl_h[1] * rel_side_arm)) if bl_h else haa
    return vaa_aa, haa_aa


def _compute_ssw(ivb_in, hb_arm_in, spin_axis, hand="R"):
    if spin_axis is None or (isinstance(spin_axis, float) and math.isnan(spin_axis)):
        return 0.0
    pfx_x_ft = (hb_arm_in / 12.0) * (1 if hand == "R" else -1)
    pfx_z_ft = ivb_in / 12.0
    axis_rad = math.radians(spin_axis)
    total = math.sqrt(pfx_x_ft**2 + pfx_z_ft**2)
    pred_x = total * math.sin(axis_rad)
    pred_z = total * -math.cos(axis_rad)
    ssw_x = pfx_x_ft - pred_x
    ssw_z = pfx_z_ft - pred_z
    return min(math.sqrt(ssw_x**2 + ssw_z**2), 1.0)


def _compute_active_spin(spin_rate, ssw_magnitude, pfx_x_ft, pfx_z_ft):
    """v5: active (transverse) spin rate. Removes gyro component."""
    if spin_rate is None or (isinstance(spin_rate, float) and math.isnan(spin_rate)):
        return MEDIANS["active_spin_rate"]
    total_break = math.sqrt(pfx_x_ft**2 + pfx_z_ft**2)
    if total_break < 0.01:
        return float(spin_rate)
    ssw_frac = min(max(ssw_magnitude / total_break, 0.0), 1.0)
    return float(spin_rate) * (1.0 - ssw_frac)


def _build_row(velo, ivb, hb_arm, spin_rate, extension, rel_height, rel_side_arm,
               vaa, haa, pitch_group, hand, is_same_hand,
               primary_velo, primary_ivb, primary_hb, ssw_magnitude=0.0):
    # v5c: reclassify shape-ambiguous breaking balls before mapping pitch_type_int.
    # This must mirror the training-time reclassification so per-type scalers
    # are correctly chosen.
    if pitch_group == "Slider" and hb_arm >= 10.0 and velo <= 87.0:
        pitch_group = "Sweeper"
    elif pitch_group == "Sweeper" and hb_arm <= 8.0:
        pitch_group = "Slider"

    pt_int = _GROUP_TO_INT.get(pitch_group, 0)
    is_lefty = 1 if hand == "L" else 0
    velo_diff = velo - primary_velo if primary_velo is not None else MEDIANS["velo_diff"]
    ivb_diff  = ivb  - primary_ivb  if primary_ivb  is not None else MEDIANS["ivb_diff"]
    hb_diff   = hb_arm - primary_hb if primary_hb   is not None else MEDIANS["hb_diff"]

    # v5c: compute VAA-AA and HAA-AA residuals
    vaa_aa, haa_aa = _apply_vaa_haa_residual(vaa, haa, rel_height, rel_side_arm,
                                               pt_int, is_lefty)

    # v5c: vaa_aa_x_velo replaces vaa_x_velo
    vaa_aa_x_velo     = vaa_aa * velo
    rel_height_x_velo = rel_height * velo
    rel_side_x_typeint = rel_side_arm * pt_int

    pfx_x_ft = (hb_arm / 12.0) * (1 if hand == "R" else -1)
    pfx_z_ft = ivb / 12.0
    active_spin_rate = _compute_active_spin(spin_rate, ssw_magnitude, pfx_x_ft, pfx_z_ft)
    rel_quadrant     = rel_height * rel_side_arm

    return {
        "start_speed":   velo,
        "spin_rate":     spin_rate,
        "ivb_in":        ivb,
        "hb_arm_in":     hb_arm,
        "vaa_aa":        vaa_aa,
        "haa_aa":        haa_aa,
        "rel_height":    rel_height,
        "rel_side_arm":  rel_side_arm,
        "extension":     extension,
        "velo_diff":     velo_diff,
        "ivb_diff":      ivb_diff,
        "hb_diff":       hb_diff,
        "pitch_type_int": pt_int,
        "is_lefty":      is_lefty,
        "is_same_hand":  is_same_hand,
        "ssw_magnitude": ssw_magnitude,
        "vaa_aa_x_velo":      vaa_aa_x_velo,
        "rel_height_x_velo":  rel_height_x_velo,
        "rel_side_x_typeint": rel_side_x_typeint,
        "active_spin_rate":   active_spin_rate,
        "rel_quadrant":       rel_quadrant,
    }


def _apply_per_type_scaling(X_df, pitch_type_int_arr):
    """Apply per-pitch-type RobustScaler to continuous features only;
    categoricals pass through unscaled."""
    cont_idx = [i for i, c in enumerate(_FEATURES) if c not in _CAT_FEATURES]
    X = X_df.values.astype(np.float64)
    X_out = X.copy()
    for grp_int in np.unique(pitch_type_int_arr):
        mask = pitch_type_int_arr == grp_int
        scaler = (_PER_TYPE_SCALERS or {}).get(int(grp_int), _FALLBACK_SCALER)
        if scaler is None:
            continue
        sub_cont = X[mask][:, cont_idx]
        sub_scaled = scaler.transform(sub_cont)
        rows = np.where(mask)[0]
        for i, col in enumerate(cont_idx):
            X_out[rows, col] = sub_scaled[:, i]
    return X_out


def _standardize_with_norms(score_raw, pitch_groups, norms):
    out = np.full_like(score_raw, 100.0, dtype=float)
    by_type = norms.get("by_type", {})
    overall = norms.get("overall", {"mean": 0.0, "sd": 1.0})
    for i, grp in enumerate(pitch_groups):
        params = by_type.get(grp, overall)
        m = params["mean"]; s = params["sd"]
        out[i] = 100.0 + ((score_raw[i] - m) / max(s, 1e-6)) * 10.0
    return out


def _standardize(raw_pred, pitch_groups=None):
    """v5: no sign flip — predictions are already in pitcher-positive convention."""
    score_raw = raw_pred
    if pitch_groups is None:
        m = _NORMS["overall"]["mean"]; s = _NORMS["overall"]["sd"]
        return 100.0 + ((score_raw - m) / max(s, 1e-6)) * 10.0
    return _standardize_with_norms(score_raw, pitch_groups, _NORMS)


def score_dm_stuff_plus(pitches, rel_height=5.8, rel_side=-1.7,
                        extension=6.4, hand="R", model_dir="models"):
    """Live single-arsenal scoring."""
    if _BUNDLE is None and not _load(model_dir):
        return {}
    rel_side_arm = -abs(rel_side) if hand == "L" else rel_side
    is_same_hand = 0

    FB_PRIORITY = ["4-Seam", "2-Seam/Sinker", "Cutter"]
    primary_velo = primary_ivb = primary_hb = None
    for fb in FB_PRIORITY:
        if fb in pitches and pitches[fb].get("velo") is not None:
            primary_velo = float(pitches[fb]["velo"])
            primary_ivb  = float(pitches[fb].get("ivb", MEDIANS["ivb_in"]))
            _hb = pitches[fb].get("hb")
            primary_hb = float(_hb) if _hb is not None else MEDIANS["hb_arm_in"]
            break
    if primary_velo is None:
        best = max(pitches.items(), key=lambda kv: kv[1].get("velo", 0) or 0,
                   default=(None, {}))
        if best[1]:
            primary_velo = float(best[1].get("velo", 91.0))
            primary_ivb  = float(best[1].get("ivb", MEDIANS["ivb_in"]))
            _hb = best[1].get("hb")
            primary_hb = float(_hb) if _hb is not None else MEDIANS["hb_arm_in"]

    rows, keys = [], []
    for grp, m in pitches.items():
        if grp not in _GROUP_TO_INT: continue
        velo = m.get("velo")
        if velo is None: continue
        velo = float(velo)
        ivb    = _v(m.get("ivb"), "ivb_in")
        hb_arm = _v(m.get("hb"),  "hb_arm_in")
        spin   = _v(m.get("spin_rate"), "spin_rate")
        spin_axis = m.get("spin_axis")

        if all(m.get(k) is not None for k in ("az", "ax", "vy0", "vz0", "vx0")):
            vaa, haa = _compute_vaa_haa(velo, m["az"], m["ax"],
                                        m["vy0"], m["vz0"], m["vx0"], hand=hand)
        elif all(m.get(k) is not None for k in ("az", "ax")):
            vaa, haa = _compute_vaa_haa(velo, m["az"], m["ax"], hand=hand)
        else:
            vaa = MEDIANS["vaa"]; haa = MEDIANS["haa"]

        ssw_mag = _compute_ssw(ivb, hb_arm, spin_axis, hand=hand) if spin_axis else 0.0

        row = _build_row(
            velo=velo, ivb=ivb, hb_arm=hb_arm,
            spin_rate=spin, extension=_v(extension, "extension"),
            rel_height=_v(rel_height, "rel_height"),
            rel_side_arm=rel_side_arm,
            vaa=vaa, haa=haa,
            pitch_group=grp, hand=hand, is_same_hand=is_same_hand,
            primary_velo=primary_velo, primary_ivb=primary_ivb, primary_hb=primary_hb,
            ssw_magnitude=ssw_mag,
        )
        rows.append(row)
        keys.append(grp)

    if not rows: return {}

    df = pd.DataFrame(rows)[_FEATURES]
    pt_int = df["pitch_type_int"].to_numpy()
    X_scaled = _apply_per_type_scaling(df, pt_int)
    raw = _BUNDLE["model"].predict(X_scaled)
    scores = _standardize(raw, keys)
    return {k: round(float(s), 1) for k, s in zip(keys, scores)}


def score_dm_stuff_plus_batch(df_pitches, model_dir="models", recalibrate=True):
    """Batch scoring with auto-recalibration of norms from production data.

    v5c changes:
      - Slider/Sweeper reclassification applied at pitch level (mirrors training)
      - vaa_aa / haa_aa / vaa_aa_x_velo computed from baselines
    """
    if _BUNDLE is None and not _load(model_dir):
        return pd.Series(np.nan, index=df_pitches.index)

    df = df_pitches.copy()

    # v5c: reclassify shape-ambiguous breaking balls BEFORE mapping pitch_type_int
    if "pitch_group" in df.columns and "hb_arm_in" in df.columns and "start_speed" in df.columns:
        to_sw = ((df["pitch_group"] == "Slider") &
                  (df["hb_arm_in"] >= 10.0) &
                  (df["start_speed"] <= 87.0))
        to_sl = ((df["pitch_group"] == "Sweeper") &
                  (df["hb_arm_in"] <= 8.0))
        if to_sw.any(): df.loc[to_sw, "pitch_group"] = "Sweeper"
        if to_sl.any(): df.loc[to_sl, "pitch_group"] = "Slider"

    df["pitch_type_int"] = df["pitch_group"].map(_GROUP_TO_INT)
    df["is_lefty"]       = (df["p_throws"] == "L").astype(int)
    if "is_same_hand" not in df.columns:
        df["is_same_hand"] = 0
    if "ssw_magnitude" not in df.columns:
        df["ssw_magnitude"] = 0.0

    # v5c: compute VAA/HAA residuals from baselines (if available)
    # Profile data already has vaa/haa columns; apply baselines vectorized.
    vaa_arr = df["vaa"].to_numpy(dtype=np.float64).copy()
    haa_arr = df["haa"].to_numpy(dtype=np.float64).copy()
    rh_arr  = df["rel_height"].to_numpy(dtype=np.float64)
    rs_arr  = df["rel_side_arm"].to_numpy(dtype=np.float64)
    pt_arr_full = df["pitch_type_int"].to_numpy()
    lt_arr_full = df["is_lefty"].to_numpy()
    vaa_aa_arr = vaa_arr.copy()
    haa_aa_arr = haa_arr.copy()
    if _VAA_HAA_BASELINES:
        for (kind, pt_int, is_lefty), (intercept, slope) in _VAA_HAA_BASELINES.items():
            mask = (pt_arr_full == pt_int) & (lt_arr_full == is_lefty)
            if not mask.any(): continue
            if kind == "vaa":
                vaa_aa_arr[mask] = vaa_arr[mask] - (intercept + slope * rh_arr[mask])
            else:  # haa
                haa_aa_arr[mask] = haa_arr[mask] - (intercept + slope * rs_arr[mask])
    df["vaa_aa"] = vaa_aa_arr
    df["haa_aa"] = haa_aa_arr

    # v5c interaction (replaces v4's vaa_x_velo)
    df["vaa_aa_x_velo"]      = df["vaa_aa"] * df["start_speed"]
    df["rel_height_x_velo"]  = df["rel_height"] * df["start_speed"]
    df["rel_side_x_typeint"] = df["rel_side_arm"] * df["pitch_type_int"]

    # v5 NEW features
    pfx_x_ft = (df["hb_arm_in"] / 12.0) * np.where(df["is_lefty"] == 1, -1, 1)
    pfx_z_ft = df["ivb_in"] / 12.0
    total_break = np.sqrt(pfx_x_ft**2 + pfx_z_ft**2).clip(lower=0.01)
    ssw_frac    = (df["ssw_magnitude"] / total_break).clip(lower=0.0, upper=1.0)
    df["active_spin_rate"] = df["spin_rate"].fillna(MEDIANS["active_spin_rate"]) * (1.0 - ssw_frac)
    df["rel_quadrant"]     = df["rel_height"] * df["rel_side_arm"]

    needed = list(_FEATURES)
    valid_mask = df["pitch_type_int"].notna() & df[needed].notna().all(axis=1)
    if not valid_mask.any():
        return pd.Series(np.nan, index=df_pitches.index)

    sub = df[valid_mask].copy()
    pt_int = sub["pitch_type_int"].to_numpy()
    X_scaled = _apply_per_type_scaling(sub[_FEATURES], pt_int)
    raw = _BUNDLE["model"].predict(X_scaled)
    # v5: no negation — predictions already in pitcher-positive convention
    score_raw = raw
    pitch_groups = sub["pitch_group"].tolist()

    if recalibrate:
        prod_norms = {"overall": {"mean": float(np.mean(score_raw)),
                                   "sd":   float(np.std(score_raw))},
                       "by_type": {}}
        groups_arr = np.array(pitch_groups)
        for grp in set(pitch_groups):
            mask = groups_arr == grp
            if mask.sum() >= 30:
                prod_norms["by_type"][grp] = {
                    "mean": float(np.mean(score_raw[mask])),
                    "sd":   float(np.std(score_raw[mask])),
                    "n":    int(mask.sum()),
                }
            elif grp in _NORMS.get("by_type", {}):
                prod_norms["by_type"][grp] = _NORMS["by_type"][grp]

        try:
            override_path = Path(model_dir) / "dm_stuff_plus_v5_norms.json"
            override_path.write_text(json.dumps(prod_norms, indent=2))
            print(f"DM Stuff+ v5: saved recalibrated norms to {override_path.name}")
            globals()["_NORMS"] = prod_norms
        except Exception as e:
            print(f"DM Stuff+ v5: could not save norms override ({e})")

        scores = _standardize_with_norms(score_raw, pitch_groups, prod_norms)
    else:
        scores = _standardize(raw, pitch_groups)

    out = pd.Series(np.nan, index=df_pitches.index, dtype=float)
    out.loc[sub.index] = scores
    return out
