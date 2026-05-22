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
_ARSENAL_CELL_STATS = None  # v5d: per-pitcher-year arsenal lookup DataFrame


_KMEANS_STATE = None    # v6++: k-means cluster centroids + scaler


def _load(model_dir="models"):
    global _BUNDLE, _FEATURES, _CAT_FEATURES, _GROUP_TO_INT, _NORMS
    global _PER_TYPE_SCALERS, _FALLBACK_SCALER, _VAA_HAA_BASELINES
    global _ARSENAL_CELL_STATS, _KMEANS_STATE
    # Prefer highest-version bundle; fall back through v8c → v8b → v8 → v6 → v5
    candidates = [
        ("v8c", Path(model_dir) / "dm_stuff_plus_v8c.joblib"),
        ("v8b", Path(model_dir) / "dm_stuff_plus_v8b.joblib"),
        ("v8",  Path(model_dir) / "dm_stuff_plus_v8.joblib"),
        ("v6",  Path(model_dir) / "dm_stuff_plus_v6.joblib"),
        ("v5",  Path(model_dir) / "dm_stuff_plus_v5.joblib"),
    ]
    p = None; tag = None
    for t, c in candidates:
        if c.exists():
            p = c; tag = t
            break
    if p is None:
        print("DM Stuff+ load failed: no bundle file found")
        return False
    try:
        b = joblib.load(p)
        _BUNDLE       = b
        _FEATURES     = b["features"]
        _CAT_FEATURES = b["cat_features"]
        _GROUP_TO_INT = b["group_to_int"]
        _NORMS        = b["norms"]
        _PER_TYPE_SCALERS = b.get("per_type_scalers", {})
        _VAA_HAA_BASELINES = b.get("vaa_haa_baselines", {})
        _ARSENAL_CELL_STATS = b.get("arsenal_cell_stats")
        # v8+ uses bb_kmeans_state + fb_kmeans_state; v6 uses kmeans_state
        _KMEANS_STATE = (b.get("bb_kmeans_state")  # v8+ breaking ball k-means
                        or b.get("kmeans_state"))  # v6 fallback (singular)

        # Only apply norms override if it matches the bundle version
        version_str = b.get("version", "")
        norms_filename = None
        if "v8" in version_str:
            norms_filename = "dm_stuff_plus_v8_norms.json"  # only if exists
        elif "v6" in version_str:
            norms_filename = "dm_stuff_plus_v6_norms.json"
        elif "v5" in version_str:
            norms_filename = "dm_stuff_plus_v5_norms.json"
        if norms_filename:
            norms_override = Path(model_dir) / norms_filename
            if norms_override.exists():
                try:
                    _NORMS = json.loads(norms_override.read_text())
                    print(f"DM Stuff+ {version_str}: using norms override {norms_filename}")
                except Exception as e:
                    print(f"DM Stuff+ {version_str}: norms override failed ({e}); using bundle norms")
        _FALLBACK_SCALER  = b.get("fallback_scaler") or b.get("scaler")
        print(f"DM Stuff+ loaded: {version_str} from {p.name}")
        return True
    except Exception as e:
        print(f"DM Stuff+ load failed: {e}")
        return False


def _compute_v6_features(spin_axis, spin_rate, velo, pitch_type_int, ivb, hb_arm,
                          spin_axis_sin=None, spin_axis_cos=None):
    """Compute v6++ features: spin_axis_sin/cos, bauer_units, sweeper_cluster_score.
    Returns dict (sweeper_cluster_score may be NaN if not breaking ball)."""
    import math as _m
    # spin axis decomposition
    if spin_axis_sin is None or spin_axis_cos is None:
        if spin_axis is None or (isinstance(spin_axis, float) and _m.isnan(spin_axis)):
            spin_axis_sin = 0.0
            spin_axis_cos = -1.0
        else:
            axis_rad = _m.radians(float(spin_axis))
            spin_axis_sin = _m.sin(axis_rad)
            spin_axis_cos = _m.cos(axis_rad)
    # Bauer Units
    safe_velo = max(float(velo), 60.0) if velo is not None else 90.0
    safe_spin = float(spin_rate) if spin_rate is not None and not (isinstance(spin_rate, float) and _m.isnan(spin_rate)) else 2300.0
    bauer = safe_spin / safe_velo
    # Sweeper cluster score
    sweeper_score = float("nan")
    if _KMEANS_STATE is not None and pitch_type_int in (
        _GROUP_TO_INT.get("Slider", -1), _GROUP_TO_INT.get("Sweeper", -1)
    ):
        try:
            cf = _KMEANS_STATE["cluster_feats"]
            means = _KMEANS_STATE["feature_means"]; stds = _KMEANS_STATE["feature_stds"]
            # Build vector using the SAME feature ordering as training
            _vals = {
                "start_speed": velo, "ivb_in": ivb, "hb_arm_in": hb_arm,
                "spin_rate": safe_spin,
                "spin_axis_sin": spin_axis_sin, "spin_axis_cos": spin_axis_cos,
            }
            vec = np.array([(_vals[c] - means[c]) / max(stds[c], 1e-6) for c in cf]).reshape(1, -1)
            centroids = _KMEANS_STATE["centroids"]
            dists = np.sqrt(((vec - centroids) ** 2).sum(axis=1))
            neg_d2 = -(dists ** 2)
            shifted = neg_d2 - neg_d2.max()
            probs = np.exp(shifted); probs = probs / probs.sum()
            sweeper_score = float(probs[_KMEANS_STATE["sweeper_cluster"]])
        except Exception:
            sweeper_score = float("nan")
    return {
        "spin_axis_sin":         spin_axis_sin,
        "spin_axis_cos":         spin_axis_cos,
        "bauer_units":           bauer,
        "sweeper_cluster_score": sweeper_score,
    }


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
               primary_velo, primary_ivb, primary_hb, ssw_magnitude=0.0,
               spin_axis=None):
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

    row = {
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
        # v5d: arsenal-context placeholders — filled by caller after all rows built
        "velo_diff_secondary":     float("nan"),
        "arsenal_size":            float("nan"),
        "arsenal_ivb_spread":      float("nan"),
        "arsenal_hb_spread":       float("nan"),
        "arsenal_ivb_max_other":   float("nan"),
        "arsenal_ivb_min_other":   float("nan"),
        "arsenal_hb_max_other":    float("nan"),
        "arsenal_hb_min_other":    float("nan"),
        "nearest_other_velo_diff": float("nan"),
        "nearest_other_ivb_diff":  float("nan"),
        "nearest_other_hb_diff":   float("nan"),
    }
    # v6++ features (added if model bundle expects them; otherwise harmless extras)
    if _FEATURES and any(f in _FEATURES for f in
                          ("spin_axis_sin", "spin_axis_cos", "bauer_units",
                           "sweeper_cluster_score")):
        v6_feats = _compute_v6_features(
            spin_axis=spin_axis, spin_rate=spin_rate, velo=velo,
            pitch_type_int=pt_int, ivb=ivb, hb_arm=hb_arm,
        )
        row.update(v6_feats)
    return row


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
            spin_axis=spin_axis,    # v6++
        )
        rows.append(row)
        keys.append(grp)

    if not rows: return {}

    # v5d: fill arsenal-context features now that all rows are built
    if len(rows) >= 2:
        # Build per-row arsenal lookup from the input pitches themselves
        velos = np.array([r["start_speed"] for r in rows])
        ivbs  = np.array([r["ivb_in"]      for r in rows])
        hbs   = np.array([r["hb_arm_in"]   for r in rows])
        pts   = np.array([r["pitch_type_int"] for r in rows])
        for i, r in enumerate(rows):
            r["arsenal_size"] = float(len(rows))
            # Others = all rows except those with same pitch_type_int as this one
            other_mask = pts != pts[i]
            if not other_mask.any():
                continue
            ov = velos[other_mask]; oi = ivbs[other_mask]; oh = hbs[other_mask]
            r["arsenal_ivb_spread"]    = float(oi.max() - oi.min())
            r["arsenal_hb_spread"]     = float(oh.max() - oh.min())
            r["arsenal_ivb_max_other"] = float(oi.max())
            r["arsenal_ivb_min_other"] = float(oi.min())
            r["arsenal_hb_max_other"]  = float(oh.max())
            r["arsenal_hb_min_other"]  = float(oh.min())
            my_v = velos[i]
            slower = ov[ov < my_v]
            if len(slower) > 0:
                r["velo_diff_secondary"] = float(my_v - slower.max())
            else:
                r["velo_diff_secondary"] = float(my_v - ov.max())
            gaps = np.abs(ov - my_v)
            j = int(gaps.argmin())
            r["nearest_other_velo_diff"] = float(my_v - ov[j])
            r["nearest_other_ivb_diff"]  = float(ivbs[i] - oi[j])
            r["nearest_other_hb_diff"]   = float(hbs[i]  - oh[j])
    elif len(rows) == 1:
        # Single-pitch arsenal: just set arsenal_size, rest stay NaN
        rows[0]["arsenal_size"] = 1.0

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

    # v8c: compute velo_diff/ivb_diff/hb_diff from primary FB if not supplied.
    # build_profiles supplies these directly, but standalone callers may not.
    if "velo_diff" not in df.columns or df["velo_diff"].isna().all():
        _FB_PRIORITY_GRPS = ["4-Seam", "2-Seam/Sinker", "Cutter"]
        # Per (pitcher OR player_name, year), find primary FB and compute diff
        _id_col = "pitcher" if "pitcher" in df.columns and df["pitcher"].notna().any() else "player_name"
        if _id_col in df.columns and "year" in df.columns:
            primary_lookup = {}
            for fb in _FB_PRIORITY_GRPS:
                fb_rows = df[df["pitch_group"] == fb]
                if len(fb_rows) > 0:
                    agg = fb_rows.groupby([_id_col, "year"]).agg(
                        primary_velo=("start_speed", "mean"),
                        primary_ivb=("ivb_in", "mean"),
                        primary_hb=("hb_arm_in", "mean"),
                    )
                    for k, v in agg.iterrows():
                        if k not in primary_lookup:
                            primary_lookup[k] = v
            df["_primary_velo"] = df.apply(
                lambda r: primary_lookup.get((r[_id_col], r["year"]), {}).get("primary_velo", r["start_speed"]),
                axis=1
            )
            df["velo_diff"] = df["start_speed"] - df["_primary_velo"]
            df = df.drop(columns=["_primary_velo"])
        else:
            df["velo_diff"] = 0.0

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

    # v6++/v8 features (computed conditionally on what the bundle expects)
    _bundle_feats = set(_FEATURES or [])

    # spin axis decomposition
    if any(f in _bundle_feats for f in ("spin_axis_sin", "spin_axis_cos")):
        if "spin_axis" in df.columns:
            axis_rad = np.radians(df["spin_axis"].fillna(180.0))
        else:
            axis_rad = np.full(len(df), np.radians(180.0))
        df["spin_axis_sin"] = np.sin(axis_rad)
        df["spin_axis_cos"] = np.cos(axis_rad)

    # Bauer Units
    if "bauer_units" in _bundle_feats:
        df["bauer_units"] = (df["spin_rate"].fillna(2300.0)
                              / df["start_speed"].clip(lower=60.0))

    # Cluster scores: training data hard-coded 1.0/0.0 for Statcast-tagged
    # pitches. At inference (where pitch_group is known from the input), use
    # hard 0/1 values rather than softmax probabilities — softmax distribution
    # is wildly off from training distribution and blows up scaling.
    slider_int  = _GROUP_TO_INT.get("Slider", -1)
    sweeper_int = _GROUP_TO_INT.get("Sweeper", -1)
    fourSeam_int = _GROUP_TO_INT.get("4-Seam", -1)
    sinker_int  = _GROUP_TO_INT.get("2-Seam/Sinker", -1)
    if "sweeper_cluster_score" in _bundle_feats:
        df["sweeper_cluster_score"] = np.where(
            df["pitch_type_int"] == sweeper_int, 1.0,
            np.where(df["pitch_type_int"] == slider_int, 0.0, np.nan)
        )
    if "four_seam_cluster_score" in _bundle_feats:
        df["four_seam_cluster_score"] = np.where(
            df["pitch_type_int"] == fourSeam_int, 1.0,
            np.where(df["pitch_type_int"] == sinker_int, 0.0, np.nan)
        )

    # v8c physics features
    if "arm_angle" in _bundle_feats:
        _vert = df["rel_height"] - 5.0
        _horiz = df["rel_side_arm"].abs().clip(lower=0.01)
        df["arm_angle"] = np.degrees(np.arctan2(_vert, _horiz))
    if "velo_x_typeint" in _bundle_feats:
        df["velo_x_typeint"] = df["start_speed"] * df["pitch_type_int"]
    if "rel_quadrant_x_velo" in _bundle_feats:
        df["rel_quadrant_x_velo"] = df["rel_quadrant"] * df["start_speed"]
    if "rel_quadrant_x_typeint" in _bundle_feats:
        df["rel_quadrant_x_typeint"] = df["rel_quadrant"] * df["pitch_type_int"]
    if "perceived_velo" in _bundle_feats:
        df["perceived_velo"] = df["start_speed"] * (1.0 + (df["extension"] - 6.5) / 55.0)
    if "velo_x_spin_rate" in _bundle_feats:
        df["velo_x_spin_rate"] = df["start_speed"] * df["spin_rate"].fillna(2300.0)
    if "velo_x_ivb" in _bundle_feats:
        df["velo_x_ivb"] = df["start_speed"] * df["ivb_in"]
    if "movement_angle_sin" in _bundle_feats or "movement_angle_cos" in _bundle_feats:
        _tm = np.sqrt(df["ivb_in"]**2 + df["hb_arm_in"]**2).clip(lower=0.01)
        df["movement_angle_sin"] = df["ivb_in"] / _tm
        df["movement_angle_cos"] = df["hb_arm_in"] / _tm
        df["total_movement"] = _tm
    elif "total_movement" in _bundle_feats:
        df["total_movement"] = np.sqrt(df["ivb_in"]**2 + df["hb_arm_in"]**2)
    if "ivb_per_spin" in _bundle_feats:
        _safe_spin = df["spin_rate"].fillna(2300.0).clip(lower=100.0)
        df["ivb_per_spin"] = df["ivb_in"] / _safe_spin * 1000.0
        df["hb_per_spin"]  = df["hb_arm_in"] / _safe_spin * 1000.0
    if "active_spin_pct" in _bundle_feats:
        _safe_spin = df["spin_rate"].fillna(2300.0).clip(lower=100.0)
        df["active_spin_pct"] = (df["active_spin_rate"] / _safe_spin).clip(lower=0.0, upper=1.0)

    # v5d: compute arsenal-context features.
    # The batch path can receive either:
    #   - Pitch-level data (with pitcher_id column) — group by pitcher+year
    #   - Profile-aggregated data (with player_name only) — group by player_name+year
    # The arsenal IS the input dataframe itself: each pitcher-year has multiple
    # rows (one per pitch type), so we just group adjacent rows.
    _ARSENAL_FEATS = [
        "velo_diff_secondary", "arsenal_size",
        "arsenal_ivb_spread", "arsenal_hb_spread",
        "arsenal_ivb_max_other", "arsenal_ivb_min_other",
        "arsenal_hb_max_other", "arsenal_hb_min_other",
        "nearest_other_velo_diff", "nearest_other_ivb_diff", "nearest_other_hb_diff",
        # v8c additions: arsenal-context features that need multi-pitch context
        "release_pt_arsenal_spread_h", "release_pt_arsenal_spread_v",
        "movement_arc_to_primary", "perceived_velo_diff_primary",
    ]
    # Pick the grouping key based on what's available in df.
    # Priority: numeric pitcher_id > player_name. Year is required.
    _group_id_col = None
    if "pitcher" in df.columns and df["pitcher"].notna().any():
        _group_id_col = "pitcher"
    elif "player_name" in df.columns:
        _group_id_col = "player_name"
    _have_shape = all(c in df.columns for c in
                      ["start_speed", "ivb_in", "hb_arm_in", "pitch_type_int"])
    _have_year = "year" in df.columns
    needs_arsenal = any(f not in df.columns for f in _ARSENAL_FEATS)
    if needs_arsenal:
        # Always initialize the columns so subsequent _FEATURES subset works,
        # even if we can't fill them.
        for f in _ARSENAL_FEATS:
            if f not in df.columns:
                df[f] = np.nan
    if needs_arsenal and _group_id_col is not None and _have_shape and _have_year:
        # Group by (pitcher OR player_name, year) and fill arsenal context
        _FB_PRIORITY_INTS = [_GROUP_TO_INT.get(g, -1) for g in
                              ["4-Seam", "2-Seam/Sinker", "Cutter"]]
        for (_, _), grp_df in df.groupby([_group_id_col, "year"], sort=False):
            if len(grp_df) < 2:
                df.loc[grp_df.index, "arsenal_size"] = 1.0
                continue
            arsenal = [
                (int(r["pitch_type_int"]), r["start_speed"], r["ivb_in"], r["hb_arm_in"],
                 r.get("rel_height", np.nan), r.get("rel_side_arm", np.nan),
                 r.get("extension", np.nan))
                for _, r in grp_df.iterrows() if pd.notna(r["pitch_type_int"])
            ]
            # v8c: arsenal-wide release-point spread
            _rh_arr = np.array([a[4] for a in arsenal if not np.isnan(a[4])])
            _rs_arr = np.array([a[5] for a in arsenal if not np.isnan(a[5])])
            _spread_v = float(np.std(_rh_arr, ddof=0)) if len(_rh_arr) > 1 else 0.0
            _spread_h = float(np.std(_rs_arr, ddof=0)) if len(_rs_arr) > 1 else 0.0
            # v8c: identify the primary FB for this pitcher-year
            primary = None
            for _fb in _FB_PRIORITY_INTS:
                for a in arsenal:
                    if a[0] == _fb:
                        primary = a   # use first FB in priority order
                        break
                if primary is not None:
                    break
            if primary is None:
                # No FB tagged — use fastest pitch as proxy
                primary = max(arsenal, key=lambda a: a[1])

            for idx, row in grp_df.iterrows():
                this_pt = int(row["pitch_type_int"]) if pd.notna(row["pitch_type_int"]) else -1
                others = [a for a in arsenal if a[0] != this_pt]
                df.at[idx, "arsenal_size"] = float(len(arsenal))
                if "release_pt_arsenal_spread_h" in _bundle_feats:
                    df.at[idx, "release_pt_arsenal_spread_h"] = _spread_h
                if "release_pt_arsenal_spread_v" in _bundle_feats:
                    df.at[idx, "release_pt_arsenal_spread_v"] = _spread_v
                # v8c: movement_arc_to_primary
                if "movement_arc_to_primary" in _bundle_feats:
                    _ivb_d = row["ivb_in"]    - primary[2]
                    _hb_d  = row["hb_arm_in"] - primary[3]
                    df.at[idx, "movement_arc_to_primary"] = float(np.sqrt(_ivb_d**2 + _hb_d**2))
                # v8c: perceived_velo_diff_primary
                if "perceived_velo_diff_primary" in _bundle_feats:
                    _ext_row = row.get("extension", 6.4)
                    _my_perc = row["start_speed"] * (1.0 + (_ext_row - 6.5) / 55.0)
                    _pr_perc = primary[1] * (1.0 + (_ext_row - 6.5) / 55.0)
                    df.at[idx, "perceived_velo_diff_primary"] = float(_my_perc - _pr_perc)
                if not others:
                    continue
                other_v = np.array([a[1] for a in others])
                other_i = np.array([a[2] for a in others])
                other_h = np.array([a[3] for a in others])
                df.at[idx, "arsenal_ivb_spread"]    = float(other_i.max() - other_i.min())
                df.at[idx, "arsenal_hb_spread"]     = float(other_h.max() - other_h.min())
                df.at[idx, "arsenal_ivb_max_other"] = float(other_i.max())
                df.at[idx, "arsenal_ivb_min_other"] = float(other_i.min())
                df.at[idx, "arsenal_hb_max_other"]  = float(other_h.max())
                df.at[idx, "arsenal_hb_min_other"]  = float(other_h.min())
                my_velo = row["start_speed"]
                slower = other_v[other_v < my_velo]
                if len(slower) > 0:
                    df.at[idx, "velo_diff_secondary"] = float(my_velo - slower.max())
                else:
                    df.at[idx, "velo_diff_secondary"] = float(my_velo - other_v.max())
                gaps = np.abs(other_v - my_velo)
                j = int(gaps.argmin())
                df.at[idx, "nearest_other_velo_diff"] = float(my_velo - other_v[j])
                df.at[idx, "nearest_other_ivb_diff"]  = float(row["ivb_in"] - other_i[j])
                df.at[idx, "nearest_other_hb_diff"]   = float(row["hb_arm_in"] - other_h[j])

    needed = list(_FEATURES)
    # v5d: arsenal-context features are allowed to be NaN
    # v6++/v8: cluster scores are NaN for non-matching pitch types
    _arsenal_set = set(_ARSENAL_FEATS) | {"sweeper_cluster_score", "four_seam_cluster_score"}
    core_needed = [f for f in needed if f not in _arsenal_set]
    valid_mask = df["pitch_type_int"].notna() & df[core_needed].notna().all(axis=1)
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
            _ver_tag = "v6" if _BUNDLE.get("version", "").endswith("v6") else "v5"
            override_path = Path(model_dir) / f"dm_stuff_plus_{_ver_tag}_norms.json"
            override_path.write_text(json.dumps(prod_norms, indent=2))
            print(f"DM Stuff+ {_ver_tag}: saved recalibrated norms to {override_path.name}")
            globals()["_NORMS"] = prod_norms
        except Exception as e:
            print(f"DM Stuff+: could not save norms override ({e})")

        scores = _standardize_with_norms(score_raw, pitch_groups, prod_norms)
    else:
        scores = _standardize(raw, pitch_groups)

    out = pd.Series(np.nan, index=df_pitches.index, dtype=float)
    out.loc[sub.index] = scores
    return out
