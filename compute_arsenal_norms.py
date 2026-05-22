"""
compute_arsenal_norms.py — post-hoc Arsenal Stuff+ standardization
=========================================================================
Computes league-wide arsenal norms (μ_arsenal, σ_arsenal) for the
Arsenal Stuff+ aggregation approach:

    Y_arsenal     = sum(raw_pred_i × usage_i)        # usage-weighted raw
    ArsenalStuff+ = 100 + (Y_arsenal - μ) / σ × 10   # league-standardized

Method:
  1. Load the most recent valid Stuff+ bundle (v8c → v8b → v8 → v6 → v5)
  2. Load profiles + zone stats to determine per-pitcher-season usage rates
  3. For each pitcher-season with ≥500 total pitches:
       a. Get model's raw prediction for each pitch type
       b. Compute Y_arsenal = sum(raw × usage)
  4. Compute sample-weighted μ and σ across all qualified arsenals
  5. Write back to bundle as bundle["norms"]["arsenal"]

Usage:
    python3 compute_arsenal_norms.py
    python3 compute_arsenal_norms.py --bundle models/dm_stuff_plus_v8c.joblib
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


PITCH_GROUPS_ORDER = [
    "4-Seam", "2-Seam/Sinker", "Cutter", "Slider", "Sweeper",
    "Curveball", "Splitter", "Changeup", "Knuckleball",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=None,
                    help="Path to Stuff+ bundle (default: auto-detect latest valid)")
    ap.add_argument("--profiles", default="pitcher_profiles.csv")
    ap.add_argument("--min-pitches", type=int, default=500,
                    help="Minimum pitches per pitcher-season to qualify")
    args = ap.parse_args()

    # ── Find bundle ─────────────────────────────────────────────────────
    if args.bundle:
        bundle_path = Path(args.bundle)
    else:
        candidates = [
            "models/dm_stuff_plus_v8c.joblib",
            "models/dm_stuff_plus_v8b.joblib",
            "models/dm_stuff_plus_v8.joblib",
            "models/dm_stuff_plus_v6.joblib",
        ]
        bundle_path = None
        for c in candidates:
            if Path(c).exists():
                bundle_path = Path(c); break
        if bundle_path is None:
            sys.exit("✗ No bundle found. Train a model first.")
    print(f"Loading bundle: {bundle_path}")
    b = joblib.load(bundle_path)
    version = b.get("version", "unknown")
    print(f"  Version: {version}")
    print(f"  Features: {len(b['features'])}")

    # ── Load profiles ────────────────────────────────────────────────────
    print(f"\nLoading profiles: {args.profiles}")
    if not Path(args.profiles).exists():
        sys.exit(f"✗ {args.profiles} not found. Run build_profiles first.")
    prof = pd.read_csv(args.profiles)
    print(f"  {len(prof):,} pitcher-seasons in profile")

    # ── Score each profile through the bundle's model ────────────────────
    # We use score_dm_stuff_plus_batch which handles all the feature
    # engineering for v8c.
    print(f"\nScoring all pitcher-seasons through the model...")
    sys.path.insert(0, str(Path(__file__).parent))
    from score_dm_stuff_plus import score_dm_stuff_plus_batch, _load as _scorer_load
    if not _scorer_load(str(bundle_path.parent)):
        sys.exit("✗ Failed to load model via score_dm_stuff_plus")

    # Build per-(pitcher, year, pitch_group) rows
    rows = []
    for _, p in prof.iterrows():
        name, yr, hand = p["player_name"], int(p["year"]), p["hand"]
        if hand not in ("R", "L"): hand = "R"
        rh = float(p.get("rel_height", 5.8)) if not pd.isna(p.get("rel_height")) else 5.8
        rs = float(p.get("rel_side", -1.7)) if not pd.isna(p.get("rel_side")) else -1.7
        ext = float(p.get("extension", 6.4)) if not pd.isna(p.get("extension")) else 6.4

        for grp in PITCH_GROUPS_ORDER:
            n_col = f"n_{grp}"
            if pd.isna(p.get(n_col)) or float(p.get(n_col, 0)) < 30:
                continue
            velo = p.get(f"velo_{grp}")
            if pd.isna(velo): continue
            ivb = p.get(f"ivb_{grp}", np.nan)
            hb = p.get(f"hb_{grp}", np.nan)
            spin = p.get(f"spin_rate_{grp}", 2300)
            vaa = p.get(f"vaa_{grp}", -5.0)
            haa = p.get(f"haa_{grp}", 0.0)
            n_pitches = float(p[n_col])
            spin_axis = p.get(f"spin_axis_{grp}", 180.0)
            rows.append({
                "player_name": name, "year": yr, "pitch_group": grp,
                "p_throws": hand,
                "start_speed": float(velo),
                "spin_rate": float(spin) if not pd.isna(spin) else 2300,
                "ivb_in": float(ivb) if not pd.isna(ivb) else 12.0,
                "hb_arm_in": float(hb) if not pd.isna(hb) else 8.0,
                "vaa": float(vaa) if not pd.isna(vaa) else -5.0,
                "haa": float(haa) if not pd.isna(haa) else 0.0,
                "rel_height": rh, "rel_side_arm": -abs(rs) if hand=="R" else abs(rs),
                "extension": ext,
                "is_same_hand": 0,  # opp-hand default for arsenal norms
                "ssw_magnitude": 0.0,
                "spin_axis": float(spin_axis) if not pd.isna(spin_axis) else 180.0,
                "n": n_pitches,
                "pitcher": hash(name) % (10**9),   # placeholder pitcher_id
            })
    score_df = pd.DataFrame(rows)
    print(f"  {len(score_df):,} pitcher-season-pitch rows to score")

    # Get RAW predictions (not standardized)
    # score_dm_stuff_plus_batch returns standardized Stuff+. We need raw.
    # Easier path: manually predict using the model.
    import score_dm_stuff_plus as scorer
    df_feat = score_df.copy()
    # Borrow the feature-engineering code from score_dm_stuff_plus_batch
    # by calling it with recalibrate=False — but we want raw, not Stuff+.
    # Hack: monkey-patch the standardize step to return raw.
    orig_standardize = scorer._standardize_with_norms
    raw_capture = {}
    def capture_raw(score_raw, pitch_groups, norms):
        raw_capture["raw"] = score_raw.copy()
        raw_capture["groups"] = list(pitch_groups)
        return orig_standardize(score_raw, pitch_groups, norms)
    scorer._standardize_with_norms = capture_raw
    _ = scorer.score_dm_stuff_plus_batch(df_feat, model_dir=str(bundle_path.parent),
                                          recalibrate=False)
    scorer._standardize_with_norms = orig_standardize
    raw_preds = raw_capture.get("raw", np.array([]))
    if len(raw_preds) == 0:
        sys.exit("✗ Failed to capture raw predictions")
    print(f"  Captured {len(raw_preds):,} raw predictions")

    # Map raw predictions back to score_df rows. The batch filter drops rows
    # missing core features; we need to align indices.
    # Re-run the validity filter to get the matched indices.
    # Simplest: assume rows with valid features in batch are the kept ones.
    # We'll use the score_df index aligned to raw_preds positionally.
    if len(raw_preds) != len(score_df):
        print(f"  ! length mismatch: {len(raw_preds)} preds vs {len(score_df)} rows")
        print(f"    using positional alignment of first {min(len(raw_preds), len(score_df))}")
    n = min(len(raw_preds), len(score_df))
    score_df = score_df.iloc[:n].copy()
    score_df["raw_pred"] = raw_preds[:n]

    # ── Aggregate per pitcher-season → arsenal-level ─────────────────────
    print(f"\nAggregating to pitcher-season arsenals...")
    arsenals = []
    for (name, yr), grp in score_df.groupby(["player_name", "year"]):
        tot_n = grp["n"].sum()
        if tot_n < args.min_pitches: continue
        # usage-weighted sum of raw predictions
        Y = float((grp["raw_pred"] * grp["n"]).sum() / tot_n)
        arsenals.append({"player_name": name, "year": yr,
                         "Y_arsenal": Y, "total_pitches": tot_n,
                         "n_pitch_types": len(grp)})
    arsenal_df = pd.DataFrame(arsenals)
    print(f"  {len(arsenal_df):,} qualified pitcher-seasons (≥{args.min_pitches} pitches)")

    # Sample-weighted μ, σ (weighted by total_pitches)
    weights = arsenal_df["total_pitches"].astype(float)
    Y = arsenal_df["Y_arsenal"].astype(float)
    w_sum = weights.sum()
    mu_arsenal = float((Y * weights).sum() / w_sum)
    var_arsenal = float(((Y - mu_arsenal) ** 2 * weights).sum() / w_sum)
    sigma_arsenal = float(np.sqrt(var_arsenal))

    print(f"\n  Arsenal norms (sample-weighted):")
    print(f"    μ_arsenal = {mu_arsenal:+.4f}")
    print(f"    σ_arsenal = {sigma_arsenal:.4f}")
    print(f"  Sanity check — distribution of Y_arsenal:")
    print(f"    p10 = {Y.quantile(0.10):+.4f}")
    print(f"    p50 = {Y.quantile(0.50):+.4f}")
    print(f"    p90 = {Y.quantile(0.90):+.4f}")

    # ── Save back to bundle ───────────────────────────────────────────────
    b.setdefault("norms", {})
    b["norms"]["arsenal"] = {
        "mean": mu_arsenal,
        "sd":   sigma_arsenal,
        "n":    int(len(arsenal_df)),
        "min_pitches": args.min_pitches,
    }
    print(f"\nSaving updated bundle...")
    joblib.dump(b, bundle_path)
    print(f"  ✓ Saved norms.arsenal to {bundle_path}")
    print(f"  ✓ Future calls: ArsenalStuff+ = 100 + (Y_arsenal - {mu_arsenal:+.4f}) / {sigma_arsenal:.4f} × 10")


if __name__ == "__main__":
    main()
