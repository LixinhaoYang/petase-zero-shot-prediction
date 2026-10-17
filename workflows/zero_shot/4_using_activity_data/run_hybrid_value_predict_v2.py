#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import re
import hashlib
from sklearn.isotonic import IsotonicRegression

AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

# -------------------------
# Helpers
# -------------------------
def sanitize_seq(s: str) -> str:
    s = str(s).strip().upper()
    return AA_RE.sub("", s)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def quantile_clip(arr, lo_q=0.02, hi_q=0.98):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None
    lo = float(np.quantile(a, lo_q))
    hi = float(np.quantile(a, hi_q))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi

def rbf_weights(d, alpha=4000.0):
    d = np.asarray(d, dtype=float)
    return np.exp(-alpha * (d ** 2))

def active_site_present(seq: str) -> bool:
    """
    Very fast heuristic for PETase/cutinase-like serine hydrolases.
    Criteria:
      - contains at least one GXSXG motif
      - contains at least one D and one H somewhere
    This is a conservative gate (penalize when clearly missing).
    """
    seq = sanitize_seq(seq)
    if len(seq) < 120:
        return False

    found = False
    for i in range(len(seq) - 4):
        if seq[i] == "G" and seq[i+2] == "S" and seq[i+4] == "G":
            found = True
            break
    if not found:
        return False
    if ("D" not in seq) or ("H" not in seq):
        return False
    return True

# -------------------------
# Isotonic fitting with fallback
# -------------------------
def fit_isotonic_xy(x, y, clip_q_lo=0.02, clip_q_hi=0.98, min_points=5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < min_points:
        return None, None, None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(x, y)

    y_clip = quantile_clip(y, clip_q_lo, clip_q_hi)
    x_rng = (float(np.min(x)), float(np.max(x)))
    return ir, y_clip, x_rng

class ScaledIso:
    def __init__(self, base_ir, scale):
        self.base_ir = base_ir
        self.scale = float(scale)

    def predict(self, x):
        return self.base_ir.predict(x) * self.scale

def fit_cluster2_models(
    tr_c2: pd.DataFrame,
    pH1: float,
    pH2_min: float,
    clip_q_lo=0.02,
    clip_q_hi=0.98,
    min_points=5,
):
    """
    Try strict isotonic fits:
      - activity_1 from pH close to pH1 (exact match)
      - activity_2 from pH >= pH2_min

    If too few points, fallback:
      - fit base isotonic on all cluster2 points
      - scale to pH1-like and high-pH-like medians
    """
    # strict subsets
    a1 = tr_c2[np.isfinite(tr_c2["pH"]) & (np.abs(tr_c2["pH"] - pH1) < 1e-6)].copy()
    a2 = tr_c2[np.isfinite(tr_c2["pH"]) & (tr_c2["pH"] >= pH2_min)].copy()

    ir1, y1_clip, x1_rng = fit_isotonic_xy(a1["sim_wt_topk"].to_numpy(), a1["value"].to_numpy(),
                                           clip_q_lo, clip_q_hi, min_points=min_points)
    ir2, y2_clip, x2_rng = fit_isotonic_xy(a2["sim_wt_topk"].to_numpy(), a2["value"].to_numpy(),
                                           clip_q_lo, clip_q_hi, min_points=min_points)

    if ir1 is not None and ir2 is not None:
        return ir1, y1_clip, x1_rng, ir2, y2_clip, x2_rng, "strict"

    # fallback
    base = tr_c2.dropna(subset=["sim_wt_topk", "value"]).copy()
    irb, yb_clip, xb_rng = fit_isotonic_xy(base["sim_wt_topk"].to_numpy(), base["value"].to_numpy(),
                                           clip_q_lo, clip_q_hi, min_points=min_points)
    if irb is None:
        raise RuntimeError("Cluster2 isotonic failed: not enough cluster2 points overall for fallback base fit.")

    base_median = float(np.median(base["value"].to_numpy()))

    # pH1-like: choose nearest pH rows to pH1, take enough points
    base["pH_dist_1"] = (base["pH"] - pH1).abs()
    pH1_near = base.sort_values("pH_dist_1").head(max(min_points, len(base)//3)).copy()
    med1 = float(np.median(pH1_near["value"].to_numpy()))
    scale1 = med1 / (base_median + 1e-12)

    # high-pH-like: pH>=pH2_min else top-pH rows
    hi = base[base["pH"] >= pH2_min].copy()
    if len(hi) < min_points:
        hi = base.sort_values("pH", ascending=False).head(max(min_points, len(base)//3)).copy()
    med2 = float(np.median(hi["value"].to_numpy()))
    scale2 = med2 / (base_median + 1e-12)

    ir1 = ScaledIso(irb, scale1)
    ir2 = ScaledIso(irb, scale2)

    y1_clip = quantile_clip(pH1_near["value"].to_numpy(), clip_q_lo, clip_q_hi) or yb_clip
    y2_clip = quantile_clip(hi["value"].to_numpy(), clip_q_lo, clip_q_hi) or yb_clip

    return ir1, y1_clip, xb_rng, ir2, y2_clip, xb_rng, "fallback_base_scaled"


# -------------------------
# Anchor interpolation (one target)
# -------------------------
def predict_anchor_one(
    test_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "cluster_group",
    k: int = 5,
    alpha: float = 4000.0,
    min_anchor_per_group: int = 3,
):
    """
    KNN-like interpolation in 1D sim space within the same cluster.
    If a cluster has too few anchors, fallback to global anchors.
    """
    anchors = anchor_df[[group_col, x_col, y_col]].dropna().copy()
    anchors[group_col] = anchors[group_col].astype(int)

    by_g = {int(g): sub for g, sub in anchors.groupby(group_col)}
    global_anchors = anchors

    y_out = np.full(len(test_df), np.nan, dtype=float)

    test_groups = test_df[group_col].astype(int).to_numpy()
    test_x = test_df[x_col].astype(float).to_numpy()

    for i in range(len(test_df)):
        g = int(test_groups[i])
        x0 = float(test_x[i])

        sub = by_g.get(g, None)
        use_sub = sub
        if use_sub is None or len(use_sub) < min_anchor_per_group:
            use_sub = global_anchors

        xa = use_sub[x_col].astype(float).to_numpy()
        ya = use_sub[y_col].astype(float).to_numpy()

        d = np.abs(xa - x0)
        kk = min(k, len(d))
        idx = np.argpartition(d, kk - 1)[:kk]

        w = rbf_weights(d[idx], alpha=alpha)
        wsum = float(np.sum(w))

        if wsum <= 0:
            y_out[i] = float(np.mean(ya[idx]))
        else:
            y_out[i] = float(np.sum(w * ya[idx]) / wsum)

    return y_out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_csv", required=True, help="Test CSV containing sequences.")
    ap.add_argument("--test_scored_csv", required=True, help="Test scored CSV with: sequence, sim_wt_topk.")
    ap.add_argument("--clustered_sequences_csv", required=True, help="Cluster assignment CSV for test sequences.")

    ap.add_argument("--activate_scored_csv", default="activate_out/activate_scored_with_wt.csv",
                    help="Anchor/training CSV with: sequence, pH, substrate, value, sim_wt_topk, temperature_C (optional).")
    ap.add_argument("--activate_cluster_csv", default="activate_out/activate_with_cluster.csv",
                    help="Cluster assignment CSV for activate sequences (sequence, cluster_group or cluster).")

    ap.add_argument("--seq_col", default="sequence", help="Sequence column in test_csv.")
    ap.add_argument("--merge_34", action="store_true", help="Merge cluster 3/4 into 34.")

    # cluster2 isotonic config
    ap.add_argument("--cluster2_id", type=int, default=2)
    ap.add_argument("--iso_substrate", default="aflim")
    ap.add_argument("--iso_temp_any", action="store_true",
                    help="If set, do not filter temperature for isotonic / anchors.")
    ap.add_argument("--iso_pH1", type=float, default=5.5)
    ap.add_argument("--iso_pH2_min", type=float, default=9.0)

    ap.add_argument("--clip_q_lo", type=float, default=0.02)
    ap.add_argument("--clip_q_hi", type=float, default=0.98)
    ap.add_argument("--min_iso_points", type=int, default=5)

    # anchor interpolation config
    ap.add_argument("--anchor_k", type=int, default=5)
    ap.add_argument("--anchor_alpha", type=float, default=4000.0)
    ap.add_argument("--min_anchor_per_group", type=int, default=3)

    # active site penalty
    ap.add_argument("--active_site_penalty", type=float, default=0.3)

    # output
    ap.add_argument("--baseline_submission_csv", default=None,
                    help="Optional: existing submission file to overwrite activity columns.")
    ap.add_argument("--out_csv", default="submission_hybrid_value.csv")

    args = ap.parse_args()

    # -------------------------
    # Load test + sim
    # -------------------------
    test = pd.read_csv(args.test_csv).copy()
    if args.seq_col not in test.columns:
        raise ValueError(f"--seq_col '{args.seq_col}' not found in test_csv columns: {test.columns.tolist()}")
    test = test.rename(columns={args.seq_col: "sequence"})
    test["sequence"] = test["sequence"].map(sanitize_seq)

    test_scored = pd.read_csv(args.test_scored_csv).copy()
    if not {"sequence", "sim_wt_topk"}.issubset(test_scored.columns):
        raise ValueError("test_scored_csv must contain columns: sequence, sim_wt_topk")
    test_scored["sequence"] = test_scored["sequence"].map(sanitize_seq)
    test_scored["sim_wt_topk"] = test_scored["sim_wt_topk"].map(safe_float)

    test = test.merge(test_scored[["sequence", "sim_wt_topk"]], on="sequence", how="left")
    if test["sim_wt_topk"].isna().any():
        nmiss = int(test["sim_wt_topk"].isna().sum())
        raise ValueError(f"{nmiss} test rows missing sim_wt_topk after merge. Check sequence sanitization/matching.")

    # -------------------------
    # Load test cluster assignment
    # -------------------------
    clu_test = pd.read_csv(args.clustered_sequences_csv).copy()
    if "sequence" not in clu_test.columns:
        seq_cands = [c for c in clu_test.columns if "seq" in c.lower()]
        if not seq_cands:
            raise ValueError("clustered_sequences_csv must contain a sequence column.")
        clu_test = clu_test.rename(columns={seq_cands[0]: "sequence"})
    clu_test["sequence"] = clu_test["sequence"].map(sanitize_seq)

    if "cluster_group" in clu_test.columns:
        clu_test["cluster_group"] = clu_test["cluster_group"].astype(int)
    elif "cluster" in clu_test.columns:
        clu_test["cluster_group"] = clu_test["cluster"].astype(int)
    else:
        raise ValueError("clustered_sequences_csv must contain 'cluster' or 'cluster_group'.")

    if args.merge_34:
        clu_test.loc[clu_test["cluster_group"].isin([3, 4]), "cluster_group"] = 34

    test = test.merge(clu_test[["sequence", "cluster_group"]], on="sequence", how="left")
    if test["cluster_group"].isna().any():
        nmiss = int(test["cluster_group"].isna().sum())
        raise ValueError(f"{nmiss} test rows missing cluster_group after merge. Check clustered_sequences_csv coverage.")
    test["cluster_group"] = test["cluster_group"].astype(int)

    # -------------------------
    # Load activate anchors + clusters
    # -------------------------
    tr = pd.read_csv(args.activate_scored_csv).copy()
    need = {"sequence", "pH", "substrate", "value", "sim_wt_topk"}
    if not need.issubset(tr.columns):
        raise ValueError(f"activate_scored_csv must contain {sorted(list(need))}, got {tr.columns.tolist()}")

    tr["sequence"] = tr["sequence"].map(sanitize_seq)
    tr["pH"] = tr["pH"].map(safe_float)
    tr["value"] = tr["value"].map(safe_float)
    tr["sim_wt_topk"] = tr["sim_wt_topk"].map(safe_float)

    if "temperature_C" in tr.columns:
        tr["temperature_C"] = tr["temperature_C"].map(safe_float)

    tr_clu = pd.read_csv(args.activate_cluster_csv).copy()
    if "sequence" not in tr_clu.columns:
        seq_cands = [c for c in tr_clu.columns if "seq" in c.lower()]
        if not seq_cands:
            raise ValueError("activate_cluster_csv must contain a sequence column.")
        tr_clu = tr_clu.rename(columns={seq_cands[0]: "sequence"})
    tr_clu["sequence"] = tr_clu["sequence"].map(sanitize_seq)

    if "cluster_group" in tr_clu.columns:
        tr_clu["cluster_group"] = tr_clu["cluster_group"].astype(int)
    elif "cluster" in tr_clu.columns:
        tr_clu["cluster_group"] = tr_clu["cluster"].astype(int)
    else:
        raise ValueError("activate_cluster_csv must contain 'cluster' or 'cluster_group'.")

    if args.merge_34:
        tr_clu.loc[tr_clu["cluster_group"].isin([3, 4]), "cluster_group"] = 34

    tr = tr.merge(tr_clu[["sequence", "cluster_group"]].drop_duplicates("sequence"),
                  on="sequence", how="left")

    # Deduplicate per sequence for anchors (avoid repeated measurements dominating)
    tr_uniq = tr.drop_duplicates("sequence").copy()

    # -------------------------
    # Prepare cluster2 training set for isotonic
    # -------------------------
    c2 = int(args.cluster2_id)
    tr_c2 = tr_uniq[tr_uniq["cluster_group"] == c2].copy()

    # substrate filter (required for your evidence)
    tr_c2 = tr_c2[tr_c2["substrate"] == args.iso_substrate].copy()

    # optional temperature filter: if NOT iso_temp_any, keep common temps if present
    if (not args.iso_temp_any) and ("temperature_C" in tr_c2.columns):
        # Keep all finite temps by default? Or restrict to 60/70 if present.
        # We'll restrict only if those exist; otherwise keep all.
        temps = sorted([t for t in tr_c2["temperature_C"].dropna().unique().tolist() if np.isfinite(t)])
        if any(t in [60.0, 70.0] for t in temps):
            tr_c2 = tr_c2[tr_c2["temperature_C"].isin([60.0, 70.0])].copy()

    # Fit models (strict then fallback)
    ir1, y1_clip, x1_rng, ir2, y2_clip, x2_rng, mode = fit_cluster2_models(
        tr_c2,
        pH1=float(args.iso_pH1),
        pH2_min=float(args.iso_pH2_min),
        clip_q_lo=args.clip_q_lo,
        clip_q_hi=args.clip_q_hi,
        min_points=args.min_iso_points,
    )

    print(f"[INFO] cluster2 isotonic mode: {mode}")
    print(f"  cluster2 train n (after filters): {len(tr_c2)}")
    print(f"  activity_1 clip: {y1_clip}  x_rng: {x1_rng}")
    print(f"  activity_2 clip: {y2_clip}  x_rng: {x2_rng}")

    # -------------------------
    # Build anchors for interpolation (non-cluster2)
    # Define activity_1 anchors: pH == iso_pH1
    # Define activity_2 anchors: pH >= iso_pH2_min
    # -------------------------
    anchors = tr_uniq.copy()
    anchors = anchors[anchors["substrate"] == args.iso_substrate].copy()

    if (not args.iso_temp_any) and ("temperature_C" in anchors.columns):
        temps = sorted([t for t in anchors["temperature_C"].dropna().unique().tolist() if np.isfinite(t)])
        if any(t in [60.0, 70.0] for t in temps):
            anchors = anchors[anchors["temperature_C"].isin([60.0, 70.0])].copy()

    anchors_a1 = anchors[np.isfinite(anchors["pH"]) & (np.abs(anchors["pH"] - float(args.iso_pH1)) < 1e-6)].copy()
    anchors_a2 = anchors[np.isfinite(anchors["pH"]) & (anchors["pH"] >= float(args.iso_pH2_min))].copy()

    anchors_a1 = anchors_a1.rename(columns={"value": "value_a1"})
    anchors_a2 = anchors_a2.rename(columns={"value": "value_a2"})

    # Keep only needed cols
    anchors_a1 = anchors_a1[["sequence", "cluster_group", "sim_wt_topk", "value_a1"]].dropna()
    anchors_a2 = anchors_a2[["sequence", "cluster_group", "sim_wt_topk", "value_a2"]].dropna()

    if len(anchors_a1) < 5 or len(anchors_a2) < 5:
        print("[WARN] very few anchors for a1 or a2 after filters.")
        print("  anchors_a1 n =", len(anchors_a1), "anchors_a2 n =", len(anchors_a2))

    # -------------------------
    # Predict
    # -------------------------
    test_pred = test.copy()
    y1_pred = np.full(len(test_pred), np.nan, dtype=float)
    y2_pred = np.full(len(test_pred), np.nan, dtype=float)

    mask_c2 = (test_pred["cluster_group"] == c2).to_numpy()
    mask_other = ~mask_c2

    # cluster2 via isotonic
    if mask_c2.any():
        x = test_pred.loc[mask_c2, "sim_wt_topk"].astype(float).to_numpy()
        y1 = ir1.predict(x)
        y2 = ir2.predict(x)

        if y1_clip is not None:
            y1 = np.clip(y1, y1_clip[0], y1_clip[1])
        if y2_clip is not None:
            y2 = np.clip(y2, y2_clip[0], y2_clip[1])

        y1_pred[mask_c2] = y1
        y2_pred[mask_c2] = y2

    # other clusters via local anchors
    if mask_other.any():
        sub_other = test_pred.loc[mask_other, :].copy()

        y1_o = predict_anchor_one(
            sub_other,
            anchors_a1.rename(columns={"value_a1": "y"}).assign(y=anchors_a1["value_a1"]),
            x_col="sim_wt_topk",
            y_col="y",
            group_col="cluster_group",
            k=args.anchor_k,
            alpha=args.anchor_alpha,
            min_anchor_per_group=args.min_anchor_per_group,
        )

        y2_o = predict_anchor_one(
            sub_other,
            anchors_a2.rename(columns={"value_a2": "y"}).assign(y=anchors_a2["value_a2"]),
            x_col="sim_wt_topk",
            y_col="y",
            group_col="cluster_group",
            k=args.anchor_k,
            alpha=args.anchor_alpha,
            min_anchor_per_group=args.min_anchor_per_group,
        )

        # Clip to global anchor ranges (stability)
        a1_clip = quantile_clip(anchors_a1["value_a1"].to_numpy(), args.clip_q_lo, args.clip_q_hi)
        a2_clip = quantile_clip(anchors_a2["value_a2"].to_numpy(), args.clip_q_lo, args.clip_q_hi)
        if a1_clip is not None:
            y1_o = np.clip(y1_o, a1_clip[0], a1_clip[1])
        if a2_clip is not None:
            y2_o = np.clip(y2_o, a2_clip[0], a2_clip[1])

        y1_pred[mask_other] = y1_o
        y2_pred[mask_other] = y2_o

    # -------------------------
    # Active-site penalty
    # -------------------------
    penalty = float(args.active_site_penalty)
    if not (0.0 <= penalty <= 1.0):
        raise ValueError("--active_site_penalty must be in [0,1]")

    present = test_pred["sequence"].map(active_site_present).to_numpy()
    missing = ~present
    if missing.any() and penalty < 1.0:
        y1_pred[missing] = y1_pred[missing] * penalty
        y2_pred[missing] = y2_pred[missing] * penalty

    test_pred["pred_activity_1"] = y1_pred
    test_pred["pred_activity_2"] = y2_pred

    # -------------------------
    # Write submission
    # -------------------------
    col_a1 = "activity_1 (μmol [TPA]/min·mg [E])"
    col_a2 = "activity_2 (μmol [TPA]/min·mg [E])"
    col_expr = "expression (mg/mL)"

    if args.baseline_submission_csv:
        sub = pd.read_csv(args.baseline_submission_csv).copy()
        if "sequence" not in sub.columns:
            raise ValueError("baseline_submission_csv must contain 'sequence' column")
        sub["sequence"] = sub["sequence"].map(sanitize_seq)
        sub = sub.merge(test_pred[["sequence", "pred_activity_1", "pred_activity_2"]], on="sequence", how="left")
        if sub["pred_activity_1"].isna().any() or sub["pred_activity_2"].isna().any():
            n1 = int(sub["pred_activity_1"].isna().sum())
            n2 = int(sub["pred_activity_2"].isna().sum())
            raise RuntimeError(f"Merge into baseline submission produced NaNs: a1={n1}, a2={n2}.")
        sub[col_a1] = sub["pred_activity_1"]
        sub[col_a2] = sub["pred_activity_2"]
        sub = sub.drop(columns=["pred_activity_1", "pred_activity_2"])
    else:
        sub = pd.DataFrame({
            "sequence": test_pred["sequence"],
            col_a1: test_pred["pred_activity_1"],
            col_a2: test_pred["pred_activity_2"],
            col_expr: np.zeros(len(test_pred), dtype=float),
        })

    sub.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("[INFO] active-site missing fraction:", float(np.mean(missing)))
    print("[INFO] cluster2 fraction:", float(np.mean(mask_c2)))

if __name__ == "__main__":
    main()

