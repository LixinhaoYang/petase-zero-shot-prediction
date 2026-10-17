#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import numpy as np
import pandas as pd

# sklearn is usually available; if not, replace with a simple monotonic fit.
from sklearn.isotonic import IsotonicRegression


# -----------------------------
# Utilities
# -----------------------------
AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

def sanitize_seq(s: str) -> str:
    s = str(s).strip().upper()
    return AA_RE.sub("", s)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def quantile_clip(y, lo_q=0.02, hi_q=0.98):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return None
    lo = float(np.quantile(y, lo_q))
    hi = float(np.quantile(y, hi_q))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi

def rbf_weights(d, alpha=4000.0):
    # d: absolute distance in sim-space (small ~ 1e-3)
    d = np.asarray(d, dtype=float)
    w = np.exp(-alpha * (d ** 2))
    return w

def active_site_present(seq: str) -> bool:
    """
    Heuristic active-site gate for PETase/cutinase-like serine hydrolases.
    - Look for a GXSXG motif (catalytic Ser motif)
    - Require at least one Asp (D) and one His (H) elsewhere
    This is intentionally conservative and fast.
    """
    seq = sanitize_seq(seq)
    if len(seq) < 120:
        return False

    # Find GXSXG motifs
    # pattern: G ? S ? G
    found_ser_motif = False
    for i in range(len(seq) - 4):
        if seq[i] == "G" and seq[i+2] == "S" and seq[i+4] == "G":
            found_ser_motif = True
            break

    if not found_ser_motif:
        return False

    # Require D and H somewhere (very loose)
    if ("D" not in seq) or ("H" not in seq):
        return False

    return True


# -----------------------------
# Core model pieces
# -----------------------------
def fit_isotonic(train_df, x_col, y_col, clip_q_lo=0.02, clip_q_hi=0.98):
    """
    Fit isotonic regression y = f(x) monotonic increasing.
    Returns: model, (y_lo, y_hi), (x_lo, x_hi)
    """
    d = train_df[[x_col, y_col]].dropna()
    x = d[x_col].astype(float).to_numpy()
    y = d[y_col].astype(float).to_numpy()

    if len(x) < 5:
        return None, None, None

    # sort by x
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Fit
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    ir.fit(x, y)

    y_clip = quantile_clip(y, clip_q_lo, clip_q_hi)
    x_lo, x_hi = float(np.min(x)), float(np.max(x))
    return ir, y_clip, (x_lo, x_hi)


def predict_cluster2_isotonic(test_df, ir1, ir2, x_col, y1_clip, y2_clip):
    """
    Apply isotonic models to cluster2 rows.
    """
    x = test_df[x_col].astype(float).to_numpy()

    y1 = ir1.predict(x) if ir1 is not None else np.full_like(x, np.nan, dtype=float)
    y2 = ir2.predict(x) if ir2 is not None else np.full_like(x, np.nan, dtype=float)

    # clip to robust quantile ranges
    if y1_clip is not None:
        y1 = np.clip(y1, y1_clip[0], y1_clip[1])
    if y2_clip is not None:
        y2 = np.clip(y2, y2_clip[0], y2_clip[1])

    return y1, y2


def predict_anchor_interpolate(
    test_df,
    anchor_df,
    x_col,
    y1_col,
    y2_col,
    group_col="cluster_group",
    k=5,
    alpha=4000.0,
    min_anchor_per_group=3,
):
    """
    For each test row, do within-cluster KNN in 1D sim space using anchors.
    Returns arrays y1_pred, y2_pred.
    """
    # Prepare anchors by group for fast lookup
    anchors = anchor_df[[group_col, x_col, y1_col, y2_col]].dropna()
    anchors[group_col] = anchors[group_col].astype(int)

    # fallback global anchors
    global_anchors = anchors.copy()

    y1_out = np.full(len(test_df), np.nan, dtype=float)
    y2_out = np.full(len(test_df), np.nan, dtype=float)

    test_groups = test_df[group_col].astype(int).to_numpy()
    test_x = test_df[x_col].astype(float).to_numpy()

    # Pre-split anchors dict
    by_g = {}
    for g, sub in anchors.groupby(group_col):
        by_g[int(g)] = sub

    for i in range(len(test_df)):
        g = int(test_groups[i])
        x0 = float(test_x[i])

        sub = by_g.get(g, None)
        use_sub = sub

        # If too few anchors in this cluster, fallback to global anchors
        if use_sub is None or use_sub.shape[0] < min_anchor_per_group:
            use_sub = global_anchors

        xa = use_sub[x_col].astype(float).to_numpy()
        d = np.abs(xa - x0)
        # take k nearest
        kk = min(k, len(d))
        idx = np.argpartition(d, kk-1)[:kk]
        d_k = d[idx]
        w = rbf_weights(d_k, alpha=alpha)
        wsum = float(np.sum(w)) if np.sum(w) > 0 else 0.0

        if wsum <= 0:
            # fallback: simple mean
            y1_out[i] = float(np.mean(use_sub[y1_col].to_numpy()[idx]))
            y2_out[i] = float(np.mean(use_sub[y2_col].to_numpy()[idx]))
        else:
            y1_out[i] = float(np.sum(w * use_sub[y1_col].to_numpy()[idx]) / wsum)
            y2_out[i] = float(np.sum(w * use_sub[y2_col].to_numpy()[idx]) / wsum)

    return y1_out, y2_out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_csv", required=True,
                    help="Test CSV with at least column 'sequence' OR provide --seq_col.")
    ap.add_argument("--test_scored_csv", required=True,
                    help="Test scored CSV that MUST include: sequence, sim_wt_topk. "
                         "You can point this to a file produced by your WT-scoring pipeline.")
    ap.add_argument("--activate_scored_csv", default="activate_out/activate_scored_with_wt.csv",
                    help="Training labeled (anchor) CSV with columns: sequence, temperature_C, pH, substrate, value, sim_wt_topk.")
    ap.add_argument("--activate_cluster_csv", default="activate_out/activate_with_cluster.csv",
                    help="CSV with columns: sequence, cluster_group (or cluster).")

    ap.add_argument("--clustered_sequences_csv", default="clustered_sequences.csv",
                    help="Reference cluster assignment for test sequences; must contain: sequence, cluster (or cluster_group).")

    ap.add_argument("--seq_col", default="sequence",
                    help="Sequence column name in test_csv if not 'sequence'.")

    ap.add_argument("--merge_34", action="store_true",
                    help="Merge cluster 3/4 into 34 via cluster_group.")
    ap.add_argument("--cluster2_id", type=int, default=2,
                    help="Cluster id treated as Cluster2 for isotonic mapping.")

    # isotonic fitting filters
    ap.add_argument("--iso_substrate", default="aflim",
                    help="Substrate name to use for isotonic fitting on cluster2 (default: aflim).")
    ap.add_argument("--iso_temp_any", action="store_true",
                    help="If set, do NOT filter temperature for isotonic fit (use all temps).")
    ap.add_argument("--iso_pH1", type=float, default=5.5,
                    help="pH used to fit activity_1 isotonic (default: 5.5 exact match).")
    ap.add_argument("--iso_pH2_min", type=float, default=9.0,
                    help="pH threshold used to fit activity_2 isotonic (default: >=9.0; your data uses 9.5).")

    ap.add_argument("--clip_q_lo", type=float, default=0.02)
    ap.add_argument("--clip_q_hi", type=float, default=0.98)

    # anchor interpolation
    ap.add_argument("--anchor_k", type=int, default=5)
    ap.add_argument("--anchor_alpha", type=float, default=4000.0)
    ap.add_argument("--min_anchor_per_group", type=int, default=3)

    # active site penalty
    ap.add_argument("--active_site_penalty", type=float, default=0.3,
                    help="Multiply predicted values by this factor if active-site heuristic fails.")

    # output
    ap.add_argument("--out_csv", default="submission_hybrid_value.csv")
    ap.add_argument("--baseline_submission_csv", default=None,
                    help="Optional baseline submission CSV to copy other columns (e.g., expression). "
                         "If provided, we overwrite activity_1 and activity_2, keep others.")

    args = ap.parse_args()

    # -----------------------------
    # Load test sequences + scored sim
    # -----------------------------
    test = pd.read_csv(args.test_csv).copy()
    if args.seq_col not in test.columns:
        raise ValueError(f"--seq_col '{args.seq_col}' not found in test_csv columns: {test.columns.tolist()}")
    test = test.rename(columns={args.seq_col: "sequence"})
    test["sequence"] = test["sequence"].map(sanitize_seq)

    test_scored = pd.read_csv(args.test_scored_csv).copy()
    if "sequence" not in test_scored.columns or "sim_wt_topk" not in test_scored.columns:
        raise ValueError("test_scored_csv must contain columns: sequence, sim_wt_topk")
    test_scored["sequence"] = test_scored["sequence"].map(sanitize_seq)

    test = test.merge(test_scored[["sequence", "sim_wt_topk"]], on="sequence", how="left")
    if test["sim_wt_topk"].isna().any():
        missing = int(test["sim_wt_topk"].isna().sum())
        raise ValueError(f"{missing} test rows missing sim_wt_topk after merge. "
                         "Ensure test_scored_csv covers all test sequences exactly (after sanitization).")

    # -----------------------------
    # Load test cluster assignment
    # -----------------------------
    clu_test = pd.read_csv(args.clustered_sequences_csv).copy()
    if "sequence" not in clu_test.columns:
        # best-effort
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
        missing = int(test["cluster_group"].isna().sum())
        raise ValueError(f"{missing} test rows missing cluster_group after merge. "
                         "Ensure clustered_sequences_csv covers all test sequences.")

    test["cluster_group"] = test["cluster_group"].astype(int)

    # -----------------------------
    # Load training anchors (activate)
    # -----------------------------
    tr = pd.read_csv(args.activate_scored_csv).copy()
    needed = {"sequence", "pH", "substrate", "value", "sim_wt_topk"}
    if not needed.issubset(set(tr.columns)):
        raise ValueError(f"activate_scored_csv must contain {sorted(list(needed))}, got {tr.columns.tolist()}")
    tr["sequence"] = tr["sequence"].map(sanitize_seq)
    tr["pH"] = tr["pH"].map(safe_float)
    tr["sim_wt_topk"] = tr["sim_wt_topk"].map(safe_float)
    tr["value"] = tr["value"].map(safe_float)

    # cluster labels for activate
    tr_clu = pd.read_csv(args.activate_cluster_csv).copy()
    if "sequence" not in tr_clu.columns:
        seq_cands = [c for c in tr_clu.columns if "seq" in c.lower()]
        if not seq_cands:
            raise ValueError("activate_cluster_csv must contain a sequence column.")
        tr_clu = tr_clu.rename(columns={seq_cands[0]: "sequence"})
    tr_clu["sequence"] = tr_clu["sequence"].map(sanitize_seq)
    if "cluster_group" not in tr_clu.columns:
        if "cluster" in tr_clu.columns:
            tr_clu["cluster_group"] = tr_clu["cluster"].astype(int)
        else:
            raise ValueError("activate_cluster_csv must contain 'cluster_group' or 'cluster'.")

    if args.merge_34:
        tr_clu.loc[tr_clu["cluster_group"].isin([3, 4]), "cluster_group"] = 34
    tr_clu["cluster_group"] = tr_clu["cluster_group"].astype(int)

    tr = tr.merge(tr_clu[["sequence", "cluster_group"]].drop_duplicates("sequence"),
                  on="sequence", how="left")

    # Keep unique sequences for fitting/anchoring to avoid repeated measurements dominating
    tr_uniq = tr.drop_duplicates("sequence").copy()

    # -----------------------------
    # Fit Cluster2 isotonic: activity_1 and activity_2
    # -----------------------------
    c2 = args.cluster2_id
    tr_c2 = tr_uniq[tr_uniq["cluster_group"] == c2].copy()

    # filter substrate
    tr_c2 = tr_c2[tr_c2["substrate"] == args.iso_substrate].copy()

    # optional temperature filter (if column exists)
    if (not args.iso_temp_any) and ("temperature_C" in tr_c2.columns):
        # keep only 60/70 by default if present (common in your sheet)
        tr_c2["temperature_C"] = tr_c2["temperature_C"].map(safe_float)
        tr_c2 = tr_c2[tr_c2["temperature_C"].isin([60.0, 70.0])].copy()

    # activity_1 training set (pH exact match with tolerance)
    tr_c2_a1 = tr_c2[np.isfinite(tr_c2["pH"]) & (np.abs(tr_c2["pH"] - args.iso_pH1) < 1e-6)].copy()
    # activity_2 training set (pH >= threshold)
    tr_c2_a2 = tr_c2[np.isfinite(tr_c2["pH"]) & (tr_c2["pH"] >= args.iso_pH2_min)].copy()

    # try strict fit first
    ir1, y1_clip, x1_rng = fit_isotonic(tr_c2_a1, "sim_wt_topk", "value", args.clip_q_lo, args.clip_q_hi)
    ir2, y2_clip, x2_rng = fit_isotonic(tr_c2_a2, "sim_wt_topk", "value", args.clip_q_lo, args.clip_q_hi)

    # fallback: fit on all cluster2 points (same substrate, maybe all temps) then rescale by pH group medians
    if ir1 is None or ir2 is None:
        print("[WARN] strict isotonic fit too few points; fallback to base isotonic + pH scaling")

        tr_c2_all = tr_c2.dropna(subset=["sim_wt_topk","value"]).copy()
        ir_base, yb_clip, xb_rng = fit_isotonic(tr_c2_all, "sim_wt_topk", "value", args.clip_q_lo, args.clip_q_hi)
        if ir_base is None:
            raise RuntimeError("Fallback isotonic still failed: not enough cluster2 points overall.")

        # compute scale factors from available pH groups (medians)
        # activity_1 scale: use pH closest to iso_pH1
        tr_c2_all["pH_dist_1"] = (tr_c2_all["pH"] - args.iso_pH1).abs()
        pH1_near = tr_c2_all.sort_values("pH_dist_1").head(max(5, len(tr_c2_all)//3))
        scale1 = float(np.median(pH1_near["value"])) / (float(np.median(tr_c2_all["value"])) + 1e-12)

        # activity_2 scale: use pH >= iso_pH2_min (or nearest if none)
        hi = tr_c2_all[tr_c2_all["pH"] >= args.iso_pH2_min]
        if len(hi) < 3:
            hi = tr_c2_all.sort_values("pH", ascending=False).head(max(5, len(tr_c2_all)//3))
        scale2 = float(np.median(hi["value"])) / (float(np.median(tr_c2_all["value"])) + 1e-12)

        # wrap base into two "models" via scaling
        class _ScaledIso:
            def __init__(self, base, scale):
                self.base = base
                self.scale = scale
            def predict(self, x):
                return self.base.predict(x) * self.scale

        ir1 = _ScaledIso(ir_base, scale1)
        ir2 = _ScaledIso(ir_base, scale2)

        # clipping ranges
        y1_clip = quantile_clip(pH1_near["value"].to_numpy(), args.clip_q_lo, args.clip_q_hi) or yb_clip
        y2_clip = quantile_clip(hi["value"].to_numpy(), args.clip_q_lo, args.clip_q_hi) or yb_clip
        x1_rng = xb_rng
        x2_rng = xb_rng

    print("[INFO] Cluster2 isotonic fitted.")
    print("  activity_1 train n =", len(tr_c2_a1), "y_clip=", y1_clip, "x_rng=", x1_rng)
    print("  activity_2 train n =", len(tr_c2_a2), "y_clip=", y2_clip, "x_rng=", x2_rng)

    # -----------------------------
    # Build anchors for non-cluster2 interpolation:
    # We need separate anchors for activity_1 and activity_2.
    # We'll define them from activate using pH subsets.
    # -----------------------------
    anchors_a1 = tr_uniq[np.isfinite(tr_uniq["pH"]) & (np.abs(tr_uniq["pH"] - args.iso_pH1) < 1e-6)].copy()
    anchors_a2 = tr_uniq[np.isfinite(tr_uniq["pH"]) & (tr_uniq["pH"] >= args.iso_pH2_min)].copy()

    # Rename value columns to keep both
    anchors_a1 = anchors_a1.rename(columns={"value": "value_a1"})
    anchors_a2 = anchors_a2.rename(columns={"value": "value_a2"})

    # Merge to get anchors that have both values where possible; if missing, still usable via fallback
    anchors = anchors_a1[["sequence", "cluster_group", "sim_wt_topk", "value_a1"]].merge(
        anchors_a2[["sequence", "cluster_group", "sim_wt_topk", "value_a2"]],
        on=["sequence", "cluster_group", "sim_wt_topk"], how="outer"
    )

    # For interpolation, require value columns present; we will handle NaNs separately by fallback
    # We'll create two anchor tables with the same schema:
    anchors_for_interp = anchors.copy()

    # -----------------------------
    # Predict on test: piecewise
    # -----------------------------
    test_pred = test.copy()

    # initialize
    y1_pred = np.full(len(test_pred), np.nan, dtype=float)
    y2_pred = np.full(len(test_pred), np.nan, dtype=float)

    # Cluster2 rows
    mask_c2 = (test_pred["cluster_group"] == c2).to_numpy()
    if mask_c2.any():
        y1_c2, y2_c2 = predict_cluster2_isotonic(
            test_pred.loc[mask_c2, :], ir1, ir2, "sim_wt_topk", y1_clip, y2_clip
        )
        y1_pred[mask_c2] = y1_c2
        y2_pred[mask_c2] = y2_c2

    # Non-cluster2 rows: anchor interpolation
    mask_other = ~mask_c2
    if mask_other.any():
        # anchors for a1: need rows with value_a1
        a1_tbl = anchors_for_interp.dropna(subset=["value_a1"]).copy()
        a2_tbl = anchors_for_interp.dropna(subset=["value_a2"]).copy()

        # For simplicity, we interpolate each target using same x_col, group_col
        y1_o, _dummy = predict_anchor_interpolate(
            test_pred.loc[mask_other, :],
            a1_tbl.rename(columns={"value_a1": "value_a1", "value_a2": "value_a2"}).assign(value_a2=np.nan),
            x_col="sim_wt_topk",
            y1_col="value_a1",
            y2_col="value_a1",  # placeholder, overwritten
            group_col="cluster_group",
            k=args.anchor_k,
            alpha=args.anchor_alpha,
            min_anchor_per_group=args.min_anchor_per_group,
        )
        _dummy, y2_o = predict_anchor_interpolate(
            test_pred.loc[mask_other, :],
            a2_tbl.rename(columns={"value_a2": "value_a2", "value_a1": "value_a1"}).assign(value_a1=np.nan),
            x_col="sim_wt_topk",
            y1_col="value_a2",  # placeholder
            y2_col="value_a2",
            group_col="cluster_group",
            k=args.anchor_k,
            alpha=args.anchor_alpha,
            min_anchor_per_group=args.min_anchor_per_group,
        )

        y1_pred[mask_other] = y1_o
        y2_pred[mask_other] = y2_o

        # clip to global anchor ranges to avoid extreme nonsense
        a1_clip = quantile_clip(a1_tbl["value_a1"].to_numpy(), args.clip_q_lo, args.clip_q_hi)
        a2_clip = quantile_clip(a2_tbl["value_a2"].to_numpy(), args.clip_q_lo, args.clip_q_hi)
        if a1_clip is not None:
            y1_pred[mask_other] = np.clip(y1_pred[mask_other], a1_clip[0], a1_clip[1])
        if a2_clip is not None:
            y2_pred[mask_other] = np.clip(y2_pred[mask_other], a2_clip[0], a2_clip[1])

    # -----------------------------
    # Active site penalty
    # -----------------------------
    penalty = float(args.active_site_penalty)
    if penalty < 0 or penalty > 1:
        raise ValueError("--active_site_penalty must be in [0,1]")

    present = test_pred["sequence"].map(active_site_present).to_numpy()
    missing_mask = ~present
    if missing_mask.any() and penalty < 1.0:
        y1_pred[missing_mask] = y1_pred[missing_mask] * penalty
        y2_pred[missing_mask] = y2_pred[missing_mask] * penalty

    test_pred["pred_activity_1"] = y1_pred
    test_pred["pred_activity_2"] = y2_pred

    # -----------------------------
    # Write submission
    # -----------------------------
    col_a1 = "activity_1 (μmol [TPA]/min·mg [E])"
    col_a2 = "activity_2 (μmol [TPA]/min·mg [E])"
    col_expr = "expression (mg/mL)"

    if args.baseline_submission_csv is not None:
        sub = pd.read_csv(args.baseline_submission_csv).copy()
        # must have sequence to merge
        if "sequence" not in sub.columns:
            raise ValueError("baseline_submission_csv must contain 'sequence' column")
        sub["sequence"] = sub["sequence"].map(sanitize_seq)
        sub = sub.merge(test_pred[["sequence", "pred_activity_1", "pred_activity_2"]], on="sequence", how="left")
        sub[col_a1] = sub["pred_activity_1"]
        sub[col_a2] = sub["pred_activity_2"]
        sub = sub.drop(columns=["pred_activity_1", "pred_activity_2"])
    else:
        # minimal submission with zeros for expression unless you have a separate predictor
        sub = pd.DataFrame({
            "sequence": test_pred["sequence"],
            col_a1: test_pred["pred_activity_1"],
            col_a2: test_pred["pred_activity_2"],
            col_expr: np.zeros(len(test_pred), dtype=float),
        })

    sub.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("[INFO] active-site missing fraction:", float(np.mean(missing_mask)))


if __name__ == "__main__":
    main()

