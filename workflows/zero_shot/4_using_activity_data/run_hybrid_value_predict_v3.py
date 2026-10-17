#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import re
from sklearn.isotonic import IsotonicRegression

AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

# -------------------------
# Basic utils
# -------------------------
def sanitize_seq(s: str) -> str:
    s = str(s).strip().upper()
    return AA_RE.sub("", s)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def unique_sequence_values(frame, columns):
    """Build a lookup without multiplying repeated input sequences or hiding conflicts."""
    lookup = frame[["sequence", *columns]].copy()
    if lookup.groupby("sequence")[columns].nunique(dropna=False).gt(1).any().any():
        raise ValueError(f"Conflicting values for repeated sequences in {columns}")
    return lookup.drop_duplicates("sequence")

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
    Fast heuristic gate:
      - at least one GXSXG motif
      - contains at least one D and one H somewhere
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
# Isotonic with fallback for cluster2
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

def fit_cluster2_models(tr_c2, pH1, pH2_min, clip_q_lo, clip_q_hi, min_points):
    """
    strict:
      a1: pH==pH1
      a2: pH>=pH2_min
    fallback:
      base isotonic on all points; scale by medians of pH-near and high-pH subsets
    """
    # strict subsets
    a1 = tr_c2[np.isfinite(tr_c2["pH"]) & (np.abs(tr_c2["pH"] - pH1) < 1e-6)].copy()
    a2 = tr_c2[np.isfinite(tr_c2["pH"]) & (tr_c2["pH"] >= pH2_min)].copy()

    ir1, y1_clip, x1_rng = fit_isotonic_xy(a1["sim_wt_topk"], a1["value"], clip_q_lo, clip_q_hi, min_points)
    ir2, y2_clip, x2_rng = fit_isotonic_xy(a2["sim_wt_topk"], a2["value"], clip_q_lo, clip_q_hi, min_points)

    if ir1 is not None and ir2 is not None:
        return ir1, y1_clip, x1_rng, ir2, y2_clip, x2_rng, "strict"

    base = tr_c2.dropna(subset=["sim_wt_topk","value"]).copy()
    irb, yb_clip, xb_rng = fit_isotonic_xy(base["sim_wt_topk"], base["value"], clip_q_lo, clip_q_hi, min_points)
    if irb is None:
        raise RuntimeError("cluster2 isotonic failed: not enough points even for fallback base fit.")

    base_median = float(np.median(base["value"].to_numpy()))

    # pH1-like subset
    if base["pH"].isna().all():
        pH1_near = base.copy()
    else:
        base2 = base.copy()
        base2["pH_dist_1"] = (base2["pH"] - pH1).abs()
        pH1_near = base2.sort_values("pH_dist_1").head(max(min_points, len(base2)//3)).copy()
        if len(pH1_near) == 0:
            pH1_near = base.copy()

    med1 = float(np.median(pH1_near["value"].to_numpy()))
    scale1 = med1 / (base_median + 1e-12)

    # high-pH subset
    if base["pH"].isna().all():
        hi = base.copy()
    else:
        hi = base[base["pH"] >= pH2_min].copy()
        if len(hi) < min_points:
            hi = base.sort_values("pH", ascending=False).head(max(min_points, len(base)//3)).copy()
        if len(hi) == 0:
            hi = base.copy()

    med2 = float(np.median(hi["value"].to_numpy()))
    scale2 = med2 / (base_median + 1e-12)

    ir1 = ScaledIso(irb, scale1)
    ir2 = ScaledIso(irb, scale2)

    y1_clip = quantile_clip(pH1_near["value"].to_numpy(), clip_q_lo, clip_q_hi) or yb_clip
    y2_clip = quantile_clip(hi["value"].to_numpy(), clip_q_lo, clip_q_hi) or yb_clip

    return ir1, y1_clip, xb_rng, ir2, y2_clip, xb_rng, "fallback_base_scaled"

# -------------------------
# Same-cluster hierarchical anchors (NO global anchors)
# -------------------------
def _select_anchors_same_cluster(anchor_all, g, substrate, ph_mode, ph_value, ph_min):
    """
    Returns list of candidate anchor DataFrames in decreasing strictness (tiers).
    anchor_all already contains only valid rows (x,y finite) + normalized columns.
    """
    sub = anchor_all[anchor_all["cluster_group"] == g]
    if len(sub) == 0:
        return []

    tiers = []

    # Tier1: cluster + substrate + pH constraint
    t1 = sub[sub["substrate"] == substrate].copy()
    if ph_mode == "near":
        t1 = t1[np.isfinite(t1["pH"]) & (np.abs(t1["pH"] - ph_value) < 1e-6)]
    elif ph_mode == "ge":
        t1 = t1[np.isfinite(t1["pH"]) & (t1["pH"] >= ph_min)]
    elif ph_mode == "any":
        pass
    else:
        raise ValueError("ph_mode must be near/ge/any")
    tiers.append(t1)

    # Tier2: cluster + substrate (ignore pH)
    t2 = sub[sub["substrate"] == substrate].copy()
    tiers.append(t2)

    # Tier3: cluster only (ignore substrate/pH)
    tiers.append(sub)

    # keep only non-empty tiers
    tiers = [t for t in tiers if len(t) > 0]
    return tiers

def predict_one_same_cluster_hier(
    test_df,
    anchor_df,
    x_col="sim_wt_topk",
    y_col="value",
    group_col="cluster_group",
    substrate_col="substrate",
    ph_col="pH",
    target_substrate="aflim",
    ph_mode="near",   # near / ge / any
    ph_value=5.5,
    ph_min=9.0,
    k=5,
    alpha=4000.0,
    min_anchor_per_tier=3,
    final_fallback_value=0.0,
):
    """
    For each test item, use SAME-CLUSTER anchors with tier fallback.
    Never borrows anchors from other clusters.
    """
    # normalize anchors
    A = anchor_df.copy()
    A[group_col] = pd.to_numeric(A[group_col], errors="coerce").astype("Int64")
    A[x_col] = pd.to_numeric(A[x_col], errors="coerce")
    A[y_col] = pd.to_numeric(A[y_col], errors="coerce")
    A[substrate_col] = A[substrate_col].astype(str).str.strip()
    A[ph_col] = pd.to_numeric(A[ph_col], errors="coerce")

    A = A.dropna(subset=[group_col, x_col, y_col]).copy()
    A[group_col] = A[group_col].astype(int)

    # normalize test
    T = test_df.copy()
    T[group_col] = pd.to_numeric(T[group_col], errors="coerce").astype("Int64")
    T[x_col] = pd.to_numeric(T[x_col], errors="coerce")
    if substrate_col in T.columns:
        T[substrate_col] = T[substrate_col].astype(str).str.strip()
    if ph_col in T.columns:
        T[ph_col] = pd.to_numeric(T[ph_col], errors="coerce")

    y_out = np.full(len(T), final_fallback_value, dtype=float)

    # Pre-check clusters present in anchors
    anchor_clusters = set(A[group_col].unique().tolist())

    for i in range(len(T)):
        if pd.isna(T.iloc[i][group_col]) or (not np.isfinite(T.iloc[i][x_col])):
            y_out[i] = final_fallback_value
            continue

        g = int(T.iloc[i][group_col])
        x0 = float(T.iloc[i][x_col])

        if g not in anchor_clusters:
            # no anchors at all for this cluster
            y_out[i] = final_fallback_value
            continue

        tiers = _select_anchors_same_cluster(
            anchor_all=A,
            g=g,
            substrate=target_substrate,
            ph_mode=ph_mode,
            ph_value=ph_value,
            ph_min=ph_min,
        )
        if len(tiers) == 0:
            y_out[i] = final_fallback_value
            continue

        # choose first tier with enough points; else choose largest available
        use = None
        for t in tiers:
            if len(t) >= min_anchor_per_tier:
                use = t
                break
        if use is None:
            use = max(tiers, key=len)

        xa = use[x_col].to_numpy(dtype=float)
        ya = use[y_col].to_numpy(dtype=float)

        d = np.abs(xa - x0)
        kk = min(k, len(d))
        idx = np.argpartition(d, kk - 1)[:kk]

        w = rbf_weights(d[idx], alpha=alpha)
        wsum = float(np.sum(w))
        if (not np.isfinite(wsum)) or wsum <= 0:
            y_out[i] = float(np.mean(ya[idx]))
        else:
            y_out[i] = float(np.sum(w * ya[idx]) / wsum)

    return y_out

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--test_scored_csv", required=True)  # sequence, sim_wt_topk
    ap.add_argument("--clustered_sequences_csv", required=True)  # sequence, cluster or cluster_group

    ap.add_argument("--activate_scored_csv", default="activate_out/activate_scored_with_wt.csv")
    ap.add_argument("--activate_cluster_csv", default="activate_out/activate_with_cluster.csv")

    ap.add_argument("--seq_col", default="sequence")
    ap.add_argument("--merge_34", action="store_true")

    ap.add_argument("--cluster2_id", type=int, default=2)

    ap.add_argument("--iso_substrate", default="aflim")
    ap.add_argument("--iso_pH1", type=float, default=5.5)
    ap.add_argument("--iso_pH2_min", type=float, default=8.5)  # use 8.5 for stability, proxy to 9.0
    ap.add_argument("--min_iso_points", type=int, default=5)
    ap.add_argument("--clip_q_lo", type=float, default=0.02)
    ap.add_argument("--clip_q_hi", type=float, default=0.98)

    ap.add_argument("--anchor_k", type=int, default=5)
    ap.add_argument("--anchor_alpha", type=float, default=4000.0)
    ap.add_argument("--min_anchor_per_tier", type=int, default=3)

    ap.add_argument("--active_site_penalty", type=float, default=0.3)

    ap.add_argument("--baseline_submission_csv", default=None)
    ap.add_argument("--out_csv", default="submission_hybrid_value.csv")

    args = ap.parse_args()

    # ---------- load test ----------
    test = pd.read_csv(args.test_csv).copy()
    if args.seq_col not in test.columns:
        raise ValueError(f"seq_col '{args.seq_col}' not found in test_csv columns: {test.columns.tolist()}")
    test = test.rename(columns={args.seq_col: "sequence"})
    test["sequence"] = test["sequence"].map(sanitize_seq)

    scored = pd.read_csv(args.test_scored_csv).copy()
    if not {"sequence","sim_wt_topk"}.issubset(scored.columns):
        raise ValueError("test_scored_csv must contain columns: sequence, sim_wt_topk")
    scored["sequence"] = scored["sequence"].map(sanitize_seq)
    scored["sim_wt_topk"] = scored["sim_wt_topk"].map(safe_float)

    test = test.merge(unique_sequence_values(scored, ["sim_wt_topk"]),
                      on="sequence", how="left", sort=False, validate="many_to_one")
    if test["sim_wt_topk"].isna().any():
        raise RuntimeError("Some test sequences missing sim_wt_topk after merge. Check sequence matching.")

    # ---------- load test clusters ----------
    clu = pd.read_csv(args.clustered_sequences_csv).copy()
    if "sequence" not in clu.columns:
        seq_cands = [c for c in clu.columns if "seq" in c.lower()]
        if not seq_cands:
            raise ValueError("clustered_sequences_csv must contain a sequence column.")
        clu = clu.rename(columns={seq_cands[0]: "sequence"})
    clu["sequence"] = clu["sequence"].map(sanitize_seq)

    if "cluster_group" in clu.columns:
        clu["cluster_group"] = clu["cluster_group"].astype(int)
    elif "cluster" in clu.columns:
        clu["cluster_group"] = clu["cluster"].astype(int)
    else:
        raise ValueError("clustered_sequences_csv must contain cluster or cluster_group")

    if args.merge_34:
        clu.loc[clu["cluster_group"].isin([3,4]), "cluster_group"] = 34

    test = test.merge(unique_sequence_values(clu, ["cluster_group"]),
                      on="sequence", how="left", sort=False, validate="many_to_one")
    if test["cluster_group"].isna().any():
        raise RuntimeError("Some test sequences missing cluster_group after merge.")
    test["cluster_group"] = test["cluster_group"].astype(int)

    # ---------- load activate anchors ----------
    tr = pd.read_csv(args.activate_scored_csv).copy()
    need = {"sequence","pH","substrate","value","sim_wt_topk"}
    if not need.issubset(tr.columns):
        raise ValueError(f"activate_scored_csv must contain {sorted(list(need))}, got {tr.columns.tolist()}")

    tr["sequence"] = tr["sequence"].map(sanitize_seq)
    tr["pH"] = pd.to_numeric(tr["pH"], errors="coerce")
    tr["substrate"] = tr["substrate"].astype(str).str.strip()
    tr["value"] = pd.to_numeric(tr["value"], errors="coerce")
    tr["sim_wt_topk"] = pd.to_numeric(tr["sim_wt_topk"], errors="coerce")

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
        raise ValueError("activate_cluster_csv must contain cluster or cluster_group")

    if args.merge_34:
        tr_clu.loc[tr_clu["cluster_group"].isin([3,4]), "cluster_group"] = 34

    tr = tr.merge(tr_clu[["sequence","cluster_group"]].drop_duplicates("sequence"), on="sequence", how="left")
    tr = tr.drop_duplicates("sequence").copy()

    # filter to target substrate for anchors (you can relax if needed)
    tr = tr[tr["substrate"] == args.iso_substrate].copy()

    # ---------- cluster2 isotonic models ----------
    c2 = int(args.cluster2_id)
    tr_c2 = tr[tr["cluster_group"] == c2].dropna(subset=["sim_wt_topk","value"]).copy()

    ir1, y1_clip, x1_rng, ir2, y2_clip, x2_rng, mode = fit_cluster2_models(
        tr_c2,
        pH1=float(args.iso_pH1),
        pH2_min=float(args.iso_pH2_min),
        clip_q_lo=args.clip_q_lo,
        clip_q_hi=args.clip_q_hi,
        min_points=args.min_iso_points,
    )

    # conservative constants for "no anchor in cluster" fallback (not global anchors)
    c2_low1 = y1_clip[0] if y1_clip is not None else float(np.quantile(tr_c2["value"].to_numpy(), 0.05))
    c2_low2 = y2_clip[0] if y2_clip is not None else float(np.quantile(tr_c2["value"].to_numpy(), 0.05))

    print(f"[INFO] cluster2 isotonic mode: {mode}, n={len(tr_c2)}")
    print(f"[INFO] cluster2 low fallback a1={c2_low1:.6g}, a2={c2_low2:.6g}")

    # ---------- predict ----------
    pred = test.copy()
    y1 = np.full(len(pred), np.nan, dtype=float)
    y2 = np.full(len(pred), np.nan, dtype=float)

    mask_c2 = (pred["cluster_group"] == c2).to_numpy()
    mask_other = ~mask_c2

    # cluster2 -> isotonic value
    if mask_c2.any():
        x = pred.loc[mask_c2, "sim_wt_topk"].to_numpy(dtype=float)
        yy1 = ir1.predict(x)
        yy2 = ir2.predict(x)
        if y1_clip is not None:
            yy1 = np.clip(yy1, y1_clip[0], y1_clip[1])
        if y2_clip is not None:
            yy2 = np.clip(yy2, y2_clip[0], y2_clip[1])
        y1[mask_c2] = yy1
        y2[mask_c2] = yy2

    # other clusters -> same-cluster hierarchical anchors
    if mask_other.any():
        sub_other = pred.loc[mask_other, :].copy()

        # build a unified anchor table (same structure) for both tasks
        anchor_all = tr[["cluster_group","sim_wt_topk","value","substrate","pH"]].dropna(subset=["cluster_group","sim_wt_topk","value"]).copy()

        y1_other = predict_one_same_cluster_hier(
            test_df=sub_other,
            anchor_df=anchor_all,
            x_col="sim_wt_topk",
            y_col="value",
            group_col="cluster_group",
            substrate_col="substrate",
            ph_col="pH",
            target_substrate=args.iso_substrate,
            ph_mode="near",
            ph_value=float(args.iso_pH1),
            ph_min=float(args.iso_pH2_min),
            k=args.anchor_k,
            alpha=args.anchor_alpha,
            min_anchor_per_tier=args.min_anchor_per_tier,
            final_fallback_value=float(c2_low1),
        )

        y2_other = predict_one_same_cluster_hier(
            test_df=sub_other,
            anchor_df=anchor_all,
            x_col="sim_wt_topk",
            y_col="value",
            group_col="cluster_group",
            substrate_col="substrate",
            ph_col="pH",
            target_substrate=args.iso_substrate,
            ph_mode="ge",
            ph_value=float(args.iso_pH1),
            ph_min=float(args.iso_pH2_min),
            k=args.anchor_k,
            alpha=args.anchor_alpha,
            min_anchor_per_tier=args.min_anchor_per_tier,
            final_fallback_value=float(c2_low2),
        )

        y1[mask_other] = y1_other
        y2[mask_other] = y2_other

    # active-site penalty
    penalty = float(args.active_site_penalty)
    if not (0.0 <= penalty <= 1.0):
        raise ValueError("--active_site_penalty must be in [0,1]")
    present = pred["sequence"].map(active_site_present).to_numpy()
    missing = ~present
    if missing.any() and penalty < 1.0:
        y1[missing] = y1[missing] * penalty
        y2[missing] = y2[missing] * penalty

    pred["pred_activity_1"] = y1
    pred["pred_activity_2"] = y2

    # sanity: no NaN
    if pred["pred_activity_1"].isna().any() or pred["pred_activity_2"].isna().any():
        n1 = int(pred["pred_activity_1"].isna().sum())
        n2 = int(pred["pred_activity_2"].isna().sum())
        raise RuntimeError(f"Prediction contains NaN values: a1 NaN={n1}, a2 NaN={n2}")

    # write submission
    col_a1 = "activity_1 (μmol [TPA]/min·mg [E])"
    col_a2 = "activity_2 (μmol [TPA]/min·mg [E])"
    col_expr = "expression (mg/mL)"

    if args.baseline_submission_csv:
        sub = pd.read_csv(args.baseline_submission_csv).copy()
        if "sequence" not in sub.columns:
            raise ValueError("baseline_submission_csv must contain 'sequence'")
        sub["sequence"] = sub["sequence"].map(sanitize_seq)
        # Historical baselines can themselves contain expanded duplicate rows.
        # Rebuild against the candidate rows instead of using their row count/order.
        baseline = unique_sequence_values(sub, [col_a1, col_a2, col_expr])
        sub = test[["sequence"]].merge(baseline, on="sequence", how="left",
                                       sort=False, validate="many_to_one")
        if sub[col_expr].isna().any():
            raise RuntimeError("Baseline submission is missing expression values for test sequences.")
        sub = sub.merge(unique_sequence_values(pred, ["pred_activity_1", "pred_activity_2"]),
                        on="sequence", how="left", sort=False, validate="many_to_one")
        if sub["pred_activity_1"].isna().any() or sub["pred_activity_2"].isna().any():
            raise RuntimeError("Merge into baseline submission produced NaNs (sequence mismatch).")
        sub[col_a1] = sub["pred_activity_1"]
        sub[col_a2] = sub["pred_activity_2"]
        sub = sub.drop(columns=["pred_activity_1","pred_activity_2"])
    else:
        sub = pd.DataFrame({
            "sequence": pred["sequence"],
            col_a1: pred["pred_activity_1"],
            col_a2: pred["pred_activity_2"],
            col_expr: np.zeros(len(pred), dtype=float),
        })

    sub.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("[INFO] cluster2 fraction:", float(np.mean(mask_c2)))
    print("[INFO] active-site missing fraction:", float(np.mean(missing)))

if __name__ == "__main__":
    main()

