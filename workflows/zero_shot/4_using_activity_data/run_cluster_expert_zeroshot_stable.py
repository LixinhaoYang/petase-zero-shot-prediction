"""
run_cluster_expert_zeroshot_stable.py

Zero-shot submission generator (stable version):
- Embeddings: ESM2 mean-pooled per-sequence embeddings.
- Clustering: uses clustered_sequences.csv to assign each test sequence to a cluster.
- Optional merge: cluster 3 and 4 -> 34 (cluster_group).
- Expression: keeps the original "cluster expert panel" logic (WT panel per cluster_group)
  and uses within-cluster percentile * cluster confidence (optional PLL fusion).
- Activity_1 / Activity_2 (most stable patch):
  - Compute sim_wt_topk against ALL WT sequences (same evidence-aligned score you validated).
  - Map to continuous [0,1] values via within-cluster percentile rank.
  - activity_1 == activity_2 for maximum robustness when substrate/temperature are unknown.

Output CSV columns:
- sequence
- activity_1 (μmol [TPA]/min·mg [E])
- activity_2 (μmol [TPA]/min·mg [E])
- expression (mg/mL)
"""

import os
import re
import json
import argparse
import hashlib
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd


# -------------------------
# Utilities
# -------------------------

def md5_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def sanitize_seq(s: str) -> str:
    s = str(s).strip().upper()
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s)
    return s


def find_sequence_column(df: pd.DataFrame) -> str:
    candidates = ["sequence", "seq", "protein_sequence", "aa_sequence", "Wt AA Sequence", "wt aa sequence"]
    cols = list(df.columns)
    low = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    # fallback: first column containing 'seq'
    for c in cols:
        if "seq" in str(c).lower():
            return c
    raise ValueError(f"Could not detect sequence column. Columns: {cols}")


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def percentile_from_rank(rank: np.ndarray) -> np.ndarray:
    """
    rank: 1..N (1 is best)
    returns: 0..1 (best -> 1, worst -> 0)
    """
    rank = np.asarray(rank, dtype=np.float32)
    n = len(rank)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    return (n - rank) / (n - 1)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def load_cache_np(cache_dir: str, key: str):
    npy = os.path.join(cache_dir, f"{key}.npy")
    meta = os.path.join(cache_dir, f"{key}.meta.json")
    if not (os.path.exists(npy) and os.path.exists(meta)):
        return None
    arr = np.load(npy)
    with open(meta, "r") as f:
        meta_obj = json.load(f)
    return arr, meta_obj


def save_cache_np(cache_dir: str, key: str, arr: np.ndarray, meta_obj: dict):
    ensure_dir(cache_dir)
    npy = os.path.join(cache_dir, f"{key}.npy")
    meta = os.path.join(cache_dir, f"{key}.meta.json")
    np.save(npy, arr)
    with open(meta, "w") as f:
        json.dump(meta_obj, f, indent=2)


# -------------------------
# ESM Embedding / PLL
# -------------------------

def load_esm_model(model_name: str, device):
    import esm
    fn = getattr(esm.pretrained, model_name, None)
    if fn is None:
        raise ValueError(f"Unknown esm pretrained model: {model_name}")
    model, alphabet = fn()
    model = model.eval().to(device)
    return model, alphabet


def embed_sequences(
    seqs: List[str],
    model,
    alphabet,
    device,
    batch_size: int = 16,
    repr_layer: Optional[int] = None,
) -> np.ndarray:
    import torch

    if repr_layer is None:
        repr_layer = max(model.num_layers, 1)

    batch_converter = alphabet.get_batch_converter()
    pad_idx = alphabet.padding_idx

    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            data = [(f"seq{i+j}", s) for j, s in enumerate(chunk)]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)

            res = model(toks, repr_layers=[repr_layer], return_contacts=False)
            reps = res["representations"][repr_layer]  # [B, T, C]

            mask = (toks != pad_idx).float()
            reps_no_bos = reps[:, 1:, :]
            mask_no_bos = mask[:, 1:]

            # drop EOS (last non-pad token after BOS removal)
            for b in range(reps_no_bos.shape[0]):
                nonpad = torch.where(mask_no_bos[b] > 0.5)[0]
                if len(nonpad) > 0:
                    last = nonpad[-1].item()
                    mask_no_bos[b, last] = 0.0

            denom = mask_no_bos.sum(dim=1, keepdim=True).clamp(min=1.0)
            emb = (reps_no_bos * mask_no_bos.unsqueeze(-1)).sum(dim=1) / denom
            outs.append(emb.detach().cpu().numpy())

    return np.concatenate(outs, axis=0).astype(np.float32)


def pseudo_pll(
    seqs: List[str],
    model,
    alphabet,
    device,
    batch_size: int = 1,
) -> np.ndarray:
    """
    Slow masked pseudo-log-likelihood per residue (mean over positions).
    Intended for optional expression fusion only.
    """
    import torch
    from torch.nn.functional import log_softmax

    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    scores = np.zeros(len(seqs), dtype=np.float32)

    with torch.no_grad():
        for i, s in enumerate(seqs):
            data = [("seq", s)]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)

            L = toks.shape[1]
            # positions: 1..L-2 (exclude BOS=0, EOS=L-1)
            pos_scores = []
            for p in range(1, L - 1):
                toks_masked = toks.clone()
                true_tok = toks_masked[0, p].item()
                toks_masked[0, p] = mask_idx

                out = model(toks_masked, repr_layers=[], return_contacts=False)
                logits = out["logits"][0, p]  # [V]
                lp = log_softmax(logits, dim=0)[true_tok].item()
                pos_scores.append(lp)

            if len(pos_scores) == 0:
                scores[i] = -100.0
            else:
                scores[i] = float(np.mean(pos_scores))

    return scores.astype(np.float32)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt_csv", required=True, help="WT CSV file (library)")
    ap.add_argument("--test_csv", required=True, help="Test CSV file to predict")
    ap.add_argument("--cluster_csv", required=True, help="CSV with columns: sequence, cluster")
    ap.add_argument("--out_csv", default="submission_cluster_expert.csv")
    ap.add_argument("--cache_dir", default="cache_cluster_expert")
    ap.add_argument("--external_cache", default=None, help="Optional external cache dir to read from first")

    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    ap.add_argument("--repr_layer", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--use_gpu", action="store_true")

    ap.add_argument("--wt_panel_m", type=int, default=5, help="WT panel size per cluster_group (expression path)")
    ap.add_argument("--knn_k", type=int, default=5, help="k for kNN averaging (used in expression and activity sim_wt_topk)")
    ap.add_argument("--merge_34", action="store_true", help="merge cluster 3 and 4 to 34")

    ap.add_argument("--use_pll", action="store_true", help="Optional: fuse PLL into expression percentile")
    ap.add_argument("--expr_pll_lam", type=float, default=0.05, help="PLL fusion weight for expression [0..1]")

    args = ap.parse_args()

    ensure_dir(args.cache_dir)

    # ---- load CSVs
    wt_df = pd.read_csv(args.wt_csv)
    test_df = pd.read_csv(args.test_csv)
    clu_df = pd.read_csv(args.cluster_csv)

    wt_seq_col = find_sequence_column(wt_df)
    test_seq_col = find_sequence_column(test_df)

    if "sequence" not in clu_df.columns:
        # try to find a sequence-like column
        clu_seq_col = find_sequence_column(clu_df)
        clu_df = clu_df.rename(columns={clu_seq_col: "sequence"})
    if "cluster" not in clu_df.columns:
        raise ValueError(f"cluster_csv must contain a 'cluster' column. Columns: {list(clu_df.columns)}")

    # sanitize sequences
    wt_seqs = [sanitize_seq(s) for s in wt_df[wt_seq_col].tolist()]
    test_seqs = [sanitize_seq(s) for s in test_df[test_seq_col].tolist()]

    # merge test with cluster labels
    df = pd.DataFrame({"sequence": test_seqs})
    df = df.merge(clu_df[["sequence", "cluster"]], on="sequence", how="left")
    if df["cluster"].isna().any():
        missing = int(df["cluster"].isna().sum())
        raise ValueError(f"{missing} test sequences have no cluster label in cluster_csv. Fix clustered_sequences.csv merge keys.")
    df["cluster"] = df["cluster"].astype(int)

    if args.merge_34:
        df["cluster_group"] = df["cluster"].replace({3: 34, 4: 34})
    else:
        df["cluster_group"] = df["cluster"]

    groups = sorted(df["cluster_group"].unique().tolist())

    # ---- embeddings cache keys
    model_name = args.model
    wt_key = f"emb_wt_{model_name}_{md5_str(''.join(wt_seqs[:50]))}_{len(wt_seqs)}"
    test_key = f"emb_test_{model_name}_{md5_str(''.join(test_seqs[:50]))}_{len(test_seqs)}"

    def try_load(key: str):
        if args.external_cache:
            got = load_cache_np(args.external_cache, key)
            if got is not None:
                return got[0]
        got = load_cache_np(args.cache_dir, key)
        if got is not None:
            return got[0]
        return None

    wt_emb = try_load(wt_key)
    test_emb = try_load(test_key)

    # ---- compute embeddings if needed
    if wt_emb is None or test_emb is None:
        import torch
        device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
        print("[INFO] computing embeddings with", model_name, "on", device)

        model, alphabet = load_esm_model(model_name, device)

        if wt_emb is None:
            wt_emb = embed_sequences(
                wt_seqs, model, alphabet, device,
                batch_size=args.batch_size, repr_layer=args.repr_layer
            )
            save_cache_np(args.cache_dir, wt_key, wt_emb, {"model": model_name, "kind": "wt_emb", "n": len(wt_seqs)})

        if test_emb is None:
            test_emb = embed_sequences(
                test_seqs, model, alphabet, device,
                batch_size=args.batch_size, repr_layer=args.repr_layer
            )
            save_cache_np(args.cache_dir, test_key, test_emb, {"model": model_name, "kind": "test_emb", "n": len(test_seqs)})

    # -------------------------
    # Expression path (cluster expert panel)
    # -------------------------

    # select WT panel per cluster_group using centroid of test embeddings in that group
    panels: Dict[int, np.ndarray] = {}
    conf: Dict[int, float] = {}

    for g in groups:
        idx = np.where(df["cluster_group"].to_numpy() == g)[0]
        if len(idx) == 0:
            continue
        centroid = test_emb[idx].mean(axis=0, keepdims=True)  # [1, D]
        sims = cosine_sim_matrix(centroid, wt_emb).ravel()  # [N_wt]
        m = min(args.wt_panel_m, len(sims))
        topm = np.argsort(-sims)[:m]
        panels[g] = topm

        # confidence: median of best similarity to panel within this group
        sims_g = cosine_sim_matrix(test_emb[idx], wt_emb[topm])  # [ng, m]
        best = np.max(sims_g, axis=1)
        conf[g] = float(np.median(best))

    # sim_knn within each group vs its panel
    sim_knn = np.zeros(len(test_seqs), dtype=np.float32)
    for g in groups:
        idx = np.where(df["cluster_group"].to_numpy() == g)[0]
        topm = panels[g]
        sims = cosine_sim_matrix(test_emb[idx], wt_emb[topm])  # [ng, m]
        k = min(args.knn_k, sims.shape[1])
        topk = np.partition(sims, -k, axis=1)[:, -k:]
        sim_knn[idx] = topk.mean(axis=1).astype(np.float32)

    # optional PLL for expression
    pll = None
    if args.use_pll:
        pll_key = f"pll_test_{model_name}_{md5_str(''.join(test_seqs[:50]))}_{len(test_seqs)}"
        pll = try_load(pll_key)
        if pll is None:
            import torch
            device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
            print("[INFO] computing pseudo-PLL on", device, "(slow)")
            model, alphabet = load_esm_model(model_name, device)
            pll = pseudo_pll(test_seqs, model, alphabet, device)
            save_cache_np(args.cache_dir, pll_key, pll, {"model": model_name, "kind": "pll_test", "n": len(test_seqs)})
        pll = pll.astype(np.float32)

    # expression = conf[g] * percentile(sim_knn) (optionally fuse PLL percentile)
    expr = np.zeros(len(test_seqs), dtype=np.float32)
    for g in groups:
        idx = np.where(df["cluster_group"].to_numpy() == g)[0]
        s = sim_knn[idx].astype(np.float32)
        r = pd.Series(s).rank(ascending=False, method="average").to_numpy(np.float32)
        p_sim = percentile_from_rank(r).astype(np.float32)

        e = p_sim
        if pll is not None:
            lam = float(np.clip(args.expr_pll_lam, 0.0, 1.0))
            p = pll[idx].astype(np.float32)
            r2 = pd.Series(p).rank(ascending=False, method="average").to_numpy(np.float32)
            p_pll = percentile_from_rank(r2).astype(np.float32)
            e = (1.0 - lam) * p_sim + lam * p_pll

        expr[idx] = float(conf.get(g, 1.0)) * e

    # -------------------------
    # Activity path (stable patch)
    # -------------------------
    # Evidence-aligned score: sim_wt_topk against ALL WT sequences
    sims_full = cosine_sim_matrix(test_emb, wt_emb)  # [N_test, N_wt]
    k_full = min(max(1, args.knn_k), sims_full.shape[1])
    topk_full = np.partition(sims_full, -k_full, axis=1)[:, -k_full:]
    sim_wt_topk = topk_full.mean(axis=1).astype(np.float32)

    # Within-cluster percentile rank mapping: higher sim => higher activity
    act1 = np.zeros(len(test_seqs), dtype=np.float32)
    act2 = np.zeros(len(test_seqs), dtype=np.float32)
    for g in groups:
        idx = np.where(df["cluster_group"].to_numpy() == g)[0]
        s = sim_wt_topk[idx].astype(np.float32)
        r = pd.Series(s).rank(ascending=False, method="average").to_numpy(np.float32)
        p = percentile_from_rank(r).astype(np.float32)  # 0..1
        act1[idx] = p
        act2[idx] = p  # identical for maximum stability

    # -------------------------
    # Write output
    # -------------------------
    out = pd.DataFrame({
        "sequence": test_seqs,
        "activity_1 (μmol [TPA]/min·mg [E])": act1.astype(np.float32),
        "activity_2 (μmol [TPA]/min·mg [E])": act2.astype(np.float32),
        "expression (mg/mL)": expr.astype(np.float32),
    })
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("[INFO] activity range:", float(np.min(act1)), float(np.max(act1)))
    print("[INFO] expression range:", float(np.min(expr)), float(np.max(expr)))


if __name__ == "__main__":
    main()

