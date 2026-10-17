import os
import json
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# -----------------------------
# Utilities
# -----------------------------
def md5_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-8)

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [N, D], b: [M, D]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def percentile_from_rank(rank: np.ndarray) -> np.ndarray:
    # rank: 1..N (float ok). convert to [0,1], higher better
    N = rank.shape[0]
    return 1.0 - (rank - 1.0) / (max(N - 1.0, 1.0))

def load_cache_np(cache_dir: str, key: str):
    npy = os.path.join(cache_dir, f"{key}.npy")
    meta = os.path.join(cache_dir, f"{key}.meta.json")
    if not (os.path.exists(npy) and os.path.exists(meta)):
        return None
    with open(meta, "r") as f:
        meta_obj = json.load(f)
    arr = np.load(npy)
    return arr, meta_obj

def save_cache_np(cache_dir: str, key: str, arr: np.ndarray, meta_obj: dict):
    ensure_dir(cache_dir)
    npy = os.path.join(cache_dir, f"{key}.npy")
    meta = os.path.join(cache_dir, f"{key}.meta.json")
    np.save(npy, arr)
    with open(meta, "w") as f:
        json.dump(meta_obj, f, indent=2)

# -----------------------------
# ESM embedding / PLL
# -----------------------------
def load_esm_model(model_name: str, device: torch.device):
    import esm
    fn = getattr(esm.pretrained, model_name, None)
    if fn is None:
        raise ValueError(f"Unknown esm pretrained model: {model_name}")
    model, alphabet = fn()
    model = model.eval()
    model = model.to(device)
    return model, alphabet

@torch.no_grad()
def embed_sequences(
    seqs: List[str],
    model,
    alphabet,
    device: torch.device,
    batch_size: int = 16,
    repr_layer: int = None,
) -> np.ndarray:
    """
    Return per-sequence embedding as mean of token representations (excluding BOS/EOS/pad).
    """
    if repr_layer is None:
        # use last layer
        repr_layer = max(model.num_layers, 1)

    batch_converter = alphabet.get_batch_converter()

    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i+batch_size]
        data = [(f"seq{i+j}", s) for j, s in enumerate(chunk)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)

        res = model(toks, repr_layers=[repr_layer], return_contacts=False)
        reps = res["representations"][repr_layer]  # [B, T, C]

        # mask: tokens that are not padding
        pad_idx = alphabet.padding_idx
        mask = (toks != pad_idx).float()

        # exclude BOS (0) and EOS (last non-pad)
        # ESM uses BOS at position 0. We'll drop position 0, and also drop last token where EOS sits.
        reps_no_bos = reps[:, 1:, :]
        mask_no_bos = mask[:, 1:]

        # For each sequence, EOS is at the position right after the last AA.
        # We can exclude the final non-pad token by setting it to 0.
        for b in range(reps_no_bos.shape[0]):
            nonpad = torch.where(mask_no_bos[b] > 0.5)[0]
            if len(nonpad) > 0:
                last = nonpad[-1].item()
                mask_no_bos[b, last] = 0.0

        # mean pool
        denom = mask_no_bos.sum(dim=1, keepdim=True).clamp(min=1.0)
        emb = (reps_no_bos * mask_no_bos.unsqueeze(-1)).sum(dim=1) / denom
        out.append(emb.detach().cpu().numpy())

    return np.concatenate(out, axis=0).astype(np.float32)

@torch.no_grad()
def pseudo_pll(
    seqs: List[str],
    model,
    alphabet,
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Compute pseudo log-likelihood per sequence by masked scoring each position.
    Returns average log-prob per residue (higher is better).
    NOTE: This is expensive. Use only if you really want PLL.
    """
    batch_converter = alphabet.get_batch_converter()
    mask_idx = alphabet.mask_idx

    pll_scores = np.zeros(len(seqs), dtype=np.float32)

    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start+batch_size]
        data = [(f"seq{start+j}", s) for j, s in enumerate(chunk)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)

        # Determine AA positions (exclude BOS/EOS/pad)
        pad = alphabet.padding_idx
        B, T = toks.shape
        mask = (toks != pad)
        # We'll identify per-seq valid AA positions: positions 1..(L) where L is last AA index before EOS.
        for b in range(B):
            # positions with non-pad
            nonpad = torch.where(mask[b])[0]
            if len(nonpad) == 0:
                continue
            # BOS at 0, EOS at last nonpad
            eos_pos = nonpad[-1].item()
            aa_positions = list(range(1, eos_pos))  # AA tokens
            if len(aa_positions) == 0:
                continue

            # score by masking one position at a time
            seq_score = 0.0
            for pos in aa_positions:
                masked = toks[b:b+1].clone()
                true_token = masked[0, pos].item()
                masked[0, pos] = mask_idx
                logits = model(masked)["logits"]  # [1, T, V]
                log_probs = torch.log_softmax(logits[0, pos], dim=-1)
                seq_score += float(log_probs[true_token].item())

            pll_scores[start + b] = seq_score / max(len(aa_positions), 1)

    return pll_scores

# -----------------------------
# Main logic: cluster experts
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wt_csv", default="pet-2025-wildtype-cds.csv")
    ap.add_argument("--test_csv", default="predictive-pet-zero-shot-test-2025.csv")
    ap.add_argument("--cluster_csv", default="clustered_sequences.csv")
    ap.add_argument("--cache_dir", default="cache_esm", help="local cache dir (in this folder)")
    ap.add_argument("--external_cache", default=None, help="optional: path to existing cache_esm to reuse")
    ap.add_argument("--model", default="esm2_t30_150M_UR50D", help="ESM pretrained fn name in esm.pretrained")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--wt_panel_m", type=int, default=5, help="top-m WT per cluster-group")
    ap.add_argument("--knn_k", type=int, default=5, help="kNN within WT panel")
    ap.add_argument("--merge_34", action="store_true", help="merge clusters 3 and 4 into one group")
    ap.add_argument("--use_pll", action="store_true", help="compute pseudo-PLL and blend into expression")
    ap.add_argument("--expr_pll_lam", type=float, default=0.05, help="lambda for expression PLL blend (rank-normalized)")
    ap.add_argument("--wt_seq_col", default=None)
    ap.add_argument("--out_csv", default="submission_cluster_expert.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    # Load CSVs
    wt = pd.read_csv(args.wt_csv)
    test = pd.read_csv(args.test_csv)
    clu = pd.read_csv(args.cluster_csv)

    # Detect sequence column
    def detect_seq_col(df: pd.DataFrame) -> str:
        for c in df.columns:
            if c.lower() in ["sequence", "seq", "protein_sequence", "aa_sequence", "wt aa sequence"]:
                return c
        raise ValueError("Cannot find a protein sequence column; pass --wt_seq_col for the WT table")

    wt_seq_col = args.wt_seq_col or detect_seq_col(wt)
    test_seq_col = detect_seq_col(test)

    if "cluster" not in clu.columns:
        raise ValueError("cluster_csv must contain a 'cluster' column")

    clu_seq_col = detect_seq_col(clu)

    # A repeated candidate is still one input row; reference duplicates must not
    # turn this lookup into a many-to-many join.
    labels = clu[[clu_seq_col, "cluster"]].rename(columns={clu_seq_col: "sequence"})
    if labels.groupby("sequence")["cluster"].nunique(dropna=False).gt(1).any():
        raise ValueError("Conflicting cluster labels for the same sequence")
    labels = labels.drop_duplicates("sequence")
    # Merge cluster labels onto test by sequence
    df = test[[test_seq_col]].rename(columns={test_seq_col: "sequence"}).copy()
    df = df.merge(
        labels,
        on="sequence",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if df["cluster"].isna().any():
        missing = int(df["cluster"].isna().sum())
        raise ValueError(f"{missing} test sequences missing cluster labels. Ensure clustered_sequences.csv covers all test sequences.")

    # cluster grouping
    if args.merge_34:
        df["cluster_group"] = df["cluster"].replace({3: 34, 4: 34})
    else:
        df["cluster_group"] = df["cluster"]

    # Build sequence lists
    wt_seqs = wt[wt_seq_col].astype(str).tolist()
    test_seqs = df["sequence"].astype(str).tolist()

    # Cache keys tied to exact sequence order
    key_wt = f"emb_wt_{args.model}_{md5_str(''.join(wt_seqs[:50]))}_{len(wt_seqs)}"
    key_test = f"emb_test_{args.model}_{md5_str(''.join(test_seqs[:50]))}_{len(test_seqs)}"
    key_pll = f"pll_test_{args.model}_{md5_str(''.join(test_seqs[:50]))}_{len(test_seqs)}"

    # Try external cache first (if provided), else local
    def try_load_any(key: str):
        if args.external_cache:
            got = load_cache_np(args.external_cache, key)
            if got is not None:
                return got[0]
        got = load_cache_np(args.cache_dir, key)
        if got is not None:
            return got[0]
        return None

    wt_emb = try_load_any(key_wt)
    test_emb = try_load_any(key_test)

    # Compute embeddings if missing
    if wt_emb is None or test_emb is None:
        print("[INFO] embeddings not found in cache; computing with ESM:", args.model)
        model, alphabet = load_esm_model(args.model, device)
        if wt_emb is None:
            wt_emb = embed_sequences(wt_seqs, model, alphabet, device, batch_size=args.batch_size)
            save_cache_np(args.cache_dir, key_wt, wt_emb, {"model": args.model, "n": len(wt_seqs), "kind": "wt_emb"})
        if test_emb is None:
            test_emb = embed_sequences(test_seqs, model, alphabet, device, batch_size=args.batch_size)
            save_cache_np(args.cache_dir, key_test, test_emb, {"model": args.model, "n": len(test_seqs), "kind": "test_emb"})
    else:
        print("[OK] loaded embeddings from cache")

    # Optionally compute PLL
    pll = None
    if args.use_pll:
        pll = try_load_any(key_pll)
        if pll is None:
            print("[INFO] PLL not found in cache; computing pseudo-PLL (this is expensive).")
            model, alphabet = load_esm_model(args.model, device)
            pll = pseudo_pll(test_seqs, model, alphabet, device, batch_size=max(1, args.batch_size // 4))
            save_cache_np(args.cache_dir, key_pll, pll, {"model": args.model, "n": len(test_seqs), "kind": "pll"})
        else:
            print("[OK] loaded PLL from cache")

    # Cluster expert selection: WT panel per group
    groups = sorted(df["cluster_group"].unique().tolist())
    print("cluster_groups:", groups)

    wt_panels: Dict[int, np.ndarray] = {}
    conf: Dict[int, float] = {}

    # Precompute WT similarity to each group centroid
    for g in groups:
        idx = np.where(df["cluster_group"].to_numpy() == g)[0]
        centroid = test_emb[idx].mean(axis=0, keepdims=True)  # [1, D]
        sims = cosine_sim_matrix(centroid, wt_emb).ravel()    # [n_wt]
        topm = np.argsort(-sims)[: args.wt_panel_m]
        wt_panels[g] = topm

        # cluster confidence: median of top1-to-panel sim for sequences in this group
        sim_to_panel = cosine_sim_matrix(test_emb[idx], wt_emb[topm]).max(axis=1)
        conf[g] = float(np.median(sim_to_panel))
        print(f" group {g}: WT_panel={topm.tolist()} conf={conf[g]:.4f}")

    # Score within each group using kNN to WT_panel
    sim_knn = np.zeros(len(test_seqs), dtype=np.float32)
    for g in groups:
        idx = np.where(df["cluster_group"].to_numpy() == g)[0]
        panel = wt_panels[g]
        sims = cosine_sim_matrix(test_emb[idx], wt_emb[panel])  # [ng, m]
        k = min(args.knn_k, sims.shape[1])
        topk = np.partition(sims, -k, axis=1)[:, -k:]
        sim_knn[idx] = topk.mean(axis=1).astype(np.float32)

    # Activity scores (simple monotonic maps)
    act1 = minmax(sim_knn)
    act2 = minmax(sim_knn ** 2)

    # Expression: either sim-only or rank-normalized blend with PLL, within group + conf scaling
    expr = np.zeros(len(test_seqs), dtype=np.float32)
    for g in groups:
        idx = np.where(df["cluster_group"].to_numpy() == g)[0]
        s = sim_knn[idx].astype(np.float32)

        # within-group percentile score
        r_sim = pd.Series(s).rank(ascending=False, method="average").to_numpy(np.float32)
        p_sim = percentile_from_rank(r_sim)

        if pll is not None:
            p = pll[idx].astype(np.float32)
            r_pll = pd.Series(p).rank(ascending=False, method="average").to_numpy(np.float32)
            p_pll = percentile_from_rank(r_pll)
            e = (1.0 - args.expr_pll_lam) * p_sim + args.expr_pll_lam * p_pll
        else:
            e = p_sim

        # cross-group calibration via conf
        expr[idx] = (conf[g] * e).astype(np.float32)

    # Final output
    out = pd.DataFrame({
        "sequence": test_seqs,
        "activity_1 (μmol [TPA]/min·mg [E])": act1.astype(np.float32),
        "activity_2 (μmol [TPA]/min·mg [E])": act2.astype(np.float32),
        "expression (mg/mL)": expr.astype(np.float32),
    })
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()

