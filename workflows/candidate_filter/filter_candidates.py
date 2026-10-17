#!/usr/bin/env python3
# ESM similarity and diversity filter. This script does not compute PLL.

import os
import re
import argparse
import hashlib
import numpy as np
import pandas as pd
from typing import List, Optional

# -------------------------
# utils
# -------------------------

def sanitize_seq(s: str) -> str:
    s = str(s).strip().upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s)

def find_seq_col(df, requested=None):
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Sequence column {requested!r} is missing")
        return requested
    for name in ["sequence", "Wt AA Sequence", "Sequence", "seq", "protein_sequence"]:
        if name in df.columns:
            return name
    candidates = [c for c in df.columns if "seq" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Cannot identify one protein sequence column; specify it explicitly")

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def select_candidates(seqs, embeddings, wt_embeddings, sim_drop=0.15,
                      n_clusters=20, cluster_k=300):
    """Filter and cap clusters while keeping sequence and embedding rows aligned."""
    from sklearn.cluster import KMeans

    if not 0 <= sim_drop < 1:
        raise ValueError("sim_drop must be in [0, 1)")
    if n_clusters < 1 or cluster_k < 1:
        raise ValueError("n_clusters and cluster_k must be positive")
    if len(seqs) == 0 or len(wt_embeddings) == 0:
        raise ValueError("Candidate and reference sequence sets must be nonempty")
    if len(seqs) != len(embeddings):
        raise ValueError("Candidate sequences and embeddings have different row counts")
    similarities = cosine_sim(embeddings, wt_embeddings)
    k = min(5, similarities.shape[1])
    topk = np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
    frame = pd.DataFrame({"sequence": seqs, "sim_wt_topk": topk})
    if sim_drop > 0:
        threshold = np.quantile(topk, sim_drop)
        frame = frame[frame["sim_wt_topk"] >= threshold].copy()
    # The original row index selects embeddings before the output index is reset.
    selected = embeddings[frame.index.to_numpy()]
    model = KMeans(n_clusters=min(n_clusters, len(frame)), random_state=0, n_init=10)
    frame["cluster"] = model.fit_predict(selected)
    frame = frame.groupby("cluster", sort=True, group_keys=False).head(cluster_k)
    return frame.sort_values("cluster", kind="stable").reset_index(drop=True)

# -------------------------
# ESM embedding
# -------------------------

def load_esm(model_name, device):
    import esm
    fn = getattr(esm.pretrained, model_name)
    model, alphabet = fn()
    model = model.eval().to(device)
    return model, alphabet

def embed(seqs: List[str], model, alphabet, device, batch=16):
    import torch
    bc = alphabet.get_batch_converter()
    pad = alphabet.padding_idx
    out = []

    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i+batch]
            data = [(f"s{i+j}", s) for j,s in enumerate(chunk)]
            _, _, toks = bc(data)
            toks = toks.to(device)
            res = model(toks, repr_layers=[model.num_layers])
            reps = res["representations"][model.num_layers]  # [B,T,C]
            mask = (toks != pad).float()
            reps = reps[:,1:,:]
            mask = mask[:,1:]
            for b in range(reps.shape[0]):
                nz = torch.where(mask[b] > 0.5)[0]
                if len(nz) > 0:
                    mask[b, nz[-1]] = 0.0
            emb = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
            out.append(emb.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)

# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="CSV with candidate protein sequences")
    ap.add_argument("--wt_csv", required=True, help="WT/reference sequence CSV")
    ap.add_argument("--out_csv", default="step1_filtered.csv")
    ap.add_argument("--candidate_seq_col", default=None)
    ap.add_argument("--wt_seq_col", default=None)

    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--use_gpu", action="store_true")

    ap.add_argument("--pll_drop", type=float, default=0.0,
                    help="Reserved compatibility option; must be 0 because PLL is not implemented")
    ap.add_argument("--sim_drop", type=float, default=0.15,
                    help="drop bottom fraction by WT similarity")

    ap.add_argument("--cluster_k", type=int, default=300,
                    help="max sequences per embedding cluster")
    ap.add_argument("--n_clusters", type=int, default=20,
                    help="kmeans clusters (rough diversity control)")

    args = ap.parse_args()
    if args.pll_drop != 0:
        ap.error("--pll_drop must be 0: this filter does not compute PLL")
    if not 0 <= args.sim_drop < 1:
        ap.error("--sim_drop must be in [0, 1)")
    if args.n_clusters < 1 or args.cluster_k < 1 or args.batch_size < 1:
        ap.error("--n_clusters, --cluster_k, and --batch_size must be positive")

    # ---- load sequences
    cand_df = pd.read_csv(args.candidates)
    wt_df   = pd.read_csv(args.wt_csv)

    ccol = find_seq_col(cand_df, args.candidate_seq_col)
    wcol = find_seq_col(wt_df, args.wt_seq_col)

    cand_seqs = [sanitize_seq(s) for s in cand_df[ccol]]
    wt_seqs   = [sanitize_seq(s) for s in wt_df[wcol]]
    if not cand_seqs or not wt_seqs or any(not seq for seq in cand_seqs + wt_seqs):
        ap.error("Candidate and reference CSVs must contain nonempty protein sequences")

    # ---- embed
    import torch
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    model, alphabet = load_esm(args.model, device)

    emb_c = embed(cand_seqs, model, alphabet, device, batch=args.batch_size)
    emb_w = embed(wt_seqs,   model, alphabet, device, batch=args.batch_size)

    df = select_candidates(cand_seqs, emb_c, emb_w, args.sim_drop,
                           args.n_clusters, args.cluster_k)

    # ---- write output (Step 1 DONE)
    df.to_csv(args.out_csv, index=False)
    print("[OK] Step 1 complete")
    print("  kept:", len(df), "sequences")
    print("  output:", args.out_csv)

if __name__ == "__main__":
    main()

