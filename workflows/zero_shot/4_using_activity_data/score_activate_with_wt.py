#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hashlib
import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def md5_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(np.float64)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

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

def load_esm_model(model_name: str, device):
    import torch
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

    out = []
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
            out.append(emb.detach().cpu().numpy())

    return np.concatenate(out, axis=0).astype(np.float32)

def sanitize_seq(s: str) -> str:
    s = str(s).strip().upper()
    # keep only standard AA letters
    import re
    s = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activate_tidy_csv", default="activate_out/activate_tidy.csv")
    ap.add_argument("--wt_csv", default="pet-2025-wildtype-cds.csv")
    ap.add_argument("--wt_seq_col", default="Wt AA Sequence")  # Canonical WT protein column.
    ap.add_argument("--out_dir", default="activate_out")
    ap.add_argument("--cache_dir", default="cache_activate_wt")
    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    ap.add_argument("--repr_layer", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--k", type=int, default=5, help="top-k WT neighbors for sim_wt_topk")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    # ---- load data
    act = pd.read_csv(args.activate_tidy_csv)
    wt = pd.read_csv(args.wt_csv)

    if args.wt_seq_col not in wt.columns:
        raise ValueError(f"WT sequence column '{args.wt_seq_col}' not found. Columns: {list(wt.columns)}")

    if "sequence" not in act.columns or "value" not in act.columns:
        raise ValueError(f"activate_tidy_csv must contain columns: sequence, value. Got: {list(act.columns)}")

    # sanitize sequences
    act = act.copy()
    act["sequence"] = act["sequence"].map(sanitize_seq)
    wt_seqs = wt[args.wt_seq_col].map(sanitize_seq).tolist()

    # unique activate sequences
    uniq_act = act["sequence"].drop_duplicates().tolist()

    # ---- device
    import torch
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    # ---- cache keys
    key_wt = f"emb_wt_{args.model}_{md5_str(''.join(wt_seqs[:50]))}_{len(wt_seqs)}"
    key_act = f"emb_act_{args.model}_{md5_str(''.join(uniq_act[:50]))}_{len(uniq_act)}"

    wt_emb = None
    act_emb = None

    got = load_cache_np(args.cache_dir, key_wt)
    if got is not None:
        wt_emb = got[0]
        print("[OK] loaded WT emb from cache:", key_wt)

    got = load_cache_np(args.cache_dir, key_act)
    if got is not None:
        act_emb = got[0]
        print("[OK] loaded ACT emb from cache:", key_act)

    if wt_emb is None or act_emb is None:
        print("[INFO] computing embeddings with", args.model)
        model, alphabet = load_esm_model(args.model, device)

        if wt_emb is None:
            wt_emb = embed_sequences(wt_seqs, model, alphabet, device, batch_size=args.batch_size, repr_layer=args.repr_layer)
            save_cache_np(args.cache_dir, key_wt, wt_emb, {"model": args.model, "kind": "wt_emb", "n": len(wt_seqs)})
        if act_emb is None:
            act_emb = embed_sequences(uniq_act, model, alphabet, device, batch_size=args.batch_size, repr_layer=args.repr_layer)
            save_cache_np(args.cache_dir, key_act, act_emb, {"model": args.model, "kind": "act_emb", "n": len(uniq_act)})

    # ---- similarity scores against WT library
    sims = cosine_sim_matrix(act_emb, wt_emb)  # [N_act_unique, N_wt]

    k = max(1, min(args.k, sims.shape[1]))
    topk = np.partition(sims, -k, axis=1)[:, -k:]               # [N, k]
    sim_wt_topk = topk.mean(axis=1).astype(np.float32)          # avg top-k
    sim_wt_max = sims.max(axis=1).astype(np.float32)            # best match
    wt_centroid = wt_emb.mean(axis=0, keepdims=True)
    sim_wt_centroid = cosine_sim_matrix(act_emb, wt_centroid).ravel().astype(np.float32)

    score_df = pd.DataFrame({
        "sequence": uniq_act,
        "sim_wt_topk": sim_wt_topk,
        "sim_wt_max": sim_wt_max,
        "sim_wt_centroid": sim_wt_centroid,
    })

    # merge scores back to row-level observations
    out = act.merge(score_df, on="sequence", how="left")

    scored_path = os.path.join(args.out_dir, "activate_scored_with_wt.csv")
    out.to_csv(scored_path, index=False)
    print("[OK] wrote:", scored_path)

    # ---- diagnostics by condition
    group_cols = []
    for c in ["pH", "temperature_C", "substrate"]:
        if c in out.columns:
            group_cols.append(c)

    diag_rows = []
    if len(group_cols) == 0:
        print("[WARN] No condition columns found; will compute overall correlation only.")
        group_iter = [(("ALL",), out)]
    else:
        group_iter = out.groupby(group_cols)

    for key, g in group_iter:
        g2 = g.dropna(subset=["value", "sim_wt_topk"])
        if len(g2) < 5:
            continue
        v = g2["value"].to_numpy(np.float32)

        row = {}
        if len(group_cols) > 0:
            if not isinstance(key, tuple):
                key = (key,)
            for col, val in zip(group_cols, key):
                row[col] = val
        row["n"] = int(len(g2))
        row["spearman_topk"] = safe_spearman(g2["sim_wt_topk"].to_numpy(np.float32), v)
        row["spearman_max"] = safe_spearman(g2["sim_wt_max"].to_numpy(np.float32), v)
        row["spearman_centroid"] = safe_spearman(g2["sim_wt_centroid"].to_numpy(np.float32), v)
        row["value_mean"] = float(np.mean(v))
        row["value_std"] = float(np.std(v))
        diag_rows.append(row)

    diag = pd.DataFrame(diag_rows)
    diag_path = os.path.join(args.out_dir, "activate_diag_by_condition.csv")
    diag.to_csv(diag_path, index=False)
    print("[OK] wrote:", diag_path)

    # also print top lines for quick look
    if len(diag) > 0:
        print("\nTop conditions by |spearman_topk|:")
        print(diag.assign(abs_sp=lambda d: d["spearman_topk"].abs()).sort_values("abs_sp", ascending=False).head(15))

if __name__ == "__main__":
    main()

