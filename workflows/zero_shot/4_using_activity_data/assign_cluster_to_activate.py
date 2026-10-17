#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, hashlib, argparse
import numpy as np
import pandas as pd

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def md5_str(s): return hashlib.md5(s.encode("utf-8")).hexdigest()
def sanitize(s):
    s = str(s).strip().upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", s)

def cosine_sim_matrix(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def load_cache_np(cache_dir, key):
    npy = os.path.join(cache_dir, f"{key}.npy")
    meta = os.path.join(cache_dir, f"{key}.meta.json")
    if not (os.path.exists(npy) and os.path.exists(meta)): return None
    return np.load(npy)

def save_cache_np(cache_dir, key, arr, meta_obj):
    ensure_dir(cache_dir)
    np.save(os.path.join(cache_dir, f"{key}.npy"), arr)
    with open(os.path.join(cache_dir, f"{key}.meta.json"), "w") as f:
        json.dump(meta_obj, f, indent=2)

def load_esm_model(model_name, device):
    import esm
    fn = getattr(esm.pretrained, model_name, None)
    if fn is None: raise ValueError(f"Unknown model: {model_name}")
    model, alphabet = fn()
    return model.eval().to(device), alphabet

def embed_sequences(seqs, model, alphabet, device, batch_size=16, repr_layer=None):
    import torch
    if repr_layer is None: repr_layer = max(model.num_layers, 1)
    batch_converter = alphabet.get_batch_converter()
    pad_idx = alphabet.padding_idx
    outs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i+batch_size]
            data = [(f"s{i+j}", s) for j,s in enumerate(chunk)]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)
            res = model(toks, repr_layers=[repr_layer], return_contacts=False)
            reps = res["representations"][repr_layer]
            mask = (toks != pad_idx).float()
            reps_no_bos = reps[:, 1:, :]
            mask_no_bos = mask[:, 1:]
            for b in range(reps_no_bos.shape[0]):
                nonpad = torch.where(mask_no_bos[b] > 0.5)[0]
                if len(nonpad) > 0:
                    mask_no_bos[b, nonpad[-1].item()] = 0.0
            denom = mask_no_bos.sum(dim=1, keepdim=True).clamp(min=1.0)
            emb = (reps_no_bos * mask_no_bos.unsqueeze(-1)).sum(dim=1) / denom
            outs.append(emb.detach().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activate_tidy", default="activate_out/activate_tidy.csv")
    ap.add_argument("--cluster_csv", default="clustered_sequences.csv")
    ap.add_argument("--out_csv", default="activate_out/activate_with_cluster.csv")
    ap.add_argument("--cache_dir", default="cache_assign_cluster")
    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    ap.add_argument("--repr_layer", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--use_gpu", action="store_true")
    ap.add_argument("--knn", type=int, default=7, help="kNN over clustered reference for voting")
    ap.add_argument("--merge_34", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.cache_dir)

    act = pd.read_csv(args.activate_tidy).copy()
    act["sequence"] = act["sequence"].map(sanitize)
    act_uniq = act["sequence"].drop_duplicates().tolist()

    clu = pd.read_csv(args.cluster_csv).copy()
    if "sequence" not in clu.columns:
        seq_col = [c for c in clu.columns if "seq" in c.lower()][0]
        clu = clu.rename(columns={seq_col:"sequence"})
    if "cluster" not in clu.columns:
        raise ValueError("cluster_csv must have column 'cluster'")
    clu["sequence"] = clu["sequence"].map(sanitize)
    clu["cluster"] = clu["cluster"].astype(int)

    ref_uniq = clu["sequence"].drop_duplicates().tolist()
    # map ref uniq -> cluster (if duplicates, keep first)
    ref_map = clu.drop_duplicates("sequence")[["sequence","cluster"]].set_index("sequence")["cluster"].to_dict()
    ref_cluster = np.array([ref_map[s] for s in ref_uniq], dtype=np.int32)

    # device
    import torch
    device = torch.device("cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    key_act = f"emb_act_{args.model}_{md5_str(''.join(act_uniq[:50]))}_{len(act_uniq)}"
    key_ref = f"emb_ref_{args.model}_{md5_str(''.join(ref_uniq[:50]))}_{len(ref_uniq)}"

    act_emb = load_cache_np(args.cache_dir, key_act)
    ref_emb = load_cache_np(args.cache_dir, key_ref)

    if act_emb is None or ref_emb is None:
        model, alphabet = load_esm_model(args.model, device)
        if ref_emb is None:
            ref_emb = embed_sequences(ref_uniq, model, alphabet, device, args.batch_size, args.repr_layer)
            save_cache_np(args.cache_dir, key_ref, ref_emb, {"kind":"ref","model":args.model,"n":len(ref_uniq)})
        if act_emb is None:
            act_emb = embed_sequences(act_uniq, model, alphabet, device, args.batch_size, args.repr_layer)
            save_cache_np(args.cache_dir, key_act, act_emb, {"kind":"act","model":args.model,"n":len(act_uniq)})

    sims = cosine_sim_matrix(act_emb, ref_emb)  # [N_act, N_ref]
    k = min(max(1, args.knn), sims.shape[1])
    topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]  # unsorted k indices

    # majority vote (tie-break by average similarity)
    pred_cluster = np.zeros(len(act_uniq), dtype=np.int32)
    vote_conf = np.zeros(len(act_uniq), dtype=np.float32)

    for i in range(len(act_uniq)):
        nbrs = topk_idx[i]
        labs = ref_cluster[nbrs]
        # counts
        vals, cnts = np.unique(labs, return_counts=True)
        best = vals[np.argmax(cnts)]
        pred_cluster[i] = int(best)
        vote_conf[i] = float(np.max(cnts) / k)

    assign_df = pd.DataFrame({"sequence": act_uniq, "cluster": pred_cluster, "vote_conf": vote_conf})
    if args.merge_34:
        assign_df["cluster_group"] = assign_df["cluster"]
        assign_df.loc[assign_df["cluster"].isin([3,4]), "cluster_group"] = 34
    else:
        assign_df["cluster_group"] = assign_df["cluster"]

    out = act.merge(assign_df, on="sequence", how="left")
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("cluster_group counts (unique seq):")
    print(assign_df["cluster_group"].value_counts().head(30))

if __name__ == "__main__":
    main()

