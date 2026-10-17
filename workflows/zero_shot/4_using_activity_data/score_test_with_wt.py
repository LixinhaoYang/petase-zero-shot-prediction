import os
import re
import math
import argparse
import hashlib
import numpy as np
import pandas as pd

import torch

try:
    import esm
except Exception as e:
    raise RuntimeError(
        "ImportError: cannot import esm. You likely need fair-esm installed.\n"
        "Try: pip install fair-esm\n"
        f"Original error: {e}"
    )

AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")

def sanitize_seq(s: str) -> str:
    s = str(s).strip().upper()
    return AA_RE.sub("", s)

def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def mean_pool(repr_tensor, tokens, pad_idx):
    """
    repr_tensor: [B, L, C]
    tokens: [B, L]
    mean over non-pad and non-special tokens (BOS=0th token, EOS token)
    """
    # mask: not pad
    mask = tokens.ne(pad_idx)
    # exclude BOS (position 0)
    mask[:, 0] = False
    # exclude EOS: token == eos_idx
    # We'll remove positions where token == eos_idx
    # (esm's alphabet has eos_idx)
    # This is handled outside by passing eos_idx
    # but we do here by excluding last non-pad token often is eos.
    return mask

def compute_embeddings(seqs, model, alphabet, device="cuda", batch_size=16):
    """
    seqs: list[str] AA sequences
    return: np.ndarray [N, C] float32
    """
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    pad_idx = alphabet.padding_idx
    eos_idx = alphabet.eos_idx

    embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            data = [(f"seq{i+j}", s) for j, s in enumerate(batch)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)

            out = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
            reps = out["representations"][model.num_layers]  # [B, L, C]

            # build mask: not pad, not BOS, not EOS
            mask = tokens.ne(pad_idx)
            mask[:, 0] = False
            mask = mask & tokens.ne(eos_idx)

            # mean pool
            mask_f = mask.unsqueeze(-1).type_as(reps)  # [B, L, 1]
            summed = (reps * mask_f).sum(dim=1)        # [B, C]
            denom = mask_f.sum(dim=1).clamp(min=1.0)   # [B, 1]
            pooled = summed / denom                    # [B, C]

            embs.append(pooled.detach().cpu().float().numpy())

    return np.concatenate(embs, axis=0).astype(np.float32)

def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def topk_mean_cosine(test_emb, wt_emb, k=5, device="cuda", wt_chunk=4096):
    """
    Compute sim_wt_topk and sim_wt_max for each test embedding vs all wt embeddings.
    Uses torch for speed; chunks WT to control memory.
    """
    test = torch.from_numpy(test_emb).to(device)
    wt = torch.from_numpy(wt_emb).to(device)

    # normalize
    test = test / (test.norm(dim=1, keepdim=True) + 1e-12)
    wt = wt / (wt.norm(dim=1, keepdim=True) + 1e-12)

    N = test.shape[0]
    sim_max = torch.full((N,), -1.0, device=device)
    sim_topk = torch.full((N, k), -1.0, device=device)

    with torch.no_grad():
        for j in range(0, wt.shape[0], wt_chunk):
            w = wt[j:j+wt_chunk]  # [M, C]
            sims = torch.matmul(test, w.t())  # [N, Mchunk]

            # update max
            chunk_max, _ = sims.max(dim=1)
            sim_max = torch.maximum(sim_max, chunk_max)

            # update topk: merge existing topk with chunk topk
            chunk_topk, _ = torch.topk(sims, k=min(k, sims.shape[1]), dim=1, largest=True, sorted=False)
            merged = torch.cat([sim_topk, chunk_topk], dim=1)  # [N, 2k]
            sim_topk, _ = torch.topk(merged, k=k, dim=1, largest=True, sorted=False)

    sim_topk_mean = sim_topk.mean(dim=1).detach().cpu().numpy().astype(np.float32)
    sim_max_np = sim_max.detach().cpu().numpy().astype(np.float32)
    return sim_topk_mean, sim_max_np

def cosine_to_centroid(test_emb, wt_emb):
    """
    sim_wt_centroid: cosine similarity between test embedding and WT centroid.
    """
    wt_centroid = wt_emb.mean(axis=0, keepdims=True).astype(np.float32)  # [1, C]
    wt_centroid = wt_centroid / (np.linalg.norm(wt_centroid, axis=1, keepdims=True) + 1e-12)
    t = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-12)
    return (t @ wt_centroid.T).reshape(-1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, help="Test CSV containing a sequence column.")
    ap.add_argument("--seq_col", default="sequence", help="Sequence column name in test CSV.")
    ap.add_argument("--wt_csv", default="pet-2025-wildtype-cds.csv", help="WT CSV path.")
    ap.add_argument("--wt_seq_col", default="Wt AA Sequence", help="WT AA sequence column.")
    ap.add_argument("--model", default="esm2_t30_150M_UR50D", help="ESM2 model name.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--wt_chunk", type=int, default=4096)
    ap.add_argument("--cache_dir", default="cache_wt_sim", help="Cache directory for embeddings.")
    ap.add_argument("--save_emb", action="store_true", help="Save test_emb.npy and wt_emb.npy into cache.")
    ap.add_argument("--out_csv", default="test_scored_with_wt.csv")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Load sequences
    test = pd.read_csv(args.test_csv)
    if args.seq_col not in test.columns:
        raise ValueError(f"seq_col '{args.seq_col}' not found in test columns: {test.columns.tolist()}")

    test_seq = test[args.seq_col].map(sanitize_seq).tolist()
    wt = pd.read_csv(args.wt_csv)
    if args.wt_seq_col not in wt.columns:
        raise ValueError(f"wt_seq_col '{args.wt_seq_col}' not found in WT columns: {wt.columns.tolist()}")
    wt_seq = wt[args.wt_seq_col].map(sanitize_seq).tolist()

    # Build cache keys (stable-ish)
    key_test = f"test_{args.model}_{len(test_seq)}_{md5(''.join(test_seq[:50]))}"
    key_wt = f"wt_{args.model}_{len(wt_seq)}_{md5(''.join(wt_seq[:50]))}"

    test_npy = os.path.join(args.cache_dir, f"{key_test}.npy")
    wt_npy = os.path.join(args.cache_dir, f"{key_wt}.npy")

    # Load model
    print(f"[INFO] loading model {args.model} on {args.device} ...")
    model, alphabet = getattr(esm.pretrained, args.model)()
    model = model.to(args.device)

    # Load or compute WT embeddings
    if os.path.exists(wt_npy):
        print("[INFO] loading cached WT embeddings:", wt_npy)
        wt_emb = np.load(wt_npy)
    else:
        print("[INFO] computing WT embeddings, n =", len(wt_seq))
        wt_emb = compute_embeddings(wt_seq, model, alphabet, device=args.device, batch_size=args.batch_size)
        if args.save_emb:
            np.save(wt_npy, wt_emb)
            print("[INFO] saved WT embeddings:", wt_npy)

    # Load or compute TEST embeddings
    if os.path.exists(test_npy):
        print("[INFO] loading cached TEST embeddings:", test_npy)
        test_emb = np.load(test_npy)
    else:
        print("[INFO] computing TEST embeddings, n =", len(test_seq))
        test_emb = compute_embeddings(test_seq, model, alphabet, device=args.device, batch_size=args.batch_size)
        if args.save_emb:
            np.save(test_npy, test_emb)
            print("[INFO] saved TEST embeddings:", test_npy)

    # Compute similarities
    print("[INFO] computing cosine similarities to WT (topk/max/centroid) ...")
    sim_topk, sim_max = topk_mean_cosine(test_emb, wt_emb, k=args.topk, device=args.device, wt_chunk=args.wt_chunk)
    sim_centroid = cosine_to_centroid(test_emb, wt_emb)

    out = pd.DataFrame({
        "sequence": test_seq,
        "sim_wt_topk": sim_topk,
        "sim_wt_max": sim_max,
        "sim_wt_centroid": sim_centroid,
    })
    out.to_csv(args.out_csv, index=False)
    print("[OK] wrote:", args.out_csv)
    print("[INFO] sim_wt_topk range:", float(np.min(sim_topk)), float(np.max(sim_topk)))

if __name__ == "__main__":
    main()

