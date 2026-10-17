import os
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import esm
from sklearn.neighbors import NearestNeighbors


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@torch.no_grad()
def esm_mean_embedding(model, alphabet, seqs, device: str, layer: int, batch_size: int) -> np.ndarray:
    """
    Mean pool residue embeddings (exclude BOS/EOS, exclude padding).
    Returns (N, D) float32.
    """
    batch_converter = alphabet.get_batch_converter()
    out = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        data = [(f"seq_{i+j}", s) for j, s in enumerate(batch)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)

        res = model(toks, repr_layers=[layer], return_contacts=False)
        rep = res["representations"][layer]  # (B, L, D)

        lengths = (toks != alphabet.padding_idx).sum(1)  # includes BOS/EOS
        for b in range(rep.size(0)):
            L = int(lengths[b].item())
            emb = rep[b, 1:L-1].mean(0)
            out.append(emb.detach().float().cpu().numpy())

    return np.stack(out).astype(np.float32)


@torch.no_grad()
def pll_per_sequence(
    model,
    alphabet,
    seq: str,
    device: str,
    mask_chunk: int = 32,
) -> float:
    r"""
    Compute per-residue pseudo-log-likelihood (PLL) for one sequence:
      PLL = (1/L) * sum_i log p(x_i | x_{\i}) approximated by masking each position.
    Returns a float (typically negative), higher is "more natural".
    """
    batch_converter = alphabet.get_batch_converter()
    data = [("seq", seq)]
    _, _, toks = batch_converter(data)  # (1, T)
    toks = toks.to(device)

    # sequence length includes BOS/EOS; residues are positions [1, L-2]
    L = int((toks != alphabet.padding_idx).sum().item())
    start = 1
    end = L - 1  # EOS index is end-1, residues stop at end-2
    residue_positions = list(range(start, end - 1))  # [1 .. L-2]
    n_res = len(residue_positions)
    if n_res <= 0:
        return float("-inf")

    total_ll = 0.0

    # True tokens for residues
    true_tokens = toks[0, :]  # (T,)

    # Mask positions in chunks to amortize forward passes
    for c0 in range(0, n_res, mask_chunk):
        pos_chunk = residue_positions[c0:c0 + mask_chunk]
        B = len(pos_chunk)

        # Create B copies of toks
        toks_rep = toks.repeat(B, 1)  # (B, T)

        # Apply mask at each copy's corresponding position
        for b, p in enumerate(pos_chunk):
            toks_rep[b, p] = alphabet.mask_idx

        # Forward pass
        out = model(toks_rep, repr_layers=[], return_contacts=False)
        logits = out["logits"]  # (B, T, vocab)

        # log prob of the original token at the masked position
        # Use log_softmax over vocab for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, vocab)

        for b, p in enumerate(pos_chunk):
            t = int(true_tokens[p].item())
            total_ll += float(log_probs[b, p, t].item())

    pll = total_ll / float(n_res)  # per-residue average
    return pll


def load_esm2(model_name: str):
    if model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        layer = 12
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        layer = 30
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model, alphabet, layer


def load_or_compute_pll(
    model,
    alphabet,
    seqs,
    device: str,
    cache_dir: str,
    cache_prefix: str,
    mask_chunk: int,
    resume: bool = True,
) -> np.ndarray:
    """
    Cache PLL per-sequence using SHA1(sequence) keys.
    Writes a jsonl cache file and a final .npy array aligned to seqs order.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Persistent map: sha1 -> pll
    map_file = cache_path / f"{cache_prefix}_pll_map.jsonl"
    out_npy = cache_path / f"{cache_prefix}_pll.npy"

    pll_map = {}
    if resume and map_file.exists():
        with open(map_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pll_map[obj["k"]] = obj["pll"]

    # Compute missing
    new_lines = []
    for i, s in enumerate(seqs):
        k = sha1(s)
        if k in pll_map:
            continue
        pll = pll_per_sequence(model, alphabet, s, device=device, mask_chunk=mask_chunk)
        pll_map[k] = pll
        new_lines.append({"k": k, "pll": pll})

        # flush periodically to avoid losing progress
        if len(new_lines) >= 50:
            with open(map_file, "a") as f:
                for obj in new_lines:
                    f.write(json.dumps(obj) + "\n")
            new_lines = []
            print(f"[INFO] PLL cached up to i={i}/{len(seqs)}")

    if new_lines:
        with open(map_file, "a") as f:
            for obj in new_lines:
                f.write(json.dumps(obj) + "\n")

    # Create array aligned to seqs
    pll_arr = np.array([pll_map[sha1(s)] for s in seqs], dtype=np.float32)
    np.save(out_npy, pll_arr)
    return pll_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="predictive-pet-zero-shot-test-2025.csv")
    ap.add_argument("--wt_csv", default="pet-2025-wildtype-cds.csv")
    ap.add_argument("--out_csv", default="predictive-pet-zero-shot-submission.csv")
    ap.add_argument("--cache_dir", default="./cache_esm")

    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--knn_alpha", type=float, default=10.0)

    ap.add_argument("--use_gpu", action="store_true")

    ap.add_argument("--use_pll", action="store_true")
    ap.add_argument("--pll_mask_chunk", type=int, default=32)

    # scoring weights
    ap.add_argument("--expr_w_pll", type=float, default=0.7)
    ap.add_argument("--expr_w_sim", type=float, default=0.3)
    ap.add_argument("--act1_w_sim", type=float, default=0.6)
    ap.add_argument("--act1_w_pll", type=float, default=0.4)

    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    test = pd.read_csv(args.test_csv)
    wt = pd.read_csv(args.wt_csv)

    test_seqs = test["sequence"].astype(str).tolist()
    wt_seqs = wt["Wt AA Sequence"].astype(str).tolist()

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] TORCH_HOME={os.environ.get('TORCH_HOME')}")
    print(f"[INFO] XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME')}")

    model, alphabet, layer = load_esm2(args.model)
    model = model.to(device).eval()

    # === Embeddings (cached) ===
    wt_emb_path = os.path.join(args.cache_dir, f"wt_emb_{args.model}.npy")
    test_emb_path = os.path.join(args.cache_dir, f"test_emb_{args.model}.npy")

    if os.path.exists(wt_emb_path):
        wt_emb = np.load(wt_emb_path)
        print(f"[INFO] loaded wt_emb {wt_emb.shape} from {wt_emb_path}")
    else:
        print(f"[INFO] computing wt embeddings: N={len(wt_seqs)}")
        wt_emb = esm_mean_embedding(model, alphabet, wt_seqs, device, layer, args.batch_size)
        np.save(wt_emb_path, wt_emb)

    if os.path.exists(test_emb_path):
        test_emb = np.load(test_emb_path)
        print(f"[INFO] loaded test_emb {test_emb.shape} from {test_emb_path}")
    else:
        print(f"[INFO] computing test embeddings: N={len(test_seqs)}")
        test_emb = esm_mean_embedding(model, alphabet, test_seqs, device, layer, args.batch_size)
        np.save(test_emb_path, test_emb)

    # === WT kNN similarity ===
    k = args.knn_k
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    nn.fit(wt_emb)
    dist, idx = nn.kneighbors(test_emb, return_distance=True)
    sim = (1.0 - dist).astype(np.float32)  # (N, k)

    # softmax weighting over similarity
    w = np.exp(args.knn_alpha * sim)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-8)
    sim_knn = (w * sim).sum(axis=1).astype(np.float32)

    # === PLL (optional) ===
    pll_test = None
    if args.use_pll:
        pll_npy = os.path.join(args.cache_dir, f"test_pll_{args.model}.npy")
        if os.path.exists(pll_npy):
            pll_test = np.load(pll_npy).astype(np.float32)
            print(f"[INFO] loaded pll_test {pll_test.shape} from {pll_npy}")
        else:
            print("[INFO] computing PLL for test sequences (cached incremental jsonl)...")
            pll_test = load_or_compute_pll(
                model=model,
                alphabet=alphabet,
                seqs=test_seqs,
                device=device,
                cache_dir=args.cache_dir,
                cache_prefix=f"test_{args.model}",
                mask_chunk=args.pll_mask_chunk,
                resume=True,
            )
            print(f"[INFO] saved pll_test to {pll_npy}")

        pll_norm = minmax(pll_test)  # higher better
    else:
        pll_norm = None

    # === Scoring heuristics ===
    sim_norm = minmax(sim_knn)

    if pll_norm is None:
        expr = sim_norm
        act1 = sim_norm
        act2 = minmax(sim_norm ** 2)
    else:
        # expression: more weight on PLL (stability/naturalness)
        expr_score = args.expr_w_pll * pll_norm + args.expr_w_sim * sim_norm
        expr = minmax(expr_score)

        # activity_1: slightly more weight on similarity (function-preserving)
        act1_score = args.act1_w_sim * sim_norm + args.act1_w_pll * pll_norm
        act1 = minmax(act1_score)

        # activity_2: strict; penalize deviations + require naturalness
        act2_score = (sim_norm ** 2) * (0.5 + 0.5 * pll_norm)
        act2 = minmax(act2_score)

    out = pd.DataFrame({
        "sequence": test["sequence"],
        "activity_1 (μmol [TPA]/min·mg [E])": act1,
        "activity_2 (μmol [TPA]/min·mg [E])": act2,
        "expression (mg/mL)": expr,
    })
    out.to_csv(args.out_csv, index=False)
    print(f"[INFO] wrote submission: {args.out_csv}")

    # Debug
    dbg = pd.DataFrame({
        "sequence": test["sequence"],
        "sim_knn": sim_knn,
        "sim_norm": sim_norm,
    })
    if pll_test is not None:
        dbg["pll"] = pll_test
        dbg["pll_norm"] = pll_norm
    dbg.to_csv(os.path.join(args.cache_dir, "debug_scores.csv"), index=False)
    print(f"[INFO] wrote debug: {os.path.join(args.cache_dir, 'debug_scores.csv')}")


if __name__ == "__main__":
    main()

