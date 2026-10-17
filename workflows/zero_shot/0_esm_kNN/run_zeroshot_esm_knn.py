import os
import argparse
import numpy as np
import pandas as pd
import torch

import esm
from sklearn.neighbors import NearestNeighbors


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-8)


@torch.no_grad()
def esm_mean_embedding(
    model,
    alphabet,
    seqs,
    device: str,
    layer: int,
    batch_size: int,
) -> np.ndarray:
    """
    Mean-pool token representations (exclude BOS/EOS, exclude padding).
    Returns (N, D) float32.
    """
    batch_converter = alphabet.get_batch_converter()
    out = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        data = [(f"seq_{i+j}", s) for j, s in enumerate(batch)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)
        
        # each residue is been convert to a embedding vector (D), and get the mean of resdues, the result is a sequence-level embedding.(averaging residue-level embedding is known as average pooling or mean pooling)
        # The overall function/stability of a sequence can be approximated by the average of the residue embeddings.
        res = model(toks, repr_layers=[layer], return_contacts=False)
        rep = res["representations"][layer]  # (B, L, D)

        lengths = (toks != alphabet.padding_idx).sum(1)  # includes BOS/EOS
        for b in range(rep.size(0)):
            L = int(lengths[b].item())
            # token indices: 0=BOS, L-1=EOS, [1:L-1) are residues
            emb = rep[b, 1:L-1].mean(0)
            out.append(emb.detach().float().cpu().numpy())

    return np.stack(out).astype(np.float32)


def load_esm2(model_name: str):
    # Expand if you want other variants
    if model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        layer = 12
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        layer = 30
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        layer = 33
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model, alphabet, layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default="predictive-pet-zero-shot-test-2025.csv")
    ap.add_argument("--wt_csv", default="pet-2025-wildtype-cds.csv")
    ap.add_argument("--out_csv", default="predictive-pet-zero-shot-submission.csv")
    ap.add_argument("--cache_dir", default="./cache_esm")
    ap.add_argument("--model", default="esm2_t12_35M_UR50D")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--knn_k", type=int, default=1)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    
    # get input data
    test = pd.read_csv(args.test_csv)
    wt = pd.read_csv(args.wt_csv)

    test_seqs = test["sequence"].astype(str).tolist()
    wt_seqs = wt["Wt AA Sequence"].astype(str).tolist()

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] device={device}")
    print(f"[INFO] TORCH_HOME={os.environ.get('TORCH_HOME')}")
    print(f"[INFO] XDG_CACHE_HOME={os.environ.get('XDG_CACHE_HOME')}")

    # load esm2, default is t12_35M, relatively small
    model, alphabet, default_layer = load_esm2(args.model)
    layer = default_layer
    model = model.to(device).eval()

    wt_emb_path = os.path.join(args.cache_dir, f"wt_emb_{args.model}.npy")
    test_emb_path = os.path.join(args.cache_dir, f"test_emb_{args.model}.npy")

    # using function esm_mean_embedding to get sequence embedding
    if os.path.exists(wt_emb_path):
        wt_emb = np.load(wt_emb_path)
        print(f"[INFO] loaded wt_emb {wt_emb.shape} from {wt_emb_path}")
    else:
        print(f"[INFO] computing wt embeddings: N={len(wt_seqs)}")
        wt_emb = esm_mean_embedding(model, alphabet, wt_seqs, device, layer, args.batch_size)
        np.save(wt_emb_path, wt_emb)
        print(f"[INFO] saved wt_emb to {wt_emb_path}")

    if os.path.exists(test_emb_path):
        test_emb = np.load(test_emb_path)
        print(f"[INFO] loaded test_emb {test_emb.shape} from {test_emb_path}")
    else:
        print(f"[INFO] computing test embeddings: N={len(test_seqs)}")
        test_emb = esm_mean_embedding(model, alphabet, test_seqs, device, layer, args.batch_size)
        np.save(test_emb_path, test_emb)
        print(f"[INFO] saved test_emb to {test_emb_path}")

    # kNN on WT embeddings
    nn = NearestNeighbors(n_neighbors=args.knn_k, metric="cosine", n_jobs=-1)
    nn.fit(wt_emb)
    dist, idx = nn.kneighbors(test_emb, return_distance=True)
    sim = (1.0 - dist).astype(np.float32)  # (N, K)

    # Use top-1 similarity for the simplest baseline
    sim1 = sim[:, 0]
    nn_wt_idx = idx[:, 0].astype(int)

    # Ranking-oriented heuristic to produce 3 properties
    # We are not predicting the actual values but constructing "ranking scores".
    # expression is usually strongly correlated with "sequence nativeness / folding ability"; activity is less sensitive under mild condition, while more sensitive when under more stringent conditions.
    expr = minmax(sim1)
    act1 = minmax(sim1 ** 1.0)
    act2 = minmax(sim1 ** 2.0)

    out = pd.DataFrame({
        "sequence": test["sequence"],
        "activity_1 (μmol [TPA]/min·mg [E])": act1,
        "activity_2 (μmol [TPA]/min·mg [E])": act2,
        "expression (mg/mL)": expr,
    })
    out.to_csv(args.out_csv, index=False)
    print(f"[INFO] wrote submission: {args.out_csv}")

    # Debug file (optional but helpful)
    dbg_path = os.path.join(args.cache_dir, "debug_nn.csv")
    dbg = pd.DataFrame({
        "sequence": test["sequence"],
        "nn_wt_index": nn_wt_idx,
        "sim": sim1,
    })
    dbg.to_csv(dbg_path, index=False)
    print(f"[INFO] wrote debug: {dbg_path}")


if __name__ == "__main__":
    main()

