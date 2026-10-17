"""
activate.xlsx pipeline:
1) Read + tidy the excel (handle merged/blank cells via forward-fill)
2) Produce a tidy CSV with columns: sequence, temperature_C, pH, substrate, value
3) Optional: compute ESM2 embeddings for unique sequences
4) Optional: compute unsupervised similarity-derived scores (kNN density, centroid similarity)
5) Diagnostics: per (pH, temperature, substrate) Spearman correlation between score and measured value
"""

import os
import re
import json
import argparse
import hashlib
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

def md5_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace, collapse internal spaces
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    return df

def coerce_float(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    # remove units like "°C", "C", "mg/mL", etc.
    s = s.replace("°", "").replace("C", "").replace("c", "")
    # keep numbers, dot, minus
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m[0])
    except Exception:
        return np.nan

def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman correlation without scipy dependency:
    Spearman(x,y) = Pearson(rank(x), rank(y))
    """
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy(np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(np.float64)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def read_activate_excel(path: str, sheet: Optional[str] = None) -> pd.DataFrame:
    # Always prefer a single sheet DataFrame.
    # If pandas returns a dict (multiple sheets), pick 'activity' if present, else first sheet.
    df = pd.read_excel(path, sheet_name=sheet if sheet is not None else "activity", header=[0, 1])

    # If user passed sheet_name=None somewhere and got a dict, handle it
    if isinstance(df, dict):
        if "activity" in df:
            df = df["activity"]
        else:
            first_key = next(iter(df.keys()))
            df = df[first_key]

    # ---- rest of your flattening logic below ----
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for a, b in df.columns:
            a = "" if pd.isna(a) else str(a).strip()
            b = "" if pd.isna(b) else str(b).strip()

            a_low = a.lower()
            b_low = b.lower()
            b_is_unnamed = (b == "" or b_low.startswith("unnamed"))
            a_is_unnamed = (a == "" or a_low.startswith("unnamed"))

            if (not b_is_unnamed) and b != "":
                name = b
            elif (not a_is_unnamed) and a != "":
                name = a
            else:
                name = f"{a}_{b}".strip("_")

            flat.append(name)

        df.columns = flat

    df = normalize_columns(df)

    # Strong renames (keep your rename_map section)
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl.startswith("name of protein"):
            rename_map[c] = "name"
        elif cl == "sequence":
            rename_map[c] = "sequence"
        elif cl == "temperature":
            rename_map[c] = "temperature"
        elif cl == "ph":
            rename_map[c] = "pH"
        elif cl == "substrate":
            rename_map[c] = "substrate"
        elif cl == "yield":
            rename_map[c] = "Yield"
        elif cl.startswith("duration"):
            rename_map[c] = "Duration(h)"

    df = df.rename(columns=rename_map)
    return df

def tidy_activate(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to create a tidy table with:
      sequence, temperature_C, pH, substrate, value
    Handles merged/blank cells by forward-fill for key columns.
    """

    df = df_raw.copy()

    # Heuristic column detection (based on what commonly appears)
    seq_col = detect_col(df, ["Sequence", "sequence", "Seq", "AA sequence", "Protein sequence"])
    temp_col = detect_col(df, ["Temperature", "Temp", "temperature"])
    ph_col   = detect_col(df, ["pH", "PH", "ph"])
    sub_col  = detect_col(df, ["substrate", "Substrate", "substrates"])

    val_col  = detect_col(df, ["Yield", "Yield ", "yield", "Activity", "activity", "Value", "value"])

    if seq_col is None:
        raise ValueError(f"Could not find a sequence column in activate.xlsx. Columns seen: {list(df.columns)}")
    if val_col is None:
        raise ValueError(f"Could not find the measurement column (Yield/Activity/Value). Columns seen: {list(df.columns)}")

    # Keep only relevant columns that exist
    keep = [seq_col, val_col]
    if temp_col: keep.append(temp_col)
    if ph_col: keep.append(ph_col)
    if sub_col: keep.append(sub_col)
    df = df[keep].copy()

    # Forward-fill likely merged cells for metadata columns
    df[seq_col] = df[seq_col].astype(str).replace({"nan": np.nan, "None": np.nan})
    df[seq_col] = df[seq_col].where(df[seq_col].str.len() > 0, np.nan)
    df[seq_col] = df[seq_col].replace({"": np.nan}).ffill()

    if temp_col:
        df[temp_col] = df[temp_col].ffill()
    if ph_col:
        df[ph_col] = df[ph_col].ffill()
    if sub_col:
        df[sub_col] = df[sub_col].ffill()

    # Normalize fields
    out = pd.DataFrame()
    out["sequence"] = df[seq_col].astype(str).str.strip()

    if temp_col:
        out["temperature_C"] = df[temp_col].apply(coerce_float)
    else:
        out["temperature_C"] = np.nan

    if ph_col:
        out["pH"] = df[ph_col].apply(coerce_float)
    else:
        out["pH"] = np.nan

    if sub_col:
        out["substrate"] = df[sub_col].astype(str).str.strip()
        out["substrate"] = out["substrate"].replace({"nan": np.nan, "None": np.nan, "": np.nan})
    else:
        out["substrate"] = np.nan

    out["value"] = df[val_col].apply(coerce_float)

    # Drop fully empty / invalid rows
    out = out.replace({"sequence": {"nan": np.nan, "": np.nan}})
    out = out.dropna(subset=["sequence", "value"]).reset_index(drop=True)

    # Simple sequence sanitization: keep only AA letters
    out["sequence"] = out["sequence"].str.upper().str.replace(r"[^ACDEFGHIKLMNPQRSTVWY]", "", regex=True)
    out = out[out["sequence"].str.len() > 0].reset_index(drop=True)

    return out

def load_esm_model(model_name: str, device: str):
    import torch
    import esm
    fn = getattr(esm.pretrained, model_name, None)
    if fn is None:
        raise ValueError(f"Unknown esm pretrained model: {model_name}")
    model, alphabet = fn()
    model = model.eval().to(device)
    return model, alphabet

def embed_sequences(seqs: List[str], model, alphabet, device: str, batch_size: int = 16, repr_layer: Optional[int] = None) -> np.ndarray:
    import torch
    if repr_layer is None:
        repr_layer = max(model.num_layers, 1)

    batch_converter = alphabet.get_batch_converter()
    pad_idx = alphabet.padding_idx

    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i+batch_size]
            data = [(f"seq{i+j}", s) for j, s in enumerate(chunk)]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)

            res = model(toks, repr_layers=[repr_layer], return_contacts=False)
            reps = res["representations"][repr_layer]  # [B, T, C]

            mask = (toks != pad_idx).float()
            reps_no_bos = reps[:, 1:, :]
            mask_no_bos = mask[:, 1:]

            # exclude EOS (last non-pad token after BOS removal)
            for b in range(reps_no_bos.shape[0]):
                nonpad = torch.where(mask_no_bos[b] > 0.5)[0]
                if len(nonpad) > 0:
                    last = nonpad[-1].item()
                    mask_no_bos[b, last] = 0.0

            denom = mask_no_bos.sum(dim=1, keepdim=True).clamp(min=1.0)
            emb = (reps_no_bos * mask_no_bos.unsqueeze(-1)).sum(dim=1) / denom
            out.append(emb.detach().cpu().numpy())

    return np.concatenate(out, axis=0).astype(np.float32)

def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T

def knn_density_score(emb: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Unsupervised "density" score: average cosine similarity to k nearest neighbors (excluding self).
    """
    sims = cosine_sim_matrix(emb, emb)  # [N,N]
    np.fill_diagonal(sims, -np.inf)
    k = min(k, emb.shape[0] - 1) if emb.shape[0] > 1 else 1
    topk = np.partition(sims, -k, axis=1)[:, -k:]
    return np.mean(topk, axis=1).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activate_xlsx", default="activate.xlsx", help="Path to activate.xlsx")
    ap.add_argument("--sheet", default=None, help="Optional sheet name")
    ap.add_argument("--out_dir", default="activate_out", help="Output directory")
    ap.add_argument("--do_esm", action="store_true", help="Compute ESM embeddings for unique sequences")
    ap.add_argument("--esm_model", default="esm2_t30_150M_UR50D", help="esm.pretrained model name")
    ap.add_argument("--repr_layer", type=int, default=None, help="Representation layer (default: last)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda", help="cuda or cpu (will fallback if cuda unavailable)")
    ap.add_argument("--cache_dir", default="cache_activate_esm", help="Cache directory for embeddings")
    ap.add_argument("--knn_k", type=int, default=10, help="k for kNN density score")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    # 1) read + tidy
    df_raw = read_activate_excel(args.activate_xlsx, sheet=args.sheet)
    tidy = tidy_activate(df_raw)

    tidy_path = os.path.join(args.out_dir, "activate_tidy.csv")
    tidy.to_csv(tidy_path, index=False)

    # 2) aggregate per sequence (basic stats)
    seq_stats = (
        tidy.groupby("sequence", as_index=False)
            .agg(
                n_obs=("value", "size"),
                value_mean=("value", "mean"),
                value_median=("value", "median"),
                value_std=("value", "std"),
                pH_min=("pH", "min"),
                pH_max=("pH", "max"),
                temp_min=("temperature_C", "min"),
                temp_max=("temperature_C", "max"),
            )
    )
    seq_stats_path = os.path.join(args.out_dir, "activate_seq_stats.csv")
    seq_stats.to_csv(seq_stats_path, index=False)

    # 3) optional embeddings + unsupervised scores
    diag_rows = []

    if args.do_esm:
        # unique sequences
        uniq = tidy["sequence"].drop_duplicates().tolist()
        key = f"activate_{args.esm_model}_{md5_str(''.join(uniq[:50]))}_{len(uniq)}"
        npy_path = os.path.join(args.cache_dir, f"{key}.npy")
        meta_path = os.path.join(args.cache_dir, f"{key}.meta.json")

        if os.path.exists(npy_path) and os.path.exists(meta_path):
            emb = np.load(npy_path)
        else:
            # device resolve
            import torch
            dev = args.device
            if dev == "cuda" and not torch.cuda.is_available():
                dev = "cpu"

            model, alphabet = load_esm_model(args.esm_model, dev)
            emb = embed_sequences(uniq, model, alphabet, dev, batch_size=args.batch_size, repr_layer=args.repr_layer)
            np.save(npy_path, emb)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "model": args.esm_model,
                        "repr_layer": args.repr_layer,
                        "n_unique": len(uniq),
                        "key": key,
                    },
                    f,
                    indent=2,
                )

        # map back
        seq2idx = {s: i for i, s in enumerate(uniq)}
        tidy["emb_idx"] = tidy["sequence"].map(seq2idx).astype(int)

        # unsupervised score per unique sequence: kNN density in embedding space
        dens = knn_density_score(emb, k=args.knn_k)
        tidy["score_knn_density"] = tidy["emb_idx"].map(lambda i: float(dens[i]))

        # (optional extra) global centroid similarity as another unsupervised score
        centroid = emb.mean(axis=0, keepdims=True)
        sim_to_cent = cosine_sim_matrix(emb, centroid).ravel().astype(np.float32)
        tidy["score_sim_centroid"] = tidy["emb_idx"].map(lambda i: float(sim_to_cent[i]))

        # 4) diagnostics per condition
        group_cols = ["pH", "temperature_C", "substrate"]
        for (ph, temp, sub), g in tidy.groupby(group_cols):
            g2 = g.dropna(subset=["value", "score_knn_density"])
            if len(g2) < 5:
                continue
            sp1 = safe_spearman(g2["score_knn_density"].to_numpy(), g2["value"].to_numpy())
            sp2 = safe_spearman(g2["score_sim_centroid"].to_numpy(), g2["value"].to_numpy())
            diag_rows.append(
                {
                    "pH": ph,
                    "temperature_C": temp,
                    "substrate": sub,
                    "n": len(g2),
                    "spearman_knn_density_vs_value": sp1,
                    "spearman_sim_centroid_vs_value": sp2,
                    "value_mean": float(np.mean(g2["value"].to_numpy())),
                    "value_std": float(np.std(g2["value"].to_numpy())),
                }
            )

        # save updated tidy with scores
        tidy_scored_path = os.path.join(args.out_dir, "activate_tidy_scored.csv")
        tidy.to_csv(tidy_scored_path, index=False)

        # save embedding pointer file
        emb_out = os.path.join(args.out_dir, "activate_emb.npy")
        np.save(emb_out, emb)

    # diagnostics output (even if empty)
    diag = pd.DataFrame(diag_rows)
    diag_path = os.path.join(args.out_dir, "activate_condition_diagnostics.csv")
    diag.to_csv(diag_path, index=False)

    print("[OK] Wrote:")
    print(" -", tidy_path)
    print(" -", seq_stats_path)
    if args.do_esm:
        print(" -", os.path.join(args.out_dir, "activate_tidy_scored.csv"))
        print(" -", os.path.join(args.out_dir, "activate_emb.npy"))
    print(" -", diag_path)

if __name__ == "__main__":
    main()

