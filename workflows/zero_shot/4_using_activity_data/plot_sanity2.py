import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config (edit if needed)
# -----------------------------
PRED_CSV = "submission_hybrid_value.csv"
SIM_CSV  = "test_scored_with_wt.csv"
CLU_CSV  = "clustered_sequences.csv"

OUTDIR = "fig_sanity"
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def spearman_corr(x, y):
    # robust spearman without scipy
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 3:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])

def show_columns(name, df):
    print(f"\n=== {name} columns ({len(df.columns)} cols) ===")
    print(df.columns.tolist())

# -----------------------------
# Load
# -----------------------------
pred = pd.read_csv(PRED_CSV)
sim  = pd.read_csv(SIM_CSV)
clu  = pd.read_csv(CLU_CSV)

show_columns("PRED", pred)
show_columns("SIM", sim)
show_columns("CLU", clu)

# Required: sequence column
seq_col_pred = pick_first_existing(pred.columns, ["sequence", "seq"])
seq_col_sim  = pick_first_existing(sim.columns,  ["sequence", "seq"])
seq_col_clu  = pick_first_existing(clu.columns,  ["sequence", "seq"])

if seq_col_pred is None:
    raise RuntimeError("Cannot find sequence column in prediction file.")
if seq_col_sim is None:
    raise RuntimeError("Cannot find sequence column in sim file.")
if seq_col_clu is None:
    raise RuntimeError("Cannot find sequence column in cluster file.")

# sim column
sim_col = pick_first_existing(sim.columns, ["sim_wt_topk", "sim_wt_max", "sim_wt_centroid"])
if sim_col is None:
    raise RuntimeError("Cannot find a similarity column in SIM file. Expected sim_wt_topk/sim_wt_max/sim_wt_centroid.")

# cluster column
clu_col = pick_first_existing(clu.columns, ["cluster_group", "cluster", "label", "labels", "cluster_id"])
if clu_col is None:
    raise RuntimeError("Cannot find a cluster column in CLU file.")

# activity columns: support multiple possible outputs
a1_col = pick_first_existing(
    pred.columns,
    [
        "activity_1 (μmol [TPA]/min·mg [E])",
        "activity_1",
        "pred_activity_1",
        "y1",
    ],
)
a2_col = pick_first_existing(
    pred.columns,
    [
        "activity_2 (μmol [TPA]/min·mg [E])",
        "activity_2",
        "pred_activity_2",
        "y2",
    ],
)

if a1_col is None or a2_col is None:
    raise RuntimeError(
        "Cannot find activity columns in prediction file. "
        f"Detected a1={a1_col}, a2={a2_col}. "
        "Please check the printed PRED columns above."
    )

print(f"\n[INFO] Using columns:")
print(f"  pred sequence: {seq_col_pred}")
print(f"  sim  sequence: {seq_col_sim}")
print(f"  clu  sequence: {seq_col_clu}")
print(f"  sim column    : {sim_col}")
print(f"  cluster column: {clu_col}")
print(f"  activity_1 col : {a1_col}")
print(f"  activity_2 col : {a2_col}")

# -----------------------------
# Merge (do NOT guess columns)
# -----------------------------
df = pred.rename(columns={seq_col_pred: "sequence"}).copy()
df = df.merge(sim.rename(columns={seq_col_sim: "sequence"})[["sequence", sim_col]], on="sequence", how="left")
df = df.merge(clu.rename(columns={seq_col_clu: "sequence"})[["sequence", clu_col]], on="sequence", how="left")

# Rename standardized names for plotting
df = df.rename(columns={sim_col: "sim", clu_col: "cluster", a1_col: "a1", a2_col: "a2"})

# Basic checks
miss_sim = df["sim"].isna().mean()
miss_clu = df["cluster"].isna().mean()
print(f"\n[INFO] Missing rate: sim={miss_sim:.3f}, cluster={miss_clu:.3f}")
if miss_sim > 0:
    print("[WARN] Some rows missing sim after merge. Check sequence formatting consistency.")
if miss_clu > 0:
    print("[WARN] Some rows missing cluster after merge. Check clustered_sequences.csv coverage.")

# Force numeric
df["sim"] = pd.to_numeric(df["sim"], errors="coerce")
df["a1"]  = pd.to_numeric(df["a1"], errors="coerce")
df["a2"]  = pd.to_numeric(df["a2"], errors="coerce")

# -----------------------------
# Plot 1: Overall scatter (a1 vs sim, a2 vs sim)
# -----------------------------
sp1 = spearman_corr(df["sim"], df["a1"])
sp2 = spearman_corr(df["sim"], df["a2"])

plt.figure(figsize=(5,4))
plt.scatter(df["sim"], df["a1"], s=8, alpha=0.6)
plt.xlabel(f"{sim_col}")
plt.ylabel(a1_col)
plt.title(f"Overall: Spearman={sp1:.3f}" if np.isfinite(sp1) else "Overall: Spearman=nan")
plt.tight_layout()
p1 = os.path.join(OUTDIR, "overall_a1_vs_sim.png")
plt.savefig(p1, dpi=200)
plt.close()
print("saved:", p1)

plt.figure(figsize=(5,4))
plt.scatter(df["sim"], df["a2"], s=8, alpha=0.6)
plt.xlabel(f"{sim_col}")
plt.ylabel(a2_col)
plt.title(f"Overall: Spearman={sp2:.3f}" if np.isfinite(sp2) else "Overall: Spearman=nan")
plt.tight_layout()
p2 = os.path.join(OUTDIR, "overall_a2_vs_sim.png")
plt.savefig(p2, dpi=200)
plt.close()
print("saved:", p2)

# -----------------------------
# Plot 2: By-cluster scatters (only clusters with enough points)
# -----------------------------
min_n = 20  # change if you want
clusters = sorted([c for c in df["cluster"].dropna().unique().tolist()])

for c in clusters:
    sub = df[df["cluster"] == c]
    if len(sub) < min_n:
        continue

    spc1 = spearman_corr(sub["sim"], sub["a1"])
    plt.figure(figsize=(4,3))
    plt.scatter(sub["sim"], sub["a1"], s=10, alpha=0.7)
    plt.xlabel(f"{sim_col}")
    plt.ylabel(a1_col)
    plt.title(f"Cluster {c} (n={len(sub)}), sp={spc1:.3f}" if np.isfinite(spc1) else f"Cluster {c} (n={len(sub)}), sp=nan")
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"cluster_{c}_a1_vs_sim.png")
    plt.savefig(out, dpi=200)
    plt.close()

    spc2 = spearman_corr(sub["sim"], sub["a2"])
    plt.figure(figsize=(4,3))
    plt.scatter(sub["sim"], sub["a2"], s=10, alpha=0.7)
    plt.xlabel(f"{sim_col}")
    plt.ylabel(a2_col)
    plt.title(f"Cluster {c} (n={len(sub)}), sp={spc2:.3f}" if np.isfinite(spc2) else f"Cluster {c} (n={len(sub)}), sp=nan")
    plt.tight_layout()
    out = os.path.join(OUTDIR, f"cluster_{c}_a2_vs_sim.png")
    plt.savefig(out, dpi=200)
    plt.close()

print("\nDone. Figures saved under:", OUTDIR)

