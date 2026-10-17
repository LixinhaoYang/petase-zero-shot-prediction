import os
import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = "submission_hybrid_value.csv"
SIM_CSV  = "test_scored_with_wt.csv"      # has: sequence, sim_wt_topk
CLU_CSV  = "clustered_sequences.csv"      # has: sequence, cluster_group (or cluster)

OUTDIR = "fig_clusters_a1"
os.makedirs(OUTDIR, exist_ok=True)

# ---- load ----
pred = pd.read_csv(PRED_CSV)
sim  = pd.read_csv(SIM_CSV)
clu  = pd.read_csv(CLU_CSV)

# ---- detect columns (no guessing) ----
def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

seq_p = pick(pred.columns, ["sequence", "seq"])
seq_s = pick(sim.columns,  ["sequence", "seq"])
seq_c = pick(clu.columns,  ["sequence", "seq"])

if seq_p is None or seq_s is None or seq_c is None:
    raise RuntimeError("Cannot find sequence column in one of the input files.")

sim_col = pick(sim.columns, ["sim_wt_topk", "sim_wt_max", "sim_wt_centroid"])
if sim_col is None:
    raise RuntimeError("Cannot find similarity column in SIM file (expected sim_wt_topk/sim_wt_max/sim_wt_centroid).")

clu_col = pick(clu.columns, ["cluster_group", "cluster", "label", "labels", "cluster_id"])
if clu_col is None:
    raise RuntimeError("Cannot find cluster column in CLU file (expected cluster_group/cluster/...).")

a1_col = pick(
    pred.columns,
    ["activity_1 (μmol [TPA]/min·mg [E])", "activity_1", "pred_activity_1", "y1"]
)
if a1_col is None:
    raise RuntimeError("Cannot find activity_1 column in prediction file.")

print("[INFO] Using columns:")
print("  seq:", seq_p, seq_s, seq_c)
print("  sim:", sim_col)
print("  clu:", clu_col)
print("  a1 :", a1_col)

# ---- merge ----
df = pred.rename(columns={seq_p: "sequence"}).copy()
df = df.merge(sim.rename(columns={seq_s: "sequence"})[["sequence", sim_col]], on="sequence", how="left")
df = df.merge(clu.rename(columns={seq_c: "sequence"})[["sequence", clu_col]], on="sequence", how="left")

df = df.rename(columns={sim_col: "sim", clu_col: "cluster", a1_col: "a1"})

# drop rows missing essentials for plotting
df = df.dropna(subset=["cluster", "sim", "a1"]).copy()

# ensure numeric
df["sim"] = pd.to_numeric(df["sim"], errors="coerce")
df["a1"]  = pd.to_numeric(df["a1"], errors="coerce")
df = df.dropna(subset=["sim", "a1"]).copy()

# ---- plot ----
min_n = 10
for g in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == g]
    if len(sub) < min_n:
        continue

    plt.figure(figsize=(4, 3))
    plt.scatter(sub["sim"], sub["a1"], s=10, alpha=0.7)
    plt.xlabel(sim_col)
    plt.ylabel(a1_col)
    plt.title(f"Cluster {g} (n={len(sub)})")
    plt.tight_layout()

    out = os.path.join(OUTDIR, f"cluster{g}_a1_vs_sim.png")
    plt.savefig(out, dpi=200)
    plt.close()

print("Done. Saved to:", OUTDIR)

