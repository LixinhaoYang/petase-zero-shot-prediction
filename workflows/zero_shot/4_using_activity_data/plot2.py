import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================
# USER SETTINGS
# =====================
SUBSTRATE = "aflim"
TEMP = 60.0
PH = 8.5

OUTDIR = "fig_conditioned"
os.makedirs(OUTDIR, exist_ok=True)

MIN_POINTS = 6     # per cluster minimum
BINS = 10          # rolling median bins

# =====================
# LOAD
# =====================
df = pd.read_csv("activate_out/activate_scored_with_wt.csv")

# cluster labels (per unique sequence)
clu = (
    pd.read_csv("activate_out/activate_with_cluster.csv")[["sequence", "cluster_group"]]
      .drop_duplicates("sequence")
)

df = df.merge(clu, on="sequence", how="left")

print("Columns:", df.columns.tolist())
assert "sim_wt_topk" in df.columns, "sim_wt_topk missing"
assert "cluster_group" in df.columns, "cluster_group missing after merge"

# =====================
# FILTER: SAME CONDITION
# =====================
sub = df[
    (df["substrate"] == SUBSTRATE) &
    (df["temperature_C"] == TEMP) &
    (df["pH"] == PH)
].dropna(subset=["sim_wt_topk", "value", "cluster_group"])

print(f"Condition: substrate={SUBSTRATE}, T={TEMP}, pH={PH}")
print("Counts by cluster_group:")
print(sub["cluster_group"].value_counts())

def rolling_median(x, y, bins=BINS):
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    xs, ys = [], []
    for i in range(bins):
        lo = int(i * n / bins)
        hi = int((i + 1) * n / bins)
        if hi - lo < 2:
            continue
        xs.append(np.median(x[lo:hi]))
        ys.append(np.median(y[lo:hi]))
    return np.array(xs), np.array(ys)

# =====================
# PLOT & SAVE
# =====================
for g in sorted(sub["cluster_group"].unique()):
    cg = sub[sub["cluster_group"] == g]
    if len(cg) < MIN_POINTS:
        print(f"cluster {g}: n={len(cg)} < {MIN_POINTS}, skipped")
        continue

    x = cg["sim_wt_topk"].to_numpy()
    y = cg["value"].to_numpy()

    sp = pd.Series(x).corr(pd.Series(y), method="spearman")

    xs, ys = rolling_median(x, y)

    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, s=35, alpha=0.8)
    plt.plot(xs, ys, color="black", linewidth=2)

    plt.xlabel("sim_wt_topk")
    plt.ylabel("activity (value)")
    plt.title(f"Cluster {g} | Spearman={sp:.3f}\n{SUBSTRATE}, {TEMP}C, pH {PH}")
    plt.tight_layout()

    # safe filename
    ph_str = str(PH).replace(".", "p")
    t_str = str(int(TEMP)) if float(TEMP).is_integer() else str(TEMP).replace(".", "p")
    fname = f"cluster{g}_{SUBSTRATE}_T{t_str}_pH{ph_str}.png"
    outpath = os.path.join(OUTDIR, fname)

    plt.savefig(outpath, dpi=200)
    plt.close()
    print("saved:", outpath)

print("Done.")

