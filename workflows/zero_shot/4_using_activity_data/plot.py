import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

scored = pd.read_csv("activate_out/activate_scored_with_wt.csv")
clu = pd.read_csv("activate_out/activate_with_cluster.csv")[["sequence","cluster_group"]].drop_duplicates("sequence")

df = scored.merge(clu, on="sequence", how="left")

def rolling_median(x, y, bins=15):
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    xs, ys = [], []
    for i in range(bins):
        lo = int(i * n / bins)
        hi = int((i + 1) * n / bins)
        if hi - lo < 3: continue
        xs.append(np.median(x[lo:hi]))
        ys.append(np.median(y[lo:hi]))
    return np.array(xs), np.array(ys)

for g in [1,2]:
    sub = df[df["cluster_group"] == g].dropna(subset=["sim_wt_topk","value"])
    if len(sub) < 10:
        print("cluster", g, "too few points")
        continue
    x = sub["sim_wt_topk"].to_numpy()
    y = sub["value"].to_numpy()
    sp = pd.Series(x).corr(pd.Series(y), method="spearman")
    xs, ys = rolling_median(x, y, bins=15)

    plt.figure(figsize=(6,4))
    plt.scatter(x, y, s=18, alpha=0.6)
    plt.plot(xs, ys, color="black", linewidth=2)
    plt.xlabel("sim_wt_topk")
    plt.ylabel("value")
    plt.title(f"Cluster {g} | Spearman={sp:.3f} | n={len(sub)}")
    plt.tight_layout()
    plt.savefig(f"fig_{g}.jpg")

