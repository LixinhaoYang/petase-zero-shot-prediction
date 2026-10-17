import pandas as pd
import numpy as np
import argparse

COL_A1 = "activity_1 (μmol [TPA]/min·mg [E])"
COL_A2 = "activity_2 (μmol [TPA]/min·mg [E])"
COL_EX = "expression (mg/mL)"

ap = argparse.ArgumentParser()
ap.add_argument("--debug", default="cache_esm/debug_scores.csv")
ap.add_argument("--base", default="../1_update_kNN/predictive-pet-zero-shot-submission.csv")
ap.add_argument("--out", default="submission_exprRankNorm.csv")
ap.add_argument("--lam", type=float, default=0.10)  # Explore values such as 0.05, 0.10, and 0.20.
args = ap.parse_args()

dbg = pd.read_csv(args.debug)
base = pd.read_csv(args.base)

assert (dbg["sequence"].values == base["sequence"].values).all(), "sequence order mismatch"

sim = dbg["sim_norm"]
pll = dbg["pll_norm"]

N = len(sim)

rank_sim = sim.rank(ascending=False, method="average").to_numpy(np.float32)
rank_pll = pll.rank(ascending=False, method="average").to_numpy(np.float32)

# convert ranks to percentile scores in [0,1], higher better
p_sim = 1.0 - (rank_sim - 1.0) / (N - 1.0)
p_pll = 1.0 - (rank_pll - 1.0) / (N - 1.0)

expr = (1.0 - args.lam) * p_sim + args.lam * p_pll

out = pd.DataFrame({
    "sequence": base["sequence"],
    COL_A1: base[COL_A1],   # keep kNN baseline activities unchanged
    COL_A2: base[COL_A2],
    COL_EX: expr.astype(np.float32),
})
out.to_csv(args.out, index=False)
print("Wrote:", args.out)

