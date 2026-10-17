import pandas as pd
import numpy as np
import argparse

COL_EX = "expression (mg/mL)"
COL_A1 = "activity_1 (μmol [TPA]/min·mg [E])"
COL_A2 = "activity_2 (μmol [TPA]/min·mg [E])"

ap = argparse.ArgumentParser()
ap.add_argument("--debug", default="cache_esm/debug_scores.csv")
ap.add_argument("--base", default="../1_update_kNN/predictive-pet-zero-shot-submission.csv")
ap.add_argument("--out", default="submission_expr_rankblend.csv")
ap.add_argument("--lam", type=float, default=0.1)  # try 0.05, 0.1, 0.2
args = ap.parse_args()

dbg = pd.read_csv(args.debug)
base = pd.read_csv(args.base)

assert (dbg["sequence"].values == base["sequence"].values).all()

sim = dbg["sim_norm"]
pll = dbg["pll_norm"]

rank_sim = sim.rank(ascending=False, method="average")
rank_pll = pll.rank(ascending=False, method="average")

expr_rank = rank_sim + args.lam * rank_pll

# convert rank back to score in [0,1] just for submission format
expr = 1.0 - (expr_rank - expr_rank.min()) / (expr_rank.max() - expr_rank.min())

out = pd.DataFrame({
    "sequence": base["sequence"],
    COL_A1: base[COL_A1],        # keep original kNN
    COL_A2: base[COL_A2],
    COL_EX: expr,
})
out.to_csv(args.out, index=False)
print("Wrote:", args.out)

