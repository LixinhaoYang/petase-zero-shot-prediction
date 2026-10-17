import argparse
import numpy as np
import pandas as pd

COL_A1 = "activity_1 (μmol [TPA]/min·mg [E])"
COL_A2 = "activity_2 (μmol [TPA]/min·mg [E])"
COL_EX = "expression (mg/mL)"

def minmax(x):
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + 1e-8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", default="cache_esm/debug_scores.csv")
    ap.add_argument("--baseline", default="../1_update_kNN/predictive-pet-zero-shot-submission.csv")
    ap.add_argument("--out", default="submission_rescored.csv")
    ap.add_argument("--expr_pll", type=float, default=0.1)  # Explore values such as 0.05, 0.1, and 0.2.
    args = ap.parse_args()

    dbg = pd.read_csv(args.debug)
    base = pd.read_csv(args.baseline)

    assert (dbg["sequence"].values == base["sequence"].values).all(), "sequence order mismatch"

    sim = dbg["sim_norm"].to_numpy(np.float32)
    pll = dbg["pll_norm"].to_numpy(np.float32)

    expr = minmax((1.0 - args.expr_pll) * sim + args.expr_pll * pll)
    act1 = minmax(sim)
    act2 = minmax(sim ** 2)

    out = pd.DataFrame({
        "sequence": base["sequence"],
        COL_A1: act1,
        COL_A2: act2,
        COL_EX: expr,
    })
    out.to_csv(args.out, index=False)
    print("Wrote:", args.out)

if __name__ == "__main__":
    main()

