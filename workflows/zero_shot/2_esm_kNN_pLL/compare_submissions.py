import argparse
import numpy as np
import pandas as pd

COL_A1 = "activity_1 (μmol [TPA]/min·mg [E])"
COL_A2 = "activity_2 (μmol [TPA]/min·mg [E])"
COL_EX = "expression (mg/mL)"

NAME2COL = {
    "activity_1": COL_A1,
    "activity_2": COL_A2,
    "expression": COL_EX,
}

def rank_diff_stats(a: pd.Series, b: pd.Series):
    r_a = a.rank(ascending=False, method="average")
    r_b = b.rank(ascending=False, method="average")
    diff = r_b - r_a
    return diff, float(np.mean(np.abs(diff))), float(np.median(np.abs(diff))), float(np.max(np.abs(diff)))

def top_overlap(a: pd.Series, b: pd.Series, Ns=(50, 100, 200, 500)):
    r_a = a.rank(ascending=False, method="average")
    r_b = b.rank(ascending=False, method="average")
    out = {}
    for N in Ns:
        top_a = set(r_a.nsmallest(N).index)
        top_b = set(r_b.nsmallest(N).index)
        out[N] = len(top_a & top_b) / N
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Compare two submission CSVs by rank changes (NDCG-oriented)."
    )
    ap.add_argument("--base", required=True, help="Baseline submission CSV (e.g., sim-only).")
    ap.add_argument("--new", required=True, help="New submission CSV (e.g., sim+PLL).")
    ap.add_argument("--metric", default="expression", choices=list(NAME2COL.keys()),
                    help="Which property to analyze for rank diff/overlap.")
    ap.add_argument("--debug", default=None, help="Optional debug_scores.csv to show top penalized/boosted.")
    ap.add_argument("--topN", default="50,100,200,500",
                    help="Comma-separated N values for Top-N overlap (default: 50,100,200,500).")
    ap.add_argument("--show_extremes", type=int, default=10,
                    help="How many most penalized/boosted to show (default: 10).")
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    new = pd.read_csv(args.new)

    # Basic sanity checks
    if "sequence" not in base.columns or "sequence" not in new.columns:
        raise SystemExit("ERROR: Both CSVs must contain a 'sequence' column.")

    # Ensure same order (strict)
    if not (base["sequence"].values == new["sequence"].values).all():
        # try align by merge
        merged = base[["sequence"] + [COL_A1, COL_A2, COL_EX]].merge(
            new[["sequence"] + [COL_A1, COL_A2, COL_EX]],
            on="sequence",
            how="inner",
            suffixes=("_base", "_new"),
        )
        if len(merged) != len(base) or len(merged) != len(new):
            raise SystemExit("ERROR: sequence sets differ between files; cannot compare safely.")
        # reorder to base order
        merged = merged.set_index("sequence").loc[base["sequence"].values].reset_index()
        base_aligned = merged[[ "sequence", f"{COL_A1}_base", f"{COL_A2}_base", f"{COL_EX}_base"]].copy()
        new_aligned  = merged[[ "sequence", f"{COL_A1}_new",  f"{COL_A2}_new",  f"{COL_EX}_new"]].copy()
        base_aligned.columns = ["sequence", COL_A1, COL_A2, COL_EX]
        new_aligned.columns  = ["sequence", COL_A1, COL_A2, COL_EX]
        base, new = base_aligned, new_aligned
        print("[WARN] sequences were not in identical order; aligned by 'sequence' values.")
    else:
        print("[OK] sequence order identical.")

    # identical check for all three metrics
    print("\n=== Value identical check ===")
    for key, col in NAME2COL.items():
        ident = (base[col].values == new[col].values).all()
        print(f"{key:>10} identical? {ident}")

    metric_col = NAME2COL[args.metric]

    # rank diffs
    diff, mean_abs, med_abs, max_abs = rank_diff_stats(base[metric_col], new[metric_col])
    print(f"\n=== Rank diff stats for {args.metric} ({metric_col}) ===")
    print(f"mean |Δrank| : {mean_abs}")
    print(f"median |Δrank|: {med_abs}")
    print(f"max |Δrank|   : {max_abs}")

    # top-N overlap
    Ns = tuple(int(x) for x in args.topN.split(",") if x.strip())
    overlaps = top_overlap(base[metric_col], new[metric_col], Ns=Ns)
    print("\n=== Top-N overlap ===")
    for N in Ns:
        print(f"Top-{N} overlap: {overlaps[N]:.3f}")

    # extremes with optional debug join
    if args.debug:
        dbg = pd.read_csv(args.debug)
        if "sequence" not in dbg.columns:
            raise SystemExit("ERROR: debug_scores.csv must contain 'sequence' column.")

        df = pd.DataFrame({
            "sequence": base["sequence"],
            "rank_base": base[metric_col].rank(ascending=False, method="average"),
            "rank_new":  new[metric_col].rank(ascending=False, method="average"),
        })
        df["delta"] = df["rank_new"] - df["rank_base"]

        df = df.join(dbg.set_index("sequence"), on="sequence")

        n = args.show_extremes
        cols_show = ["delta"]
        for cand in ["sim_knn", "sim_norm", "pll", "pll_norm"]:
            if cand in df.columns:
                cols_show.append(cand)

        print(f"\n=== Most penalized by NEW (rank down), top {n} ===")
        print(df.sort_values("delta", ascending=False).head(n)[cols_show])

        print(f"\n=== Most boosted by NEW (rank up), top {n} ===")
        print(df.sort_values("delta", ascending=True).head(n)[cols_show])

    else:
        print("\n[INFO] --debug not provided; skipping penalized/boosted diagnostics.")

if __name__ == "__main__":
    main()

