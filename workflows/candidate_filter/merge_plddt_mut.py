import re
import pandas as pd
import argparse

def extract_idx(x: str) -> int:
    """
    Extract 2200 from identifiers such as 'seq_02200.pdb' or 'row_02200'.
    """
    m = re.search(r"(\d+)", str(x))
    if not m:
        return -1
    return int(m.group(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plddt_csv", default="plddt_rank.csv")
    ap.add_argument("--mut_csv", default="mutation_counts.csv")
    ap.add_argument("--out", default="plddt_with_mutations.csv")
    ap.add_argument("--sort", default="mean_plddt",
                    help="sort key: mean_plddt | p10_mean_plddt | total_edits | min_plddt")
    ap.add_argument("--descending", action="store_true")
    args = ap.parse_args()

    plddt = pd.read_csv(args.plddt_csv)
    mut = pd.read_csv(args.mut_csv)

    # Align seq_XXXXX and row_XXXXX using their numeric identifiers.
    plddt["idx"] = plddt["pdb"].apply(extract_idx)
    mut["idx"] = mut["id"].apply(extract_idx)

    # merge
    merged = plddt.merge(mut, on="idx", how="left", validate="one_to_one")

    # Check for missing mutation records.
    miss = merged["total_edits"].isna().sum()
    print(f"Merged rows: {len(merged)}  missing mutation rows: {miss}")

    # Sort the output.
    if args.sort not in merged.columns:
        raise SystemExit(f"--sort '{args.sort}' not in columns: {merged.columns.tolist()}")
    merged = merged.sort_values(args.sort, ascending=not args.descending)

    merged.to_csv(args.out, index=False)
    print("Wrote:", args.out)

    # Print the top 20 rows.
    show_cols = ["basename","pdb","n_res","mean_plddt","p10_mean_plddt","min_plddt",
                 "id","length","subs","ins","dels","total_edits"]
    show_cols = [c for c in show_cols if c in merged.columns]
    print("\nTop 20:")
    print(merged[show_cols].head(20).to_string(index=False))

if __name__ == "__main__":
    main()

