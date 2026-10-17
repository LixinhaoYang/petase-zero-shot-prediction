import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_CSV = "submission_hybrid_value.csv"
OUTFIG = "fig_active_site_penalty.png"

# -----------------------------
# Active-site heuristic (self-contained)
# -----------------------------
def active_site_present(seq: str) -> bool:
    """
    Very lightweight PETase / serine-hydrolase heuristic:
    - GXSXG motif (nucleophilic serine)
    - at least one Asp (D) and one His (H) somewhere
    This is NOT structural validation, only a conservative filter.
    """
    if not isinstance(seq, str):
        return False
    seq = seq.upper()
    if len(seq) < 50:
        return False

    has_ser = re.search(r"G.A.G", seq) is not None
    has_D = "D" in seq
    has_H = "H" in seq
    return has_ser and has_D and has_H

# -----------------------------
# Helpers
# -----------------------------
def pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

# -----------------------------
# Load
# -----------------------------
df = pd.read_csv(PRED_CSV)

# detect columns (no guessing)
seq_col = pick(df.columns, ["sequence", "seq"])
a1_col = pick(
    df.columns,
    [
        "activity_1 (μmol [TPA]/min·mg [E])",
        "activity_1",
        "pred_activity_1",
        "y1",
    ],
)

if seq_col is None or a1_col is None:
    raise RuntimeError(
        "Cannot find required columns in submission file.\n"
        f"Detected seq_col={seq_col}, a1_col={a1_col}\n"
        f"Columns available: {df.columns.tolist()}"
    )

print("[INFO] Using columns:")
print("  sequence :", seq_col)
print("  activity1:", a1_col)

# -----------------------------
# Compute active-site flag
# -----------------------------
df = df[[seq_col, a1_col]].copy()
df["active_site"] = df[seq_col].apply(active_site_present)
df["a1"] = pd.to_numeric(df[a1_col], errors="coerce")

# drop NaNs
df = df.dropna(subset=["a1"])

# -----------------------------
# Prepare data for boxplot
# -----------------------------
vals_present = df.loc[df["active_site"], "a1"].to_numpy()
vals_missing = df.loc[~df["active_site"], "a1"].to_numpy()

print(f"[INFO] counts: active_site_present={len(vals_present)}, missing={len(vals_missing)}")

if len(vals_present) < 3 or len(vals_missing) < 3:
    print("[WARN] One group has too few points; boxplot may be unstable.")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(4, 3))
plt.boxplot(
    [vals_present, vals_missing],
    labels=["Active site present", "Active site missing"],
    showfliers=True
)
plt.ylabel(a1_col)
plt.title("Effect of active-site heuristic")
plt.tight_layout()
plt.savefig(OUTFIG, dpi=200)
plt.close()

print("saved:", OUTFIG)

