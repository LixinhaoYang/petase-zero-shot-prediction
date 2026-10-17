import pandas as pd


f1 = "predictive-pet-zero-shot-submission.csv"
f2 = "../0_esm_kNN/predictive-pet-zero-shot-submission.csv"

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)

assert (df1["sequence"] == df2["sequence"]).all()

cols = [
    "activity_1 (μmol [TPA]/min·mg [E])",
    "activity_2 (μmol [TPA]/min·mg [E])",
    "expression (mg/mL)",
]

out = df1[["sequence"]].copy()

for c in cols:
    out[f"{c}_rank_top1"] = df1[c].rank(ascending=False, method="average")
    out[f"{c}_rank_knn"]  = df2[c].rank(ascending=False, method="average")
    out[f"{c}_rank_diff"] = (
        out[f"{c}_rank_knn"] - out[f"{c}_rank_top1"]
    )

out.to_csv("rank_diff_top1_vs_knn.csv", index=False)
print("saved rank_diff_top1_vs_knn.csv")

