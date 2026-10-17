import pandas as pd
import matplotlib.pyplot as plt

# load similarity features
sim = pd.read_csv("test_scored_with_wt.csv")[["sequence", "sim_wt_topk"]]

# load predictions
pred = pd.read_csv("submission_hybrid_value.csv")

# merge
df = pred.merge(sim, on="sequence", how="left")

assert "sim_wt_topk" in df.columns, "sim_wt_topk missing after merge"

plt.figure(figsize=(5,4))
plt.scatter(
    df["sim_wt_topk"],
    df["activity_1 (μmol [TPA]/min·mg [E])"],
    s=8,
    alpha=0.6
)
plt.xlabel("Similarity to WT (sim_wt_topk)")
plt.ylabel("Predicted activity_1")
plt.tight_layout()
plt.savefig("fig_activity1_vs_similarity.png", dpi=200)
plt.close()

print("saved: fig_activity1_vs_similarity.png")

