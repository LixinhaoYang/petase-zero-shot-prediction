# PETase Zero-Shot Analysis and Prediction

An archive of my PETase work for the **2025 Protein Engineering Tournament**.

The competition was discontinued, so the competition results here only cover the zero-shot phase. I also kept some exploratory work on activity calibration and candidate filtering.

The original scripts are on Georgia Tech's GitHub. This repository is a reorganized copy archived after the project ended.

## What I worked on

- ESM2 sequence embeddings, clustering, and family analysis.
- Zero-shot ranking using similarity to wild-type sequences, weighted kNN, and pseudo-log-likelihood (PLL).
- Cluster-specific wild-type reference panels.
- Exploratory activity calibration and candidate filtering.

Docking, Rosetta, and MD were considered as possible next steps, but those workflows were not implemented here.

## Files

```text
data/        Original inputs and processed tables
notebooks/   Sequence classification and analysis
workflows/   Prediction and candidate-filtering scripts
results/     Saved predictions, figures, alignments, and analysis tables
scripts/     Run helpers and checks
docs/        Additional working notes
```

The zero-shot outputs are relative ranking scores. Although the CSV columns use the competition's activity and expression labels, these scores should not be read as measured activity or expression values.

