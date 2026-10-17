# Validation record

Repository preparation and local validation: **2026-10-16**, macOS ARM64, Python **3.11.15**, CPU execution. Extra packages were installed in a temporary validation environment; the existing Conda environment was not modified.

## Checks completed

| Check | Observed result |
| --- | --- |
| Repository checker | Python source compilation, clean notebooks, local documentation links, source mappings, immutable artifact checksums, input row counts, and cluster-table order passed |
| Original source preservation | All 265 recorded original non-Git files retained their original SHA-256 checksums |
| Regression suite | 6 tests passed: filtered embedding-row alignment, small WT panels/caps, invalid input handling, protein-column detection, repeated-sequence joins, and conflicting metadata rejection |
| Activity calibration | Ran v3 on all 4,988 candidates using the included saved similarities and anchors; submission has exactly the original sequence order and finite, nonnegative numeric values |
| Experimental workbook preprocessing | Read `activate.xlsx` and rebuilt the tidy activity table: 163 observations and 15 distinct sequences |
| Real ESM baseline | Cached ESM2-35M, 6 WT references and 4 candidate rows (including a repeated sequence); 4 output rows in the original order |
| Real ESM weighted kNN | Same small input, `knn_k=5`; 4 output rows in the original order |
| Real ESM cluster expert | Same small input; canonical WT header accepted and repeated candidate rows preserved; 4 output rows in the original order |
| Real ESM candidate filter | Same small input, `n_clusters=2`, `cluster_k=10`, default similarity cutoff; retained 3 valid candidate rows with finite similarity scores |
| Classification notebook | Cleared execution state; compiled all 112 Python code cells with IPython magic excluded; exercised the input-staging script |
| Git packaging | All manifest destinations are included by Git's ignore rules; new run files are ignored; independent `main` repository initialized with no remote configured |

The real-model checks used existing cached ESM2-35M weights and CPU execution; they did not download weights or rerun the full candidate set through ESM.

## Validation environment

These are observed versions in the validation environment, not a complete dependency lockfile:

| Package | Version |
| --- | --- |
| NumPy | 1.26.4 |
| pandas | 3.0.3 |
| PyTorch | 2.2.2 |
| fair-esm | 2.0.0 |
| scikit-learn | 1.9.1 |
| SciPy | 1.17.1 |
| openpyxl | 3.1.5 |

## Repeating the portable checks

```bash
python scripts/check_repository.py
python -m unittest discover -s tests -v
python scripts/run_workflow.py activity
python scripts/check_submission.py outputs/activity/submission.csv \
  --reference data/raw/predictive-pet-zero-shot-test-2025.csv
```

The GitHub Actions workflow runs these checks with CPU-only validation dependencies and no model download. It has been configured locally; a hosted GitHub Actions run has not occurred because this folder has not been published.

## Scope

No independent experimental accuracy benchmark, full 150M/650M run, full masked-PLL reproduction, CUDA/SLURM execution, remote BLAST/UniProt query, MAFFT/DIAMOND pipeline, or end-to-end notebook execution was performed. The activity fit used its existing `fallback_base_scaled` mode with seven retained cluster-2 anchors. Passing file/schema checks and smoke runs establishes operational behavior, not biological accuracy. Historical results remain unchanged, including historical repeated-row artifacts.

