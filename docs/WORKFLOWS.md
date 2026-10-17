# Workflow recipes

Run these commands from the repository root with the selected environment active. New outputs belong in `outputs/`; the launcher creates its run directory. Use `--dry-run` to inspect a launch without importing ML libraries. Use a fresh `--output-dir` for changed input sequences or embedding settings because historical cache keys do not always encode every setting.

## Prediction variants

| Launcher | Retained implementation | Behavior |
| --- | --- | --- |
| `baseline` | `0_esm_kNN/run_zeroshot_esm_knn.py` | ESM2-35M mean embedding, cosine nearest WT, normalized heuristic scores |
| `weighted-knn` | `1_update_kNN/run_zeroshot_esm_knn.py` | Softmax-weighted WT similarities; launcher defaults to five neighbors |
| `pll` | `2_esm_kNN_pLL/run_zeroshot_esm_knn_pll.py` | Actual masked pseudo-log-likelihood blended with similarity |
| `cluster-expert` | `3_petase_specific/run_cluster_expert_zeroshot.py` | ESM2-150M, WT panel per cluster group, within-group expression ranks; clusters 3/4 merged |
| `activity` | `4_using_activity_data/run_hybrid_value_predict_v3.py` | Labeled activity anchors, isotonic fitting and fallback rules; uses saved ESM similarities |
| `filter` | `candidate_filter/filter_candidates.py` | WT-similarity quantile filter plus KMeans diversity cap; does not compute PLL |

All prediction variants except `filter` write `outputs/<workflow>/submission.csv`. The filter writes `filtered_candidates.csv` with `sequence`, `sim_wt_topk`, and `cluster` columns. Cluster labels are embedding-cluster identifiers, not biological family assignments.

```bash
python scripts/run_workflow.py baseline
python scripts/run_workflow.py weighted-knn --knn_k 5
python scripts/run_workflow.py pll --gpu --model esm2_t30_150M_UR50D --batch_size 8
python scripts/run_workflow.py cluster-expert --gpu
python scripts/run_workflow.py activity
python scripts/run_workflow.py filter --gpu --sim_drop 0.15 --n_clusters 20 --cluster_k 300
```

Masked PLL performs many forward passes and can take substantially longer than embedding-only scoring. The default CPU mode is useful for small checks; larger runs can use `--gpu` with a CUDA-enabled environment. Check a completed submission:

```bash
python scripts/check_submission.py outputs/activity/submission.csv \
  --reference data/raw/predictive-pet-zero-shot-test-2025.csv
```

The `activity` launcher copies expression values from the historical cluster-expert submission and replaces the activity columns with newly calculated values. For new candidate inputs, supply matching `--test_scored_csv`, `--activate_scored_csv`, and `--baseline_submission_csv` paths as forwarded arguments. New candidates also need a complete matching cluster table through `--cluster-csv`. The saved default scores cover only the bundled candidate set.

## Activity-data preparation

The following rebuilds the tidy table, WT similarities, and activity-sequence cluster assignments. This requires the core requirements, including `openpyxl`. The first command parses the workbook without loading ESM; the remaining commands calculate embeddings. Add `--use_gpu` to the latter two scripts to request CUDA.

```bash
mkdir -p outputs/activity-prep

python workflows/zero_shot/4_using_activity_data/run_activate_pipeline.py \
  --activate_xlsx data/raw/activate.xlsx \
  --out_dir outputs/activity-prep \
  --cache_dir outputs/activity-prep/cache-tidy

python workflows/zero_shot/4_using_activity_data/score_activate_with_wt.py \
  --activate_tidy_csv outputs/activity-prep/activate_tidy.csv \
  --wt_csv data/raw/pet-2025-wildtype-cds.csv \
  --out_dir outputs/activity-prep \
  --cache_dir outputs/activity-prep/cache-wt

python workflows/zero_shot/4_using_activity_data/assign_cluster_to_activate.py \
  --activate_tidy outputs/activity-prep/activate_tidy.csv \
  --cluster_csv data/processed/clustered_sequences.csv \
  --out_csv outputs/activity-prep/activate_with_cluster.csv \
  --cache_dir outputs/activity-prep/cache-cluster --merge_34

python workflows/zero_shot/4_using_activity_data/score_test_with_wt.py \
  --test_csv data/raw/predictive-pet-zero-shot-test-2025.csv \
  --wt_csv data/raw/pet-2025-wildtype-cds.csv \
  --out_csv outputs/activity-prep/test_scored_with_wt.csv \
  --cache_dir outputs/activity-prep/cache-test --device cpu

python scripts/run_workflow.py activity --output-dir outputs/activity-refit \
  --test_scored_csv outputs/activity-prep/test_scored_with_wt.csv \
  --activate_scored_csv outputs/activity-prep/activate_scored_with_wt.csv \
  --activate_cluster_csv outputs/activity-prep/activate_with_cluster.csv
```

`run_hybrid_value_predict.py` and `_v2.py`, the stable cluster-expert alternative, and plotting scripts remain in their historical source directories for comparison. Their original local defaults can refer to paths from the old layout. Inspect their constants/options and pass current paths explicitly; the supported launcher uses v3. The documented calibration uses labels and is not a zero-shot method.

## PLL rescoring and comparisons

The rescoring helpers accept explicit baseline/debug paths. The archived weighted-kNN submission was identical to the original baseline and is retained once at the baseline location.

```bash
mkdir -p outputs/rescore
python workflows/zero_shot/2_esm_kNN_pLL/rescore_expr_rankblend_norm.py \
  --base results/zero_shot/0_esm_kNN/predictive-pet-zero-shot-submission.csv \
  --debug results/zero_shot/2_esm_kNN_pLL/diagnostics/debug_scores.csv \
  --out outputs/rescore/submission.csv --lam 0.05

python workflows/zero_shot/2_esm_kNN_pLL/compare_submissions.py \
  --base results/zero_shot/0_esm_kNN/predictive-pet-zero-shot-submission.csv \
  --new outputs/rescore/submission.csv --metric expression
```

These comparisons measure agreement and rank changes between submissions, not accuracy against experimental ground truth.

## Classification notebook

```bash
python -m pip install -r requirements-analysis.txt
python scripts/prepare_classification.py
jupyter lab notebooks/classification.ipynb
```

Run the notebook setup cell first. It locates the repository, stages the original inputs and archived intermediate results in `outputs/classification/`, and makes that the working directory without overwriting existing files. The notebook is exploratory and contains alternative analysis branches; run the sections needed for a specific analysis. It includes a comparison against `PETase.xlsx`, not just the unlabeled candidate analysis.

MAFFT alignment sections require the `mafft` executable. Search/annotation sections require DIAMOND, downloaded Swiss-Prot or UniRef databases, and network access for UniProt taxonomy requests. Search databases were empty placeholders in the original folder and are not bundled. Follow the corresponding notebook section to generate search results before rerunning dependent cells. Genome-scale databases and model weights should remain outside Git.

## Candidate-filter utilities and future stages

`--wt-csv data/raw/wildtype-augmented-393.csv` selects the separate augmented reference set. The default launcher uses the canonical 313-row WT/CDS file. The filter accepts CSV only, and `--candidate_seq_col`/`--wt_seq_col` override protein-column detection. Its `--pll_drop` compatibility flag must be zero.

The exploratory utilities are retained alongside the filter:

- `count_mutations_vs_wt_csv.py`: global alignment to the **first WT row only**, not nearest-WT assignment. It has one-based fallback row IDs; choose explicit IDs when combining with external structure files.
- `merge_plddt_mut.py`: joins supplied pLDDT and mutation tables by an integer extracted from their identifiers. Verify that both ID schemes use the same origin before interpreting results.
- `batch_qblast.py`: original standalone remote BLAST utility. It expects `blast_query.fasta` in the working directory and sends sequences to NCBI only when explicitly executed. It needs Biopython from the analysis requirements; it is not run by the launcher or CI.

No structure-prediction, docking, Rosetta, or MD implementation was present. The pLDDT tables are historical inputs/results from an external process, not evidence that those stages are reproducible from this repository.

## SLURM

Activate the environment and submit from the repository root:

```bash
mkdir -p outputs/slurm
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/run_baseline.sbatch
```

Replace the account and partition with values for your cluster. The template uses the active environment's `python` and the submission directory; it does not load site-specific modules or activate a private Conda path.

## Local Git use

This folder can be used as an independent repository. If it has not already been initialized:

```bash
git init -b main
git add .
git status --short
git commit -m "Prepare PETase research workflows"
```

The parent workspace's Git history is not needed. No remote is configured or pushed by these preparation scripts.

