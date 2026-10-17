# PETase sequence analysis and prediction

Research workflows developed for the predictive phase of the **2025 Protein Engineering Tournament**: ESM2 sequence analysis, wild-type nearest-neighbor scoring, cluster-specific reference panels, activity-data calibration, and candidate filtering.

**Project status:** The competition was discontinued, so this repository reports competition results from the zero-shot phase only. Additional calibration and candidate-filtering analyses are included as exploratory work.

**Archive provenance:** The original scripts are hosted on Georgia Tech's GitHub. This repository is a migrated archive prepared after the project was discontinued, with the code, data, and historical results reorganized for preservation.

[Workflow recipes](docs/WORKFLOWS.md) · [Data and provenance](docs/DATA.md) · [Validation](docs/VALIDATION.md)

The zero-shot workflows produce **relative ranking scores**. Their output columns retain the tournament's activity and expression labels, including units, but the normalized scores are not experimentally calibrated measurements. The activity-calibrated workflow uses labeled observations and must be evaluated separately from zero-shot methods. No benchmark accuracy is claimed here.

## Repository layout

```text
data/raw/                    Original tournament inputs and experimental workbooks
data/processed/              Shared cluster labels, similarity scores, and activity anchors
notebooks/                  Classification notebook and GPU diagnostic, with outputs cleared
workflows/zero_shot/         Baseline, weighted kNN, PLL, cluster expert, and activity variants
workflows/candidate_filter/  Similarity/diversity filter and exploratory analysis utilities
scripts/                    Portable workflow launcher, notebook setup, and repository checks
results/                    Historical tables, submissions, alignments, and figures
docs/                       Recipes, provenance, limitations, and source-file manifest
outputs/                    New runs (created locally; ignored by Git)
```

## Installation

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, run `conda env create -f environment.yml` and `conda activate petase`. For notebooks, plotting, BLAST utilities, and phylogenetic analysis, install `requirements-analysis.txt`. MAFFT and DIAMOND are separate external tools used by optional notebook sections.

ESM workflows download pretrained weights on first use. CPU is the default; add `--gpu` to request CUDA on a compatible machine. CUDA availability still depends on the installed PyTorch build and host drivers. The activity recipe below uses the included precomputed scores and needs no model download.

## Quick start

```bash
# Check files, provenance, notebook state, and Python syntax without ML dependencies.
python scripts/check_repository.py

# Preview a command without loading a model or creating run files.
python scripts/run_workflow.py baseline --dry-run

# ESM2-35M baseline. All new files go under outputs/baseline/.
python scripts/run_workflow.py baseline

# Softmax-weighted kNN with five wild-type neighbors.
python scripts/run_workflow.py weighted-knn --knn_k 5

# Cluster-specific wild-type panels; merges historical clusters 3 and 4.
python scripts/run_workflow.py cluster-expert --gpu

# Refit the activity calibration from included anchors and saved similarities.
python scripts/run_workflow.py activity

# Similarity filtering and embedding-space diversity control (not PLL).
python scripts/run_workflow.py filter --gpu
```

The launcher resolves shared inputs relative to this repository and gives each workflow a separate output/cache directory. Override inputs with `--test-csv`, `--wt-csv`, and `--cluster-csv`; extra arguments are forwarded to the selected script. For example:

```bash
python scripts/run_workflow.py baseline --output-dir outputs/my-run \
  --test-csv path/to/candidates.csv --batch_size 8
```

See [workflow recipes](docs/WORKFLOWS.md) for actual masked PLL, rescoring, experimental-data preparation, notebooks, and SLURM usage. The proposed Rosetta/docking/MD workflow was a research plan; implementations of those stages are not present in this repository.

## Historical results and reproducibility

`results/` contains selected historical artifacts copied from the source workspace, not outputs rerun or validated against experimental ground truth during repository preparation. Identical files are stored once. Raw and derived data retain their original values. The candidate filter includes a documented correction to keep filtered sequences aligned with embedding rows; its new output can differ from archived results. See [release notes](docs/RELEASE_NOTES.md).

`docs/source_manifest.csv` maps every original workspace file to its retained location or exclusion reason, with its original SHA-256 checksum. Large/regenerable embeddings, logs, checkpoints, empty database placeholders, site-specific job files, and the parent Git history are excluded.

The source workspace supplied no code license or upstream redistribution license metadata. This preparation does not assign a license to the code, tournament data, or experimental workbooks; provenance details are in [DATA.md](docs/DATA.md).

