# Data and provenance

The supplied source workspace identifies this project as the predictive phase of the **2025 Protein Engineering Tournament**. No original download URL, DOI, code license, or data redistribution license was supplied. This document records observable file provenance without assigning additional permissions or inventing citations.

## Raw inputs

| File under `data/raw/` | Contents | Source in the original workspace |
| --- | --- | --- |
| `predictive-pet-zero-shot-test-2025.csv` | 4,988 candidate rows; 4,987 distinct sequences; original target columns | `Data_zero_shot/` |
| `pet-2025-wildtype-cds.csv` | 313 distinct WT proteins, with `Wt AA Sequence` and `CDS` columns | `Data_zero_shot/` |
| `wildtype-sequences-only-313.csv` | Same 313 proteins in the same order, with a single `Sequence` column | `1_zero_shot_rebuilt/search/step1/` |
| `wildtype-augmented-393.csv` | 393 rows, 389 distinct proteins, including all 313 canonical WT proteins; additional-reference provenance unspecified | `1_zero_shot_rebuilt/temp/pet-2025-wildtype-cds-new.csv` |
| `activate.xlsx` | Experimental activity workbook used by the activity preprocessing scripts | Workspace root; identical duplicate in `4_using_activity_data/` |
| `yield.xlsx` | Original yield workbook; not consumed by the default activity recipe | Workspace root |
| `PETase.xlsx` | Combined experimental reference workbook used by optional notebook comparisons | `0_classification/` |
| `pet-2025-experimental-methods.pdf` | Supplied experimental-methods reference, copied unchanged | `Data_zero_shot/` |
| `pet-2025-expression-vector.gb` | Supplied expression-vector reference, copied unchanged | `Data_zero_shot/` |

The one repeated candidate row is intentionally retained. New supported submissions must have exactly 4,988 rows in the original order. Sequence-based metadata lookups in the cluster-expert and v3 activity scripts now validate many-to-one joins, avoiding the repeated-row expansion seen in historical results.

## Processed inputs

| File under `data/processed/` | Purpose |
| --- | --- |
| `clustered_sequences.csv` | 4,988 original candidate rows with `cluster`, `umap_x`, and `umap_y`; shared by cluster-expert and activity workflows |
| `test_scored_with_wt.csv` | Saved candidate-to-WT similarity features; the activity recipe uses `sim_wt_topk` |
| `activate_tidy.csv` | 163 long-format observations from 15 distinct experimental sequences; includes temperature, pH, substrate, and value |
| `activate_tidy_scored.csv` | Historical ESM density/centroid scores added to the tidy observations |
| `activate_scored_with_wt.csv` | 163 activity observations with WT similarity features, used for calibration |
| `activate_with_cluster.csv` | Cluster assignments and vote confidence for the activity observations |
| `activate_seq_stats.csv` | Summary statistics for the 15 experimental sequences |

These tables are historical derived artifacts, not newly measured data. `scripts/run_workflow.py activity` uses the saved similarities and labels to exercise the calibration stage without recomputing ESM embeddings. For a new sequence set, generate matching features and labels and pass their paths explicitly.

## Historical results

Classification tables, annotations, alignments, trees, figures, and zero-shot submissions are retained under `results/`. Experimental filtering, BLAST, mutation, and pLDDT tables are retained under `results/candidate_filter/`. The structure files and the code that generated the pLDDT summaries were not present in the source folder. The final filled CSV in that directory is an archived exploratory artifact, not the validated default submission.

Identical files are retained once; for example, the original weighted-kNN submission was byte-for-byte identical to the baseline submission. The manifest records both source names and their shared destination. Archived numeric data were not edited.

## Manifest

[`source_manifest.csv`](source_manifest.csv) covers all 265 original non-Git files. Each record lists its original relative path, original size and SHA-256, destination when retained, and an action:

- `included`: copied unchanged to the new layout.
- `adapted`: retained source with a documented code or notebook adjustment.
- `deduplicated`: identical bytes are retained at the destination listed.
- `excluded`: regenerable cache, log, checkpoint, empty placeholder, site-specific job file, or documentation replaced by the consolidated version.

No parent Git history is copied. The original workspace can be verified without modifying it:

```bash
python scripts/check_repository.py --source-root ..
```

