# Historical results

These files are archived from the original research workspace. They are not a benchmark report and were not regenerated as part of repository preparation.

- `classification/`: cluster tables, UMAP plots, taxonomic annotations, MSA/logo artifacts, and trees.
- `zero_shot/`: historical submissions, similarity/PLL debug tables, activity diagnostics, and plots. Numeric subdirectories retain the original experiment order.
- `candidate_filter/`: exploratory filtering, BLAST, mutation, and pLDDT summaries. The underlying structure-generation jobs and structures are not included.

Required shared intermediate tables are stored once in `data/processed/`. See `docs/source_manifest.csv` for exact mappings, including duplicate results that share one retained copy. Fresh runs write to the ignored `outputs/` directory.

