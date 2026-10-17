# Repository preparation notes

Prepared from the existing PETase research workspace on 2026-10-16. The original files and parent Git history remain in place.

## Layout and portability

- Moved source into `workflows/` and cleaned notebooks into `notebooks/`.
- Standardized documentation, source comments, and notebook annotations in English.
- Stored identical input files once in `data/raw/` or `data/processed/`.
- Archived historical tables and figures under `results/`; diagnostic CSVs previously inside embedding caches now live in `diagnostics/` directories.
- Added a Python 3.11 environment, optional analysis dependencies, a root-level launcher, input/schema checks, submission validation, and a GitHub Actions workflow.
- Replaced site-specific SLURM jobs with one template that contains no private account, queue, or filesystem path.
- Removed notebook outputs, execution counts, widgets, checkpoint copies, and local session metadata. Added a setup cell that stages historical intermediates into an ignored notebook working directory.
- Excluded embedding/model caches, machine logs, empty search/database placeholders, and old Git metadata. The source manifest records their original presence.

## Corrections in retained source

1. **Candidate-filter row alignment:** the old filter reset the candidate index before selecting embedding rows. After dropping candidates, it could cluster the wrong vectors. The filter now selects vectors with the original row indices. Small WT panels use the available references instead of always requiring five. Invalid filter fractions and empty inputs are rejected before model loading. The input is documented as CSV; the reserved `--pll_drop` option accepts only zero because no PLL calculation exists in this script.
2. **Repeated-sequence joins and WT header:** one candidate appears twice in the input. The cluster-expert script and the v3 activity script now use unique, conflict-checked metadata lookups and many-to-one merges. The activity output is rebuilt against the candidate rows, even when its archived expression baseline contains expanded duplicates. Original input multiplicity and order are preserved. The cluster-expert WT detector now accepts the canonical `Wt AA Sequence` header.
3. **PLL-helper syntax:** made a mathematical docstring raw so its backslash is literal and Python does not emit an invalid-escape warning.

Baseline scoring, weighted kNN, PLL scoring, experimental observations, and historical output values otherwise retain their original logic or bytes. Earlier activity variants and plotting scripts are retained as historical alternatives; the documented activity launcher selects v3. New corrected runs can differ from archived outputs.

## Research limitations retained from the source

- Normalized zero-shot scores retain physical-unit column names for submission compatibility; they are ranking heuristics, not calibrated measurements.
- The historical classification notebook contains alternative branches and depends on optional external tools and databases. It is not presented as an unattended end-to-end pipeline.
- Several original embedding caches use model names or a prefix of the sequence list as keys. Use a new `--output-dir` when changing sequence content/order, model settings, or representation layers. Pre-existing caches were excluded.
- The activity v3 script keeps the first observation per anchor sequence before fitting; it does not model all repeated experimental conditions. It also contains sparse-data fallbacks and an active-site motif heuristic. Its values need independent biological evaluation.
- The candidate filter computes WT similarity and embedding clusters. Masked PLL is available in the dedicated PLL workflow. Docking, Rosetta, and MD remain proposed future stages.
- No independent accuracy benchmark or full GPU reproduction was performed during repository preparation. The exact checks performed are recorded in [VALIDATION.md](VALIDATION.md).

