#!/bin/bash
# Set site-specific account/partition options when calling sbatch.
# Submit from the repository root after activating your Python environment:
#   mkdir -p outputs/slurm
#   sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/slurm/run_baseline.sbatch
#SBATCH --job-name=petase-baseline
#SBATCH --output=outputs/slurm/%x.%j.out
#SBATCH --error=outputs/slurm/%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
python scripts/run_workflow.py baseline --gpu

