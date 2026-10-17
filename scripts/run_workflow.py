#!/usr/bin/env python3
"""Launch the retained workflows with repository-relative inputs and isolated outputs."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ZERO = ROOT / 'workflows' / 'zero_shot'
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'


def build_command(args, extra):
    output = args.output_dir or ROOT / 'outputs' / args.workflow
    output = output.resolve()
    test = (args.test_csv or RAW / 'predictive-pet-zero-shot-test-2025.csv').resolve()
    wt = (args.wt_csv or RAW / 'pet-2025-wildtype-cds.csv').resolve()
    clusters = (args.cluster_csv or PROCESSED / 'clustered_sequences.csv').resolve()
    common = ['--test_csv', str(test), '--wt_csv', str(wt),
              '--out_csv', str(output / 'submission.csv'), '--cache_dir', str(output / 'cache')]

    if args.workflow in {'baseline', 'weighted-knn', 'pll'}:
        script = {
            'baseline': ZERO / '0_esm_kNN' / 'run_zeroshot_esm_knn.py',
            'weighted-knn': ZERO / '1_update_kNN' / 'run_zeroshot_esm_knn.py',
            'pll': ZERO / '2_esm_kNN_pLL' / 'run_zeroshot_esm_knn_pll.py',
        }[args.workflow]
        flags = common
        if args.workflow == 'weighted-knn':
            flags += ['--knn_k', '5']
        if args.workflow == 'pll':
            flags += ['--use_pll']
    elif args.workflow == 'cluster-expert':
        script = ZERO / '3_petase_specific' / 'run_cluster_expert_zeroshot.py'
        flags = common + ['--cluster_csv', str(clusters), '--merge_34']
    elif args.workflow == 'activity':
        if args.gpu:
            raise ValueError('The activity recipe uses saved similarities and runs on CPU; omit --gpu.')
        script = ZERO / '4_using_activity_data' / 'run_hybrid_value_predict_v3.py'
        flags = [
            '--test_csv', str(test),
            '--test_scored_csv', str(PROCESSED / 'test_scored_with_wt.csv'),
            '--clustered_sequences_csv', str(clusters),
            '--activate_scored_csv', str(PROCESSED / 'activate_scored_with_wt.csv'),
            '--activate_cluster_csv', str(PROCESSED / 'activate_with_cluster.csv'),
            '--baseline_submission_csv', str(ROOT / 'results' / 'zero_shot' / '3_petase_specific' /
                                              'submission_cluster_expert_simonly.csv'),
            '--merge_34', '--out_csv', str(output / 'submission.csv'),
        ]
    else:
        script = ROOT / 'workflows' / 'candidate_filter' / 'filter_candidates.py'
        flags = ['--candidates', str(test), '--wt_csv', str(wt),
                 '--out_csv', str(output / 'filtered_candidates.csv')]
    if args.gpu:
        flags.append('--use_gpu')
    return [sys.executable, str(script), *flags, *extra], output


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('workflow', choices=['baseline', 'weighted-knn', 'pll', 'cluster-expert', 'activity', 'filter'])
    parser.add_argument('--dry-run', action='store_true', help='Print the command without creating files or running it.')
    parser.add_argument('--output-dir', type=Path, help='Run directory (default: outputs/<workflow>).')
    parser.add_argument('--test-csv', type=Path, help='Candidate input CSV; used as --candidates for the filter.')
    parser.add_argument('--wt-csv', type=Path, help='Wild-type input CSV (not used by saved-score activity calibration).')
    parser.add_argument('--cluster-csv', type=Path, help='Cluster labels for cluster-expert or activity.')
    parser.add_argument('--gpu', action='store_true', help='Request CUDA for an ESM workflow.')
    args, extra = parser.parse_known_args()
    try:
        command, output = build_command(args, extra)
    except ValueError as exc:
        parser.error(str(exc))
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())

