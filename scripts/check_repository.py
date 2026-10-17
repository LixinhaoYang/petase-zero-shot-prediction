#!/usr/bin/env python3
"""Dependency-free checks for the prepared research repository."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import warnings

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.venv', 'venv', '__pycache__', '.ipynb_checkpoints', 'outputs', '.pytest_cache'}
SECRET = re.compile(r'AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}'
                    r'|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
TEXT = {'.py', '.md', '.txt', '.yml', '.yaml', '.csv', '.json', '.ipynb', '.sh', '.sbatch', '.gb'}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, help='Optionally verify every original file against its recorded checksum.')
    args = parser.parse_args()
    errors = []
    files = [p for p in ROOT.rglob('*') if p.is_file() and not SKIP.intersection(p.relative_to(ROOT).parts)]
    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > 25 * 1024**2:
            errors.append(f'{relative}: exceeds this repository\'s 25 MiB per-file budget')
        if path.suffix in TEXT:
            source = path.read_text(encoding='utf-8')
            if SECRET.search(source):
                errors.append(f'{relative}: possible embedded credential')
        if path.suffix == '.py':
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', SyntaxWarning)
                    warnings.simplefilter('error', DeprecationWarning)
                    compile(path.read_text(), str(relative), 'exec')
            except (SyntaxError, SyntaxWarning, DeprecationWarning) as exc:
                errors.append(f'{relative}: {exc}')
        if path.suffix == '.ipynb':
            notebook = json.loads(path.read_text())
            if notebook.get('nbformat') != 4:
                errors.append(f'{relative}: expected notebook format 4')
            for i, cell in enumerate(notebook.get('cells', [])):
                if cell['cell_type'] == 'code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                    errors.append(f'{relative}: cell {i} retains execution state')
            if 'widgets' in notebook.get('metadata', {}):
                errors.append(f'{relative}: retained widget metadata')
        if path.suffix == '.md':
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text()):
                if '://' in target or target.startswith('#') or target.startswith('mailto:'):
                    continue
                target = target.split('#', 1)[0]
                if not (path.parent / target).exists():
                    errors.append(f'{relative}: broken local link {target}')

    manifest = read_csv(ROOT / 'docs/source_manifest.csv')
    for row in manifest:
        destination = row['destination']
        if destination:
            target = ROOT / destination
            if not target.is_file():
                errors.append(f'Manifest target missing: {destination}')
            elif row['action'] in {'included', 'deduplicated'}:
                if hashlib.sha256(target.read_bytes()).hexdigest() != row['source_sha256']:
                    errors.append(f'Unmodified artifact checksum differs: {destination}')
        if args.source_root:
            original = args.source_root / row['source_path']
            if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != row['source_sha256']:
                errors.append(f'Original source changed or missing: {row["source_path"]}')

    try:
        candidates = read_csv(ROOT / 'data/raw/predictive-pet-zero-shot-test-2025.csv')
        wildtypes = read_csv(ROOT / 'data/raw/pet-2025-wildtype-cds.csv')
        labels = read_csv(ROOT / 'data/processed/clustered_sequences.csv')
        if len(candidates) != 4988 or len(wildtypes) != 313:
            errors.append('Unexpected row count in canonical tournament inputs')
        if [r['sequence'] for r in candidates] != [r['sequence'] for r in labels]:
            errors.append('Shared cluster table does not preserve candidate sequence order')
        if not all(r.get('cluster') for r in labels):
            errors.append('Missing cluster labels')
    except (KeyError, OSError) as exc:
        errors.append(f'Input schema check failed: {exc}')

    if errors:
        print('\n'.join('FAIL: ' + e for e in errors))
        return 1
    size = sum(p.stat().st_size for p in files) / 1024**2
    print(f'PASS: {len(files)} release files ({size:.2f} MiB); Python syntax, clean notebooks, '
          f'local documentation links, {len(manifest)} source mappings, and canonical input schemas.')
    if args.source_root:
        print('PASS: all recorded original files retain their original SHA-256 checksums.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

