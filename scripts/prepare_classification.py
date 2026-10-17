#!/usr/bin/env python3
"""Stage notebook inputs and historical intermediates in an ignored working directory."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    destination = ROOT / 'outputs' / 'classification'
    destination.mkdir(parents=True, exist_ok=True)
    sources = [(ROOT / 'results' / 'classification', destination)]
    for source, target in sources:
        for path in source.rglob('*'):
            if path.is_file():
                out = target / path.relative_to(source)
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    shutil.copy2(path, out)
    for path in [
        ROOT / 'data/raw/pet-2025-wildtype-cds.csv',
        ROOT / 'data/raw/predictive-pet-zero-shot-test-2025.csv',
        ROOT / 'data/raw/PETase.xlsx',
        ROOT / 'data/processed/clustered_sequences.csv',
    ]:
        out = destination / path.name
        if not out.exists():
            shutil.copy2(path, out)
    for name in ['MSA', 'Phylo', 'logos', 'search']:
        (destination / name).mkdir(exist_ok=True)
    return destination


if __name__ == '__main__':
    print(prepare())

