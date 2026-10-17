#!/usr/bin/env python3
"""Check submission columns, sequence order, and finite nonnegative numeric values."""

import argparse
import csv
import math
from pathlib import Path

COLUMNS = ['sequence', 'activity_1 (μmol [TPA]/min·mg [E])',
           'activity_2 (μmol [TPA]/min·mg [E])', 'expression (mg/mL)']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('--reference', type=Path, required=True, help='Candidate CSV in expected row order.')
    args = parser.parse_args()
    with args.submission.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            parser.error(f'Expected columns {COLUMNS}; got {reader.fieldnames}')
        rows = list(reader)
    with args.reference.open(newline='', encoding='utf-8-sig') as handle:
        expected = [row['sequence'] for row in csv.DictReader(handle)]
    if not rows:
        parser.error('Submission is empty.')
    if [row['sequence'] for row in rows] != expected:
        parser.error('Submission sequence order/count differs from the reference.')
    for index, row in enumerate(rows, start=2):
        for column in COLUMNS[1:]:
            try:
                value = float(row[column])
            except (ValueError, TypeError):
                parser.error(f'Invalid numeric value at row {index}, column {column}')
            if not math.isfinite(value) or value < 0:
                parser.error(f'Non-finite or negative value at row {index}, column {column}')
    print(f'PASS: {len(rows)} rows; expected order and finite, nonnegative values.')


if __name__ == '__main__':
    main()

