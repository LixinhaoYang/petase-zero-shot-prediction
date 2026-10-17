"""Sequence metadata lookups must preserve repeated candidate rows and reject conflicts."""

from pathlib import Path
import runpy
import unittest

import pandas as pd

PATH = (Path(__file__).resolve().parents[1] /
        'workflows/zero_shot/4_using_activity_data/run_hybrid_value_predict_v3.py')
UNIQUE = runpy.run_path(str(PATH))['unique_sequence_values']


class ActivityLookupTest(unittest.TestCase):
    def test_duplicate_metadata_does_not_expand_candidate_rows(self):
        candidates = pd.DataFrame({'sequence': ['b', 'a', 'b']})
        metadata = pd.DataFrame({'sequence': ['a', 'b', 'b', 'b'], 'score': [1., 2., 2., 2.]})
        result = candidates.merge(UNIQUE(metadata, ['score']), on='sequence',
                                  how='left', sort=False, validate='many_to_one')
        self.assertEqual(result['sequence'].tolist(), ['b', 'a', 'b'])
        self.assertEqual(result['score'].tolist(), [2., 1., 2.])

    def test_inconsistent_metadata_is_not_silently_discarded(self):
        metadata = pd.DataFrame({'sequence': ['a', 'a'], 'score': [1., 2.]})
        with self.assertRaisesRegex(ValueError, 'Conflicting values'):
            UNIQUE(metadata, ['score'])


if __name__ == '__main__':
    unittest.main()
