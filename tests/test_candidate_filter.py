"""Regressions for the filter's sequence/embedding alignment and small WT panels."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PATH = Path(__file__).resolve().parents[1] / 'workflows/candidate_filter/filter_candidates.py'
SPEC = importlib.util.spec_from_file_location('candidate_filter', PATH)
FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILTER)


class CandidateSelectionTest(unittest.TestCase):
    def test_filtered_sequences_use_their_own_embedding_rows(self):
        sequences = ['discard', 'east', 'north', 'near-east']
        embeddings = np.array([[-1., 0.], [1., 0.], [0., 1.], [.99, .01]])
        reference = np.array([[1., 0.]])
        seen = []

        def recording_kmeans(**kwargs):
            model = KMeans(**kwargs)
            original = model.fit_predict

            def record(values):
                seen.append(values.copy())
                return original(values)

            model.fit_predict = record
            return model

        with patch('sklearn.cluster.KMeans', side_effect=recording_kmeans):
            result = FILTER.select_candidates(sequences, embeddings, reference,
                                               sim_drop=.25, n_clusters=2)
        self.assertNotIn('discard', result['sequence'].tolist())
        np.testing.assert_array_equal(seen[0], embeddings[1:])
        labels = result.set_index('sequence')['cluster']
        self.assertEqual(labels['east'], labels['near-east'])
        self.assertNotEqual(labels['east'], labels['north'])

    def test_one_reference_sequence_and_cluster_cap(self):
        result = FILTER.select_candidates(['a', 'b', 'c'], np.eye(3), np.ones((1, 3)),
                                           sim_drop=0, n_clusters=1, cluster_k=2)
        self.assertEqual(result['sequence'].tolist(), ['a', 'b'])
        self.assertTrue(np.isfinite(result['sim_wt_topk']).all())

    def test_invalid_inputs_fail_before_clustering(self):
        for drop in [-.1, 1.0]:
            with self.assertRaises(ValueError):
                FILTER.select_candidates(['a'], np.ones((1, 2)), np.ones((1, 2)), sim_drop=drop)
        with self.assertRaises(ValueError):
            FILTER.select_candidates([], np.empty((0, 2)), np.ones((1, 2)))
        with self.assertRaises(ValueError):
            FILTER.select_candidates(['a'], np.ones((2, 2)), np.ones((1, 2)))

    def test_protein_column_selection_is_explicit(self):
        frame = pd.DataFrame(columns=['CDS', 'Wt AA Sequence'])
        self.assertEqual(FILTER.find_seq_col(frame), 'Wt AA Sequence')
        with self.assertRaises(ValueError):
            FILTER.find_seq_col(pd.DataFrame(columns=['query_seq', 'reference_seq']))


if __name__ == '__main__':
    unittest.main()
