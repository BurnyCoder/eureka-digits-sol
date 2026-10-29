"""Synthetic tests for paired analysis; these never fit a model or read study scores."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reports'))
from compare_artifacts import paired_summary


class PairedAnalysis(unittest.TestCase):
    """Probe pairing, seed averaging and degenerate-interval interpretation."""

    def records(self, predictions):
        """Construct aligned synthetic records with no connection to digits results."""
        return [dict(seed=i, example_ids=[7, 8], y_true=[0, 1], y_pred=row)
                for i, row in enumerate(predictions)]

    def test_identical_correctness_is_inconclusive(self):
        """A degenerate interval is not evidence of equivalence or benefit."""
        rows = self.records([[0, 1], [1, 1], [0, 0]])
        result = paired_summary(rows, rows, resamples=100, seed=3)
        self.assertEqual(result['mean_difference'], 0)
        self.assertEqual(result['conditional_interval'], [0, 0])
        self.assertEqual(result['interpretation'], 'inconclusive')

    def test_shared_examples_preserve_cross_seed_cancellation(self):
        """Opposite effects on each example cancel only if resampling stays paired."""
        treated = self.records([[0, 0], [1, 1]])
        baseline = self.records([[1, 1], [0, 0]])
        result = paired_summary(treated, baseline, resamples=100, seed=3)
        self.assertEqual(result['conditional_interval'], [0, 0])

    def test_constant_harm_and_seed_sd(self):
        """Known constant harm detects reversed subtraction and wrong SD semantics."""
        treated = self.records([[1, 0], [1, 0], [1, 0]])
        baseline = self.records([[0, 1], [0, 1], [0, 1]])
        result = paired_summary(treated, baseline, resamples=100, seed=3)
        self.assertEqual(result['mean_difference'], -1)
        self.assertEqual(result['seed_difference_sample_sd'], 0)
        self.assertEqual(result['conditional_interval'], [-1, -1])
        self.assertEqual(result['interpretation'], 'direction_contradicted_conditionally')

    def test_misaligned_examples_or_labels_rejected(self):
        """Arithmetic without paired IDs and labels can silently mislead."""
        left = self.records([[0, 1], [0, 1]])
        right = self.records([[0, 1], [0, 1]])
        right[0]['example_ids'].reverse()
        with self.assertRaises(AssertionError):
            paired_summary(left, right, resamples=100, seed=3)
        right = self.records([[0, 1], [0, 1]])
        right[0]['y_true'].reverse()
        with self.assertRaises(AssertionError):
            paired_summary(left, right, resamples=100, seed=3)


if __name__ == '__main__':
    unittest.main()
