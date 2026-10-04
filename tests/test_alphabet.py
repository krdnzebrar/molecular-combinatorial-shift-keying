"""Fast checks for modulation alphabet sizes and uniform samplers."""

import math
import unittest
from collections import Counter

import numpy as np

from modulation.alphabet import sample_ordered_subset, sample_sparse_pattern


class AlphabetTests(unittest.TestCase):
    def test_alphabet_sizes_match_table(self):
        expected_mocsk = {2: 5, 3: 16, 4: 65, 5: 326, 10: 9_864_101}
        expected_emocsk = {2: 7, 3: 34, 4: 209, 5: 1_546, 10: 234_662_231}
        for k, expected in expected_mocsk.items():
            actual = sum(math.factorial(k) // math.factorial(k - j)
                         for j in range(k + 1))
            self.assertEqual(actual, expected)
        for k, expected in expected_emocsk.items():
            actual = sum(math.comb(k, j) ** 2 * math.factorial(j)
                         for j in range(k + 1))
            self.assertEqual(actual, expected)

    def test_mtpsk_three_molecule_bit_count_is_exact(self):
        mtpsk_bits = math.log2(math.factorial(3))
        self.assertAlmostEqual(mtpsk_bits, math.log2(6), places=14)
        self.assertAlmostEqual(mtpsk_bits,
                               2.584962500721156, places=14)

    def test_ordered_subset_sampler_is_uniform_at_k3(self):
        self._assert_uniform(sample_ordered_subset, expected_symbols=16)

    def test_sparse_pattern_sampler_is_uniform_at_k3(self):
        self._assert_uniform(sample_sparse_pattern, expected_symbols=34)

    def _assert_uniform(self, sampler, expected_symbols):
        sample_count = 32_000
        np.random.seed(0)
        counts = Counter(sampler(3) for _ in range(sample_count))
        self.assertEqual(len(counts), expected_symbols)
        expected_per_symbol = sample_count / expected_symbols
        tolerance = 5 * math.sqrt(expected_per_symbol)
        self.assertTrue(all(abs(count - expected_per_symbol) < tolerance
                            for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
