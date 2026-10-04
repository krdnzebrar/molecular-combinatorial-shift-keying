"""Noisy-channel K=5 regression and threshold ablation on N500 templates."""

import contextlib
import functools
import io
import os
import unittest

import numpy as np

import experiments.sweep as sweep
import modulation.combination_fixed as mocsk
import modulation.permutation as permutation
import modulation.sparse as emocsk
from config import CUSTOM_DIR
from detection.channel_threshold import expected_pulse_threshold


class NoisyK5Regression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_dir = os.path.join(CUSTOM_DIR, "N500")
        if not os.path.isfile(os.path.join(cls.template_dir, "Exp_099.csv")):
            raise unittest.SkipTest(
                f"Need 100 channel templates in {cls.template_dir} for this layer")

    def setUp(self):
        self.old_log = sweep._SWEEP_DETAIL_LOG
        sweep._SWEEP_DETAIL_LOG = os.devnull
        self.old_savers = [module.save_signal_csv for module in
                           (permutation, mocsk, emocsk)]
        for module in (permutation, mocsk, emocsk):
            module.save_signal_csv = lambda *args, **kwargs: None

    def tearDown(self):
        for module, saver in zip((permutation, mocsk, emocsk), self.old_savers):
            module.save_signal_csv = saver
        sweep._SWEEP_DETAIL_LOG = self.old_log
        emocsk.expected_pulse_threshold = expected_pulse_threshold

    def test_k5_ts_regression_and_threshold_fraction_ablation(self):
        schemes = ("permutation", "combo_fixed", "sparse")
        errors_by_point = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for scheme in schemes:
                for ts in (0.6, 0.8, 1.0):
                    counts = sweep.sweep_ts_pooled(
                        scheme, [ts], 500, num_molecule_types=5,
                        seq_length=500, num_trials=1, n_exp_templates=100,
                        rng_seed=42, generate_templates=False,
                        exp_root=CUSTOM_DIR)
                    errors_by_point[(scheme, ts)] = int(counts["errors"][0])
                    self.assertEqual(int(counts["totals"][0]), 500)

            ablation = {}
            for fraction in (0.3, 0.4, 0.5, 0.6):
                emocsk.expected_pulse_threshold = functools.partial(
                    expected_pulse_threshold, fraction=fraction)
                counts = sweep.sweep_ts_pooled(
                    "sparse", [1.0], 500, num_molecule_types=5,
                    seq_length=500, num_trials=1, n_exp_templates=100,
                    rng_seed=42, generate_templates=False,
                    exp_root=CUSTOM_DIR)
                ablation[fraction] = int(counts["errors"][0])
                self.assertEqual(int(counts["totals"][0]), 500)

        print("K=5 errors / 500 at Ts=0.6, 0.8, 1.0:", errors_by_point)
        print("E-MoCSK errors / 500 at threshold fractions:", ablation)

    def test_slot_zero_miss_rate_by_previous_last_slot_molecule(self):
        np.random.seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, transmissions, _, decoded = emocsk.generate_sparse_bit_sequence_transmission(
                bit_sequence="1" * 500,
                num_molecule_types=5,
                exp_path=self.template_dir,
                num_experiments=100,
                normalization=10,
                delay_between_symbols=1.0,
                delay_between_molecules=0.2,
                plot=False,
            )

        same = [0, 0]
        other = [0, 0]
        for index in range(1, len(transmissions)):
            molecule = transmissions[index]["pattern"][0]
            if molecule is None:
                continue
            group = same if transmissions[index - 1]["pattern"][-1] == molecule else other
            group[1] += 1
            group[0] += decoded[index]["permutation"][0] == "_"
        print("slot-0 under-detection (misses / present):",
              {"same previous-last molecule": same,
               "other": other})


if __name__ == "__main__":
    unittest.main()
