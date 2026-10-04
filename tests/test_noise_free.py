"""Optional deterministic decoder grid using the mean N1000 template."""

import contextlib
import io
import os
import tempfile
import unittest

import numpy as np

import experiments.sweep as sweep
import modulation.combination_fixed as mocsk
import modulation.encoder as encoder
import modulation.permutation as permutation
import modulation.sparse as emocsk
from config import CUSTOM_DIR
from utils.signal import load_experiment_data, moving_average


class MeanTemplateDecoderGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_dir = os.path.join(CUSTOM_DIR, "N1000")
        if not os.path.isfile(os.path.join(cls.template_dir, "Exp_099.csv")):
            raise unittest.SkipTest(
                f"Need 100 channel templates in {cls.template_dir} for this layer")
        data = [load_experiment_data(
            os.path.join(cls.template_dir, f"Exp_{i:03d}.csv"), 10)
            for i in range(100)]
        cls.time = data[0][0]
        cls.mean_hits = np.mean([item[1] for item in data], axis=0)

    def test_mean_template_ser_grid(self):
        original_loader = encoder.load_experiment_data
        original_savers = [module.save_signal_csv for module in
                           (permutation, mocsk, emocsk)]
        original_thresholds = [module.expected_pulse_threshold for module in
                               (mocsk, emocsk)]
        old_detail_log = sweep._SWEEP_DETAIL_LOG
        with tempfile.TemporaryDirectory(prefix="mcsk-noise-free-") as temp_dir:
            try:
                encoder.load_experiment_data = lambda _path, _normalization: (
                    self.time, self.mean_hits)
                def mean_pulse_threshold(_path, _normalization, _dt_bin, window):
                    smoothed = moving_average(self.mean_hits, max(1, int(window)))
                    return 0.5 * max(float(smoothed.max() - smoothed[0]), 0.0)

                for module in (permutation, mocsk, emocsk):
                    module.save_signal_csv = lambda *args, **kwargs: None
                mocsk.expected_pulse_threshold = mean_pulse_threshold
                emocsk.expected_pulse_threshold = mean_pulse_threshold
                sweep._SWEEP_DETAIL_LOG = os.path.join(temp_dir, "detail.log")
                schemes = ("permutation", "combo_fixed", "sparse")
                ts_values = tuple(i / 10 for i in range(2, 11))
                grid = {}
                for k in range(2, 11):
                    for ts in ts_values:
                        for scheme in schemes:
                            np.random.seed(0)
                            with contextlib.redirect_stdout(io.StringIO()):
                                errors, total = sweep._single_trial_counts(
                                    scheme, "1" * 100, k, ts, self.template_dir,
                                    100, 10, ts / k)
                            self.assertEqual(total, 100)
                            grid[(scheme, k, ts)] = errors / total
                for scheme in schemes:
                    print(f"\n{scheme} noise-free SER, rows K=2..10; columns Ts=0.2..1.0")
                    for k in range(2, 11):
                        print(k, [grid[(scheme, k, ts)] for ts in ts_values])
                for scheme in schemes:
                    for k in range(2, 11):
                        self.assertEqual(grid[(scheme, k, 1.0)], 0.0,
                                         f"{scheme}, K={k}, Ts=1.0")
            finally:
                encoder.load_experiment_data = original_loader
                for module, saver in zip((permutation, mocsk, emocsk), original_savers):
                    module.save_signal_csv = saver
                mocsk.expected_pulse_threshold, emocsk.expected_pulse_threshold = original_thresholds
                sweep._SWEEP_DETAIL_LOG = old_detail_log


if __name__ == "__main__":
    unittest.main()
