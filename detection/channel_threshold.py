"""Channel-derived thresholds that do not depend on transmitted labels."""

import os

import numpy as np

from utils.signal import load_experiment_data, moving_average


_THRESHOLD_CACHE = {}


def expected_pulse_threshold(exp_path, normalization, dt_bin, smooth_window,
                             n_templates=20, fraction=0.5):
    """Return a fraction of the mean smoothed, baseline-subtracted pulse peak."""
    key = (os.path.abspath(exp_path), int(normalization), float(dt_bin),
           int(smooth_window), int(n_templates), float(fraction))
    if key in _THRESHOLD_CACHE:
        return _THRESHOLD_CACHE[key]

    peaks = []
    for i in range(int(n_templates)):
        path = os.path.join(exp_path, f"Exp_{i:03d}.csv")
        if not os.path.isfile(path):
            if not peaks:
                raise FileNotFoundError(f"Missing channel template: {path}")
            break
        time_axis, hits = load_experiment_data(path, normalization)
        if not len(hits):
            continue
        smoothed = moving_average(hits, max(1, int(smooth_window)))
        peak_idx = int(np.argmax(smoothed))
        # Templates start before the first molecule arrives, so sample that
        # baseline instead of the rising edge immediately before the peak.
        baseline_idx = 0
        peaks.append(max(float(smoothed[peak_idx] - smoothed[baseline_idx]), 0.0))

    if not peaks:
        raise ValueError(f"No usable templates found in {exp_path}")
    threshold = float(fraction * np.mean(peaks))
    _THRESHOLD_CACHE[key] = threshold
    return threshold
