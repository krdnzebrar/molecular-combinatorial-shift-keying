"""Shared local-rise scoring for slot based moving-average decoders."""

import numpy as np


def slot_scores(smoothed_signals, peak_center, half_window, dt_bin,
                delay_between_molecules):
    """Score each molecule by its local peak rise above the preceding valley.

    The baseline window ends half a slot before the expected peak and uses
    its minimum, avoiding a same-molecule pulse peak from the preceding slot.
    """
    n_samples = smoothed_signals.shape[1]
    peak_start = max(0, int((peak_center - half_window) / dt_bin))
    peak_end = min(n_samples, int(np.ceil((peak_center + half_window) / dt_bin)) + 1)
    if peak_end <= peak_start:
        return np.zeros(smoothed_signals.shape[0], dtype=float)

    valley = peak_center - 0.5 * delay_between_molecules
    b0 = max(0, int((valley - half_window) / dt_bin))
    b1 = max(b0 + 1, int(valley / dt_bin) + 1)
    b1 = min(b1, n_samples)
    if b0 >= b1:
        baseline = smoothed_signals[:, min(max(b0, 0), n_samples - 1)]
    else:
        baseline = smoothed_signals[:, b0:b1].min(axis=1)
    peaks = smoothed_signals[:, peak_start:peak_end].max(axis=1)
    return np.maximum(peaks - baseline, 0.0)
