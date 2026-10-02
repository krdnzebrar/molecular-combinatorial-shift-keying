"""
utils/signal.py
===============
Shared signal processing utilities.
ONE definition of each function — imported everywhere.
"""

import numpy as np


def moving_average(x, w):
    """Causal moving average with convolution."""
    return np.convolve(x, np.ones(w) / w, mode='same')


def adaptive_smoothing_window(slot_delay, dt_bin, maximum=40):
    """Keep smoothing below one third of the inter-slot time."""
    if dt_bin <= 0:
        raise ValueError("dt_bin must be positive")
    return max(1, min(int(maximum), int(float(slot_delay) / (3.0 * float(dt_bin)))))


def load_experiment_data(exp_path, normalization=1):
    """
    Load a single Exp_xxx.csv template and optionally bin the data.

    Parameters
    ----------
    exp_path : str
        Path to CSV with columns [Time, Number of Molecules].
    normalization : int
        Bin-averaging factor. If > 1, adjacent steps are aggregated.

    Returns
    -------
    time_sec : np.ndarray
    hits : np.ndarray
    """
    data = np.genfromtxt(exp_path, delimiter=',', skip_header=1)
    time_sec = data[:, 0]
    hits = data[:, 1]

    if normalization > 1:
        usable_len = (len(time_sec) // normalization) * normalization
        time_binned = time_sec[:usable_len].reshape(-1, normalization).mean(axis=1)
        hits_binned = hits[:usable_len].reshape(-1, normalization).sum(axis=1)
        return time_binned, hits_binned

    return time_sec, hits
