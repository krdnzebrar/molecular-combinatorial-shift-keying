"""
detection/threshold_decoder.py
==============================
Threshold-based MoCSK decoder using a channel-calibrated fixed threshold.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import SMOOTHING_WINDOW, T_PEAK_THEORY
from utils.signal import moving_average, adaptive_smoothing_window
from detection.slot_scoring import slot_scores as _slot_scores


def _combination_slot_scores(smoothed_signals, bit_idx, num_molecule_types,
                              dt_bin, delay_between_symbols,
                              delay_between_molecules):
    """Rise scores indexed by MoCSK release slot and molecule type."""
    half_window = max(dt_bin, 0.45 * delay_between_molecules)
    start_t = bit_idx * delay_between_symbols
    scores = np.zeros((num_molecule_types, num_molecule_types), dtype=float)
    for slot_idx in range(num_molecule_types):
        center = start_t + slot_idx * delay_between_molecules + T_PEAK_THEORY
        scores[slot_idx] = _slot_scores(
            smoothed_signals, center, half_window, dt_bin,
            delay_between_molecules)
    return scores


def _decode_mocsk_prefix(scores, threshold, molecule_names):
    """Choose the best ordered prefix of occupied MoCSK release slots."""
    best_count, best_objective, best_cols = 0, 0.0, []
    for active_count in range(1, scores.shape[0] + 1):
        rows, cols = linear_sum_assignment(-scores[:active_count])
        objective = float(np.sum(scores[rows, cols] - threshold))
        if objective > best_objective:
            best_count = active_count
            best_objective = objective
            best_cols = cols[np.argsort(rows)].tolist()
    return ''.join(molecule_names[m] for m in best_cols[:best_count])


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-threshold decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_combinations_fixed(molecule_signals, bit_sequence, dt_bin,
                               num_molecule_types, molecule_names,
                               delay_between_symbols,
                               fixed_threshold, delay_between_molecules=0.4):
    """
    Fixed threshold decoder for combinations using an unlabeled calibration.

    Returns
    -------
    decoded_symbols : list of dict
    smoothed_signals : np.ndarray
        The smoothed signal matrix (for plotting).
    """
    smoothed_signals = np.zeros_like(molecule_signals)
    smooth_window = adaptive_smoothing_window(
        delay_between_molecules, dt_bin, SMOOTHING_WINDOW)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], smooth_window)

    decoded_symbols = []

    for bit_idx, bit in enumerate(bit_sequence):
        if bit != '1':
            continue

        slot_scores = _combination_slot_scores(
            smoothed_signals, bit_idx, num_molecule_types, dt_bin,
            delay_between_symbols, delay_between_molecules)
        peak_vals = np.max(slot_scores, axis=0)
        decoded_combo = _decode_mocsk_prefix(
            slot_scores, fixed_threshold, molecule_names)

        decoded_symbols.append({
            'bit_position': bit_idx,
            'decoded_combo': decoded_combo,
            'peak_vals': peak_vals,
            'threshold': fixed_threshold,
            'peak_value': float(np.max(peak_vals)),
        })

    return decoded_symbols, smoothed_signals
