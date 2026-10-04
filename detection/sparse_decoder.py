"""
detection/sparse_decoder.py
===========================
Fixed threshold + joint molecule/empty-slot assignment decoder for sparse patterns.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import R0, RR, D, SMOOTHING_WINDOW
from utils.signal import moving_average, adaptive_smoothing_window
from detection.slot_scoring import slot_scores as _shared_slot_scores


def decode_sparse_fixed(molecule_signals, bit_sequence, dt_bin,
                         num_molecule_types, molecule_names,
                         delay_between_symbols, delay_between_molecules,
                         fixed_threshold):
    """
    Jointly assign molecule types and empty-slot dummy choices.
    """
    t_peak_theory = (R0 - RR) ** 2 / (6 * D)
    half_w = delay_between_molecules * 0.45

    smoothed_signals = np.zeros_like(molecule_signals)
    smooth_window = adaptive_smoothing_window(
        delay_between_molecules, dt_bin, SMOOTHING_WINDOW)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], smooth_window)

    decoded_symbols = []

    for bit_idx, bit in enumerate(bit_sequence):
        if bit != '1':
            continue

        symbol_start_t = bit_idx * delay_between_symbols
        decoded_pattern = [None] * num_molecule_types
        score_matrix = np.full((num_molecule_types, 2 * num_molecule_types), -np.inf)
        for slot_idx in range(num_molecule_types):
            slot_send_t = symbol_start_t + slot_idx * delay_between_molecules
            peak_center = slot_send_t + t_peak_theory

            rises = _shared_slot_scores(
                smoothed_signals, peak_center, half_w, dt_bin,
                delay_between_molecules)
            score_matrix[slot_idx, :num_molecule_types] = rises - fixed_threshold
            score_matrix[slot_idx, num_molecule_types:] = 0.0

        rows, cols = linear_sum_assignment(-score_matrix)
        for slot_idx, col in zip(rows, cols):
            if col < num_molecule_types and score_matrix[slot_idx, col] > 0:
                decoded_pattern[slot_idx] = int(col)

        decoded_string = ''.join(
            molecule_names[m] if m is not None else '_'
            for m in decoded_pattern
        )
        sym_s = max(int(symbol_start_t / dt_bin), 0)
        sym_e = min(int((symbol_start_t + delay_between_symbols) / dt_bin),
                    smoothed_signals.shape[1] - 1)

        decoded_symbols.append({
            'bit_position': bit_idx,
            'permutation': decoded_string,
            'peak_value': float(np.max(smoothed_signals[:, sym_s:sym_e]))
        })

    return decoded_symbols
