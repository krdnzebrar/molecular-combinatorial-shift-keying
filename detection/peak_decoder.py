"""
detection/peak_decoder.py
=========================
Smoothed-peak arrival-order decoder for the permutation scheme.

Detects the transmitted permutation with one local arrival window per
release slot. Each slot is assigned the strongest not-yet-used molecule
channel, limiting tail-only channels from changing the decoded order.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import R0, RR, D, SMOOTHING_WINDOW
from utils.signal import moving_average, adaptive_smoothing_window
from detection.slot_scoring import slot_scores


def robust_decode(molecule_signals, bit_sequence, dt_bin, num_molecule_types,
                  molecule_names, delay_between_symbols, delay_between_molecules):
    """
    Decode permutation symbols by smoothed peak detection.

    For each '1' bit:
      1. Smooth each channel with a window sized to the slot spacing.
      2. Search near the expected arrival peak for each release slot.
      3. Assign one unused molecule type to each slot in release order.

    Returns
    -------
    decoded_symbols : list of dict
        Each has keys: 'permutation', 'peak_value', 'molecule_peaks', 'peak_time'.
    """
    t_peak_theory = (R0 - RR) ** 2 / (6 * D)
    # Keep smoothing shorter than the spacing between release slots so peaks
    # from adjacent molecules do not merge at short Ts or large alphabets.
    smooth_window = adaptive_smoothing_window(
        delay_between_molecules, dt_bin, SMOOTHING_WINDOW)
    smoothed_signals = np.zeros_like(molecule_signals)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], smooth_window)

    decoded_symbols = []

    for bit_idx, bit in enumerate(bit_sequence):
        if bit != '1':
            continue

        symbol_start_t = bit_idx * delay_between_symbols
        half_window = max(dt_bin, 0.45 * delay_between_molecules)
        score_matrix = np.full((num_molecule_types, num_molecule_types), -np.inf)
        peak_indices = np.zeros((num_molecule_types, num_molecule_types), dtype=int)
        for slot_idx in range(num_molecule_types):
            expected_peak_t = (symbol_start_t + slot_idx * delay_between_molecules
                               + t_peak_theory)
            rises = slot_scores(smoothed_signals, expected_peak_t, half_window,
                                dt_bin, delay_between_molecules)
            for m_type in range(num_molecule_types):
                # Keep the local peak time for diagnostics; assignment uses
                # the same valley-subtracted rise as the other decoders.
                idx_start = max(0, int((expected_peak_t - half_window) / dt_bin))
                idx_end = min(smoothed_signals.shape[1],
                              int(np.ceil((expected_peak_t + half_window) / dt_bin)) + 1)
                window_data = smoothed_signals[m_type, idx_start:idx_end]
                if not len(window_data):
                    continue
                local_peak_idx = int(np.argmax(window_data))
                peak_idx = idx_start + local_peak_idx
                score_matrix[slot_idx, m_type] = rises[m_type]
                peak_indices[slot_idx, m_type] = peak_idx

        # Assign all molecule types to all slots jointly. Greedy per-slot picks
        # can consume a strong channel tail early and cascade errors afterward.
        rows, cols = linear_sum_assignment(-score_matrix)
        molecule_for_slot = np.empty(num_molecule_types, dtype=int)
        molecule_for_slot[rows] = cols
        mol_arrivals = [
            {
                'name': molecule_names[molecule_for_slot[slot_idx]],
                'peak_time': peak_indices[slot_idx, molecule_for_slot[slot_idx]] * dt_bin,
                'peak_value': float(smoothed_signals[
                    molecule_for_slot[slot_idx],
                    peak_indices[slot_idx, molecule_for_slot[slot_idx]]]),
            }
            for slot_idx in range(num_molecule_types)
        ]

        permutation = ''.join(m['name'] for m in mol_arrivals)

        decoded_symbols.append({
            'peak_time': symbol_start_t + t_peak_theory,
            'permutation': permutation,
            'molecule_peaks': mol_arrivals,
            'peak_value': max([m['peak_value'] for m in mol_arrivals]) if mol_arrivals else 0
        })

    return decoded_symbols
