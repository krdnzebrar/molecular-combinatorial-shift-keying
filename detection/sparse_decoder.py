"""
detection/sparse_decoder.py
===========================
Fixed threshold + joint molecule/empty-slot assignment decoder for sparse patterns.
"""

import numpy as np

from config import R0, RR, D, SMOOTHING_WINDOW
from utils.signal import moving_average, adaptive_smoothing_window


def calibrate_sparse_threshold(smoothed_signals, symbol_transmissions, dt_bin,
                                num_molecule_types, delay_between_symbols,
                                delay_between_molecules):
    """
    Calibrate threshold using tight SLOT-based windows (not whole-symbol windows).

    Builds an unlabeled score distribution over every candidate slot and
    molecule channel, then splits its largest gap. Empty slots are included.
    """
    t_peak_theory = (R0 - RR) ** 2 / (6 * D)
    half_w = delay_between_molecules * 0.45

    slot_scores = []

    for symbol in symbol_transmissions:
        bit_pos = symbol['bit_position']
        symbol_start = bit_pos * delay_between_symbols

        for slot_idx in range(num_molecule_types):
            slot_send_t = symbol_start + slot_idx * delay_between_molecules
            peak_center = slot_send_t + t_peak_theory

            s = max(int((peak_center - half_w) / dt_bin), 0)
            e = min(int((peak_center + half_w) / dt_bin),
                    smoothed_signals.shape[1] - 1)
            if s >= e:
                continue

            baseline_idx = max(0, s - max(1, int(half_w / dt_bin)))
            for m in range(num_molecule_types):
                peak = float(np.max(smoothed_signals[m, s:e]))
                slot_scores.append(max(peak - float(smoothed_signals[m, baseline_idx]), 0.0))

    sorted_scores = np.sort(np.asarray(slot_scores, dtype=float))
    gaps = np.diff(sorted_scores)
    if not len(gaps) or np.max(gaps) <= 0:
        threshold = 0.5 * float(np.max(sorted_scores)) if len(sorted_scores) else 0.0
    else:
        split_idx = int(np.argmax(gaps))
        threshold = 0.5 * (sorted_scores[split_idx] + sorted_scores[split_idx + 1])

    print(f"\n── Fixed Threshold Calibration ──────────────────────────────")
    print(f"   Unlabeled slot scores : {len(slot_scores)}")
    print(f"   Fixed threshold       : {threshold:.4f} (largest-gap split)")
    print(f"─────────────────────────────────────────────────────────────\n")

    return threshold, float("nan"), float("nan")


def decode_sparse_fixed(molecule_signals, bit_sequence, dt_bin,
                         num_molecule_types, molecule_names,
                         delay_between_symbols, delay_between_molecules,
                         all_patterns, pattern_strings,
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

    pattern_lookup = ({p: s for p, s in zip(all_patterns, pattern_strings)}
                      if all_patterns is not None and pattern_strings is not None else {})
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

            s = max(int((peak_center - half_w) / dt_bin), 0)
            e = min(int((peak_center + half_w) / dt_bin),
                    smoothed_signals.shape[1] - 1)
            if s >= e:
                score_matrix[slot_idx, num_molecule_types:] = 0.0
                continue
            baseline_idx = max(0, s - max(1, int(half_w / dt_bin)))
            for m in range(num_molecule_types):
                peak_val = float(np.max(smoothed_signals[m, s:e]))
                baseline = float(smoothed_signals[m, baseline_idx])
                score_matrix[slot_idx, m] = peak_val - baseline - fixed_threshold
            score_matrix[slot_idx, num_molecule_types:] = 0.0

        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-score_matrix)
        for slot_idx, col in zip(rows, cols):
            if col < num_molecule_types and score_matrix[slot_idx, col] > 0:
                decoded_pattern[slot_idx] = int(col)

        decoded_tuple = tuple(decoded_pattern)
        decoded_string = ''.join(
            molecule_names[m] if m is not None else '_'
            for m in decoded_pattern
        )
        decoded_name = pattern_lookup.get(decoded_tuple, decoded_string)

        sym_s = max(int(symbol_start_t / dt_bin), 0)
        sym_e = min(int((symbol_start_t + delay_between_symbols) / dt_bin),
                    smoothed_signals.shape[1] - 1)

        decoded_symbols.append({
            'bit_position': bit_idx,
            'permutation': decoded_name,
            'peak_value': float(np.max(smoothed_signals[:, sym_s:sym_e]))
        })

    return decoded_symbols
