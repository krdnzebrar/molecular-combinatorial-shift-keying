"""
detection/threshold_decoder.py
==============================
Threshold-based decoders for the combination scheme.

Two variants:
  1. Adaptive: per-symbol threshold from peak rises over the pre-symbol level.
  2. Fixed: global unlabeled threshold calibrated from observed peak rises.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import SMOOTHING_WINDOW, T_PEAK_THEORY
from utils.signal import moving_average, adaptive_smoothing_window


def _combination_slot_scores(smoothed_signals, bit_idx, num_molecule_types,
                              dt_bin, delay_between_symbols,
                              delay_between_molecules):
    """Rise scores indexed by MoCSK release slot and molecule type."""
    half_window = max(dt_bin, 0.45 * delay_between_molecules)
    start_t = bit_idx * delay_between_symbols
    scores = np.zeros((num_molecule_types, num_molecule_types), dtype=float)
    for slot_idx in range(num_molecule_types):
        center = start_t + slot_idx * delay_between_molecules + T_PEAK_THEORY
        s = max(0, int((center - half_window) / dt_bin))
        e = min(smoothed_signals.shape[1], int(np.ceil((center + half_window) / dt_bin)) + 1)
        if e <= s:
            continue
        baseline_idx = max(0, s - max(1, int(half_window / dt_bin)))
        peaks = np.max(smoothed_signals[:, s:e], axis=1)
        scores[slot_idx] = np.maximum(peaks - smoothed_signals[:, baseline_idx], 0.0)
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
# Fixed threshold calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_fixed_threshold(smoothed_signals, symbol_transmissions, dt_bin,
                               num_molecule_types, delay_between_symbols,
                               delay_between_molecules=0.4):
    """
    Calibrate a single fixed threshold from all transmitted symbols.

    Uses an unlabeled two-cluster split over local peak rises. The sent
    combination labels are deliberately not consulted.
    """
    peak_rises = []

    for symbol in symbol_transmissions:
        bit_pos = symbol['bit_position']
        scores = _combination_slot_scores(
            smoothed_signals, bit_pos, num_molecule_types, dt_bin,
            delay_between_symbols, delay_between_molecules)
        peak_rises.extend(scores.ravel().tolist())

    if not peak_rises:
        return 0.0
    sorted_rises = np.sort(np.asarray(peak_rises, dtype=float))
    gaps = np.diff(sorted_rises)
    if not len(gaps) or np.max(gaps) <= 0:
        threshold = 0.5 * float(np.max(sorted_rises))
    else:
        split_idx = int(np.argmax(gaps))
        threshold = 0.5 * (sorted_rises[split_idx] + sorted_rises[split_idx + 1])

    print(f"\n── Unlabeled Threshold Calibration ──────────────────────────")
    print(f"   Peak-rise samples : {len(peak_rises)}")
    print(f"   Fixed threshold   : {threshold:.4f} (largest-gap split)")
    print(f"─────────────────────────────────────────────────────────────\n")

    return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_combinations_adaptive(molecule_signals, bit_sequence, dt_bin,
                                  num_molecule_types, molecule_names,
                                  delay_between_symbols, delay_between_molecules,
                                  all_combos, combo_strings):
    """
    Blind per-symbol ratio threshold decoder for combinations.

    For each symbol:
      1. Get smoothed peak of every molecule channel in the symbol window.
      2. Subtract the pre-symbol level from each peak.
      3. Keep channels whose rise is at least half the largest rise.
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

        # A ratio threshold can retain every molecule when all K are present;
        # the largest-gap split always forced a nonempty absent cluster.
        threshold = 0.5 * float(np.max(slot_scores)) if np.any(slot_scores > 0) else np.inf
        peak_vals = np.max(slot_scores, axis=0)
        decoded_combo = _decode_mocsk_prefix(slot_scores, threshold, molecule_names)

        decoded_symbols.append({
            'bit_position': bit_idx,
            'decoded_combo': decoded_combo,
            'peak_vals': peak_vals,
            'threshold': threshold,
            'peak_value': float(np.max(peak_vals)),
        })

    return decoded_symbols


# ─────────────────────────────────────────────────────────────────────────────
# Fixed-threshold decoder
# ─────────────────────────────────────────────────────────────────────────────

def decode_combinations_fixed(molecule_signals, bit_sequence, dt_bin,
                               num_molecule_types, molecule_names,
                               delay_between_symbols,
                               all_combos, combo_strings,
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
