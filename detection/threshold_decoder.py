"""
detection/threshold_decoder.py
==============================
Threshold-based decoders for the combination scheme.

Two variants:
  1. Adaptive: per-symbol threshold from the largest gap in peak values.
  2. Fixed: global threshold calibrated from ground-truth labels.
"""

import numpy as np

from config import SMOOTHING_WINDOW
from utils.signal import moving_average


# ─────────────────────────────────────────────────────────────────────────────
# Fixed threshold calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_fixed_threshold(smoothed_signals, symbol_transmissions, dt_bin,
                               num_molecule_types, delay_between_symbols):
    """
    Calibrate a single fixed threshold from all transmitted symbols.

    For each molecule channel in each symbol window, records whether it
    was present (sent) or absent (not sent), then computes:

        threshold = absent_mean + 0.30 × (present_mean − absent_mean)

    Placing it closer to absent avoids false positives from ISI tails.
    """
    present_peaks = []
    absent_peaks = []

    for symbol in symbol_transmissions:
        bit_pos = symbol['bit_position']
        sent_set = set(symbol['combo'])
        symbol_start = bit_pos * delay_between_symbols
        search_end = symbol_start + delay_between_symbols

        s = max(int(symbol_start / dt_bin), 0)
        e = min(int(search_end / dt_bin), smoothed_signals.shape[1] - 1)

        for m in range(num_molecule_types):
            peak = np.max(smoothed_signals[m, s:e])
            if m in sent_set:
                present_peaks.append(peak)
            else:
                absent_peaks.append(peak)

    mean_present = np.mean(present_peaks) if present_peaks else 1.0
    mean_absent = np.mean(absent_peaks) if absent_peaks else 0.0
    threshold = mean_absent + 0.30 * (mean_present - mean_absent)

    print(f"\n── Fixed Threshold Calibration ──────────────────────────────")
    print(f"   Samples  : {len(present_peaks)} present, {len(absent_peaks)} absent")
    print(f"   Mean present peak : {mean_present:.4f}")
    print(f"   Mean absent  peak : {mean_absent:.4f}")
    print(f"   Fixed threshold   : {threshold:.4f}  (= absent + 30% × gap)")
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
    Adaptive per-symbol threshold decoder for combinations.

    For each symbol:
      1. Get smoothed peak of every molecule channel in the symbol window.
      2. Sort peak values, find the largest gap.
      3. Set threshold at the midpoint of the gap.
      4. Channels above threshold → detected as present.
    """
    smoothed_signals = np.zeros_like(molecule_signals)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], SMOOTHING_WINDOW)

    combo_lookup = {c: s for c, s in zip(all_combos, combo_strings)}
    decoded_symbols = []

    for bit_idx, bit in enumerate(bit_sequence):
        if bit != '1':
            continue

        symbol_start_t = bit_idx * delay_between_symbols
        search_end_t = symbol_start_t + delay_between_symbols
        s = max(int(symbol_start_t / dt_bin), 0)
        e = min(int(search_end_t / dt_bin), smoothed_signals.shape[1] - 1)

        peak_vals = np.array([
            np.max(smoothed_signals[m, s:e]) for m in range(num_molecule_types)
        ])

        # Adaptive threshold: midpoint of the largest gap
        sorted_vals = np.sort(peak_vals)
        gaps = np.diff(sorted_vals)
        split_idx = np.argmax(gaps)
        threshold = (sorted_vals[split_idx] + sorted_vals[split_idx + 1]) / 2.0

        detected = [m for m in range(num_molecule_types) if peak_vals[m] > threshold]
        detected_tuple = tuple(sorted(detected))
        detected_str = ''.join(molecule_names[m] for m in detected_tuple)

        decoded_combo = combo_lookup.get(detected_tuple, detected_str or '?')

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
                               fixed_threshold):
    """
    Fixed threshold decoder for combinations.

    Same logic as adaptive, but uses a single pre-calibrated threshold.

    Returns
    -------
    decoded_symbols : list of dict
    smoothed_signals : np.ndarray
        The smoothed signal matrix (for plotting).
    """
    smoothed_signals = np.zeros_like(molecule_signals)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], SMOOTHING_WINDOW)

    combo_lookup = {c: s for c, s in zip(all_combos, combo_strings)}
    decoded_symbols = []

    for bit_idx, bit in enumerate(bit_sequence):
        if bit != '1':
            continue

        symbol_start_t = bit_idx * delay_between_symbols
        search_end_t = symbol_start_t + delay_between_symbols
        s = max(int(symbol_start_t / dt_bin), 0)
        e = min(int(search_end_t / dt_bin), smoothed_signals.shape[1] - 1)

        peak_vals = np.array([
            np.max(smoothed_signals[m, s:e]) for m in range(num_molecule_types)
        ])

        detected = [m for m in range(num_molecule_types)
                     if peak_vals[m] > fixed_threshold]
        detected_tuple = tuple(sorted(detected))
        detected_str = ''.join(molecule_names[m] for m in detected_tuple)
        decoded_combo = combo_lookup.get(detected_tuple, detected_str or '?')

        decoded_symbols.append({
            'bit_position': bit_idx,
            'decoded_combo': decoded_combo,
            'peak_vals': peak_vals,
            'threshold': fixed_threshold,
            'peak_value': float(np.max(peak_vals)),
        })

    return decoded_symbols, smoothed_signals
