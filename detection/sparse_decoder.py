"""
detection/sparse_decoder.py
===========================
Fixed threshold + molecule exclusion decoder for sparse patterns.

Key improvement over basic threshold: each molecule can appear AT MOST
ONCE per symbol.  Once a molecule wins a slot, it's excluded from all
remaining slots — preventing diffusion tails from being re-detected.
"""

import numpy as np

from config import R0, RR, D, SMOOTHING_WINDOW
from utils.signal import moving_average


def calibrate_sparse_threshold(smoothed_signals, symbol_transmissions, dt_bin,
                                num_molecule_types, delay_between_symbols,
                                delay_between_molecules):
    """
    Calibrate threshold using tight SLOT-based windows (not whole-symbol windows).

    For each filled (symbol, slot), looks in a tight window centred on the
    exact expected peak time.  Labels the sent molecule as 'present' and
    all others as 'absent'.

    threshold = absent_mean + 0.30 × (present_mean − absent_mean)
    """
    t_peak_theory = (R0 - RR) ** 2 / (6 * D)
    half_w = delay_between_molecules * 0.8  # tight: one slot only

    present_peaks = []
    absent_peaks = []

    for symbol in symbol_transmissions:
        bit_pos = symbol['bit_position']
        pattern = symbol['pattern']
        symbol_start = bit_pos * delay_between_symbols

        for slot_idx in range(num_molecule_types):
            sent_mol = pattern[slot_idx]
            if sent_mol is None:
                continue  # skip empty slots

            slot_send_t = symbol_start + slot_idx * delay_between_molecules
            peak_center = slot_send_t + t_peak_theory

            s = max(int((peak_center - half_w) / dt_bin), 0)
            e = min(int((peak_center + half_w) / dt_bin),
                    smoothed_signals.shape[1] - 1)
            if s >= e:
                continue

            for m in range(num_molecule_types):
                peak = float(np.max(smoothed_signals[m, s:e]))
                if m == sent_mol:
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
    print(f"   Fixed threshold   : {threshold:.4f}  (absent + 30% × gap)")
    print(f"─────────────────────────────────────────────────────────────\n")

    return threshold, mean_present, mean_absent


def decode_sparse_fixed(molecule_signals, bit_sequence, dt_bin,
                         num_molecule_types, molecule_names,
                         delay_between_symbols, delay_between_molecules,
                         all_patterns, pattern_strings,
                         fixed_threshold):
    """
    Slot-by-slot decoder with molecule exclusion.

    Per slot:
      1. Tight window around expected peak time of this slot.
      2. Among molecules NOT yet used in this symbol, pick the highest
         smoothed peak that exceeds fixed_threshold.
      3. If none exceeds threshold → slot stays empty (_).
    """
    t_peak_theory = (R0 - RR) ** 2 / (6 * D)
    half_w = delay_between_molecules * 0.8  # same as calibration

    smoothed_signals = np.zeros_like(molecule_signals)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], SMOOTHING_WINDOW)

    pattern_lookup = {p: s for p, s in zip(all_patterns, pattern_strings)}
    decoded_symbols = []

    for bit_idx, bit in enumerate(bit_sequence):
        if bit != '1':
            continue

        symbol_start_t = bit_idx * delay_between_symbols
        decoded_pattern = [None] * num_molecule_types
        used_molecules = set()  # ← exclusion set per symbol

        for slot_idx in range(num_molecule_types):
            slot_send_t = symbol_start_t + slot_idx * delay_between_molecules
            peak_center = slot_send_t + t_peak_theory

            s = max(int((peak_center - half_w) / dt_bin), 0)
            e = min(int((peak_center + half_w) / dt_bin),
                    smoothed_signals.shape[1] - 1)
            if s >= e:
                continue

            best_m, best_val = None, fixed_threshold  # threshold = floor

            for m in range(num_molecule_types):
                if m in used_molecules:
                    continue  # ← skip already-used molecules
                peak_val = float(np.max(smoothed_signals[m, s:e]))
                if peak_val > best_val:
                    best_val = peak_val
                    best_m = m

            if best_m is not None:
                decoded_pattern[slot_idx] = best_m
                used_molecules.add(best_m)  # ← lock out for rest of symbol

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
