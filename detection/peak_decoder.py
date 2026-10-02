"""
detection/peak_decoder.py
=========================
Smoothed-peak arrival-order decoder for the permutation scheme.

Detects the transmitted permutation by finding the peak time of each
molecule channel's smoothed signal within the symbol window, then
sorting by arrival time.
"""

import numpy as np

from config import R0, RR, D, SMOOTHING_WINDOW
from utils.signal import moving_average


def robust_decode(molecule_signals, bit_sequence, dt_bin, num_molecule_types,
                  molecule_names, delay_between_symbols, delay_between_molecules):
    """
    Decode permutation symbols by smoothed peak detection.

    For each '1' bit:
      1. Smooth every molecule channel with moving average (window=40).
      2. Find the peak of each channel within the symbol window.
      3. Sort channels by peak arrival time → decoded permutation.

    Returns
    -------
    decoded_symbols : list of dict
        Each has keys: 'permutation', 'peak_value', 'molecule_peaks', 'peak_time'.
    """
    t_peak_theory = (R0 - RR) ** 2 / (6 * D)

    smoothed_signals = np.zeros_like(molecule_signals)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], SMOOTHING_WINDOW)

    decoded_symbols = []

    for bit_idx, bit in enumerate(bit_sequence):
        if bit != '1':
            continue

        symbol_start_t = bit_idx * delay_between_symbols
        search_end_t = symbol_start_t + delay_between_symbols

        idx_start = int(symbol_start_t / dt_bin)
        idx_end = int(search_end_t / dt_bin)

        mol_arrivals = []
        for m_type in range(num_molecule_types):
            window_data = smoothed_signals[m_type, idx_start:idx_end]
            if len(window_data) > 0:
                local_peak_idx = np.argmax(window_data)
                peak_time = (idx_start + local_peak_idx) * dt_bin
                peak_val = window_data[local_peak_idx]

                mol_arrivals.append({
                    'name': molecule_names[m_type],
                    'peak_time': peak_time,
                    'peak_value': peak_val
                })

        # Sort by arrival time → decoded permutation order
        mol_arrivals.sort(key=lambda x: x['peak_time'])
        permutation = ''.join([m['name'] for m in mol_arrivals])

        decoded_symbols.append({
            'peak_time': symbol_start_t + t_peak_theory,
            'permutation': permutation,
            'molecule_peaks': mol_arrivals,
            'peak_value': max([m['peak_value'] for m in mol_arrivals]) if mol_arrivals else 0
        })

    return decoded_symbols
