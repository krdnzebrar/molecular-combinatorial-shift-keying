"""
modulation/encoder.py
=====================
Common signal construction logic shared by all modulation schemes.

The signal-building loop was previously copy-pasted in all 4 scheme
functions. Now it's ONE function that handles permutations, combinations,
and sparse patterns (with None entries).
"""

import os
import csv

import numpy as np

from config import DT, T
from utils.signal import load_experiment_data


def build_signal(
    symbol_transmissions,
    bit_sequence,
    num_molecule_types,
    exp_path,
    num_experiments,
    normalization,
    delay_between_symbols,
    delay_between_molecules,
    slot_key='permutation_indices',
):
    """
    Build time-domain received signals by overlaying experiment CSV templates.

    Common to all schemes — the only difference is which key in
    ``symbol_transmissions`` holds the molecule indices for each slot.
    Handles None entries (sparse patterns) by skipping them.

    Parameters
    ----------
    symbol_transmissions : list of dict
        Each dict has ``'bit_position'`` and ``slot_key``.
    slot_key : str
        Key that holds the per-slot molecule indices in each dict.

    Returns
    -------
    combined_signal : np.ndarray
    time_axis : np.ndarray
    molecule_signals : np.ndarray, shape (num_molecule_types, num_bins)
    """
    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    dt_bin = normalization * DT

    # Calculate required time axis length
    if symbol_transmissions:
        last_start = symbol_transmissions[-1]['bit_position'] * delay_between_symbols
        last_mol_delay = (num_molecule_types - 1) * delay_between_molecules
        max_time = last_start + last_mol_delay + T
    else:
        max_time = len(bit_sequence) * delay_between_symbols

    num_bins = int(max_time / dt_bin)
    combined_signal = np.zeros(num_bins)
    molecule_signals = np.zeros((num_molecule_types, num_bins))
    time_axis = np.arange(num_bins) * dt_bin
    template_cache = {}

    print("\n=== Processing Transmissions ===")
    for sym_idx, symbol in enumerate(symbol_transmissions):
        bit_pos = symbol['bit_position']
        indices = symbol[slot_key]
        symbol_start = bit_pos * delay_between_symbols

        print(f"\nSymbol {sym_idx} at bit {bit_pos}: {symbol['permutation_names']}")

        for slot_idx, molecule_type in enumerate(indices):
            if molecule_type is None:
                continue  # sparse: skip empty slots

            exp_id = np.random.randint(0, num_experiments)
            exp_file = os.path.join(exp_path, f"Exp_{exp_id:03d}.csv")

            if not os.path.exists(exp_file):
                print(f"  Warning: {exp_file} not found, skipping")
                continue

            if exp_id not in template_cache:
                template_cache[exp_id] = load_experiment_data(exp_file, normalization)
            time_data, hits_data = template_cache[exp_id]
            total_delay = symbol_start + slot_idx * delay_between_molecules

            print(f"  Slot {slot_idx}: Molecule {molecule_names[molecule_type]} "
                  f"at t={total_delay:.2f}s  (Exp_{exp_id:03d})")

            for t_idx, hit_count in enumerate(hits_data):
                bin_idx = int((time_data[t_idx] + total_delay) / dt_bin)
                if bin_idx < num_bins:
                    combined_signal[bin_idx] += hit_count
                    molecule_signals[molecule_type, bin_idx] += hit_count

    return combined_signal, time_axis, molecule_signals


def save_signal_csv(combined_signal, time_axis, molecule_signals,
                    molecule_names, output_file):
    """Save combined + per-molecule signals to CSV."""
    num_molecule_types = len(molecule_names)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = (['Time_s', 'Total_Molecules'] +
                  [f'Molecule_{molecule_names[i]}' for i in range(num_molecule_types)])
        writer.writerow(header)
        for i in range(len(time_axis)):
            row = ([time_axis[i], combined_signal[i]] +
                   [molecule_signals[j, i] for j in range(num_molecule_types)])
            writer.writerow(row)
    print(f"\nCombined signal saved to: {output_file}")
