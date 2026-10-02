"""
modulation/permutation.py
=========================
Permutation-based MoCSK scheme.

Each '1' bit carries a random full permutation of all K molecule types.
Decoded by smoothed-peak arrival order detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import DT, CUSTOM_DIR
from utils.signal import moving_average
from modulation.alphabet import get_all_permutations
from modulation.encoder import build_signal, save_signal_csv
from detection.peak_decoder import robust_decode


def generate_bit_sequence_transmission(
    bit_sequence,
    num_molecule_types=5,
    exp_path=None,
    num_experiments=100,
    normalization=10,
    delay_between_symbols=2.5,
    delay_between_molecules=0.4,
    plot=True,
):
    """
    Full permutation scheme pipeline: encode → transmit → decode → compare.

    Parameters
    ----------
    bit_sequence : str
        Binary string, e.g. '10110'.
    num_molecule_types : int
        Alphabet size K (default 5 → A,B,C,D,E).
    exp_path : str
        Directory containing Exp_xxx.csv templates.
    num_experiments : int
        Number of available template files.
    normalization : int
        Bin-averaging factor for templates.
    delay_between_symbols : float
        Symbol period Ts (seconds).
    delay_between_molecules : float
        Inter-molecule delay within a symbol (seconds).
    plot : bool
        If True, show 4-panel diagnostic plot.

    Returns
    -------
    combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
    """
    if exp_path is None:
        exp_path = os.path.join(CUSTOM_DIR, "N1000")

    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    all_perms, perm_strings = get_all_permutations(num_molecule_types)

    print(f"Total permutations available: {len(all_perms)}")
    print(f"Bit sequence length: {len(bit_sequence)}")
    print(f"Number of '1's (symbols to send): {bit_sequence.count('1')}\n")

    # ── Encoding: assign random permutation to each '1' ──────────────────
    symbol_transmissions = []
    for bit_idx, bit in enumerate(bit_sequence):
        if bit == '1':
            perm_idx = np.random.randint(0, len(all_perms))
            selected_perm = all_perms[perm_idx]
            perm_names = ''.join(molecule_names[i] for i in selected_perm)

            symbol_transmissions.append({
                'bit_position': bit_idx,
                'permutation_indices': selected_perm,
                'permutation_names': perm_names,
            })
            print(f"Bit {bit_idx}: Sending permutation {perm_names} "
                  f"(indices: {selected_perm})")

    # ── Signal building ──────────────────────────────────────────────────
    combined_signal, time_axis, molecule_signals = build_signal(
        symbol_transmissions, bit_sequence, num_molecule_types,
        exp_path, num_experiments, normalization,
        delay_between_symbols, delay_between_molecules,
        slot_key='permutation_indices',
    )

    # ── Decoding ─────────────────────────────────────────────────────────
    dt_bin = normalization * DT
    print("\n=== DECODING WITH ROBUST SMOOTHED PEAK DETECTION ===")
    decoded_symbols = robust_decode(
        molecule_signals, bit_sequence, dt_bin,
        num_molecule_types, molecule_names,
        delay_between_symbols, delay_between_molecules,
    )

    print(f"\n=== Decoded {len(decoded_symbols)} symbols ===\n")
    for i, symbol in enumerate(decoded_symbols):
        print(f"Symbol {i}:")
        print(f"  Decoded permutation: {symbol['permutation']}")
        for mol in symbol['molecule_peaks']:
            print(f"    {mol['name']} at t={mol['peak_time']:.3f}s "
                  f"(value={mol['peak_value']:.2f})")

    # ── Compare with ground truth ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== COMPARISON WITH GROUND TRUTH ===")
    print("=" * 70)

    for i, (sent, decoded) in enumerate(zip(symbol_transmissions, decoded_symbols)):
        match = ("✓ CORRECT" if sent['permutation_names'] == decoded['permutation']
                 else "✗ WRONG")
        print(f"Symbol {i}: Sent={sent['permutation_names']}, "
              f"Decoded={decoded['permutation']} {match}")

    total = max(len(symbol_transmissions), 1)
    accuracy = sum(1 for s, d in zip(symbol_transmissions, decoded_symbols)
                   if s['permutation_names'] == d['permutation']) / total * 100
    print(f"\nDecoding Accuracy: {accuracy:.1f}%")

    # ── Plotting ─────────────────────────────────────────────────────────
    if plot:
        colors = ['red', 'blue', 'green', 'orange', 'pink']
        fig = plt.figure(figsize=(16, 14))

        plt.subplot(4, 1, 1)
        bit_array = np.array([int(b) for b in bit_sequence])
        bit_times = np.arange(len(bit_sequence)) * delay_between_symbols
        plt.step(bit_times, bit_array, where='post', linewidth=2, color='blue')
        plt.xlabel('Time (s)'); plt.ylabel('Bit Value')
        plt.title('Input Bit Sequence'); plt.grid(True); plt.ylim([-0.1, 1.1])

        plt.subplot(4, 1, 2)
        for m in range(num_molecule_types):
            plt.plot(time_axis, molecule_signals[m],
                     label=f'Molecule {molecule_names[m]}',
                     linewidth=1, color=colors[m % len(colors)], alpha=0.7)
        plt.xlabel('Time (s)'); plt.ylabel('Received Molecules')
        plt.title('Individual Molecule Type Signals'); plt.legend(); plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(time_axis, combined_signal, linewidth=1.5, color='black',
                 label='Combined Signal')
        for sym in decoded_symbols:
            plt.plot(sym['peak_time'], sym['peak_value'], 'ro', markersize=10)
            plt.text(sym['peak_time'], sym['peak_value'],
                     f"  {sym['permutation']}", fontsize=10, verticalalignment='bottom')
        plt.xlabel('Time (s)'); plt.ylabel('Total Received Molecules')
        plt.title('Combined Signal with Decoded Permutations')
        plt.legend(); plt.grid(True)

        plt.subplot(4, 1, 4)
        smoothed = moving_average(combined_signal, 10)
        plt.plot(time_axis, smoothed, linewidth=2, color='purple',
                 label='Smoothed Signal')
        for sym in decoded_symbols:
            plt.plot(sym['peak_time'], sym['peak_value'], 'go', markersize=8)
        plt.xlabel('Time (s)'); plt.ylabel('Smoothed Signal')
        plt.title('Smoothed Combined Signal (Moving Average)')
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.show()

    # ── Save CSV ─────────────────────────────────────────────────────────
    output_file = os.path.join(exp_path, "combined_bit_sequence_signal.csv")
    save_signal_csv(combined_signal, time_axis, molecule_signals,
                    molecule_names, output_file)

    return combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
