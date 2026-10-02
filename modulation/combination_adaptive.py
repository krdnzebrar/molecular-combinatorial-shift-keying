"""
modulation/combination_adaptive.py
===================================
Combination scheme with adaptive per-symbol threshold.

Each '1' bit carries a random non-empty subset of the molecule alphabet.
Decoded by finding the largest gap in per-channel smoothed peak values.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import DT, CUSTOM_DIR
from utils.signal import moving_average
from modulation.alphabet import get_all_combinations
from modulation.encoder import build_signal, save_signal_csv
from detection.threshold_decoder import decode_combinations_adaptive


def generate_combination_transmission(
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
    Full combination (adaptive threshold) pipeline.

    Returns
    -------
    combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
    """
    if exp_path is None:
        exp_path = os.path.join(CUSTOM_DIR, "N1000")

    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    all_combos, combo_strings = get_all_combinations(num_molecule_types)

    print(f"Total combinations available: {len(all_combos)}")
    print(f"Bit sequence length: {len(bit_sequence)}")
    print(f"Number of '1's (symbols to send): {bit_sequence.count('1')}\n")

    # ── Encoding ─────────────────────────────────────────────────────────
    symbol_transmissions = []
    for bit_idx, bit in enumerate(bit_sequence):
        if bit == '1':
            idx = np.random.randint(0, len(all_combos))
            symbol_transmissions.append({
                'bit_position': bit_idx,
                'combo': all_combos[idx],
                'permutation_names': combo_strings[idx],
            })
            print(f"Bit {bit_idx}: Sending {combo_strings[idx]}")

    # ── Signal building ──────────────────────────────────────────────────
    combined_signal, time_axis, molecule_signals = build_signal(
        symbol_transmissions, bit_sequence, num_molecule_types,
        exp_path, num_experiments, normalization,
        delay_between_symbols, delay_between_molecules,
        slot_key='combo',
    )

    # ── Decoding ─────────────────────────────────────────────────────────
    dt_bin = normalization * DT
    print("\n=== DECODING WITH ADAPTIVE THRESHOLD ===")
    decoded_symbols = decode_combinations_adaptive(
        molecule_signals, bit_sequence, dt_bin,
        num_molecule_types, molecule_names,
        delay_between_symbols, delay_between_molecules,
        all_combos, combo_strings,
    )

    print(f"\n=== Decoded {len(decoded_symbols)} symbols ===\n")
    for i, sym in enumerate(decoded_symbols):
        print(f"Symbol {i}:  Decoded={sym['decoded_combo']}  "
              f"threshold={sym['threshold']:.3f}  "
              f"peaks={np.round(sym['peak_vals'], 2)}")

    # ── Compare ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== COMPARISON WITH GROUND TRUTH ===")
    print("=" * 70)

    for i, (sent, dec) in enumerate(zip(symbol_transmissions, decoded_symbols)):
        match = ("✓ CORRECT" if sent['permutation_names'] == dec['decoded_combo']
                 else "✗ WRONG")
        print(f"Symbol {i:3d}: Sent={sent['permutation_names']:6s}  "
              f"Decoded={dec['decoded_combo']:6s}  {match}")

    errors = sum(1 for s, d in zip(symbol_transmissions, decoded_symbols)
                 if s['permutation_names'] != d['decoded_combo'])
    total = max(len(symbol_transmissions), 1)
    accuracy = (1 - errors / total) * 100
    ser = errors / total
    print(f"\nDecoding Accuracy : {accuracy:.1f}%")
    print(f"SER               : {ser:.4f}  ({errors}/{total} errors)")

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
        for sent, dec in zip(symbol_transmissions, decoded_symbols):
            t_mark = sent['bit_position'] * delay_between_symbols
            y_mark = dec['peak_value']
            color = 'green' if sent['permutation_names'] == dec['decoded_combo'] else 'red'
            plt.plot(t_mark, y_mark, 'o', markersize=10, color=color)
            plt.text(t_mark, y_mark, f"  {dec['decoded_combo']}", fontsize=8,
                     verticalalignment='bottom', color=color)
        plt.xlabel('Time (s)'); plt.ylabel('Total Received Molecules')
        plt.title('Combined Signal with Decoded Combinations  (green=correct, red=error)')
        plt.legend(); plt.grid(True)

        plt.subplot(4, 1, 4)
        smoothed = moving_average(combined_signal, 10)
        plt.plot(time_axis, smoothed, linewidth=2, color='purple',
                 label='Smoothed Signal')
        for dec in decoded_symbols:
            t_mark = dec['bit_position'] * delay_between_symbols
            plt.plot(t_mark, dec['peak_value'], 'go', markersize=8)
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
