"""
modulation/combination_fixed.py
================================
MoCSK ordered-subset scheme with an unlabeled global threshold.

Each active interval carries a uniformly sampled ordered subset, including
the empty symbol. The threshold comes from the expected channel pulse response.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from config import DT, CUSTOM_DIR, SMOOTHING_WINDOW
from utils.signal import moving_average, adaptive_smoothing_window
from modulation.alphabet import sample_ordered_subset
from modulation.encoder import build_signal, save_signal_csv
from detection.threshold_decoder import (
    decode_combinations_fixed,
)
from detection.channel_threshold import expected_pulse_threshold


def generate_combination_transmission(
    bit_sequence,
    num_molecule_types=5,
    exp_path=None,
    num_experiments=100,
    normalization=10,
    delay_between_symbols=2.5,
    delay_between_molecules=0.4,
    plot=True,
    save_signal=True,
):
    """
    Full combination (fixed threshold) pipeline.

    Returns
    -------
    combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
    """
    if exp_path is None:
        exp_path = os.path.join(CUSTOM_DIR, "N1000")

    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    print("MoCSK ordered-subset alphabet size: " + str(sum(
        math.factorial(num_molecule_types)
        // math.factorial(num_molecule_types - size)
        for size in range(num_molecule_types + 1)
    )))
    print(f"Bit sequence length          : {len(bit_sequence)}")
    print(f"Number of '1's               : {bit_sequence.count('1')}\n")

    # ── Encoding ─────────────────────────────────────────────────────────
    symbol_transmissions = []
    for bit_idx, bit in enumerate(bit_sequence):
        if bit == '1':
            selected_combo = sample_ordered_subset(num_molecule_types)
            combo_string = ''.join(molecule_names[m] for m in selected_combo)
            symbol_transmissions.append({
                'bit_position': bit_idx,
                'combo': selected_combo,
                'permutation_names': combo_string,
            })
            print(f"Bit {bit_idx:3d}: Sending {combo_string or 'EMPTY'}")

    # ── Signal building ──────────────────────────────────────────────────
    combined_signal, time_axis, molecule_signals = build_signal(
        symbol_transmissions, bit_sequence, num_molecule_types,
        exp_path, num_experiments, normalization,
        delay_between_symbols, delay_between_molecules,
        slot_key='combo',
    )

    # ── Pre-smooth for calibration ───────────────────────────────────────
    dt_bin = normalization * DT
    smoothed_signals = np.zeros_like(molecule_signals)
    smooth_window = adaptive_smoothing_window(
        delay_between_molecules, dt_bin, SMOOTHING_WINDOW)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], smooth_window)

    # ── Calibrate fixed threshold ────────────────────────────────────────
    fixed_threshold = expected_pulse_threshold(
        exp_path, normalization, dt_bin, smooth_window)

    # ── Decoding ─────────────────────────────────────────────────────────
    print("=== DECODING WITH FIXED THRESHOLD ===")
    decoded_symbols, _ = decode_combinations_fixed(
        molecule_signals, bit_sequence, dt_bin,
        num_molecule_types, molecule_names,
        delay_between_symbols,
        fixed_threshold,
        delay_between_molecules=delay_between_molecules,
    )

    print(f"\n=== Decoded {len(decoded_symbols)} symbols ===\n")
    for i, sym in enumerate(decoded_symbols):
        peaks_str = '  '.join(
            f"{molecule_names[m]}:{sym['peak_vals'][m]:.2f}"
            for m in range(num_molecule_types)
        )
        print(f"Symbol {i:3d}:  Decoded={sym['decoded_combo']:6s}  "
              f"thr={fixed_threshold:.3f}   [{peaks_str}]")

    # ── Compare ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== COMPARISON WITH GROUND TRUTH ===")
    print("=" * 70)

    errors = 0
    for i, (sent, dec) in enumerate(zip(symbol_transmissions, decoded_symbols)):
        match = ("✓ CORRECT" if sent['permutation_names'] == dec['decoded_combo']
                 else "✗ WRONG")
        print(f"Symbol {i:3d}: Sent={sent['permutation_names']:6s}  "
              f"Decoded={dec['decoded_combo']:6s}  {match}")
        if sent['permutation_names'] != dec['decoded_combo']:
            errors += 1

    total = max(len(symbol_transmissions), 1)
    ser = errors / total
    accuracy = (1 - ser) * 100
    print(f"\nDecoding Accuracy : {accuracy:.1f}%")
    print(f"Fixed Threshold   : {fixed_threshold:.4f}")
    print(f"SER               : {ser:.4f}  ({errors}/{total} errors)")

    # ── Plotting ─────────────────────────────────────────────────────────
    if plot:
        colors = ['red', 'blue', 'green', 'orange', 'magenta']
        fig = plt.figure(figsize=(16, 16))

        plt.subplot(4, 1, 1)
        bit_array = np.array([int(b) for b in bit_sequence])
        bit_times = np.arange(len(bit_sequence)) * delay_between_symbols
        plt.step(bit_times, bit_array, where='post', linewidth=2, color='blue')
        plt.xlabel('Time (s)'); plt.ylabel('Bit Value')
        plt.title('Input Bit Sequence'); plt.grid(True); plt.ylim([-0.1, 1.1])

        plt.subplot(4, 1, 2)
        for m in range(num_molecule_types):
            plt.plot(time_axis, smoothed_signals[m],
                     label=f'Molecule {molecule_names[m]}',
                     linewidth=1.2, color=colors[m % len(colors)], alpha=0.85)
        plt.axhline(fixed_threshold, color='black', linewidth=1.5,
                    linestyle='--', label=f'Fixed threshold ({fixed_threshold:.3f})')
        plt.xlabel('Time (s)'); plt.ylabel('Smoothed Molecules')
        plt.title('Smoothed Individual Molecule Signals + Fixed Threshold')
        plt.legend(fontsize=8); plt.grid(True)

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
        sym_indices = np.arange(len(decoded_symbols))
        for m in range(num_molecule_types):
            peaks = [sym['peak_vals'][m] for sym in decoded_symbols]
            plt.scatter(sym_indices, peaks, color=colors[m % len(colors)],
                        s=25, alpha=0.7, label=f'Mol {molecule_names[m]}')
        plt.axhline(fixed_threshold, color='black', linewidth=2,
                    linestyle='--', label=f'Fixed threshold ({fixed_threshold:.3f})')
        plt.xlabel('Symbol index'); plt.ylabel('Smoothed Peak Value')
        plt.title('Per-Symbol Smoothed Peak Values vs Fixed Threshold')
        plt.legend(fontsize=8); plt.grid(True)

        plt.tight_layout()
        plt.show()

    # ── Save CSV ─────────────────────────────────────────────────────────
    output_file = os.path.join(exp_path, "combined_bit_sequence_signal.csv")
    if save_signal:
        save_signal_csv(combined_signal, time_axis, molecule_signals,
                        molecule_names, output_file)

    return combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
