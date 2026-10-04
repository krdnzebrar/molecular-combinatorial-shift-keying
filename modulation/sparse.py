"""
modulation/sparse.py
====================
Sparse pattern scheme with fixed threshold + molecule exclusion.

Each '1' bit carries a random sparse pattern (partial slot fill with
permutation of molecule types).  Decoded slot-by-slot with molecule
exclusion to prevent diffusion tail re-detection.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from config import DT, R0, RR, D, CUSTOM_DIR, SMOOTHING_WINDOW
from utils.signal import moving_average, adaptive_smoothing_window
from modulation.encoder import build_signal, save_signal_csv
from detection.sparse_decoder import decode_sparse_fixed
from detection.channel_threshold import expected_pulse_threshold
from modulation.alphabet import sample_sparse_pattern
from detection.slot_scoring import slot_scores


def generate_sparse_bit_sequence_transmission(
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
    Full sparse pattern pipeline: encode → transmit → decode → compare.

    Returns
    -------
    combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
    """
    if exp_path is None:
        exp_path = os.path.join(CUSTOM_DIR, "N1000")

    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    # The full sparse alphabet grows super-exponentially. Sample uniformly
    # from its members without constructing every pattern.
    pattern_count = sum(
        math.comb(num_molecule_types, active) ** 2 * math.factorial(active)
        for active in range(num_molecule_types + 1))
    print(f"Total sparse patterns available : {pattern_count}")
    print(f"Bit sequence length             : {len(bit_sequence)}")
    print(f"Number of '1's                  : {bit_sequence.count('1')}\n")

    # ── Encoding ─────────────────────────────────────────────────────────
    symbol_transmissions = []
    for bit_idx, bit in enumerate(bit_sequence):
        if bit == '1':
            pattern = sample_sparse_pattern(num_molecule_types)
            pattern_string = ''.join(molecule_names[m] if m is not None else '_' for m in pattern)
            symbol_transmissions.append({
                'bit_position': bit_idx,
                'pattern': pattern,
                'permutation_names': pattern_string,
            })
            print(f"Bit {bit_idx:3d}: Sending {pattern_string}")

    # ── Signal building ──────────────────────────────────────────────────
    combined_signal, time_axis, molecule_signals = build_signal(
        symbol_transmissions, bit_sequence, num_molecule_types,
        exp_path, num_experiments, normalization,
        delay_between_symbols, delay_between_molecules,
        slot_key='pattern',
    )

    # ── Pre-smooth for calibration ───────────────────────────────────────
    dt_bin = normalization * DT
    smoothed_signals = np.zeros_like(molecule_signals)
    smooth_window = adaptive_smoothing_window(
        delay_between_molecules, dt_bin, SMOOTHING_WINDOW)
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], smooth_window)

    # ── Calibrate from unlabeled single-pulse channel templates ──────────
    fixed_threshold = expected_pulse_threshold(
        exp_path, normalization, dt_bin, smooth_window)

    # ── Decoding ─────────────────────────────────────────────────────────
    print("=== DECODING WITH FIXED THRESHOLD + MOLECULE EXCLUSION ===")
    decoded_symbols = decode_sparse_fixed(
        molecule_signals, bit_sequence, dt_bin,
        num_molecule_types, molecule_names,
        delay_between_symbols, delay_between_molecules,
        fixed_threshold,
    )

    # ── Compare ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== COMPARISON WITH GROUND TRUTH ===")
    print("=" * 70)

    errors = 0
    for i, (sent, dec) in enumerate(zip(symbol_transmissions, decoded_symbols)):
        match = ("✓ CORRECT" if sent['permutation_names'] == dec['permutation']
                 else "✗ WRONG")
        print(f"Symbol {i:3d}: Sent={sent['permutation_names']}  "
              f"Decoded={dec['permutation']}  {match}")
        if sent['permutation_names'] != dec['permutation']:
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
                    linestyle='--', label=f'Threshold ({fixed_threshold:.3f})')
        plt.xlabel('Time (s)'); plt.ylabel('Smoothed Molecules')
        plt.title('Smoothed Molecule Signals + Fixed Threshold')
        plt.legend(fontsize=8); plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(time_axis, combined_signal, linewidth=1.5, color='black',
                 label='Combined Signal')
        for sent, dec in zip(symbol_transmissions, decoded_symbols):
            t_mark = sent['bit_position'] * delay_between_symbols
            y_mark = dec['peak_value']
            color = 'green' if sent['permutation_names'] == dec['permutation'] else 'red'
            plt.plot(t_mark, y_mark, 'o', markersize=10, color=color)
            plt.text(t_mark, y_mark, f"  {dec['permutation']}", fontsize=7,
                     verticalalignment='bottom', color=color)
        plt.xlabel('Time (s)'); plt.ylabel('Total Received Molecules')
        plt.title('Combined Signal  (green=correct, red=error)')
        plt.legend(); plt.grid(True)

        # Panel 4: present vs absent channel peak scatter
        plt.subplot(4, 1, 4)
        t_peak_theory = (R0 - RR)**2 / (6 * D)
        half_w = delay_between_molecules * 0.45
        all_present, all_absent = [], []
        for symbol in symbol_transmissions:
            bit_pos = symbol['bit_position']
            pattern = symbol['pattern']
            symbol_start = bit_pos * delay_between_symbols
            for slot_idx in range(num_molecule_types):
                sent_mol = pattern[slot_idx]
                if sent_mol is None:
                    continue
                slot_send_t = symbol_start + slot_idx * delay_between_molecules
                peak_center = slot_send_t + t_peak_theory
                rises = slot_scores(smoothed_signals, peak_center, half_w,
                                    dt_bin, delay_between_molecules)
                for m, peak in enumerate(rises):
                    if m == sent_mol:
                        all_present.append(float(peak))
                    else:
                        all_absent.append(float(peak))

        plt.scatter(range(len(all_present)), all_present, s=10, color='green',
                    alpha=0.6, label='Present channel peaks')
        plt.scatter(range(len(all_absent)), all_absent, s=4, color='gray',
                    alpha=0.3, label='Absent channel peaks')
        plt.axhline(fixed_threshold, color='red', linewidth=2, linestyle='--',
                    label=f'Fixed threshold ({fixed_threshold:.3f})')
        plt.xlabel('Sample index'); plt.ylabel('Smoothed Peak Value')
        plt.title('Present vs Absent Channel Peaks — Fixed Threshold Separation')
        plt.legend(fontsize=8); plt.grid(True)

        plt.tight_layout()
        plt.show()

    # ── Save CSV ─────────────────────────────────────────────────────────
    output_file = os.path.join(exp_path, "combined_bit_sequence_signal.csv")
    if save_signal:
        save_signal_csv(combined_signal, time_axis, molecule_signals,
                        molecule_names, output_file)

    return combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
