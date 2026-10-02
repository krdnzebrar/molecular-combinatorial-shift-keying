"""
modulation/sparse.py
====================
Sparse pattern scheme with fixed threshold + molecule exclusion.

Each '1' bit carries a random sparse pattern (partial slot fill with
permutation of molecule types).  Decoded slot-by-slot with molecule
exclusion to prevent diffusion tail re-detection.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from config import DT, R0, RR, D, CUSTOM_DIR, SMOOTHING_WINDOW
from utils.signal import moving_average
from modulation.alphabet import get_all_sparse_patterns
from modulation.encoder import build_signal, save_signal_csv
from detection.sparse_decoder import calibrate_sparse_threshold, decode_sparse_fixed


def generate_sparse_bit_sequence_transmission(
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
    Full sparse pattern pipeline: encode → transmit → decode → compare.

    Returns
    -------
    combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
    """
    if exp_path is None:
        exp_path = os.path.join(CUSTOM_DIR, "N1000")

    molecule_names = [chr(65 + i) for i in range(num_molecule_types)]
    all_patterns, pattern_strings = get_all_sparse_patterns(num_molecule_types)

    print(f"Total sparse patterns available : {len(all_patterns)}")
    print(f"Bit sequence length             : {len(bit_sequence)}")
    print(f"Number of '1's                  : {bit_sequence.count('1')}\n")

    # ── Encoding ─────────────────────────────────────────────────────────
    symbol_transmissions = []
    for bit_idx, bit in enumerate(bit_sequence):
        if bit == '1':
            idx = np.random.randint(0, len(all_patterns))
            symbol_transmissions.append({
                'bit_position': bit_idx,
                'pattern': all_patterns[idx],
                'permutation_names': pattern_strings[idx],
            })
            print(f"Bit {bit_idx:3d}: Sending {pattern_strings[idx]}")

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
    for i in range(num_molecule_types):
        smoothed_signals[i] = moving_average(molecule_signals[i], SMOOTHING_WINDOW)

    # ── Calibrate fixed threshold ────────────────────────────────────────
    fixed_threshold, mean_present, mean_absent = calibrate_sparse_threshold(
        smoothed_signals, symbol_transmissions, dt_bin,
        num_molecule_types, delay_between_symbols, delay_between_molecules,
    )

    # ── Decoding ─────────────────────────────────────────────────────────
    print("=== DECODING WITH FIXED THRESHOLD + MOLECULE EXCLUSION ===")
    decoded_symbols = decode_sparse_fixed(
        molecule_signals, bit_sequence, dt_bin,
        num_molecule_types, molecule_names,
        delay_between_symbols, delay_between_molecules,
        all_patterns, pattern_strings,
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
        half_w = delay_between_molecules * 0.8
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
                s = max(int((peak_center - half_w) / dt_bin), 0)
                e = min(int((peak_center + half_w) / dt_bin),
                        smoothed_signals.shape[1] - 1)
                if s >= e:
                    continue
                for m in range(num_molecule_types):
                    peak = float(np.max(smoothed_signals[m, s:e]))
                    if m == sent_mol:
                        all_present.append(peak)
                    else:
                        all_absent.append(peak)

        plt.scatter(range(len(all_present)), all_present, s=10, color='green',
                    alpha=0.6, label='Present channel peaks')
        plt.scatter(range(len(all_absent)), all_absent, s=4, color='gray',
                    alpha=0.3, label='Absent channel peaks')
        plt.axhline(fixed_threshold, color='red', linewidth=2, linestyle='--',
                    label=f'Fixed threshold ({fixed_threshold:.3f})')
        plt.axhline(mean_present, color='green', linewidth=1, linestyle=':',
                    label=f'Mean present ({mean_present:.3f})')
        plt.axhline(mean_absent, color='gray', linewidth=1, linestyle=':',
                    label=f'Mean absent ({mean_absent:.3f})')
        plt.xlabel('Sample index'); plt.ylabel('Smoothed Peak Value')
        plt.title('Present vs Absent Channel Peaks — Fixed Threshold Separation')
        plt.legend(fontsize=8); plt.grid(True)

        plt.tight_layout()
        plt.show()

    # ── Save CSV ─────────────────────────────────────────────────────────
    output_file = os.path.join(exp_path, "combined_bit_sequence_signal.csv")
    save_signal_csv(combined_signal, time_axis, molecule_signals,
                    molecule_names, output_file)

    return combined_signal, time_axis, symbol_transmissions, molecule_signals, decoded_symbols
