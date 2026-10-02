"""
evaluation/metrics.py
=====================
Evaluation metrics for molecular communication experiments.

Provides:
  - Bit accuracy and symbol accuracy comparison (from Kontrol.py).
  - Wilson score confidence interval for binomial proportions
    (from mc_modulation_ser_sweep.py).
  - SER/BER computation helpers.
"""

import os
import math

import numpy as np
import pandas as pd


# ============================================================
# WILSON SCORE CONFIDENCE INTERVAL
# ============================================================

def wilson_ci(k, n, z=1.96):
    """
    95% Wilson score confidence interval for a binomial proportion.

    Correctly handles k=0 and k=n (unlike naive normal approximation).

    Parameters
    ----------
    k : int
        Number of successes (errors).
    n : int
        Number of trials (total symbols/bits).
    z : float
        Z-score for confidence level (1.96 → 95%).

    Returns
    -------
    phat : float
        Point estimate p̂ = k/n.
    lo : float
        Lower bound of the CI.
    hi : float
        Upper bound of the CI.
    """
    if n <= 0:
        return float("nan"), float("nan"), float("nan")

    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom

    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return phat, lo, hi


# ============================================================
# BIT / SYMBOL ACCURACY COMPARISON
# ============================================================

def compare_bit_accuracy(original_bits, decoded_results_csv):
    """
    Compare ground truth bits against decoded results.

    Parameters
    ----------
    original_bits : str
        Ground truth binary string.
    decoded_results_csv : str
        Path to decoded results CSV with column 'Decoded_Symbol'.

    Returns
    -------
    bit_accuracy : float
    total_bits : int
    correct_bits : int
    """
    df = pd.read_csv(decoded_results_csv)
    total_bits = len(original_bits)
    correct_bits = 0

    for i in range(total_bits):
        true_bit = original_bits[i]
        decoded_sym = str(df.loc[i, 'Decoded_Symbol'])
        decoded_bit = "0" if decoded_sym == "SILENCE" else "1"

        if true_bit == decoded_bit:
            correct_bits += 1

    return correct_bits / total_bits, total_bits, correct_bits


def compare_symbol_accuracy(ground_truth_symbols, decoded_results_csv):
    """
    Compare ground truth symbols against decoded symbols.

    Parameters
    ----------
    ground_truth_symbols : list of str
        Ground truth symbol list (e.g. ['AB', 'SILENCE', 'CDE', ...]).
    decoded_results_csv : str
        Path to decoded results CSV.

    Returns
    -------
    symbol_accuracy : float
    total_symbols : int
    correct_symbols : int
    """
    df = pd.read_csv(decoded_results_csv)
    total = len(ground_truth_symbols)
    correct = 0

    for i in range(total):
        true_sym = ground_truth_symbols[i]
        decoded_sym = str(df.loc[i, 'Decoded_Symbol'])

        if true_sym == decoded_sym:
            correct += 1

    return correct / total, total, correct


def full_evaluation(gt_bit_file, gt_symbol_file, decoded_results_csv):
    """
    Full bit + symbol accuracy evaluation (refactored from Kontrol.py).

    Parameters
    ----------
    gt_bit_file : str
        Path to ground_truth_sequence.txt.
    gt_symbol_file : str
        Path to ground_truth_symbols.txt.
    decoded_results_csv : str
        Path to Decoded_Results.csv.

    Returns
    -------
    results : dict
        Keys: bit_accuracy, symbol_accuracy, total_bits,
        correct_bits, correct_symbols, errors (list of dicts).
    """
    if not os.path.exists(gt_bit_file) or not os.path.exists(decoded_results_csv):
        raise FileNotFoundError(
            "Required files not found. Run simulator and decoder first."
        )

    with open(gt_bit_file, "r") as f:
        original_bits = f.read().strip()

    with open(gt_symbol_file, "r") as f:
        true_symbols = f.read().strip().split('\n')

    df = pd.read_csv(decoded_results_csv)

    total_bits = len(original_bits)
    correct_bits = 0
    correct_symbols = 0
    errors = []

    print(f"--- {total_bits} BIT / SYMBOL COMPARISON ---\n")
    print(f"{'Bit':>4} | {'Ground Truth':>12} | {'Decoded':>12} | {'Bit OK':>6} | {'Symbol OK':>9}")
    print("-" * 58)

    for i in range(total_bits):
        true_bit = original_bits[i]
        true_sym = true_symbols[i]
        decoded_sym = str(df.loc[i, 'Decoded_Symbol'])
        decoded_bit = "0" if decoded_sym == "SILENCE" else "1"

        bit_ok = true_bit == decoded_bit
        symbol_ok = true_sym == decoded_sym

        if bit_ok:
            correct_bits += 1
        if symbol_ok:
            correct_symbols += 1

        if not bit_ok or not symbol_ok:
            bit_flag = "OK" if bit_ok else "ERROR"
            symbol_flag = "OK" if symbol_ok else "WRONG"
            print(f"{i:>4} | {true_sym:>12} | {decoded_sym:>12} | {bit_flag:>6} | {symbol_flag:>9}")
            errors.append({
                'index': i,
                'true_symbol': true_sym,
                'decoded_symbol': decoded_sym,
                'bit_error': not bit_ok,
                'symbol_error': not symbol_ok,
            })

    print("-" * 58)
    bit_acc = correct_bits / total_bits
    sym_acc = correct_symbols / total_bits
    print(f"\nBit    Accuracy: {bit_acc * 100:.2f}%  ({correct_bits}/{total_bits})")
    print(f"Symbol Accuracy: {sym_acc * 100:.2f}%  ({correct_symbols}/{total_bits})")

    return {
        'bit_accuracy': bit_acc,
        'symbol_accuracy': sym_acc,
        'total_bits': total_bits,
        'correct_bits': correct_bits,
        'correct_symbols': correct_symbols,
        'errors': errors,
    }


# ============================================================
# METHODOLOGY NOTE FOR PAPER
# ============================================================

def print_methodology_note(n_tx_list, ts_list, seq_length, num_trials, seed, mode="SER"):
    """
    Prints (and returns) a ready-to-paste methodology note for the
    Performance Evaluation section of a paper.
    """
    total_units = seq_length * num_trials
    unit_name = "symbols" if mode == "SER" else "bits"
    note = (
        "\n[Performance Evaluation -- methodology note]\n"
        f"Error rates are obtained via Monte Carlo simulation of the particle-based "
        f"absorbing-receiver channel (Brownian motion; no closed-form shortcut is "
        f"used). For each (N_Tx, T_s) operating point in N_Tx ∈ {n_tx_list}, "
        f"T_s ∈ {ts_list} s, we transmit {num_trials} independent random sequences "
        f"of {seq_length} {unit_name} each (RNG seed={seed}), pooling "
        f"{total_units} {unit_name} per point before computing the {mode}. Each "
        f"reported {mode} is accompanied by its 95% Wilson score confidence "
        f"interval; operating points where zero errors were observed across all "
        f"{total_units} pooled {unit_name} are reported as the 95% upper confidence "
        f"bound on the {mode} rather than as a literal zero, since a literal zero "
        f"is neither statistically meaningful from a finite sample nor "
        f"representable on a logarithmic axis.\n"
    )
    print(note)
    return note
