"""
bit_generation/bit_generator.py
===============================
Random bit sequence generation and conversion to received symbol
observations using a pre-simulated indexed molecule pool.

This module provides:
  - File naming helpers for simulation artefacts.
  - Random bit sequence generation.
  - Conversion of indexed molecule pool → received symbol observations.
"""

import os

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print

from channel.diffusion import (
    load_hits_file,
    simulate_indexed_hits,
)


# ============================================================
# FILE-NAME HELPERS
# ============================================================

def float_tag(x, ndigits=8):
    s = f"{float(x):.{ndigits}g}"
    s = s.replace("-", "m")
    s = s.replace(".", "p")
    s = s.replace("+", "")
    return s


def int_tag(n):
    return str(int(n))


def name_from_distance(d):
    return f"d_{float_tag(d)}"


def make_raw_brownian_tag(radius, distance, diffusion_coef, step_time, memory):
    return (
        f"{name_from_distance(distance)}"
        f"_r{float_tag(radius)}"
        f"_D{float_tag(diffusion_coef)}"
        f"_dt{float_tag(step_time)}"
        f"_MEM{float_tag(memory)}"
    )


def make_bit_pool_filename(sim_dir, radius, distance, diffusion_coef, step_time, memory, n_pool):
    tag = make_raw_brownian_tag(radius, distance, diffusion_coef, step_time, memory)
    return os.path.join(
        sim_dir,
        f"Hits_bit_pool_indexed_noXYZ_{tag}_N{int_tag(n_pool)}.pt",
    )


def make_bits_filename(sim_dir, radius, distance, diffusion_coef, step_time,
                       memory, ts, memory_taps, n_bits, n_emit_bit):
    tag = (
        f"{make_raw_brownian_tag(radius, distance, diffusion_coef, step_time, memory)}"
        f"_TS{float_tag(ts)}"
        f"_MT{memory_taps}"
        f"_Nbit{n_bits}"
        f"_Nemitbit{n_emit_bit}"
    )
    return os.path.join(sim_dir, f"{tag}_bits_sequence.csv")


def make_received_summary_filename(sim_dir, radius, distance, diffusion_coef,
                                   step_time, memory, ts, memory_taps,
                                   n_bits, n_emit_bit):
    tag = (
        f"{make_raw_brownian_tag(radius, distance, diffusion_coef, step_time, memory)}"
        f"_TS{float_tag(ts)}"
        f"_MT{memory_taps}"
        f"_Nbit{n_bits}"
        f"_Nemitbit{n_emit_bit}"
    )
    return os.path.join(sim_dir, f"{tag}_received_summary.csv")


def make_received_patterns_filename(sim_dir, radius, distance, diffusion_coef,
                                    step_time, memory, ts, memory_taps,
                                    n_bits, n_emit_bit):
    tag = (
        f"{make_raw_brownian_tag(radius, distance, diffusion_coef, step_time, memory)}"
        f"_TS{float_tag(ts)}"
        f"_MT{memory_taps}"
        f"_Nbit{n_bits}"
        f"_Nemitbit{n_emit_bit}"
    )
    return os.path.join(sim_dir, f"{tag}_received_hit_patterns.npz")


# ============================================================
# BIT-POOL MANAGEMENT
# ============================================================

def ensure_bit_pool_file(
    pool_file,
    radius,
    distance,
    diffusion_coef,
    step_time,
    memory,
    n_pool,
    device=None,
    batch_size=500_000,
):
    """
    Returns the path to the bit-pool hit file. Generates it via
    Brownian simulation if it doesn't already exist.
    """
    if os.path.exists(pool_file):
        print("\nUsing existing bit-pool file:")
        print(" ", pool_file)
        return pool_file

    print("\nBit-pool file does not exist. Generating:")
    print(" ", pool_file)

    return simulate_indexed_hits(
        filename=pool_file,
        radius=radius,
        total_time=memory,
        step_time=step_time,
        diffusion_coef=diffusion_coef,
        distance=distance,
        nof_molecules=n_pool,
        device=device,
        override=False,
        batch_size=batch_size,
    )


# ============================================================
# RANDOM BIT SEQUENCE GENERATION
# ============================================================

def generate_bit_sequence(n_bits, p_one=0.5, seed=2026):
    """
    Generates a random binary sequence.

    Returns
    -------
    bits : np.ndarray of int8
    """
    rng = np.random.default_rng(seed)
    bits = rng.binomial(n=1, p=p_one, size=n_bits).astype(np.int8)
    return bits


def save_bit_sequence(bits, filepath):
    """Saves bits to a CSV file with columns [bit_index, bit]."""
    bits_df = pd.DataFrame({
        "bit_index": np.arange(len(bits), dtype=np.int64),
        "bit": bits,
    })
    bits_df.to_csv(filepath, index=False)
    display(bits_df.head(30))
    print(f"\nBit sequence saved to: {filepath}")
    return bits_df


# ============================================================
# INDEXED MOLECULE POOL → RECEIVED SYMBOL OBSERVATIONS
# ============================================================

def generate_received_symbols_from_indexed_impulse(
    hit_file,
    bits,
    n_emit_bit,
    memory_taps,
    ts,
    index_mode="disjoint_blocks",
):
    """
    Creates received symbol observations from the indexed molecule pool.

    Parameters
    ----------
    hit_file : str
        Path to the .pt bit-pool hit file.
    bits : array-like of int
        The transmitted bit sequence.
    n_emit_bit : int
        Number of molecules emitted per bit.
    memory_taps : int
        Number of ISI memory taps.
    ts : float
        Symbol duration in seconds.
    index_mode : str
        'disjoint_blocks' (default) or 'first_block'.

    Returns
    -------
    received_symbol_hits : list of dict
        Each dict has keys: rx_symbol, t_rel, source_bit, tap.
    """
    if not os.path.exists(hit_file):
        raise FileNotFoundError(f"Missing hit file: {hit_file}")

    hits = load_hits_file(hit_file)

    t_imp = hits["t"].astype(np.float64)
    idx_imp = hits["idx"]

    if idx_imp is None:
        raise ValueError("Bit-pool file must contain molecule indices `idx`.")

    idx_imp = idx_imp.astype(np.int64)

    bits = np.asarray(bits, dtype=np.int8)
    n_bits = len(bits)

    if index_mode not in {"disjoint_blocks", "first_block"}:
        raise ValueError("index_mode must be 'disjoint_blocks' or 'first_block'.")

    if index_mode == "disjoint_blocks":
        needed = n_bits * int(n_emit_bit)

        if hits["nof_molecules"] is not None and needed > hits["nof_molecules"]:
            raise ValueError(
                f"Need at least {needed} indexed molecules for "
                f"{n_bits} bits × {n_emit_bit} molecules/bit, "
                f"but file has {hits['nof_molecules']} emitted molecules."
            )

    received_symbol_hits = []

    for n in range(n_bits):
        collected_t_rel = []
        collected_source_bit = []
        collected_tap = []

        for tap in range(memory_taps):
            k = n - tap

            if k < 0 or k >= n_bits:
                continue

            if bits[k] == 0:
                continue

            if index_mode == "disjoint_blocks":
                idx_low = k * int(n_emit_bit)
                idx_high = (k + 1) * int(n_emit_bit)
            else:
                idx_low = 0
                idx_high = int(n_emit_bit)

            t_low = tap * ts
            t_high = (tap + 1) * ts

            mask = (
                (idx_imp >= idx_low)
                & (idx_imp < idx_high)
                & (t_imp >= t_low)
                & (t_imp < t_high)
            )

            selected = np.flatnonzero(mask)

            if len(selected) == 0:
                continue

            # Relative time inside receive symbol n
            t_rel = t_imp[selected] - t_low

            collected_t_rel.append(t_rel)
            collected_source_bit.extend([k] * len(selected))
            collected_tap.extend([tap] * len(selected))

        if len(collected_t_rel) == 0:
            t_rel_n = np.empty(0, dtype=np.float64)
        else:
            t_rel_n = np.concatenate(collected_t_rel).astype(np.float64)

        received_symbol_hits.append({
            "rx_symbol": n,
            "t_rel": t_rel_n,
            "source_bit": np.array(collected_source_bit, dtype=np.int64),
            "tap": np.array(collected_tap, dtype=np.int64),
        })

    return received_symbol_hits


# ============================================================
# SAVE / SUMMARISE RECEIVED OBSERVATIONS
# ============================================================

def save_received_summary(received_symbol_hits, bits, memory_taps, filepath):
    """Saves a debug summary table of received observations."""
    summary_rows = []

    for n, obs in enumerate(received_symbol_hits):
        row = {
            "rx_symbol": n,
            "tx_bit": int(bits[n]),
            "n_observed_hits": len(obs["t_rel"]),
        }

        for tap in range(memory_taps):
            bit_index = n - tap

            row[f"bit_n_minus_{tap}"] = (
                int(bits[bit_index]) if 0 <= bit_index < len(bits) else 0
            )

            row[f"hits_from_tap_{tap}"] = int(np.sum(obs["tap"] == tap))

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(filepath, index=False)
    display(df.head(30))
    print(f"\nReceived summary saved to: {filepath}")
    return df


def save_received_patterns(received_symbol_hits, bits, n_emit_bit, ts,
                           memory, memory_taps, bit_pool_hit_file, filepath):
    """Saves full received hit patterns to a compressed .npz file."""
    np.savez_compressed(
        filepath,
        rx_symbol=np.array(
            [obs["rx_symbol"] for obs in received_symbol_hits],
            dtype=np.int64,
        ),
        t_rel=np.array(
            [obs["t_rel"] for obs in received_symbol_hits],
            dtype=object,
        ),
        source_bit=np.array(
            [obs["source_bit"] for obs in received_symbol_hits],
            dtype=object,
        ),
        tap=np.array(
            [obs["tap"] for obs in received_symbol_hits],
            dtype=object,
        ),
        bits=bits,
        n_emit_bit=int(n_emit_bit),
        ts=float(ts),
        memory=float(memory),
        memory_taps=int(memory_taps),
        bit_pool_hit_file=str(bit_pool_hit_file),
    )
    print(f"\nReceived hit patterns saved to: {filepath}")
