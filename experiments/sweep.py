"""
experiments/sweep.py
====================
Monte-Carlo SER/BER vs Ts sweep across the 4 MoCSK-family schemes:
  1. Permutation (smoothed-peak arrival order)
  2. Combination with fixed calibrated threshold
  3. Combination with adaptive gap-based threshold
  4. Sparse pattern with fixed threshold and molecule exclusion

Features:
  - N_Tx sweep ([500, 1000] by default) wired to per-N cached experiment templates.
  - Pooled Wilson 95% confidence intervals across all trials per point.
  - Zero-error safe: displays 95% upper bound markers (0/n) instead of dropping or NaN on log plots.
  - Generates publication-ready figures saved in results/figures/.
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Ensure repository root is on sys.path
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from config import CUSTOM_DIR, NORMALIZATION, R0, RR, D, T, DT
from channel.diffusion import ensure_experiment_templates, get_device
from evaluation.metrics import wilson_ci, print_methodology_note

from modulation.permutation import generate_bit_sequence_transmission as run_permutation
from modulation.combination_fixed import generate_combination_transmission as run_combo_fixed
from modulation.combination_adaptive import generate_combination_transmission as run_combo_adaptive
from modulation.sparse import generate_sparse_bit_sequence_transmission as run_sparse


SCHEME_RUNNERS = {
    "permutation": run_permutation,
    "combo_fixed": run_combo_fixed,
    "combo_adaptive": run_combo_adaptive,
    "sparse": run_sparse,
}

_SCHEME_COLORS = {
    "permutation": "#378ADD",
    "combo_fixed": "#1D9E75",
    "combo_adaptive": "#D85A30",
    "sparse": "#7F77DD",
}

_SCHEME_LABELS = {
    "permutation": "Permutation",
    "combo_fixed": "Combination (fixed thr.)",
    "combo_adaptive": "Combination (adaptive thr.)",
    "sparse": "Sparse",
}

_LINE_STYLES = ["-", "--", ":", "-."]
_MARKERS = ["o", "s", "^", "D", "v"]


def _single_trial_counts(
    scheme,
    bit_sequence,
    num_molecule_types,
    ts,
    exp_path,
    num_experiments,
    normalization,
    delay_between_molecules,
    mode="ser",
):
    """
    Run one trial of a given scheme and return (errors, total_units).
    For 'ser', units = number of '1' symbols transmitted.
    For 'ber', units = len(bit_sequence).
    """
    runner = SCHEME_RUNNERS.get(scheme)
    if runner is None:
        raise ValueError(f"Unknown scheme: {scheme}. Choose from {list(SCHEME_RUNNERS.keys())}")

    kwargs = dict(
        bit_sequence=bit_sequence,
        num_molecule_types=num_molecule_types,
        exp_path=exp_path,
        num_experiments=num_experiments,
        normalization=normalization,
        delay_between_symbols=ts,
        delay_between_molecules=delay_between_molecules,
        plot=False,
    )

    _, _, transmissions, _, decoded = runner(**kwargs)

    if scheme in ("permutation", "sparse"):
        errors = sum(
            1 for s, d in zip(transmissions, decoded)
            if s["permutation_names"] != d["permutation"]
        )
    elif scheme in ("combo_fixed", "combo_adaptive"):
        errors = sum(
            1 for s, d in zip(transmissions, decoded)
            if s["permutation_names"] != d["decoded_combo"]
        )
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    if mode == "ser":
        total_units = len(transmissions)
    elif mode == "ber":
        total_units = len(bit_sequence)
    else:
        raise ValueError("mode must be 'ser' or 'ber'")

    return errors, total_units


def sweep_ts_pooled(
    scheme,
    ts_values,
    n_tx,
    *,
    num_molecule_types=5,
    seq_length=100,
    num_trials=10,
    exp_root=None,
    n_exp_templates=100,
    normalization=NORMALIZATION,
    delay_between_molecules=0.4,
    device=None,
    rng_seed=None,
    mode="ser",
    min_informative_units=30,
):
    """
    Sweeps Ts for a single (scheme, n_tx) configuration, pooling counts across trials.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    if exp_root is None:
        exp_root = CUSTOM_DIR

    exp_path = os.path.join(exp_root, f"N{int(n_tx)}")
    ensure_experiment_templates(
        n_tx=n_tx,
        exp_dir=exp_path,
        n_exp=n_exp_templates,
        radius=RR,
        total_time=T,
        step_time=DT,
        diffusion_coef=D,
        distance=R0,
        device=device,
    )

    ts_out, err_out, tot_out = [], [], []

    for ts in ts_values:
        total_errors = 0
        total_units = 0

        for _ in range(num_trials):
            bit_sequence = "".join(str(np.random.randint(0, 2)) for _ in range(seq_length))
            if mode == "ser" and bit_sequence.count("1") == 0:
                continue

            try:
                errors, units = _single_trial_counts(
                    scheme=scheme,
                    bit_sequence=bit_sequence,
                    num_molecule_types=num_molecule_types,
                    ts=ts,
                    exp_path=exp_path,
                    num_experiments=n_exp_templates,
                    normalization=normalization,
                    delay_between_molecules=delay_between_molecules,
                    mode=mode,
                )
            except Exception as exc:
                print(f"  [WARN] scheme={scheme} N_Tx={n_tx} Ts={ts:.2f} -> {exc}")
                continue

            total_errors += errors
            total_units += units

        phat, lo, hi = wilson_ci(total_errors, total_units)

        flag = ""
        if total_errors == 0 and total_units > 0:
            unit_name = "symbols" if mode == "ser" else "bits"
            flag = f"  <- 0 errors / {total_units} {unit_name}; 95% CI upper bound shown"
            if total_units < min_informative_units:
                flag += f" [WARNING: n={total_units} < {min_informative_units}]"

        print(
            f"  {scheme:15s}  N_Tx={n_tx:5d}  Ts={ts:.2f}  "
            f"{mode.upper()}={phat:.3e}  CI95=[{lo:.3e}, {hi:.3e}]  "
            f"n={total_units}{flag}"
        )

        ts_out.append(ts)
        err_out.append(total_errors)
        tot_out.append(total_units)

    return {
        "ts": np.array(ts_out, dtype=float),
        "errors": np.array(err_out, dtype=np.int64),
        "totals": np.array(tot_out, dtype=np.int64),
    }


def plot_rate_vs_ts(
    results,
    mode="ser",
    title=None,
    save_path=None,
    figsize=(10, 6),
    log_scale=True,
):
    """
    Plot pooled SER/BER vs Ts with Wilson confidence intervals and zero-error upper bounds.
    """
    ylabel = "Symbol Error Rate (SER)" if mode == "ser" else "Bit Error Rate (BER)"
    if title is None:
        title = f"{'SER' if mode == 'ser' else 'BER'} vs Symbol Period $T_s$ (pooled, 95% Wilson CI)"

    fig, ax = plt.subplots(figsize=figsize)
    scheme_legend_handles = []

    for scheme, n_dict in results.items():
        color = _SCHEME_COLORS.get(scheme, "black")
        s_lbl = _SCHEME_LABELS.get(scheme, scheme)

        n_values = sorted(n_dict.keys())
        for n_idx, n_tx in enumerate(n_values):
            data = n_dict[n_tx]
            ts_arr = np.asarray(data["ts"], dtype=float)
            errs = np.asarray(data["errors"], dtype=np.int64)
            tots = np.asarray(data["totals"], dtype=np.int64)

            phat = np.full(len(ts_arr), np.nan)
            lo = np.full(len(ts_arr), np.nan)
            hi = np.full(len(ts_arr), np.nan)
            for i, (k, n) in enumerate(zip(errs, tots)):
                if n > 0:
                    phat[i], lo[i], hi[i] = wilson_ci(int(k), int(n))

            zero_mask = (errs == 0) & (tots > 0)
            nz_mask = (~zero_mask) & (~np.isnan(phat))

            lstyle = _LINE_STYLES[n_idx % len(_LINE_STYLES)]
            marker = _MARKERS[n_idx % len(_MARKERS)]

            if np.any(nz_mask):
                ax.plot(
                    ts_arr[nz_mask], phat[nz_mask],
                    color=color, linestyle=lstyle, marker=marker,
                    markersize=5, linewidth=1.8,
                    label=f"{s_lbl} N_Tx={n_tx}",
                )
                yerr_lo = np.clip(phat[nz_mask] - lo[nz_mask], 1e-12, None)
                yerr_hi = np.clip(hi[nz_mask] - phat[nz_mask], 1e-12, None)
                ax.errorbar(
                    ts_arr[nz_mask], phat[nz_mask],
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor=color, alpha=0.35, capsize=3,
                )

            if np.any(zero_mask):
                ax.plot(
                    ts_arr[zero_mask], hi[zero_mask],
                    marker="v", linestyle="None",
                    markerfacecolor="none", markeredgecolor=color,
                    markersize=8, markeredgewidth=1.5,
                )
                for x, y, n in zip(ts_arr[zero_mask], hi[zero_mask], tots[zero_mask]):
                    ax.annotate(
                        f"0/{int(n)}", (x, y),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=7, color=color, ha="center",
                    )

        scheme_legend_handles.append(
            mlines.Line2D([], [], color=color, linewidth=2.5, label=s_lbl)
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Symbol Period $T_s$ (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=10)

    scheme_legend = ax.legend(
        handles=scheme_legend_handles, loc="upper right",
        fontsize=9, title="Scheme", title_fontsize=9, framealpha=0.85,
    )
    ax.add_artist(scheme_legend)

    n_all = sorted({n for n_dict in results.values() for n in n_dict.keys()})
    n_handles = [
        mlines.Line2D(
            [], [], color="gray",
            linestyle=_LINE_STYLES[i % len(_LINE_STYLES)],
            marker=_MARKERS[i % len(_MARKERS)],
            markersize=5, label=f"N_Tx = {n}",
        )
        for i, n in enumerate(n_all)
    ]
    ax.legend(
        handles=n_handles, loc="lower left",
        fontsize=9, title="Molecules per Tx", title_fontsize=9, framealpha=0.85,
    )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {save_path}")
    return fig


def run_ser_vs_ts_sweep(
    schemes=None,
    n_tx_list=None,
    ts_values=None,
    num_molecule_types=5,
    seq_length=100,
    num_trials=10,
    exp_root=None,
    n_exp_templates=100,
    title=None,
    save_path=None,
    log_scale=True,
    rng_seed=42,
):
    """
    Run full Monte Carlo SER vs Ts sweep across schemes and N_Tx values.
    """
    if schemes is None:
        schemes = ["permutation", "combo_fixed", "combo_adaptive", "sparse"]
    if n_tx_list is None:
        n_tx_list = [500, 1000]
    if ts_values is None:
        ts_values = list(np.round(np.arange(0.20, 0.40 + 1e-9, 0.05), 2))

    print("=" * 70)
    print("SER vs Ts Sweep (Pooled, Wilson 95% CI)")
    print(f"  Schemes   : {schemes}")
    print(f"  N_Tx      : {n_tx_list}")
    print(f"  Ts grid   : {ts_values}")
    print(f"  per point : {num_trials} trials x {seq_length} symbols "
          f"= {num_trials * seq_length} symbols pooled")
    print("=" * 70)

    results = {}
    for scheme in schemes:
        results[scheme] = {}
        for n_tx in n_tx_list:
            print(f"\n>> scheme={scheme}  N_Tx={n_tx}")
            results[scheme][n_tx] = sweep_ts_pooled(
                scheme=scheme,
                ts_values=ts_values,
                n_tx=n_tx,
                num_molecule_types=num_molecule_types,
                seq_length=seq_length,
                num_trials=num_trials,
                exp_root=exp_root,
                n_exp_templates=n_exp_templates,
                rng_seed=rng_seed,
                mode="ser",
            )

    fig = plot_rate_vs_ts(results, mode="ser", title=title, save_path=save_path, log_scale=log_scale)
    plt.show()

    print_methodology_note(n_tx_list, ts_values, seq_length, num_trials, rng_seed, mode="SER")
    return results


if __name__ == "__main__":
    figures_dir = os.path.join(PROJECT_DIR, "results", "figures")
    out_fig = os.path.join(figures_dir, "ser_vs_ts_sweep.png")

    run_ser_vs_ts_sweep(
        schemes=["permutation", "combo_fixed", "combo_adaptive", "sparse"],
        n_tx_list=[500, 1000],
        ts_values=[0.20, 0.25, 0.30, 0.35, 0.40],
        num_molecule_types=5,
        seq_length=50,
        num_trials=5,
        save_path=out_fig,
    )
