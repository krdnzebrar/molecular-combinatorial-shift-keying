"""
experiments/sweep.py
====================
Monte-Carlo SER/BER vs Ts sweep across the 4 MoCSK-family schemes:
  1. Permutation (smoothed-peak arrival order)
  2. Combination with fixed calibrated threshold
  3. Combination with adaptive gap-based threshold
  4. Sparse pattern with fixed threshold and molecule exclusion

Features:
  - N_Tx sweep (100, 200, ..., 1000 by default) wired to per-N cached templates.
  - Pooled Wilson 95% confidence intervals across all trials per point.
  - Zero-error safe: displays 95% upper bound markers (0/n) instead of dropping or NaN on log plots.
  - Generates publication-ready figures saved in results/figures/.
"""

import os
import sys
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

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
    "combo_fixed": "MoCSK (unlabeled fixed threshold)",
    "combo_adaptive": "MoCSK (adaptive ratio threshold)",
    "sparse": "E-MoCSK (slot selection)",
}

_LINE_STYLES = ["-", "--", ":", "-."]
_MARKERS = ["o", "s", "^", "D", "v"]


def information_bits_per_symbol(scheme, num_molecule_types):
    """Return log2 of the scheme's nominal alphabet size."""
    k = int(num_molecule_types)
    if scheme == "permutation":
        log2_m = math.lgamma(k + 1) / math.log(2)
    elif scheme in ("combo_fixed", "combo_adaptive"):
        alphabet_size = sum(math.factorial(k) // math.factorial(k - j)
                            for j in range(k + 1))
        log2_m = math.log2(alphabet_size)
    elif scheme == "sparse":
        # E-MoCSK: includes k=0 (the all-empty symbol).
        alphabet_size = sum(math.comb(k, j) ** 2 * math.factorial(j) for j in range(k + 1))
        log2_m = math.log2(alphabet_size)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return log2_m


def slot_delay_for_symbol(ts, num_molecule_types):
    """Use a uniform release grid, including across consecutive symbol boundaries."""
    k = int(num_molecule_types)
    if k <= 1:
        return min(float(max_delay), float(ts))
    # Keeping Δ=Ts/K makes the gap from the last slot of one active symbol
    # to the first slot of the next consecutive symbol equal to every other gap.
    return float(ts) / k


def _save_sweep_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    columns = ["sweep", "scheme", "num_molecule_types", "n_tx", "ts", "errors",
               "total_symbols", "ser", "ser_ci_low", "ser_ci_high",
               "bits_per_symbol", "goodput_bits_per_second"]
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_point(scheme, k, n_tx, counts, sweep):
    rows = []
    for idx, (errors, total) in enumerate(zip(counts["errors"], counts["totals"])):
        ser, low, high = wilson_ci(int(errors), int(total))
        bps = information_bits_per_symbol(scheme, k)
        ts = float(counts["ts"][idx])
        rows.append({
            "sweep": sweep, "scheme": scheme, "num_molecule_types": k,
            "n_tx": n_tx, "ts": ts, "errors": int(errors),
            "total_symbols": int(total), "ser": ser, "ser_ci_low": low,
            "ser_ci_high": high, "bits_per_symbol": bps,
            "goodput_bits_per_second": (1.0 - ser) * bps / ts if total else float("nan"),
        })
    return rows


def run_requested_sweeps(
    *, schemes=None, molecule_type_counts=range(2, 11), n_tx_values=range(100, 1001, 100),
    seq_length=100, num_trials=10, n_exp_templates=100, exp_root=None,
    delay_between_molecules=None, rng_seed=42, ts_sweep_n_tx=500,
    ts_values=None,
    generate_templates=True,
    results_dir=None,
):
    """Run the Ts and N_Tx sweeps, save PDF plots and a combined CSV."""
    if schemes is None:
        schemes = ["permutation", "combo_fixed", "combo_adaptive", "sparse"]
    ks = [int(k) for k in molecule_type_counts]
    nts = [int(n) for n in n_tx_values]
    if ts_values is None:
        ts_values = np.round(np.arange(0.2, 1.0 + 1e-9, 0.1), 1).tolist()
    else:
        ts_values = [float(ts) for ts in ts_values]
    if not ks or any(k < 1 for k in ks):
        raise ValueError("molecule_type_counts must contain positive integers")
    if not nts or any(n < 1 for n in nts):
        raise ValueError("n_tx_values must contain positive integers")
    if not ts_values or any(ts <= 0 for ts in ts_values):
        raise ValueError("ts_values must contain positive values")
    results_dir = results_dir or os.path.join(PROJECT_DIR, "results")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    rows = []

    # Ts sweep: one curve per K and method, with N_Tx fixed.
    ts_data = {}
    fixed_n_tx = int(ts_sweep_n_tx)
    for scheme in schemes:
        ts_data[scheme] = {}
        for k in ks:
            ts_data[scheme][k] = sweep_ts_pooled(
                scheme, ts_values, fixed_n_tx, num_molecule_types=k,
                seq_length=seq_length, num_trials=num_trials, exp_root=exp_root,
                n_exp_templates=n_exp_templates, delay_between_molecules=delay_between_molecules,
                rng_seed=rng_seed, mode="ser", generate_templates=generate_templates)
            rows.extend(_summarize_point(scheme, k, fixed_n_tx, ts_data[scheme][k], "ts"))

    # N_Tx sweep: one curve per K and method, at its minimum Ts=1/K.
    ntx_data = {}
    for scheme in schemes:
        ntx_data[scheme] = {}
        for k in ks:
            ntx_data[scheme][k] = {}
            for n_tx in nts:
                counts = sweep_ts_pooled(
                    scheme, [1.0 / k], n_tx, num_molecule_types=k,
                    seq_length=seq_length, num_trials=num_trials, exp_root=exp_root,
                    n_exp_templates=n_exp_templates, delay_between_molecules=delay_between_molecules,
                    rng_seed=rng_seed, mode="ser", generate_templates=generate_templates)
                ntx_data[scheme][k][n_tx] = counts
                rows.extend(_summarize_point(scheme, k, n_tx, counts, "n_tx"))

    _plot_metric_families(ts_data, ks, schemes, "ts", os.path.join(figures_dir, "ser_goodput_vs_ts.pdf"), fixed_n_tx)
    _plot_metric_families(ntx_data, ks, schemes, "n_tx", os.path.join(figures_dir, "ser_goodput_vs_n_tx.pdf"), None)
    _save_sweep_csv(os.path.join(results_dir, "sweep_results.csv"), rows)
    print(f"Saved figures and pooled results under: {results_dir}")
    return {"ts": ts_data, "n_tx": ntx_data, "rows": rows}


def _plot_metric_families(data, ks, schemes, x_kind, path, fixed_n_tx):
    """Write a two-page PDF with one small multiple per K on each page."""
    ncols = min(3, len(ks))
    nrows = int(math.ceil(len(ks) / ncols))
    xlabel = "Symbol period $T_s$ (s)" if x_kind == "ts" else "$N_{\\rm Tx}$ molecules"
    if x_kind == "ts":
        condition = f"$N_{{\\rm Tx}}={fixed_n_tx}$"
    else:
        condition = "$T_s=1/K$ s for each K"

    with PdfPages(path) as pdf:
        for metric in ("ser", "goodput"):
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False
            )
            plotted_handles = {}
            for facet_idx, k in enumerate(ks):
                ax = axes.flat[facet_idx]
                for scheme in schemes:
                    if scheme not in data or k not in data[scheme]:
                        continue
                    if x_kind == "ts":
                        counts = data[scheme][k]
                        points = [
                            (float(ts), counts, idx)
                            for idx, ts in enumerate(counts["ts"])
                        ]
                    else:
                        points = [
                            (float(n_tx), data[scheme][k][n_tx], 0)
                            for n_tx in sorted(data[scheme][k])
                        ]

                    xs, ys = [], []
                    for x, counts, idx in points:
                        errors = int(counts["errors"][idx])
                        total = int(counts["totals"][idx])
                        # Failed/missing points must not appear as zero-error results.
                        if total <= 0:
                            continue
                        ser, _, ci_high = wilson_ci(errors, total)
                        ts = float(counts["ts"][idx])
                        y = (ser if ser > 0 else ci_high) if metric == "ser" else (
                            (1.0 - ser) * information_bits_per_symbol(scheme, k) / ts
                        )
                        xs.append(x)
                        ys.append(y)

                    if not xs:
                        continue
                    line, = ax.plot(
                        xs, ys, marker="o", markersize=3.5, linewidth=1.35,
                        label=_SCHEME_LABELS.get(scheme, scheme),
                        color=_SCHEME_COLORS.get(scheme, "black"),
                    )
                    plotted_handles.setdefault(scheme, line)

                ax.set_title(f"K={k}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(
                    "SER (zero-error points show 95% upper bound)"
                    if metric == "ser" else "SER-based goodput (bits/s)"
                )
                if metric == "ser":
                    ax.set_yscale("log")
                ax.grid(True, which="both", alpha=0.3)

            for idx in range(len(ks), nrows * ncols):
                axes.flat[idx].set_visible(False)

            title_metric = "SER" if metric == "ser" else "SER-based goodput"
            x_label = "$T_s$" if x_kind == "ts" else "$N_{\\rm Tx}$"
            fig.suptitle(f"{title_metric} vs {x_label} ({condition})", y=0.995)
            if plotted_handles:
                fig.legend(
                    list(plotted_handles.values()),
                    [_SCHEME_LABELS.get(s, s) for s in plotted_handles],
                    loc="upper center", bbox_to_anchor=(0.5, 0.965),
                    ncol=min(4, len(plotted_handles)), fontsize=8,
                )
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Figure saved -> {path}")


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
    For 'ser', units = number of modulation symbols transmitted.
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
    delay_between_molecules=None,
    device=None,
    rng_seed=None,
    mode="ser",
    min_informative_units=30,
    generate_templates=True,
):
    """
    Sweeps Ts for a single (scheme, n_tx) configuration, pooling counts across trials.
    Every symbol interval carries one random modulation symbol.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)

    if exp_root is None:
        exp_root = CUSTOM_DIR

    exp_path = os.path.join(exp_root, f"N{int(n_tx)}")
    if generate_templates:
        template_limit = 100 if n_exp_templates is None else int(n_exp_templates)
        ensure_experiment_templates(
            n_tx=n_tx,
            exp_dir=exp_path,
            n_exp=template_limit,
            radius=RR,
            total_time=T,
            step_time=DT,
            diffusion_coef=D,
            distance=R0,
            device=device,
        )
        effective_n_exp_templates = template_limit
    else:
        existing = set(os.listdir(exp_path)) if os.path.isdir(exp_path) else set()
        contiguous_count = 0
        while f"Exp_{contiguous_count:03d}.csv" in existing:
            contiguous_count += 1
        effective_n_exp_templates = contiguous_count if n_exp_templates is None else int(n_exp_templates)
        if contiguous_count < effective_n_exp_templates:
            raise FileNotFoundError(
                f"Reuse-only mode needs {effective_n_exp_templates} consecutive templates in {exp_path}; found {contiguous_count}."
            )

    ts_out, err_out, tot_out = [], [], []

    for ts in ts_values:
        slot_delay = (slot_delay_for_symbol(ts, num_molecule_types)
            if delay_between_molecules is None else float(delay_between_molecules))
        total_errors = 0
        total_units = 0

        for _ in range(num_trials):
            # One modulation symbol is sent in every period. The modulation
            # alphabet itself includes the empty MoCSK/E-MoCSK symbol.
            bit_sequence = "1" * seq_length

            try:
                errors, units = _single_trial_counts(
                    scheme=scheme,
                    bit_sequence=bit_sequence,
                    num_molecule_types=num_molecule_types,
                    ts=ts,
                    exp_path=exp_path,
                    num_experiments=effective_n_exp_templates,
                    normalization=normalization,
                    delay_between_molecules=slot_delay,
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
    import argparse

    parser = argparse.ArgumentParser(description="Run the requested SER/goodput sweeps.")
    parser.add_argument("--trials", type=int, default=10, help="Monte Carlo trials per point (default: 10)")
    parser.add_argument("--symbols", type=int, default=100, help="modulation symbols per trial (default: 100)")
    parser.add_argument("--templates", default="100", help="templates per N_Tx, or 'auto' to reuse all consecutive existing files")
    parser.add_argument("--ts-n-tx", type=int, default=500, help="fixed N_Tx for the Ts sweep (default: 500)")
    parser.add_argument("--ts-values", default="0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="comma-separated Ts sweep values in seconds")
    parser.add_argument("--exp-root", help="directory containing N100, N500, ... template folders")
    parser.add_argument("--n-tx-values", default="100,200,300,400,500,600,700,800,900,1000",
                        help="comma-separated N_Tx values for the N_Tx sweep")
    parser.add_argument("--molecule-types", default="2,3,4,5,6,7,8,9,10",
                        help="comma-separated molecule type counts")
    parser.add_argument("--reuse-only", action="store_true",
                        help="use existing templates only and fail instead of generating missing ones")
    args = parser.parse_args()
    run_requested_sweeps(
        seq_length=args.symbols,
        num_trials=args.trials,
        n_exp_templates=None if args.templates.lower() == "auto" else int(args.templates),
        ts_sweep_n_tx=args.ts_n_tx,
        exp_root=args.exp_root,
        n_tx_values=[int(x) for x in args.n_tx_values.split(",")],
        molecule_type_counts=[int(x) for x in args.molecule_types.split(",")],
        ts_values=[float(x) for x in args.ts_values.split(",")],
        generate_templates=not args.reuse_only,
    )
