"""Generate channel templates without running modulation sweeps."""

import argparse
import os

from channel.diffusion import ensure_experiment_templates, get_device
from config import CUSTOM_DIR, D, DT, R0, RR, T


def main():
    parser = argparse.ArgumentParser(
        description="Prepare reusable diffusion-channel templates for NTx values."
    )
    parser.add_argument(
        "--n-tx-values", default="100,200,300,400,500,600,700,800,900,1000",
        help="comma-separated transmitter molecule counts",
    )
    parser.add_argument("--templates", type=int, default=100,
                        help="number of templates per NTx value (default: 100)")
    parser.add_argument("--output-root", default=CUSTOM_DIR,
                        help="folder in which NTx template folders are created")
    parser.add_argument("--template-method", choices=("brownian", "first-passage"),
                        default="brownian",
                        help="stepwise Brownian walk or exact first-passage sampling")
    args = parser.parse_args()

    n_tx_values = [int(value) for value in args.n_tx_values.split(",")]
    if not n_tx_values or any(value <= 0 for value in n_tx_values):
        parser.error("--n-tx-values must contain positive integers")
    if args.templates <= 0:
        parser.error("--templates must be positive")

    output_root = os.path.abspath(args.output_root)
    if args.template_method == "first-passage":
        output_root = os.path.join(output_root, "first-passage")

    device = get_device() if args.template_method == "brownian" else None
    for n_tx in n_tx_values:
        ensure_experiment_templates(
            n_tx=n_tx,
            exp_dir=os.path.join(output_root, f"N{n_tx}"),
            n_exp=args.templates,
            radius=RR,
            total_time=T,
            step_time=DT,
            diffusion_coef=D,
            distance=R0,
            device=device,
            method=args.template_method,
        )
    print(f"Template generation complete: {output_root}")


if __name__ == "__main__":
    main()
