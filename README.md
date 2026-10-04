# Molecular Combinatorial Shift Keying (MoCSK)

A unified, modular research repository for Molecular Combinatorial Shift Keying (MoCSK) and enhanced multi-molecule communication schemes over diffusion channels.

---

## Architecture Overview

All schemes share the same physical constants (`config.py`), signal processing tools (`utils/signal.py`), channel models (`channel/diffusion.py`), and evaluation metrics (`evaluation/metrics.py`).

```
molecular-combinatorial-shift-keying/
│
├── config.py                          # Single source of truth for physical constants & paths
├── requirements.txt                   # Project dependencies
├── README.md                          # Documentation and usage guide
├── .gitignore
│
├── utils/
│   ├── __init__.py
│   └── signal.py                      # moving_average, load_experiment_data
│
├── channel/
│   ├── __init__.py
│   └── diffusion.py                   # Brownian-motion simulator & template generator
│
├── bit_generation/
│   ├── __init__.py
│   └── bit_generator.py               # Random bit sequence generation & helpers
│
├── modulation/
│   ├── __init__.py
│   ├── alphabet.py                    # Permutations, combinations, sparse patterns
│   ├── encoder.py                     # Multi-molecule transmission signal builder
│   ├── permutation.py                 # Permutation MoCSK scheme pipeline
│   ├── combination_fixed.py           # Combination MoCSK with calibrated fixed threshold
│   └── sparse.py                      # Sparse pattern MoCSK with molecule exclusion
│
├── detection/
│   ├── __init__.py
│   ├── peak_decoder.py                # Smoothed-peak arrival order decoder
│   ├── threshold_decoder.py           # MoCSK decoder with a calibrated fixed threshold
│   └── sparse_decoder.py              # Slot-by-slot decoder with molecule lockout
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                     # Wilson score CI, accuracy comparison, methodology notes
│
├── experiments/
│   ├── __init__.py
│   └── sweep.py                       # Monte-Carlo SER/BER vs Ts sweep with Wilson CI
│
├── notebooks/
│   └── simulation.ipynb               # Clean interactive demonstration notebook
│
└── results/
    └── figures/                       # Publication-ready plots (SER vs Ts, etc.)
```

---

## Modulation & Detection Schemes

| Scheme | Carrier Symbol | Detection Mechanism | Key Strength |
|---|---|---|---|
| **Permutation MoCSK** (`permutation.py`) | Full permutation of $K$ molecule types | Joint assignment of molecule channels to release slots using local peak rises | Preserves release order |
| **MoCSK** (`combination_fixed.py`) | Ordered subset in the first $k$ slots, including the empty symbol | Joint assignment to slots using local peak rises and a channel-derived threshold | Supports variable-length ordered subsets |
| **E-MoCSK** (`sparse.py`) | Sparse slot and molecule patterns, including the all-empty symbol | Joint assignment with molecule columns and empty-slot dummy columns | Supports sparse patterns |

All three moving-average decoders use a shared local-rise score. The expected
channel peak response locates each slot window. The baseline is the minimum
smoothed signal in a window ending half a slot before that peak, and the
decision threshold is one half of the isolated-pulse channel peak. This
detector is limited when $\Delta=T_s/K$ is shorter than the pulse width because
adjacent pulses overlap; the discrete maximum-likelihood detector is intended
to address that regime and is not yet implemented here.

---

## Physical Parameters (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| $r_0$ | 10 µm | Transmitter–receiver center-to-center distance |
| $r_r$ | 5 µm | Spherical absorbing receiver radius |
| $D$ | 79.4 µm²/s | Diffusion coefficient |
| $\Delta t$ | 10⁻⁴ s | Simulation time step |
| $T$ | 5.0 s | Impulse response duration |
| $t_{\text{peak}}^{\text{theory}}$ | $\frac{(r_0 - r_r)^2}{6D} \approx 0.0525\text{ s}$ | Theoretical peak arrival time |
| `NORMALIZATION` | 10 | Binning factor for experiment data |
| `SMOOTHING_WINDOW` | 40 | Moving-average window for peak detection |

---

## Quick Start & Usage

### 1. Interactive Notebook
Launch Jupyter and open:
```
notebooks/simulation.ipynb
```
The notebook demonstrates each scheme individually, produces multi-panel diagnostic plots, and runs a comparative SER vs $T_s$ sweep.

### 2. Running a Scheme in Python
```python
from modulation import run_permutation, run_combination_fixed, run_sparse

# Run Permutation MoCSK
res_perm = run_permutation(
    bit_sequence="10110",
    num_molecule_types=5,
    delay_between_symbols=2.5,
    plot=True
)

# Run Combination MoCSK with fixed calibrated threshold
res_combo = run_combination_fixed(
    bit_sequence="10110",
    num_molecule_types=5,
    delay_between_symbols=2.5,
    plot=True
)
```

### 3. Running Monte Carlo SER and Goodput Sweeps
```bash
python -m experiments.sweep
```
This runs the requested two families of plots for $K=2,\ldots,10$ comparing the three implemented schemes:

- **SER and goodput vs. $T_s$:** $T_s=0.2,0.3,\ldots,1.0$ s for every $K=2,\ldots,10$, with $N_{\text{Tx}}=500$ held fixed.
- **SER and BER-approximate goodput vs. $N_{\text{Tx}}$:** $N_{\text{Tx}}=100,200,\ldots,1000$, with $T_s=1.0$ s for every $K$.
- For the sweeps, releases use uniform spacing $\Delta=T_s/K$, including across consecutive symbols; moving-average width shrinks when slots are close. The receiver decodes with the expected channel peak delay, so decisions can use a short look-ahead into the next interval.
- Pools symbol errors across Monte Carlo trials and calculates 95% Wilson intervals.
- The sweep transmits one modulation symbol in every symbol period, with the empty symbol represented explicitly in the MoCSK/E-MoCSK alphabets. It reports SER and uses the uniform-symbol-error BER approximation for the BER-approximate goodput.
- The `ber_approx` CSV field and `goodput_bits_per_second` use the uniform-symbol-error formula; they are estimates, not bitwise BER measurements.
- Saves each vector PDF to `results/figures/ser_goodput_vs_ts.pdf` and `results/figures/ser_goodput_vs_n_tx.pdf`. Each PDF has a SER page and a BER-approximate goodput page, with one panel per molecule count. Pooled point-by-point data and confidence intervals are saved to `results/sweep_results.csv`.

For a quicker exploratory run, lower the trial, sequence, and template counts:
```bash
python -m experiments.sweep --trials 2 --symbols 20 --templates 10
```

To reuse existing channel templates without generating missing files, pass their parent directory, restrict the NTx sweep to available `N...` folders, and enable reuse-only mode. The Ts sweep also requires the folder given by `--ts-n-tx`. For example, if the prior templates are in `../modulation_research/netlab/custom`:
```bash
python -m experiments.sweep --exp-root "../modulation_research/netlab/custom" --n-tx-values 100,500,1000 --templates auto --molecule-types 5 --trials 2 --symbols 20 --reuse-only
```
Reuse-only mode fails instead of silently generating templates if a requested folder is missing or has too few CSV files. Existing folders contain channel templates, not precomputed SER/goodput points, so the modulation and decoding trials still need to run before the PDFs and summary CSV can be plotted.

The default full run can take a long time because it covers many configurations. The permutation, MoCSK, and sparse encoders sample directly from their alphabets rather than materializing every possible symbol. MoSK, BCSK, and the paper's discrete maximum-likelihood detector are not implemented yet.

### Progressive validation

Run the checks from cheapest to most expensive. These use the standard library's
`unittest`, so pytest is not required.

1. Alphabet sizes, exact bit counts, and K=3 sampler uniformity:

   ```bash
   .venv/bin/python -m unittest tests.test_alphabet -v
   ```

2. Noise-free decoder grid using the mean of 100 N1000 channel templates. It
   covers K=2..10 and Ts=0.2..1.0 and prints one SER grid per scheme:

   ```bash
   .venv/bin/python -m unittest tests.test_noise_free -v
   ```

3. For a smaller noisy-channel smoke run before the full regression, use
   `--molecule-types 5 --n-tx-values 500 --ts-values 0.6,0.8,1.0 --trials 1 --symbols 500 --reuse-only`.
   Then raise trials and symbols to at least 10 × 200 (2,000 symbols per point)
   before interpreting sub-1% SER. Use `--exp-root` if templates are outside
   `../custom`.

   For the fixed-seed K=5 noisy regression, threshold-fraction ablation, and
   slot-0 miss-rate split, run:

   ```bash
   .venv/bin/python -m unittest tests.test_noisy_regression -v
   ```

4. After generating a sweep with the current code, validate its CSV and
   metadata:

   ```bash
   .venv/bin/python -m unittest tests.test_saved_results -v
   ```

   It checks the current valley-scoring metadata, expected row count, complete
   pooled symbol counts, finite fields, and reports adjacent points with
   non-overlapping Wilson intervals. Older CSVs and logs should be regenerated.

5. Scale up only after these pass: start with K=2,5,10; Ts=0.2,0.6,1.0;
   N_Tx=100,500,1000; and 2 trials × 50 symbols. Then run the full grid. Keep
   the result metadata and detail log with the generated CSV and PDFs.
