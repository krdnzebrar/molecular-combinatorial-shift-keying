# Molecular Combinatorial Shift Keying (MoCSK)

A unified, modular research repository for Molecular Combinatorial Shift Keying (MoCSK) and enhanced multi-molecule communication schemes over diffusion channels.

---

## Architecture Overview

All schemes share the same physical constants (`config.py`), signal processing tools (`utils/signal.py`), channel models (`channel/diffusion.py`), and evaluation metrics (`evaluation/metrics.py`). There is **zero duplication** across modules.

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
│   ├── combination_adaptive.py        # Combination MoCSK with adaptive gap threshold
│   ├── combination_fixed.py           # Combination MoCSK with calibrated fixed threshold
│   └── sparse.py                      # Sparse pattern MoCSK with molecule exclusion
│
├── detection/
│   ├── __init__.py
│   ├── peak_decoder.py                # Smoothed-peak arrival order decoder
│   ├── threshold_decoder.py           # Adaptive gap and calibrated fixed threshold decoders
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
| **Permutation MoCSK** (`permutation.py`) | Full permutation of $K$ molecule types | Smoothed peak arrival order (`peak_decoder.py`) | High symbol entropy without requiring threshold calibration |
| **Combination MoCSK (Adaptive)** (`combination_adaptive.py`) | Subset of $K$ molecule alphabet | Dynamic threshold at largest inter-peak gap (`threshold_decoder.py`) | Self-adjusting to fluctuating channel conditions |
| **Combination MoCSK (Fixed)** (`combination_fixed.py`) | Subset of $K$ molecule alphabet | Calibrated global threshold: $\mu_{\text{absent}} + 0.30(\mu_{\text{present}} - \mu_{\text{absent}})$ | Prevents false positives caused by ISI tails across symbols |
| **Sparse Pattern MoCSK** (`sparse.py`) | Partial slot fill with molecule permutation | Slot-by-slot detection with molecule exclusion (`sparse_decoder.py`) | Eliminates tail re-detection; robust at high symbol rates |

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
This runs the requested two families of plots for $K=2,\ldots,10$ comparing all four implemented schemes:

- **SER and goodput vs. $T_s$:** $T_s=0.2,0.3,\ldots,1.0$ s for every $K=2,\ldots,10$, with $N_{\text{Tx}}=500$ held fixed.
- **SER and goodput vs. $N_{\text{Tx}}$:** $N_{\text{Tx}}=100,200,\ldots,1000$, with $T_s=1/K$ for each $K$.
- For the sweeps, releases use uniform spacing $\Delta=T_s/K$, including across consecutive symbols; moving-average width shrinks when slots are close. The receiver decodes with the expected channel peak delay, so decisions can use a short look-ahead into the next interval.
- Pools symbol errors across Monte Carlo trials and calculates 95% Wilson intervals.
- The sweep transmits one modulation symbol in every symbol period, with the empty symbol represented explicitly in the MoCSK/E-MoCSK alphabets. Goodput is $(1-\mathrm{SER})\log_2(M)/T_s$ bits/s. It is a SER-based estimate, not a BER measurement.
- Saves each vector PDF to `results/figures/ser_goodput_vs_ts.pdf` and `results/figures/ser_goodput_vs_n_tx.pdf`. Each PDF has a SER page and a SER-based goodput page, with one panel per molecule count so methods can be compared without overlaying all $K$ values. Pooled point-by-point data and confidence intervals are saved to `results/sweep_results.csv`.

For a quicker exploratory run, lower the trial, sequence, and template counts:
```bash
python -m experiments.sweep --trials 2 --symbols 20 --templates 10
```

To reuse existing channel templates without generating missing files, pass their parent directory, restrict the NTx sweep to available `N...` folders, and enable reuse-only mode. The Ts sweep also requires the folder given by `--ts-n-tx`. For example, if the prior templates are in `../modulation_research/netlab/custom`:
```bash
python -m experiments.sweep --exp-root "../modulation_research/netlab/custom" --n-tx-values 100,500,1000 --templates auto --molecule-types 5 --trials 2 --symbols 20 --reuse-only
```
Reuse-only mode fails instead of silently generating templates if a requested folder is missing or has too few CSV files. Existing folders contain channel templates, not precomputed SER/goodput points, so the modulation and decoding trials still need to run before the PDFs and summary CSV can be plotted.

The default full run can take a long time because it covers many configurations. The permutation and sparse encoders sample directly from their alphabets rather than materializing every possible symbol. The project still does not implement the paper's discrete maximum-likelihood detector or BER-based goodput; the plots compare the four existing moving-average/threshold-based scheme implementations and use SER-based goodput.
