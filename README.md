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

### 3. Running Monte Carlo SER vs $T_s$ Sweep
```bash
python -m experiments.sweep
```
Features:
- Sweeps $T_s \in [0.20, 0.40]\text{ s}$ and $N_{\text{Tx}} \in \{500, 1000\}$ molecules.
- Pools errors across all trials per point.
- Evaluates 95% Wilson score confidence intervals.
- Safely handles zero errors on logarithmic plots by showing 95% upper bound markers (`0/n`).
- Saves figures directly to `results/figures/ser_vs_ts_sweep.png`.
