"""
config.py
=========
Single source of truth for all physical constants and project paths.
Imported by every module — eliminates the 8× duplicated constant blocks.
"""

import os

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "custom"))

# ── Physical parameters (analytical model — used by decoders) ─────────────────
R0 = 10        # Tx–Rx center-to-center distance (µm)
RR = 5         # Receiver radius (µm)
D  = 79.4      # Diffusion coefficient (µm²/s)
T  = 5.0       # Impulse-response tail / total time horizon (s)
DT = 0.0001    # Simulation step time (s)

# ── Derived constants ─────────────────────────────────────────────────────────
T_PEAK_THEORY = (R0 - RR) ** 2 / (6 * D)   # ≈ 0.0525 s — expected peak time

# ── Signal processing defaults ────────────────────────────────────────────────
NORMALIZATION    = 10     # Bin-averaging factor for experiment CSVs
SMOOTHING_WINDOW = 40     # Moving-average window for decoder smoothing
