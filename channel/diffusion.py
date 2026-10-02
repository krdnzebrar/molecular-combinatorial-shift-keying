"""
channel/diffusion.py
====================
Brownian-motion diffusion channel simulator.

Provides two simulation approaches:
  1. Indexed Brownian hit simulator (from molecule_sim / bit_generator)
     - Saves per-molecule first-hit times for use with MoSK/MoCSK decoders.
  2. Experiment template generator (from Custom_Simulator / mc_modulation_ser_sweep)
     - Saves per-step aggregate hit counts as CSV templates.

Both share the same underlying physics: 3-D Brownian motion of point
molecules toward a fully-absorbing spherical receiver.
"""

import os
import csv
import math
import time

import numpy as np
import torch
from scipy.special import erfc, erfcinv


# ============================================================
# DEFAULT PHYSICAL PARAMETERS
# ============================================================

DEFAULT_RADIUS = 5.0
DEFAULT_DISTANCE = 12.5
DEFAULT_DIFFUSION_COEF = 79.4
DEFAULT_STEP_TIME = 1e-4
DEFAULT_TOTAL_TIME = 5.0          # impulse-response horizon


# ============================================================
# DEVICE SELECTION
# ============================================================

def get_device():
    """Pick the fastest available torch backend: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# (1) INDEXED BROWNIAN HIT SIMULATOR
#     Saves per-molecule (idx, t) first-hit events.
# ============================================================

def segment_sphere_first_tau(p0, p1, radius):
    """
    First-intersection time parameter τ of the line segment

        p(τ) = p0 + τ·(p1 − p0),   τ ∈ [0, 1]

    with a sphere of the given radius centred at the origin.
    """
    d = p1 - p0

    a = (d * d).sum(dim=1)
    b = 2.0 * (p0 * d).sum(dim=1)
    c = (p0 * p0).sum(dim=1) - radius * radius

    disc = b * b - 4.0 * a * c
    disc = torch.clamp(disc, min=0.0)

    sqrt_disc = torch.sqrt(disc)

    tau = (-b - sqrt_disc) / (2.0 * torch.clamp(a, min=1e-12))
    tau = torch.clamp(tau, 0.0, 1.0)

    return tau


def simulate_indexed_hits(
    filename,
    radius=DEFAULT_RADIUS,
    total_time=DEFAULT_TOTAL_TIME,
    step_time=DEFAULT_STEP_TIME,
    diffusion_coef=DEFAULT_DIFFUSION_COEF,
    distance=DEFAULT_DISTANCE,
    nof_molecules=10_000_000,
    device=None,
    override=False,
    batch_size=500_000,
):
    """
    Simulates Brownian motion and saves only first-hit events.

    Saved fields (torch .pt file):
        idx           : emitted molecule index
        t             : first-hit time
        nof_molecules : total emitted molecules

    Unreceived molecules are not saved.
    """
    if device is None:
        device = get_device()

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    if os.path.exists(filename) and not override:
        print(f"File already exists, skipping simulation:")
        print(" ", filename)
        return filename

    radius_t = torch.as_tensor(radius, device=device, dtype=torch.float32)
    step_time_f = float(step_time)
    diffusion_coef_t = torch.as_tensor(diffusion_coef, device=device, dtype=torch.float32)
    distance_f = float(distance)
    total_time_f = float(total_time)

    nof_steps = int(math.ceil(total_time_f / step_time_f))
    sigma = torch.sqrt(2.0 * diffusion_coef_t * step_time_f)

    idx_out = []
    times_out = []

    total_hits = 0
    start_clock = time.time()

    print("\nStarting indexed Brownian simulation")
    print("filename      :", filename)
    print("nof_molecules :", int(nof_molecules))
    print("total_time    :", total_time_f)
    print("step_time     :", step_time_f)
    print("nof_steps     :", nof_steps)
    print("batch_size    :", int(batch_size))
    print("device        :", device)

    for start_idx in range(0, int(nof_molecules), int(batch_size)):
        end_idx = min(start_idx + int(batch_size), int(nof_molecules))
        n_chunk = end_idx - start_idx

        positions = torch.zeros((n_chunk, 3), device=device, dtype=torch.float32)
        positions[:, 2] = distance_f

        alive = torch.ones(n_chunk, device=device, dtype=torch.bool)

        mol_idx = torch.arange(
            start_idx,
            end_idx,
            device=device,
            dtype=torch.int64,
        )

        with torch.no_grad():
            for step in range(nof_steps):
                if not alive.any():
                    break

                prev = positions.clone()
                positions.add_(torch.randn_like(positions) * sigma)

                inside = positions.pow(2).sum(dim=1) <= radius_t * radius_t
                new_hits = inside & alive

                if new_hits.any():
                    tau = segment_sphere_first_tau(
                        prev[new_hits],
                        positions[new_hits],
                        radius_t,
                    )

                    t_hit = (float(step) + tau) * step_time_f

                    valid = t_hit < total_time_f

                    if valid.any():
                        idx_out.append(mol_idx[new_hits][valid].detach().cpu())
                        times_out.append(t_hit[valid].detach().cpu())

                        total_hits += int(valid.sum().item())

                    alive[new_hits] = False

                if step % 5000 == 0:
                    elapsed = time.time() - start_clock
                    print(
                        f"chunk {start_idx}:{end_idx}, "
                        f"step {step}/{nof_steps}, "
                        f"total_hits={total_hits}, "
                        f"alive_in_chunk={int(alive.sum().item())}, "
                        f"elapsed={elapsed:.1f}s"
                    )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if total_hits == 0:
        data = {
            "idx": torch.empty(0, dtype=torch.int64),
            "t": torch.empty(0, dtype=torch.float32),
            "nof_molecules": int(nof_molecules),
        }
    else:
        idx_cat = torch.cat(idx_out, dim=0).to(torch.int64)
        t_cat = torch.cat(times_out, dim=0).to(torch.float32)

        order = torch.argsort(idx_cat)

        data = {
            "idx": idx_cat[order],
            "t": t_cat[order],
            "nof_molecules": int(nof_molecules),
        }

    torch.save(data, filename)

    print("\nFinished simulation")
    print("Total received molecules saved:", total_hits)
    print("Total emitted molecules indexed:", int(nof_molecules))
    print("Saved to:", filename)

    return filename


def load_hits_file(hit_file):
    """
    Loads an indexed hit file (.pt).

    Returns dict with keys: t, idx, nof_molecules, has_xyz.
    """
    if not hit_file.endswith(".pt"):
        raise ValueError("hit_file must be a .pt file.")

    data = torch.load(hit_file, map_location="cpu")

    if "t" not in data:
        raise ValueError(f"{hit_file} must contain key 't'.")

    t = data["t"].detach().cpu().numpy().astype(np.float64)

    idx = None
    if "idx" in data:
        idx = data["idx"].detach().cpu().numpy().astype(np.int64)

    nof_molecules = None
    if "nof_molecules" in data:
        nof_molecules = int(data["nof_molecules"])

    return {
        "t": t,
        "idx": idx,
        "nof_molecules": nof_molecules,
        "has_xyz": "xyz" in data,
    }


# ============================================================
# (2) EXPERIMENT TEMPLATE GENERATOR
#     Saves aggregate per-step hit counts as CSV.
# ============================================================

def simulate_experiment_csv(
    exp_id,
    n_tx,
    exp_dir,
    radius=DEFAULT_RADIUS,
    total_time=DEFAULT_TOTAL_TIME,
    step_time=DEFAULT_STEP_TIME,
    diffusion_coef=DEFAULT_DIFFUSION_COEF,
    distance=DEFAULT_DISTANCE,
    device=None,
    override=False,
    method="brownian",
):
    """
    Brownian-motion simulation of `n_tx` molecules. Saves per-step
    aggregate hit count as Exp_xxx.csv.
    """
    if device is None:
        device = get_device()

    os.makedirs(exp_dir, exist_ok=True)
    filename = os.path.join(exp_dir, f"Exp_{exp_id:03d}.csv")

    if os.path.exists(filename) and not override:
        return filename

    n_steps = int(round(total_time / step_time))

    if method == "first-passage":
        if distance <= radius:
            raise ValueError("distance must exceed receiver radius")
        if step_time <= 0 or total_time <= 0 or diffusion_coef <= 0:
            raise ValueError("step_time, total_time, and diffusion_coef must be positive")

        # Exact first-arrival distribution for a point source and a perfectly
        # absorbing sphere in unbounded 3-D diffusion. This avoids advancing
        # every molecule through all 50,000 time steps.
        cdf_end = (radius / distance) * erfc(
            (distance - radius) / math.sqrt(4.0 * diffusion_coef * total_time)
        )
        uniforms = np.random.random(int(n_tx))
        received = uniforms < cdf_end
        if np.any(received):
            erfc_argument = uniforms[received] * distance / radius
            erfc_argument = np.maximum(erfc_argument, np.finfo(float).tiny)
            hit_times = ((distance - radius) ** 2) / (
                4.0 * diffusion_coef * erfcinv(erfc_argument) ** 2
            )
            # Record arrivals in the first discrete sample at or after the
            # continuous hitting time, matching the template's time grid.
            hit_bins = np.ceil(hit_times / step_time).astype(np.int64) - 1
            hit_bins = np.clip(hit_bins, 0, n_steps - 1)
            hit_counts = np.bincount(hit_bins, minlength=n_steps).astype(float)
        else:
            hit_counts = np.zeros(n_steps, dtype=float)

        time_axis = (np.arange(n_steps, dtype=float) + 1.0) * step_time
        output = np.column_stack((time_axis, hit_counts))
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Number of Molecules"])
            writer.writerows(output)
        return filename

    if method != "brownian":
        raise ValueError("method must be 'brownian' or 'first-passage'")

    positions = torch.zeros((int(n_tx), 3), device=device)
    positions[:, 2] = distance
    alive = torch.ones(int(n_tx), dtype=torch.bool, device=device)
    sigma = math.sqrt(2.0 * diffusion_coef * step_time)
    radius_sq = radius * radius

    hit_counts = torch.zeros(n_steps, device=device)

    with torch.no_grad():
        for step in range(n_steps):
            if not alive.any():
                break
            positions += torch.randn_like(positions) * sigma
            inside = positions.pow(2).sum(dim=1) <= radius_sq
            new_hits = inside & alive

            hit_counts[step] = new_hits.sum()
            alive[new_hits] = False

    time_axis = (torch.arange(n_steps, device=device, dtype=torch.float32) + 1.0) * step_time
    out_rows = torch.stack([time_axis, hit_counts], dim=1).cpu().tolist()

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Number of Molecules"])
        writer.writerows(out_rows)

    return filename


def ensure_experiment_templates(
    n_tx,
    exp_dir,
    n_exp=100,
    radius=DEFAULT_RADIUS,
    total_time=DEFAULT_TOTAL_TIME,
    step_time=DEFAULT_STEP_TIME,
    diffusion_coef=DEFAULT_DIFFUSION_COEF,
    distance=DEFAULT_DISTANCE,
    device=None,
    override=False,
    method="brownian",
):
    """
    Returns a directory of `n_exp` Exp_xxx.csv templates generated with
    exactly `n_tx` molecules per release event. Creates them if they
    don't already exist.
    """
    os.makedirs(exp_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(exp_dir)
        if f.startswith("Exp_") and f.endswith(".csv")
    ]

    if len(existing) >= n_exp and not override:
        print(f"[cache] N_Tx={n_tx}: using {len(existing)} existing templates in {exp_dir}")
        return exp_dir

    _dev = device if device is not None else get_device()
    print(f"[gen]   N_Tx={n_tx}: generating {n_exp} {method} templates in {exp_dir} (device={_dev}) ...")
    t0 = time.time()
    for i in range(n_exp):
        simulate_experiment_csv(
            exp_id=i,
            n_tx=n_tx,
            exp_dir=exp_dir,
            radius=radius,
            total_time=total_time,
            step_time=step_time,
            diffusion_coef=diffusion_coef,
            distance=distance,
            device=device,
            override=override,
            method=method,
        )
    print(f"[gen]   N_Tx={n_tx}: done in {time.time() - t0:.1f}s")
    return exp_dir


def load_experiment_data(exp_path, normalization=1):
    """
    Loads a single Exp_xxx.csv template and optionally bins the data.

    Parameters
    ----------
    exp_path : str
        Path to a CSV file with columns [Time, Number of Molecules].
    normalization : int
        Bin-averaging factor. If > 1, adjacent time steps are binned.

    Returns
    -------
    time_sec : np.ndarray
    hits : np.ndarray
    """
    data = np.genfromtxt(exp_path, delimiter=",", skip_header=1)
    time_sec = data[:, 0]
    hits = data[:, 1]

    if normalization > 1:
        usable_len = (len(time_sec) // normalization) * normalization
        time_binned = time_sec[:usable_len].reshape(-1, normalization).mean(axis=1)
        hits_binned = hits[:usable_len].reshape(-1, normalization).sum(axis=1)
        return time_binned, hits_binned
    return time_sec, hits
