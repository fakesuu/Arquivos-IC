"""
niching_utils.py
-----------------
Shared, optimizer-agnostic niching primitives used by the niched variants
of BIPOP-CMA-ES, LM-CMA-ES (LM-MA-ES formulation) and L-SHADE:

    - clearing()   : Petrowski's Clearing algorithm (fitness sharing by
                      truncation within a niche radius)
    - speciate()    : greedy radius-based clustering, used for bookkeeping
                      / merging redundant search distributions
    - nearest_crowding_partner() : DE-style crowding replacement

All fitness values are assumed to be MINIMIZED.
"""
from __future__ import annotations
import numpy as np


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix. X: (n, dim)."""
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    return np.maximum(d2, 0.0)


def clearing(X: np.ndarray, fitness: np.ndarray, radius: float,
             capacity: int = 1) -> np.ndarray:
    """
    Petrowski's clearing algorithm.

    Individuals are processed best-to-worst. Within `radius` of a given
    "dominant" individual, only the best `capacity` individuals keep their
    true fitness; the rest are cleared (effective fitness -> +inf), so
    they lose selection pressure inside that niche and the population is
    pushed to spread across multiple basins.

    Returns a NEW array of effective fitness values; originals untouched.
    """
    n = len(fitness)
    order = np.argsort(fitness)  # best first (minimization)
    cleared = fitness.copy().astype(float)
    alive_count = np.zeros(n, dtype=int)
    d2 = pairwise_sq_dists(X)
    r2 = radius * radius

    dominant_idx: list[int] = []
    for i in order:
        if cleared[i] == np.inf:
            continue
        claimed = False
        for j in dominant_idx:
            if d2[i, j] <= r2:
                claimed = True
                if alive_count[j] < capacity:
                    alive_count[j] += 1
                else:
                    cleared[i] = np.inf
                break
        if not claimed:
            dominant_idx.append(i)
            alive_count[i] = 1
    return cleared


def speciate(X: np.ndarray, fitness: np.ndarray, radius: float) -> list[list[int]]:
    """
    Greedy radius-based speciation for bookkeeping / merge decisions.
    Returns species as lists of indices into X, best-seed-first.
    """
    order = np.argsort(fitness)
    d2 = pairwise_sq_dists(X)
    r2 = radius * radius
    seeds: list[int] = []
    species: list[list[int]] = []
    for i in order:
        best_seed, best_d = -1, np.inf
        for k, s in enumerate(seeds):
            if d2[i, s] <= r2 and d2[i, s] < best_d:
                best_seed, best_d = k, d2[i, s]
        if best_seed == -1:
            seeds.append(i)
            species.append([i])
        else:
            species[best_seed].append(i)
    return species


def nearest_crowding_partner(x: np.ndarray, pop: np.ndarray,
                              candidate_idx: np.ndarray) -> int:
    """Index (into `pop`) of the member closest to `x` among `candidate_idx`.
    Used for DE/L-SHADE crowding replacement (a child only competes with
    its nearest neighbours, not the whole population)."""
    d = np.sum((pop[candidate_idx] - x[None, :]) ** 2, axis=1)
    return int(candidate_idx[np.argmin(d)])


def reflect_into_bounds(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Reflective ("bounce") boundary repair, applied in place of clipping.

    Clipping collapses every out-of-bounds sample onto the boundary
    itself, which (a) piles up probability mass exactly on the edge and
    (b) for CMA-ES/LM-MA-ES in particular, silently corrupts the
    mean/step-size feedback the next generation's adaptation relies on,
    since the "effective" step actually taken no longer matches the step
    the search distribution generated. Reflection instead folds the
    excess distance back into the box, so points that overshoot land
    somewhere plausible inside the domain instead of stacking at lo/hi.

    Implemented as a triangle-wave fold via `mod`, so a point that
    overshoots by more than the box width (common with a large CMA-ES
    sigma near the start of a run) bounces off both walls the right
    number of times in one vectorized pass, rather than being repaired
    with a single reflection that would still leave it out of bounds.

    X      : (..., dim) array of candidate points (any leading shape)
    bounds : (dim, 2) array of [lo, hi] per dimension
    """
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    width = hi - lo
    width = np.where(width <= 0, 1e-12, width)  # guard against degenerate/zero-width dims
    period = 2.0 * width
    y = np.mod(X - lo, period)                  # numpy's mod is already non-negative here
    folded = np.where(y > width, period - y, y)
    result = lo + folded
    return np.clip(result, lo, hi)               # float-precision safety net at the exact edges


def export_gen_log_csv(gen_log: list[dict], path: str) -> None:
    """
    Write a per-generation log (list of dicts, as accumulated in
    `.gen_log` by NichedBIPOPCMAES / NichedLMMAES / NichedLSHADE) to a
    CSV file. Nested dicts (per-regime or per-operator probabilities)
    are flattened into separate columns named '<key>_<subkey>', e.g.
    op_prob={'pbest1': 0.4, ...} becomes columns op_prob_pbest1, ...
    """
    import csv
    flat_rows = []
    fieldnames: list[str] = []
    for row in gen_log:
        flat = {}
        for k, v in row.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    flat[f"{k}_{sk}"] = sv
            else:
                flat[k] = v
        flat_rows.append(flat)
        for k in flat:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow(row)


def default_niche_radius(bounds: np.ndarray, dim: int, target_niches: int = 10) -> float:
    """Heuristic starting niche radius: a fraction of the search-space
    diagonal, shrinking with sqrt(dim) so high-D spaces don't get one
    giant niche. Treat this as a tunable hyperparameter, not a law."""
    span = np.mean(bounds[:, 1] - bounds[:, 0])
    return 0.5 * span / (target_niches ** (1.0 / max(dim, 1)))
