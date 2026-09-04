# GNBG niching evolutionary baselines — corrected NumPy 2 / IOHGNBG integration
# Algorithms: BIPOP-CMA-ES, LM-CMA-ES, L-SHADE
# Includes: niching, success-conditioned operator adaptation, reflective bounds,
# population-size reduction, FE/generation logging, GNBG integration, and
# target tracking at error <= 1e-8.
#
# GNBG integration deliberately bypasses iohgnbg.get_problem()/ioh.wrap_problem.
# IOHGNBG v0.0.2 loads the official GECCO_2025/fN.mat data and evaluates
# GNBG.fitness through the IOH C++ wrapper. This file uses the same official
# .mat fields and objective expression directly in NumPy, with explicit
# scalar extraction, avoiding the NumPy/pybind11 conversion failure.

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, Any

import numpy as np

Array = np.ndarray
EPS = 1e-30


def reflect_bounds(x: Array, lo: Array, hi: Array) -> Array:
    """Reflect arbitrary points into a closed hyper-rectangle.

    Reflection is applied component-wise and can handle arbitrarily large
    overshoots (multiple traversals of the interval), unlike simple clipping.
    This is used immediately before every objective evaluation/generation output
    so all three optimizers remain strictly feasible.
    """
    x = np.asarray(x, dtype=float).copy()
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        raise ValueError("bounds must be finite")
    if np.any(hi <= lo):
        raise ValueError("upper bounds must be strictly greater than lower bounds")
    if np.any(~np.isfinite(x)):
        raise ValueError("candidate contains NaN or infinite values")
    span = hi - lo
    y = np.mod(x - lo, 2.0 * span)
    y = np.where(y <= span, y, 2.0 * span - y)
    # Protect against floating-point roundoff at the endpoints.
    return np.minimum(np.maximum(lo + y, lo), hi)


def normalized_improvement(old_f: Array, new_f: Array) -> Array:
    """Stable, scale-aware improvement score in [roughly 0, 1]."""
    old_f = np.asarray(old_f, float)
    new_f = np.asarray(new_f, float)
    denom = np.maximum(np.abs(old_f), np.abs(new_f)) + 1e-12
    return np.maximum(0.0, (old_f - new_f) / denom)


@dataclass
class OperatorCredit:
    """Exponentially smoothed credit for a small operator portfolio."""

    n_ops: int
    learning_rate: float = 0.2
    exploration: float = 0.08
    credits: Array | None = None

    def __post_init__(self) -> None:
        if self.n_ops < 1:
            raise ValueError("n_ops must be positive")
        if self.credits is None:
            self.credits = np.ones(self.n_ops, dtype=float)
        else:
            self.credits = np.asarray(self.credits, float).copy()

    def probabilities(self) -> Array:
        """Softmax-like probabilities with a nonzero exploration floor."""
        c = self.credits - np.max(self.credits)
        p = np.exp(np.clip(c, -20.0, 20.0))
        p /= np.sum(p)
        p = (1.0 - self.exploration) * p + self.exploration / self.n_ops
        return p

    def sample(self, rng: np.random.Generator, n: int) -> Array:
        return rng.choice(self.n_ops, size=n, p=self.probabilities())

    def update(self, rewards: Array, counts: Array) -> None:
        rewards = np.asarray(rewards, float)
        counts = np.asarray(counts, int)
        for i in range(self.n_ops):
            if counts[i] > 0:
                mean_reward = rewards[i] / counts[i]
                self.credits[i] = (
                    (1.0 - self.learning_rate) * self.credits[i]
                    + self.learning_rate * mean_reward
                )


class NicheArchive:
    """Decision-space fitness-sharing archive; distance is normalized RMS."""

    def __init__(
        self,
        lower: Array,
        upper: Array,
        radius: float = 0.15,
        max_size: int = 500,
        pressure: float = 1.0,
    ):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.span = self.hi - self.lo
        self.radius = float(radius)
        self.max_size = int(max_size)
        self.pressure = float(pressure)
        self.X: list[Array] = []
        self.F: list[float] = []

    def norm(self, X: Array) -> Array:
        return (np.asarray(X) - self.lo) / self.span

    def rms_dist(self, X: Array, Y: Array) -> Array:
        Xn = self.norm(np.atleast_2d(X))
        Yn = self.norm(np.atleast_2d(Y))
        d2 = ((Xn[:, None, :] - Yn[None, :, :]) ** 2).mean(axis=2)
        return np.sqrt(d2)

    def update(self, X: Array, F: Array, max_add: Optional[int] = None) -> None:
        X = np.atleast_2d(X)
        F = np.asarray(F, float).ravel()
        order = np.argsort(F)
        nadd = len(order) if max_add is None else min(len(order), max_add)
        for idx in order[:nadd]:
            x, f = X[idx].copy(), float(F[idx])
            if not self.X:
                self.X.append(x)
                self.F.append(f)
                continue
            d = self.rms_dist(x, np.asarray(self.X))[0]
            j = int(np.argmin(d))
            if d[j] < self.radius:
                if f < self.F[j]:
                    self.X[j], self.F[j] = x, f
            else:
                self.X.append(x)
                self.F.append(f)
        self._trim()

    def _trim(self) -> None:
        if len(self.X) <= self.max_size:
            return
        E = np.asarray(self.X)
        F = np.asarray(self.F)
        # Keep both objective quality and spatial coverage when trimming.
        keep: list[int] = []
        remaining = np.ones(len(F), dtype=bool)
        while len(keep) < self.max_size:
            ids = np.flatnonzero(remaining)
            if len(keep) == 0:
                pick = int(ids[np.argmin(F[ids])])
            else:
                D = self.rms_dist(E[ids], E[np.asarray(keep)])
                nearest = D.min(axis=1)
                score = 0.7 * (F[ids] - F.min()) / (np.std(F) + 1e-12) - 0.3 * nearest
                pick = int(ids[np.argmin(score)])
            keep.append(pick)
            remaining[pick] = False
        self.X = [self.X[i] for i in keep]
        self.F = [self.F[i] for i in keep]

    def elites(self) -> tuple[Array, Array]:
        if not self.X:
            return np.empty((0, self.lo.size)), np.empty(0)
        o = np.argsort(self.F)
        return np.asarray([self.X[i] for i in o]), np.asarray([self.F[i] for i in o])

    def select(self, X: Array, F: Array, k: int) -> Array:
        """Greedy rank + sharing selection. Lower score is better."""
        X = np.atleast_2d(X)
        F = np.asarray(F, float).ravel()
        n = len(F)
        k = min(int(k), n)
        if k <= 0:
            return np.empty(0, dtype=int)
        rank_order = np.argsort(F)
        rank = np.empty(n, int)
        rank[rank_order] = np.arange(n)
        rnorm = rank / max(1, n - 1)
        chosen = [int(rank_order[0])]
        remaining = np.ones(n, dtype=bool)
        remaining[chosen[0]] = False
        while len(chosen) < k:
            ids = np.flatnonzero(remaining)
            d = self.rms_dist(X[ids], X[np.asarray(chosen)])
            sharing = np.maximum(0.0, 1.0 - d / max(self.radius, 1e-12)) ** 2
            crowd = sharing.sum(axis=1)
            score = rnorm[ids] + self.pressure * crowd
            pick = int(ids[np.argmin(score)])
            chosen.append(pick)
            remaining[pick] = False
        return np.asarray(chosen, int)

    def niche_reward(self, X: Array) -> Array:
        """Reward points that are far from the current niche archive."""
        X = np.atleast_2d(X)
        if not self.X:
            return np.ones(len(X))
        D = self.rms_dist(X, np.asarray(self.X))
        nearest = D.min(axis=1)
        return np.clip(nearest / max(self.radius, 1e-12), 0.0, 1.0)



def weighted_mean(X: Array, w: Array) -> Array:
    return np.sum(X * w[:, None], axis=0)


def linear_population_size(start: int, end: int, progress: float) -> int:
    """Linearly reduce population size as progress moves from 0 to 1."""
    start = int(start)
    end = int(end)
    if start < 1 or end < 1 or end > start:
        raise ValueError("population schedule requires 1 <= end <= start")
    p = float(np.clip(progress, 0.0, 1.0))
    return int(max(end, round(start + (end - start) * p)))


class CMAES:
    """Full-covariance CMA-ES with operator-credit adaptation and niching.

    Operators:
      0: covariance-shaped sampling (standard CMA-ES)
      1: isotropic sampling (global exploration)

    Operator credits are updated from objective improvement, selection frequency,
    and niche novelty. A small exploration floor prevents premature lock-in.
    """

    def __init__(
        self,
        x0: Array,
        sigma0: float,
        lam: int,
        lower: Array,
        upper: Array,
        niche: NicheArchive,
        rng: np.random.Generator,
        operator_exploration: float = 0.08,
        min_lam: Optional[int] = None,
    ):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.mean = reflect_bounds(np.asarray(x0, float), self.lo, self.hi)
        self.sigma = float(sigma0)
        self.lam = int(lam)
        self.start_lam = int(lam)
        self.min_lam = int(max(4, min_lam if min_lam is not None else max(4, int(math.ceil(0.35 * lam)))))
        self.min_lam = min(self.min_lam, self.start_lam)
        self.d = self.mean.size
        self.rng = rng
        self.niche = niche
        self._refresh_population_params()
        self.ps = np.zeros(self.d)
        self.pc = np.zeros(self.d)
        self.C = np.eye(self.d)
        self.B = np.eye(self.d)
        self.D = np.ones(self.d)
        self.BD = self.B * self.D[None, :]
        self.inv_sqrt_C = np.eye(self.d)
        self.eig_age = 0
        self.chiN = self.d**0.5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d**2))
        self.best_x = self.mean.copy()
        self.best_f = math.inf
        self.operator = OperatorCredit(
            n_ops=2,
            learning_rate=0.25,
            exploration=operator_exploration,
        )
        self.sigma_gain = 0.0

    def _refresh_population_params(self) -> None:
        """Recompute rank-based CMA parameters after a population resize."""
        self.mu = max(1, self.lam // 2)
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cc = (4 + self.mueff / self.d) / (self.d + 4 + 2 * self.mueff / self.d)
        self.cs = (self.mueff + 2) / (self.d + self.mueff + 5)
        self.c1 = 2 / ((self.d + 1.4142) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((self.d + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(
            0, math.sqrt((self.mueff - 1) / (self.d + 1)) - 1
        ) + self.cs

    def resize_population(self, new_lam: int) -> None:
        """Shrink the CMA population without ever increasing it."""
        new_lam = int(np.clip(new_lam, self.min_lam, self.lam))
        if new_lam < self.lam:
            self.lam = new_lam
            self._refresh_population_params()

    def ask(self, n: Optional[int] = None) -> tuple[Array, Array]:
        n = self.lam if n is None else int(n)
        if n < 1 or n > self.lam:
            raise ValueError("n must be in [1, lam]")
        ops = self.operator.sample(self.rng, n)
        z = self.rng.standard_normal((n, self.d))
        y_cov = z @ self.BD.T
        Y = y_cov.copy()
        isotropic = ops == 1
        Y[isotropic] = z[isotropic]
        X = reflect_bounds(self.mean + self.sigma * Y, self.lo, self.hi)
        return X, ops

    def tell(self, X: Array, F: Array, ops: Array) -> None:
        F = np.asarray(F, float)
        ops = np.asarray(ops, int)
        old_mean = self.mean.copy()
        old_best = self.best_f if np.isfinite(self.best_f) else float(np.max(F))

        self.niche.update(X, F, max_add=self.lam)
        ksel = min(self.mu, len(F))
        idx = self.niche.select(X, F, ksel)
        w = self.w[:len(idx)]
        w = w / np.sum(w)
        self.mean = weighted_mean(X[idx], w)
        y_w = (self.mean - old_mean) / max(self.sigma, 1e-12)

        vals, B = np.linalg.eigh(self.C)
        vals = np.maximum(vals, 1e-20)
        self.B = B
        self.D = np.sqrt(vals)
        self.BD = self.B * self.D[None, :]
        self.inv_sqrt_C = (self.B / self.D[None, :]) @ self.B.T

        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * (self.inv_sqrt_C @ y_w)
        hsig = float(
            np.linalg.norm(self.ps)
            / math.sqrt(1 - (1 - self.cs) ** (2 * (self.eig_age + 1)))
            / self.chiN
            < (1.4 + 2 / (self.d + 1))
        )
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * y_w

        Y = (X[idx] - old_mean) / max(self.sigma, 1e-12)
        rank_mu = np.zeros_like(self.C)
        for wi, yi in zip(w, Y):
            rank_mu += wi * np.outer(yi, yi)
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * np.outer(self.pc, self.pc)
            + self.cmu * rank_mu
        )

        # Standard CSA feedback plus explicit success feedback. If the selected
        # offspring improve strongly, allow larger steps; if they stagnate, shrink.
        median_f = float(np.median(F))
        selected_reward = float(np.mean(normalized_improvement(np.full(len(idx), median_f), F[idx])))
        path_term = np.linalg.norm(self.ps) / self.chiN - 1.0
        self.sigma_gain = 0.85 * self.sigma_gain + 0.15 * (path_term + 2.0 * selected_reward)
        self.sigma *= math.exp((self.cs / self.damps) * path_term + 0.10 * self.sigma_gain)
        self.sigma = float(np.clip(self.sigma, 1e-12, 0.5 * np.max(self.hi - self.lo)))

        # Operator feedback: reward actual selected improvements and niche novelty.
        rewards = np.zeros(2, dtype=float)
        counts = np.zeros(2, dtype=int)
        novelty = self.niche.niche_reward(X)
        base_ref = np.median(F)
        for i in range(self.lam):
            if i in idx:
                reward = float(max(0.0, (base_ref - F[i]) / (abs(base_ref) + 1e-12)))
                reward = 0.5 * reward + 0.5 * float(novelty[i])
                rewards[ops[i]] += reward
                counts[ops[i]] += 1
        self.operator.update(rewards, counts)

        j = int(np.argmin(F))
        if F[j] < self.best_f:
            self.best_f = float(F[j])
            self.best_x = X[j].copy()
        self.eig_age += 1

        # Mild success-driven pressure on the isotropic operator when no global
        # progress was made, without allowing it to dominate covariance learning.
        gen_improvement = max(0.0, (old_best - float(np.min(F))) / (abs(old_best) + 1e-12))
        if gen_improvement < 1e-8:
            self.operator.credits[1] += 0.10


class BIPOPCMAES:
    """BIPOP-style restarts with adaptive regime credit and niche archive."""

    def __init__(
        self,
        x0: Array,
        sigma0: float,
        lower: Array,
        upper: Array,
        niche_radius: float = 0.15,
        seed: int = 0,
    ):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.d = self.lo.size
        self.rng = np.random.default_rng(seed)
        self.x0 = reflect_bounds(np.asarray(x0, float), self.lo, self.hi)
        self.sigma0 = float(sigma0)
        self.niche = NicheArchive(
            self.lo, self.hi, radius=niche_radius, max_size=1000, pressure=0.8
        )
        # Restart-regime operators:
        # 0 = large-population / broad sigma, 1 = small-population / local restart.
        self.regime = OperatorCredit(2, learning_rate=0.30, exploration=0.15)
        self.restart = 0
        self.large_k = 0

    def minimize(
        self,
        fun: Callable[[Array], float],
        budget: int,
        generation_logger: Optional[GenerationLogger] = None,
        optimum: float = 0.0,
        target: float = 1e-8,
    ) -> tuple[Array, float, NicheArchive]:
        evals = 0
        best_x = self.x0.copy()
        best_f = math.inf
        base = max(4, 4 + int(3 * math.log(self.d + 1)))

        while evals < budget:
            regime = int(self.rng.choice(2, p=self.regime.probabilities()))
            before_best = best_f
            before_niches = len(self.niche.X)

            if regime == 0:
                lam = base * (2 ** self.large_k)
                sigma = self.sigma0 * (2.0 ** self.large_k)
                self.large_k += 1
                mean = self.x0.copy()
                if self.niche.X and self.restart > 0:
                    E, _ = self.niche.elites()
                    D = self.niche.rms_dist(E, np.asarray([self.x0]))[:, 0]
                    far = np.flatnonzero(D > self.niche.radius)
                    if len(far):
                        mean = E[far[int(np.argmax(D[far]))]].copy()
            else:
                max_small = max(base * 2, base * (2 ** max(0, self.large_k - 1)))
                lam = int(
                    max(
                        4,
                        round(
                            math.exp(
                                self.rng.uniform(math.log(base), math.log(max_small))
                            )
                        ),
                    )
                )
                if self.niche.X:
                    E, _ = self.niche.elites()
                    mean = E[self.rng.integers(len(E))].copy()
                    mean += self.rng.normal(0, 0.1 * (self.hi - self.lo), size=self.d)
                    mean = reflect_bounds(mean, self.lo, self.hi)
                else:
                    mean = self.x0.copy()
                sigma = self.sigma0 * float(self.rng.lognormal(0.0, 0.5))

            lam = max(4, int(lam))
            min_lam = max(4, int(math.ceil(0.35 * lam)))
            opt = CMAES(
                mean, max(sigma, 1e-12), lam, self.lo, self.hi, self.niche, self.rng,
                min_lam=min_lam,
            )

            local_evals = 0
            local_budget = budget - evals
            initial_lam = lam
            ngen = 0
            while evals < budget and ngen < 2000:
                progress = local_evals / max(1, local_budget)
                target_lam = linear_population_size(initial_lam, min_lam, progress)
                opt.resize_population(target_lam)
                n_eval = min(opt.lam, budget - evals)
                eval_start = evals
                regime_probs = self.regime.probabilities()
                cma_probs = opt.operator.probabilities()
                X, ops = opt.ask(n_eval)
                X = reflect_bounds(X, self.lo, self.hi)
                F = np.asarray([fun(x) for x in X], float)
                evals += len(F)
                local_evals += len(F)
                opt.tell(X, F, ops)
                if opt.best_f < best_f:
                    best_f = opt.best_f
                    best_x = opt.best_x.copy()
                if generation_logger is not None:
                    generation_logger.record(
                        evaluation_start=eval_start,
                        evaluation_end=evals,
                        best_value=best_f,
                        optimum=optimum,
                        target=target,
                        operator_probabilities={
                            "restart_regime": {
                                "large_population": float(regime_probs[0]),
                                "small_population": float(regime_probs[1]),
                            },
                            "cma_operator": {
                                "covariance": float(cma_probs[0]),
                                "isotropic": float(cma_probs[1]),
                            },
                        },
                        operator_counts={
                            "cma_covariance": int(np.sum(ops == 0)),
                            "cma_isotropic": int(np.sum(ops == 1)),
                        },
                        metadata={
                            "restart_index": int(self.restart),
                            "restart_regime": int(regime),
                            "population_size": int(opt.lam),
                            "population_size_start": int(initial_lam),
                            "population_size_min": int(min_lam),
                            "population_reduction_progress": float(progress),
                            "sigma": float(opt.sigma),
                        },
                    )
                ngen += 1

            # Credit the restart regime by both objective improvement and creation
            # of spatially distinct niches. This is feedback across restart runs.
            improvement = 0.0
            if np.isfinite(before_best) and np.isfinite(best_f):
                improvement = max(0.0, (before_best - best_f) / (abs(before_best) + 1e-12))
            niche_gain = max(0, len(self.niche.X) - before_niches)
            reward = improvement + 0.10 * niche_gain / max(1, local_evals // max(1, lam))
            self.regime.credits[regime] = (
                0.70 * self.regime.credits[regime] + 0.30 * reward
            )
            self.restart += 1

        return best_x, best_f, self.niche


class LMCMAES:
    """Limited-memory CMA-ES with adaptive operator credits and niching.

    Operators:
      0: learned low-rank covariance transform
      1: isotropic exploration

    The low-rank update strength is also adapted from successful selected steps.
    """

    def __init__(
        self,
        x0: Array,
        sigma0: float,
        lam: int,
        lower: Array,
        upper: Array,
        niche: NicheArchive,
        memory: int = 10,
        seed: int = 0,
        operator_exploration: float = 0.08,
        min_lam: Optional[int] = None,
    ):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.mean = reflect_bounds(np.asarray(x0, float), self.lo, self.hi)
        self.sigma = float(sigma0)
        self.lam = int(lam)
        self.start_lam = int(lam)
        self.min_lam = int(max(4, min_lam if min_lam is not None else max(4, int(math.ceil(0.35 * lam)))))
        self.min_lam = min(self.min_lam, self.start_lam)
        self.d = self.mean.size
        self.rng = np.random.default_rng(seed)
        self.niche = niche
        self.m = int(max(1, memory))
        self._refresh_population_params()
        self.ps = np.zeros(self.d)
        self.pc = np.zeros(self.d)
        self.hist: list[Array] = []
        self.a = np.zeros(0)
        self.U = np.empty((self.d, 0))
        self.chiN = self.d**0.5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d**2))
        self.best_x = self.mean.copy()
        self.best_f = math.inf
        self.operator = OperatorCredit(2, learning_rate=0.25, exploration=operator_exploration)
        self.cov_strength = 0.20
        self.sigma_gain = 0.0

    def _refresh_population_params(self) -> None:
        self.mu = max(1, self.lam // 2)
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        self.cs = (self.mueff + 2) / (self.d + self.mueff + 5)
        self.cc = (4 + self.mueff / self.d) / (self.d + 4 + 2 * self.mueff / self.d)
        self.damps = 1 + 2 * max(
            0, math.sqrt((self.mueff - 1) / (self.d + 1)) - 1
        ) + self.cs

    def resize_population(self, new_lam: int) -> None:
        new_lam = int(np.clip(new_lam, self.min_lam, self.lam))
        if new_lam < self.lam:
            self.lam = new_lam
            self._refresh_population_params()

    def _A(self, Z: Array, inverse: bool = False) -> Array:
        if self.U.shape[1] == 0:
            return Z
        coeff = (
            1 / np.sqrt(1 + self.a) - 1
            if inverse
            else np.sqrt(1 + self.a) - 1
        )
        return Z + (Z @ self.U) * coeff[None, :] @ self.U.T

    def ask(self, n: Optional[int] = None) -> tuple[Array, Array]:
        n = self.lam if n is None else int(n)
        if n < 1 or n > self.lam:
            raise ValueError("n must be in [1, lam]")
        ops = self.operator.sample(self.rng, n)
        Z = self.rng.standard_normal((n, self.d))
        Y = self._A(Z, False)
        Y[ops == 1] = Z[ops == 1]
        X = reflect_bounds(self.mean + self.sigma * Y, self.lo, self.hi)
        return X, ops

    def tell(self, X: Array, F: Array, ops: Array) -> None:
        F = np.asarray(F, float)
        ops = np.asarray(ops, int)
        old = self.mean.copy()
        old_best = self.best_f if np.isfinite(self.best_f) else float(np.max(F))

        self.niche.update(X, F, max_add=self.lam)
        ksel = min(self.mu, len(F))
        idx = self.niche.select(X, F, ksel)
        w = self.w[:len(idx)]
        w = w / np.sum(w)
        self.mean = weighted_mean(X[idx], w)
        yw = (self.mean - old) / max(self.sigma, 1e-12)

        inv_y = self._A(yw[None, :], True)[0]
        self.ps = (1 - self.cs) * self.ps + math.sqrt(
            self.cs * (2 - self.cs) * self.mueff
        ) * inv_y
        hsig = float(
            np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** 2) / self.chiN
            < 1.4 + 2 / (self.d + 1)
        )
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(
            self.cc * (2 - self.cc) * self.mueff
        ) * yw

        u = self.pc.copy()
        nu = np.linalg.norm(u)
        if nu > 1e-12:
            self.hist.append(u / nu)
        self.hist = self.hist[-self.m :]

        if self.hist:
            M = np.column_stack(self.hist)
            Q, _ = np.linalg.qr(M, mode="reduced")
            self.U = Q
            Y = (X[idx] - old) / max(self.sigma, 1e-12)
            proj = Y @ self.U
            var = np.sum(w[:, None] * (proj**2), axis=0)
            target = np.clip(var - 1.0, -0.8, 4.0)
            if self.a.size != target.size:
                self.a = target.copy()
            else:
                # cov_strength is itself learned from successful steps.
                self.a = (1 - self.cov_strength) * self.a + self.cov_strength * target

        median_f = float(np.median(F))
        success = normalized_improvement(np.full(len(idx), median_f), F[idx])
        mean_success = float(np.mean(success))
        path_term = np.linalg.norm(self.ps) / self.chiN - 1.0
        self.sigma_gain = 0.85 * self.sigma_gain + 0.15 * (path_term + 2 * mean_success)
        self.sigma *= math.exp((self.cs / self.damps) * path_term + 0.10 * self.sigma_gain)
        self.sigma = float(np.clip(self.sigma, 1e-12, 0.5 * np.max(self.hi - self.lo)))

        # Successful covariance-shaped selected steps increase covariance learning;
        # unsuccessful generations shift toward isotropic exploration.
        selected_ops = ops[idx]
        if np.any(selected_ops == 0):
            cov_reward = float(np.mean(success[selected_ops == 0]))
        else:
            cov_reward = 0.0
        if np.any(selected_ops == 1):
            iso_reward = float(np.mean(success[selected_ops == 1]))
        else:
            iso_reward = 0.0
        self.cov_strength = float(np.clip(
            0.90 * self.cov_strength + 0.10 * (0.15 + 1.5 * cov_reward - 0.75 * iso_reward),
            0.03,
            0.50,
        ))

        rewards = np.zeros(2, float)
        counts = np.zeros(2, int)
        novelty = self.niche.niche_reward(X)
        for i in idx:
            reward = 0.5 * float(normalized_improvement(np.array([median_f]), np.array([F[i]]))[0])
            reward += 0.5 * float(novelty[i])
            rewards[ops[i]] += reward
            counts[ops[i]] += 1
        self.operator.update(rewards, counts)

        if old_best <= float(np.min(F)) + 1e-12:
            self.operator.credits[1] += 0.08

        j = int(np.argmin(F))
        if F[j] < self.best_f:
            self.best_f = float(F[j])
            self.best_x = X[j].copy()

    def minimize(
        self,
        fun: Callable[[Array], float],
        budget: int,
        generation_logger: Optional[GenerationLogger] = None,
        optimum: float = 0.0,
        target: float = 1e-8,
    ) -> tuple[Array, float, NicheArchive]:
        evals = 0
        initial_lam = self.lam
        while evals < budget:
            progress = evals / max(1, budget)
            target_lam = linear_population_size(initial_lam, self.min_lam, progress)
            self.resize_population(target_lam)
            n_eval = min(self.lam, budget - evals)
            eval_start = evals
            op_probs = self.operator.probabilities()
            X, ops = self.ask(n_eval)
            X = reflect_bounds(X, self.lo, self.hi)
            F = np.asarray([fun(x) for x in X], float)
            evals += len(F)
            self.tell(X, F, ops)
            if generation_logger is not None:
                generation_logger.record(
                    evaluation_start=eval_start,
                    evaluation_end=evals,
                    best_value=self.best_f,
                    optimum=optimum,
                    target=target,
                    operator_probabilities={
                        "covariance": float(op_probs[0]),
                        "isotropic": float(op_probs[1]),
                    },
                    operator_counts={
                        "covariance": int(np.sum(ops == 0)),
                        "isotropic": int(np.sum(ops == 1)),
                    },
                    metadata={
                        "population_size": int(self.lam),
                        "population_size_start": int(initial_lam),
                        "population_size_min": int(self.min_lam),
                        "population_reduction_progress": float(progress),
                        "sigma": float(self.sigma),
                        "cov_strength": float(self.cov_strength),
                    },
                )
        return self.best_x, self.best_f, self.niche


class LSHADE:
    """Niching L-SHADE with strategy-specific success-history adaptation.

    Strategies:
      0: current-to-pbest/1 + archive
      1: rand/1 + archive
      2: current-to-rand/1 + archive

    Each strategy maintains its own F/CR memory. Strategy credits are updated from
    successful improvement and niche novelty, and are used for future sampling.
    The p-best fraction is also adapted from recent success feedback.
    """

    def __init__(
        self,
        lower: Array,
        upper: Array,
        max_pop: int = 100,
        min_pop: int = 4,
        H: int = 10,
        niche_radius: float = 0.15,
        seed: int = 0,
    ):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        # Validate bounds once and keep a dimension-consistent representation.
        if self.lo.ndim != 1 or self.hi.shape != self.lo.shape:
            raise ValueError("lower and upper must be 1-D arrays of equal shape")
        if np.any(~np.isfinite(self.lo)) or np.any(~np.isfinite(self.hi)) or np.any(self.hi <= self.lo):
            raise ValueError("bounds must be finite with upper > lower")
        self.d = len(self.lo)
        self.max_pop = int(max_pop)
        self.min_pop = int(min_pop)
        self.H = int(H)
        self.rng = np.random.default_rng(seed)
        self.niche = NicheArchive(
            self.lo, self.hi, radius=niche_radius, max_size=1000, pressure=0.8
        )
        self.archive_X: list[Array] = []
        self.archive_F: list[float] = []
        self.n_strategies = 3
        self.strategy = OperatorCredit(
            n_ops=self.n_strategies,
            learning_rate=0.25,
            exploration=0.08,
        )
        self.MF = np.full((self.n_strategies, self.H), 0.5, dtype=float)
        self.MCR = np.full((self.n_strategies, self.H), 0.5, dtype=float)
        self.memory_ptr = np.zeros(self.n_strategies, dtype=int)
        self.p_rate = 0.20
        self.p_gain = 0.0

    def _sample_F_CR(self, strategy: int) -> tuple[float, float]:
        r = int(self.rng.integers(self.H))
        for _ in range(50):
            Fi = float(self.MF[strategy, r] + 0.1 * self.rng.standard_cauchy())
            if Fi > 0:
                break
        Fi = float(np.clip(Fi, 1e-6, 1.0))
        CRi = float(np.clip(
            self.MCR[strategy, r] + 0.1 * self.rng.standard_normal(),
            0.0,
            1.0,
        ))
        return Fi, CRi

    def _pick_distinct(self, N: int, exclude: set[int], pool_len: int) -> int:
        candidates = np.asarray([i for i in range(pool_len) if i not in exclude], dtype=int)
        if len(candidates) == 0:
            candidates = np.asarray([i for i in range(pool_len) if i != next(iter(exclude))], dtype=int)
        return int(self.rng.choice(candidates))

    def _mutate(
        self,
        strategy: int,
        i: int,
        pbest: int,
        P: Array,
        pool: Array,
        Fi: float,
    ) -> Array:
        N = len(P)
        if strategy == 0:
            exclude = {i, pbest}
            r1 = self._pick_distinct(N, exclude, N)
            r2 = self._pick_distinct(len(pool), {r1, i, pbest}, len(pool))
            x2 = pool[r2]
            v = P[i] + Fi * (P[pbest] - P[i]) + Fi * (P[r1] - x2)
        elif strategy == 1:
            r1 = self._pick_distinct(N, {i}, N)
            r2 = self._pick_distinct(len(pool), {i, r1}, len(pool))
            r3 = self._pick_distinct(len(pool), {i, r1, r2}, len(pool))
            v = pool[r1] + Fi * (pool[r2] - pool[r3])
        else:
            # current-to-rand/1 with three distinct donors from P ∪ archive.
            # Keep the target out of the donor pool when it comes from P.
            r1 = self._pick_distinct(len(pool), {i}, len(pool))
            r2 = self._pick_distinct(len(pool), {i, r1}, len(pool))
            r3 = self._pick_distinct(len(pool), {i, r1, r2}, len(pool))
            K = float(self.rng.random())
            v = P[i] + K * (pool[r1] - P[i]) + Fi * (pool[r2] - pool[r3])
        return reflect_bounds(v, self.lo, self.hi)

    def _update_memory(
        self,
        strategy: int,
        F_success: Array,
        CR_success: Array,
        improvements: Array,
    ) -> None:
        if len(F_success) == 0:
            return
        w = improvements / (np.sum(improvements) + EPS)
        mf = np.sum(w * F_success**2) / (np.sum(w * F_success) + EPS)
        mcr = np.sum(w * CR_success)
        p = self.memory_ptr[strategy]
        self.MF[strategy, p] = float(np.clip(mf, 0.0, 1.0))
        self.MCR[strategy, p] = float(np.clip(mcr, 0.0, 1.0))
        self.memory_ptr[strategy] = (p + 1) % self.H

    def minimize(
        self,
        fun: Callable[[Array], float],
        budget: int,
        init: Optional[Array] = None,
        generation_logger: Optional[GenerationLogger] = None,
        optimum: float = 0.0,
        target: float = 1e-8,
    ) -> tuple[Array, float, NicheArchive]:
        if budget < self.min_pop:
            raise ValueError(f"budget must be >= min_pop ({self.min_pop})")

        N = max(self.min_pop, min(self.max_pop, budget))
        if init is None:
            P = self.rng.uniform(self.lo, self.hi, size=(N, self.d))
        else:
            P = np.asarray(init, float).copy()
            if len(P) != N:
                raise ValueError("init must have shape (population_size, dimension)")
            P = reflect_bounds(P, self.lo, self.hi)

        F = np.asarray([fun(x) for x in P], float)
        evals = len(F)
        self.niche.update(P, F, max_add=N)
        best_f_so_far = float(np.min(F))
        initial_N = N
        if generation_logger is not None:
            generation_logger.record(
                evaluation_start=0,
                evaluation_end=evals,
                best_value=best_f_so_far,
                optimum=optimum,
                target=target,
                operator_probabilities={
                    "current_to_pbest_1": float(self.strategy.probabilities()[0]),
                    "rand_1": float(self.strategy.probabilities()[1]),
                    "current_to_rand_1": float(self.strategy.probabilities()[2]),
                },
                operator_counts={
                    "initial_population": int(evals),
                },
                metadata={
                    "phase": "initial_population",
                    "population_size": int(N),
                    "p_rate": float(self.p_rate),
                },
            )
        last_success_rate = 0.0

        while evals < budget and N >= self.min_pop:
            oldP = P.copy()
            oldF = F.copy()
            M = min(N, budget - evals)
            eval_start = evals
            strategy_probs = self.strategy.probabilities()
            active = np.arange(M, dtype=int)
            pbest_count = max(2, int(math.ceil(self.p_rate * N)))
            pbest_count = min(pbest_count, N)
            pbest_pool = np.argsort(F)[:pbest_count]
            pbest_sel = self.niche.select(P[pbest_pool], F[pbest_pool], max(1, min(pbest_count, pbest_count // 2 + 1)))

            trials = []
            sampled_F = np.empty(N)
            sampled_CR = np.empty(N)
            sampled_strategy = np.empty(N, dtype=int)
            pool = np.concatenate([P, np.asarray(self.archive_X)]) if self.archive_X else P

            for i in active:
                s = int(self.rng.choice(self.n_strategies, p=strategy_probs))
                Fi, CRi = self._sample_F_CR(s)
                pbest = int(pbest_pool[int(self.rng.choice(pbest_sel))])
                v = self._mutate(s, i, pbest, P, pool, Fi)
                ui = P[i].copy()
                jrand = int(self.rng.integers(self.d))
                mask = self.rng.random(self.d) < CRi
                mask[jrand] = True
                ui[mask] = v[mask]
                # Crossover is between feasible parent/mutant coordinates, but
                # reflect again as a final invariant for future operator changes.
                trials.append(reflect_bounds(ui, self.lo, self.hi))
                sampled_F[i] = Fi
                sampled_CR[i] = CRi
                sampled_strategy[i] = s

            T = reflect_bounds(np.asarray(trials), self.lo, self.hi)
            FT = np.asarray([fun(x) for x in T], float)
            evals += len(FT)
            best_f_so_far = min(best_f_so_far, float(np.min(FT)))
            self.niche.update(T, FT, max_add=min(len(T), N))

            parentF = oldF[active]
            success = FT < parentF
            success_idx = np.flatnonzero(success)
            last_success_rate = float(np.mean(success))

            # Success-history and strategy credit are operator-specific.
            for s in range(self.n_strategies):
                ids = success_idx[sampled_strategy[success_idx] == s]
                if len(ids) == 0:
                    self.strategy.credits[s] *= 0.97
                    continue
                imp = parentF[ids] - FT[ids]
                self._update_memory(
                    s,
                    sampled_F[ids],
                    sampled_CR[ids],
                    np.maximum(imp, 0.0),
                )
                novelty = self.niche.niche_reward(T[ids])
                quality = np.maximum(imp, 0.0) / (np.abs(parentF[ids]) + 1e-12)
                reward = float(np.mean(0.6 * quality + 0.4 * novelty))
                self.strategy.credits[s] = (
                    0.75 * self.strategy.credits[s] + 0.25 * (0.05 + reward)
                )

                old_points = oldP[ids]
                self.archive_X.extend([x.copy() for x in old_points])
                self.archive_F.extend([float(parentF[i]) for i in ids])

            # Survivor selection is niche-aware, even among non-improving candidates.
            if np.any(success):
                P[active[success]] = T[success]
                F[active[success]] = FT[success]
            if len(self.archive_X) > 2 * initial_N:
                keep = np.argsort(self.archive_F)[: 2 * initial_N]
                self.archive_X = [self.archive_X[i] for i in keep]
                self.archive_F = [self.archive_F[i] for i in keep]

            if self.niche.X and len(self.niche.X) > 1:
                E, EF = self.niche.elites()
                D = self.niche.rms_dist(E, P)
                occupied = np.any(D < 0.5 * self.niche.radius, axis=1)
                missing = np.flatnonzero(~occupied)
                worst = np.argsort(F)[::-1]
                for q, mi in enumerate(missing[: max(1, N // 10)]):
                    wi = int(worst[q])
                    P[wi] = E[mi]
                    F[wi] = EF[mi]

            # Feedback-adjust p-best fraction: stagnation broadens the p-best pool;
            # productive generations tighten it for exploitation.
            progress = float(np.max(parentF) - np.max(F[active])) / (np.std(parentF) + 1e-12)
            target_p = 0.10 + 0.25 * np.clip(1.0 - last_success_rate, 0.0, 1.0)
            self.p_gain = 0.85 * self.p_gain + 0.15 * (target_p - self.p_rate)
            self.p_rate = float(np.clip(self.p_rate + 0.10 * self.p_gain + 0.02 * np.tanh(progress), 0.05, 0.35))

            targetN = int(round(
                initial_N - (initial_N - self.min_pop) * min(1.0, evals / max(1, budget))
            ))
            targetN = max(self.min_pop, min(N, targetN))
            if targetN < N:
                keep = self.niche.select(P, F, targetN)
                P, F = P[keep], F[keep]
                N = targetN

            if generation_logger is not None:
                generation_logger.record(
                    evaluation_start=eval_start,
                    evaluation_end=evals,
                    best_value=best_f_so_far,
                    optimum=optimum,
                    target=target,
                    operator_probabilities={
                        "current_to_pbest_1": float(strategy_probs[0]),
                        "rand_1": float(strategy_probs[1]),
                        "current_to_rand_1": float(strategy_probs[2]),
                    },
                    operator_counts={
                        "current_to_pbest_1": int(np.sum(sampled_strategy[active] == 0)),
                        "rand_1": int(np.sum(sampled_strategy[active] == 1)),
                        "current_to_rand_1": int(np.sum(sampled_strategy[active] == 2)),
                    },
                    metadata={
                        "population_size_before": int(len(oldP)),
                        "population_size_after": int(N),
                        "population_size_start": int(initial_N),
                        "population_size_min": int(self.min_pop),
                        "population_reduction_progress": float(evals / max(1, budget)),
                        "p_rate": float(self.p_rate),
                        "success_rate": float(last_success_rate),
                    },
                )

        j = int(np.argmin(F))
        return P[j].copy(), float(F[j]), self.niche


# ------------------------- evaluation logging / GNBG integration -------------------------

TARGET_ERROR = 1e-8


@dataclass
class RunResult:
    """Outcome plus evaluation-level stopping information."""
    algorithm: str
    problem_id: int | str
    run: int
    evaluations: int
    best_x: Array
    best_f: float
    optimum: float
    final_error: float
    target_fe: Optional[int]
    log_path: Optional[str] = None
    generation_log_path: Optional[str] = None


class EvaluationLogger:
    """Streaming CSV logger: one row per objective evaluation.

    Columns follow the GNBG competition's best-so-far view: FE, objective,
    best-so-far objective, and absolute error. Decision vectors are optional
    because storing 500k x 30 vectors per run is unnecessarily large.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        optimum: float = 0.0,
        target: float = TARGET_ERROR,
        algorithm: str = "",
        problem_id: int | str = "",
        run: int = 0,
        flush_every: int = 1000,
        store_x: bool = False,
    ):
        self.path = path
        self.optimum = float(optimum)
        self.target = float(target)
        self.algorithm = str(algorithm)
        self.problem_id = problem_id
        self.run = int(run)
        self.flush_every = max(1, int(flush_every))
        self.store_x = bool(store_x)
        self.evaluations = 0
        self.best_f = math.inf
        self.target_fe: Optional[int] = None
        self.rows: list[dict[str, Any]] = []
        self._fh = None
        self._writer = None
        if path is not None:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            self._fh = open(path, "w", newline="", encoding="utf-8")
            fields = [
                "algorithm", "problem_id", "run", "evaluation",
                "value", "best_value", "error", "target_reached"
            ]
            if self.store_x:
                fields.append("x")
            self._writer = csv.DictWriter(self._fh, fieldnames=fields)
            self._writer.writeheader()

    def record(self, x: Array, value: float) -> float:
        self.evaluations += 1
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError("objective returned NaN or infinite value")
        if value < self.best_f:
            self.best_f = value
        error = abs(self.best_f - self.optimum)
        if self.target_fe is None and error <= self.target:
            self.target_fe = self.evaluations
        row = {
            "algorithm": self.algorithm,
            "problem_id": self.problem_id,
            "run": self.run,
            "evaluation": self.evaluations,
            "value": value,
            "best_value": self.best_f,
            "error": error,
            "target_reached": int(self.target_fe is not None),
        }
        if self.store_x:
            row["x"] = json.dumps(np.asarray(x, dtype=float).tolist(), separators=(",", ":"))
        if self._writer is not None:
            self._writer.writerow(row)
            if self.evaluations % self.flush_every == 0:
                self._fh.flush()
        else:
            self.rows.append(row)
        return value

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EvaluationLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class GenerationLogger:
    """Streaming CSV logger with one row per optimizer generation."""

    def __init__(self, path: Optional[str] = None, algorithm: str = "",
                 problem_id: int | str = "", run: int = 0):
        self.path = path
        self.algorithm = str(algorithm)
        self.problem_id = problem_id
        self.run = int(run)
        self.generation = 0
        self.rows: list[dict[str, Any]] = []
        self._fh = None
        self._writer = None
        if path is not None:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            self._fh = open(path, "w", newline="", encoding="utf-8")
            fields = [
                "algorithm", "problem_id", "run", "generation",
                "evaluation_start", "evaluation_end",
                "evaluations_this_generation", "best_value", "error",
                "target_reached", "operator_probabilities",
                "operator_counts", "metadata",
            ]
            self._writer = csv.DictWriter(self._fh, fieldnames=fields)
            self._writer.writeheader()

    def record(self, evaluation_start: int, evaluation_end: int,
               best_value: float, optimum: float, target: float,
               operator_probabilities: dict[str, Any],
               operator_counts: Optional[dict[str, int]] = None,
               metadata: Optional[dict[str, Any]] = None) -> None:
        self.generation += 1
        row = {
            "algorithm": self.algorithm,
            "problem_id": self.problem_id,
            "run": self.run,
            "generation": self.generation,
            "evaluation_start": int(evaluation_start),
            "evaluation_end": int(evaluation_end),
            "evaluations_this_generation": int(evaluation_end - evaluation_start),
            "best_value": float(best_value),
            "error": float(abs(best_value - optimum)),
            "target_reached": int(abs(best_value - optimum) <= target),
            "operator_probabilities": json.dumps(
                operator_probabilities,
                sort_keys=True, separators=(",", ":")
            ),
            "operator_counts": json.dumps(
                {k: int(v) for k, v in (operator_counts or {}).items()},
                sort_keys=True, separators=(",", ":")
            ),
            "metadata": json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")),
        }
        if self._writer is not None:
            self._writer.writerow(row)
            self._fh.flush()
        else:
            self.rows.append(row)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "GenerationLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class BudgetedObjective:
    """Wrap a GNBG black-box callable with exact FE accounting and logging."""

    def __init__(
        self,
        fun: Callable[[Array], float],
        budget: int,
        optimum: float,
        logger: EvaluationLogger,
    ):
        self.fun = fun
        self.budget = int(budget)
        self.logger = logger
        self.optimum = float(optimum)

    @property
    def evaluations(self) -> int:
        return self.logger.evaluations

    @property
    def best_f(self) -> float:
        return self.logger.best_f

    @property
    def best_x(self) -> Optional[Array]:
        # Filled lazily by solve_with_objective, which observes every x.
        return getattr(self, "_best_x", None)

    def __call__(self, x: Array) -> float:
        if self.evaluations >= self.budget:
            raise RuntimeError("function-evaluation budget exhausted")
        x = np.asarray(x, dtype=float)
        value = float(self.fun(x))
        old_best = self.best_f
        self.logger.record(x, value)
        if value < old_best:
            self._best_x = x.copy()
        return value


def _as_python_scalar(value: Any, name: str) -> float:
    """Convert a scalar or one-element array to a Python float safely."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(
            f"{name} must contain exactly one numeric value; got shape {arr.shape}"
        )
    try:
        out = float(arr.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {type(value).__name__}") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite; got {out}")
    return out


@dataclass
class GNBGProblem:
    """Direct, NumPy-based GNBG instance using the official ``.mat`` data.

    This intentionally does not call ``iohgnbg.get_problem`` or ``ioh.wrap_problem``.
    IOHGNBG v0.0.2 wraps ``GNBG.fitness`` through a pybind11-backed IOH callable;
    with current NumPy/IOH combinations that path can fail before the optimizer
    receives the objective value. The official source defines the benchmark from
    the ``GECCO_2025/fN.mat`` parameters, so evaluating that formula directly
    avoids the incompatible C++ conversion layer while preserving the benchmark
    data and objective transformation.
    """

    problem_id: int | str
    fun: Callable[[Array], float]
    lower: Array
    upper: Array
    optimum: float
    metadata: Any = None
    raw_optimum_value: float = 0.0
    max_evals: int | None = None
    optimum_position: Array | None = None

    @property
    def dimension(self) -> int:
        return int(self.lower.size)


class DirectGNBG:
    """Source-faithful GNBG evaluator without the IOH C++ wrapper.

    The mathematical expression and data layout follow IOHGNBG v0.0.2
    ``gnbg_problem.py`` / ``gnbg_base.py``. Only scalar extraction and the
    scalar-return interface are made explicit for NumPy 2.x compatibility.
    """

    def __init__(
        self,
        *,
        max_evals: int,
        acceptance_threshold: float,
        dimension: int,
        comp_num: int,
        min_coordinate: float,
        max_coordinate: float,
        comp_min_pos: Array,
        comp_sigma: Array,
        comp_h: Array,
        mu: Array,
        omega: Array,
        lam: Array,
        rotation_matrix: Array,
        optimum_value: float,
        optimum_position: Array,
    ) -> None:
        self.MaxEvals = int(max_evals)
        self.AcceptanceThreshold = float(acceptance_threshold)
        self.Dimension = int(dimension)
        self.CompNum = int(comp_num)
        self.MinCoordinate = float(min_coordinate)
        self.MaxCoordinate = float(max_coordinate)
        self.CompMinPos = np.asarray(comp_min_pos, dtype=float)
        self.CompSigma = np.asarray(comp_sigma, dtype=float)
        self.CompH = np.asarray(comp_h, dtype=float)
        self.Mu = np.asarray(mu, dtype=float)
        self.Omega = np.asarray(omega, dtype=float)
        self.Lambda = np.asarray(lam, dtype=float)
        self.RotationMatrix = np.asarray(rotation_matrix, dtype=float)
        self.OptimumValue = float(optimum_value)
        self.OptimumPosition = np.asarray(optimum_position, dtype=float)

        if self.CompMinPos.shape != (self.CompNum, self.Dimension):
            raise ValueError(
                f"Component_MinimumPosition has shape {self.CompMinPos.shape}; "
                f"expected ({self.CompNum}, {self.Dimension})"
            )
        if self.CompSigma.size != self.CompNum:
            raise ValueError(
                f"ComponentSigma has {self.CompSigma.size} values; expected {self.CompNum}"
            )
        if self.CompH.shape != (self.CompNum, self.Dimension):
            raise ValueError(
                f"Component_H has shape {self.CompH.shape}; "
                f"expected ({self.CompNum}, {self.Dimension})"
            )
        if self.Mu.shape[0] != self.CompNum or self.Mu.shape[1] != 2:
            raise ValueError(
                f"Mu has shape {self.Mu.shape}; expected ({self.CompNum}, 2)"
            )
        if self.Omega.shape[0] != self.CompNum or self.Omega.shape[1] != 4:
            raise ValueError(
                f"Omega has shape {self.Omega.shape}; expected ({self.CompNum}, 4)"
            )
        if self.Lambda.size != self.CompNum:
            raise ValueError(
                f"Lambda has {self.Lambda.size} values; expected {self.CompNum}"
            )
        if self.RotationMatrix.ndim not in (2, 3):
            raise ValueError(
                f"RotationMatrix must be 2-D or 3-D; got {self.RotationMatrix.shape}"
            )
        if self.RotationMatrix.ndim == 3:
            expected = (self.Dimension, self.Dimension, self.CompNum)
            if self.RotationMatrix.shape != expected:
                raise ValueError(
                    f"RotationMatrix has shape {self.RotationMatrix.shape}; expected {expected}"
                )
        else:
            expected = (self.Dimension, self.Dimension)
            if self.RotationMatrix.shape != expected:
                raise ValueError(
                    f"RotationMatrix has shape {self.RotationMatrix.shape}; expected {expected}"
                )
        if self.OptimumPosition.size != self.Dimension:
            raise ValueError(
                f"OptimumPosition has {self.OptimumPosition.size} values; expected {self.Dimension}"
            )
        if not np.all(np.isfinite(self.CompMinPos)):
            raise ValueError("GNBG component minima contain non-finite values")
        if not np.all(np.isfinite(self.CompSigma)):
            raise ValueError("GNBG component sigma contains non-finite values")
        if not np.all(np.isfinite(self.CompH)):
            raise ValueError("GNBG component heights contain non-finite values")
        if not np.all(np.isfinite(self.Mu)) or not np.all(np.isfinite(self.Omega)):
            raise ValueError("GNBG transform parameters contain non-finite values")
        if not np.all(np.isfinite(self.Lambda)):
            raise ValueError("GNBG lambda contains non-finite values")

    def transform(self, X: Array, alpha: Array, beta: Array) -> Array:
        """The transformation used verbatim by IOHGNBG v0.0.2."""
        X = np.asarray(X, dtype=float)
        alpha = np.asarray(alpha, dtype=float).reshape(-1)
        beta = np.asarray(beta, dtype=float).reshape(-1)
        if alpha.size != 2 or beta.size != 4:
            raise ValueError(
                f"GNBG transform expects Alpha(2), Beta(4); got {alpha.shape}, {beta.shape}"
            )
        Y = X.copy()
        positive = X > 0
        Y[positive] = np.log(X[positive])
        Y[positive] = np.exp(
            Y[positive]
            + alpha[0]
            * (
                np.sin(beta[0] * Y[positive])
                + np.sin(beta[1] * Y[positive])
            )
        )
        negative = X < 0
        Y[negative] = np.log(-X[negative])
        Y[negative] = -np.exp(
            Y[negative]
            + alpha[1]
            * (
                np.sin(beta[2] * Y[negative])
                + np.sin(beta[3] * Y[negative])
            )
        )
        return Y

    def raw_fitness(self, X: Array) -> float | Array:
        """Evaluate GNBG's raw fitness exactly as defined in its source."""
        arr = np.asarray(X, dtype=float)
        scalar_input = arr.ndim == 1
        if arr.ndim == 0:
            raise ValueError("GNBG input must be a 1-D search point or 2-D batch")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"GNBG input must be 1-D or 2-D; got {arr.shape}")
        if arr.shape[1] != self.Dimension:
            raise ValueError(
                f"GNBG point has dimension {arr.shape[1]}; expected {self.Dimension}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("GNBG input contains NaN or infinite values")

        result = np.empty(arr.shape[0], dtype=float)
        for jj in range(arr.shape[0]):
            x = arr[jj, :].reshape(-1, 1)
            values = np.empty(self.CompNum, dtype=float)
            for k in range(self.CompNum):
                if self.RotationMatrix.ndim == 3:
                    rotation_matrix = self.RotationMatrix[:, :, k]
                else:
                    rotation_matrix = self.RotationMatrix

                comp_min = self.CompMinPos[k, :].reshape(-1, 1)
                alpha = self.Mu[k, :]
                beta = self.Omega[k, :]
                comp_h = self.CompH[k, :].reshape(-1)
                sigma = _as_python_scalar(self.CompSigma[k], "CompSigma[k]")
                exponent = _as_python_scalar(self.Lambda[k], "Lambda[k]")

                # These products are the exact matrix operations used by
                # IOHGNBG v0.0.2 GNBG.fitness().
                a = self.transform(
                    (x - comp_min).T @ rotation_matrix.T,
                    alpha,
                    beta,
                )
                b = self.transform(
                    rotation_matrix @ (x - comp_min),
                    alpha,
                    beta,
                )
                quadratic = _as_python_scalar(
                    a @ np.diag(comp_h) @ b,
                    "GNBG quadratic form",
                )
                values[k] = sigma + quadratic**exponent

            result[jj] = float(np.min(values))

        if scalar_input:
            return float(result[0])
        return result

    def shifted_fitness(self, X: Array) -> float | Array:
        """Apply the official IOHGNBG objective shift ``fitness - OptimumValue``."""
        value = self.raw_fitness(X)
        return np.asarray(value) - self.OptimumValue if np.ndim(value) else float(value - self.OptimumValue)


def _find_gnbg_instance_file(problem_id: int, instances_folder: Optional[str] = None) -> str:
    """Locate the official GNBG ``fN.mat`` instance without importing IOH."""
    filename = f"f{int(problem_id)}.mat"

    if instances_folder is not None:
        candidate = os.path.abspath(os.path.join(instances_folder, filename))
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"GNBG instance not found: {candidate}"
        )

    # IOHGNBG v0.0.2's documented package layout is
    # <package>/static/GECCO_2025/fN.mat. We locate the installed package
    # using importlib metadata rather than importing iohgnbg.get_problem().
    import importlib.util

    spec = importlib.util.find_spec("iohgnbg")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "IOHGNBG is not installed. Install it with: python -m pip install iohgnbg"
        )

    for base in spec.submodule_search_locations:
        candidate = os.path.join(base, "static", "GECCO_2025", filename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    searched = [
        os.path.join(base, "static", "GECCO_2025", filename)
        for base in spec.submodule_search_locations
    ]
    raise FileNotFoundError(
        "Could not locate the official GNBG instance file. Searched:\n"
        + "\n".join(searched)
        + "\nPass --instances-folder PATH to the directory containing fN.mat."
    )


def _load_official_gnbg_data(problem_id: int, instances_folder: Optional[str] = None) -> tuple[DirectGNBG, str]:
    """Load a GNBG instance using the same MATLAB fields as IOHGNBG v0.0.2."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "GNBG direct loading requires SciPy. Install it with: python -m pip install scipy"
        ) from exc

    path = _find_gnbg_instance_file(problem_id, instances_folder)
    data = loadmat(path)
    if "GNBG" not in data:
        raise ValueError(f"{path} does not contain the expected 'GNBG' MATLAB structure")
    gnbg = data["GNBG"]

    # These extractions mirror the official get_problem() implementation.
    def official_scalar_field(name: str) -> float:
        field = gnbg[name]
        values = np.array([item[0] for item in field.flatten()])
        if values.size != 1:
            values = np.asarray(values).reshape(-1)
        return _as_python_scalar(values.reshape(-1)[0], f"GNBG.{name}")

    max_evals = int(round(official_scalar_field("MaxEvals")))
    acceptance_threshold = official_scalar_field("AcceptanceThreshold")
    dimension = int(round(official_scalar_field("Dimension")))
    comp_num = int(round(official_scalar_field("o")))
    min_coordinate = official_scalar_field("MinCoordinate")
    max_coordinate = official_scalar_field("MaxCoordinate")

    comp_min_pos = np.asarray(gnbg["Component_MinimumPosition"][0, 0], dtype=float)
    comp_sigma = np.asarray(gnbg["ComponentSigma"][0, 0], dtype=float)
    comp_h = np.asarray(gnbg["Component_H"][0, 0], dtype=float)
    mu = np.asarray(gnbg["Mu"][0, 0], dtype=float)
    omega = np.asarray(gnbg["Omega"][0, 0], dtype=float)
    lam = np.asarray(gnbg["lambda"][0, 0], dtype=float)
    rotation_matrix = np.asarray(gnbg["RotationMatrix"][0, 0], dtype=float)
    optimum_value = official_scalar_field("OptimumValue")
    optimum_position = np.asarray(gnbg["OptimumPosition"][0, 0], dtype=float).reshape(-1)

    evaluator = DirectGNBG(
        max_evals=max_evals,
        acceptance_threshold=acceptance_threshold,
        dimension=dimension,
        comp_num=comp_num,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        comp_min_pos=comp_min_pos,
        comp_sigma=comp_sigma,
        comp_h=comp_h,
        mu=mu,
        omega=omega,
        lam=lam,
        rotation_matrix=rotation_matrix,
        optimum_value=optimum_value,
        optimum_position=optimum_position,
    )

    lo = np.full(dimension, min_coordinate, dtype=float)
    hi = np.full(dimension, max_coordinate, dtype=float)
    if np.any(hi <= lo):
        raise ValueError("GNBG lower/upper coordinates are invalid")

    # IOHGNBG's get_problem() wraps ``gnbg.fitness(x) - OptimumValue``.
    # Therefore the transformed GNBG objective has optimum exactly 0 by
    # construction; the raw known optimum is retained for diagnostics.
    problem = GNBGProblem(
        problem_id=int(problem_id),
        fun=lambda x, _e=evaluator: _as_python_scalar(
            _e.shifted_fitness(np.asarray(x, dtype=float)),
            "GNBG objective value",
        ),
        lower=lo,
        upper=hi,
        optimum=0.0,
        metadata={
            "source": "IOHGNBG v0.0.2 GECCO_2025 .mat",
            "instance_file": path,
            "raw_optimum_value": optimum_value,
            "acceptance_threshold": acceptance_threshold,
        },
        raw_optimum_value=optimum_value,
        max_evals=max_evals,
        optimum_position=optimum_position.copy(),
    )
    return problem, path


def load_gnbg_problem(problem_id: int, instances_folder: Optional[str] = None) -> GNBGProblem:
    """Load an official GNBG problem without using the failing IOH wrapper.

    ``instances_folder`` should contain the official files ``f1.mat``, ...,
    ``f24.mat``. When omitted, the function locates the GECCO_2025 static data
    shipped by the installed IOHGNBG package.
    """
    problem, _ = _load_official_gnbg_data(problem_id, instances_folder)
    return problem


def solve_gnbg_problem(
    algorithm: str,
    problem: GNBGProblem,
    budget: int,
    seed: int = 0,
    run: int = 0,
    log_dir: str = "gnbg_logs",
    target: float = TARGET_ERROR,
    instances_folder: Optional[str] = None,
) -> RunResult:
    """Run one of the three optimizers on one GNBG instance.

    The benchmark remains a black box: the optimizer sees only ``fun(x)`` and
    the box bounds. The known optimum is used exclusively for post-hoc error
    logging/target detection, never for search decisions.
    """
    if problem.max_evals is not None and budget > problem.max_evals:
        raise ValueError(
            f"Requested budget {budget} exceeds the GNBG instance MaxEvals={problem.max_evals}"
        )
    lo, hi = problem.lower, problem.upper
    name = algorithm.strip().lower()
    aliases = {
        "bipop": "bipop-cma-es", "bipop-cma": "bipop-cma-es", "cma": "bipop-cma-es",
        "lm-cma": "lm-cma-es", "lm": "lm-cma-es",
        "l-shade": "l-shade", "lshade": "l-shade",
    }
    name = aliases.get(name, name)
    safe_name = name.replace("/", "_").replace(" ", "_")
    log_path = os.path.join(log_dir, safe_name, f"f{problem.problem_id}_run{run}.csv")
    generation_log_path = os.path.join(
        log_dir, safe_name, f"f{problem.problem_id}_run{run}_generations.csv"
    )
    logger = EvaluationLogger(
        path=log_path,
        optimum=problem.optimum,
        target=target,
        algorithm=name,
        problem_id=problem.problem_id,
        run=run,
    )
    generation_logger = GenerationLogger(
        path=generation_log_path,
        algorithm=name,
        problem_id=problem.problem_id,
        run=run,
    )
    objective = BudgetedObjective(problem.fun, budget, problem.optimum, logger)
    try:
        if name == "bipop-cma-es":
            sigma0 = 0.20 * float(np.min(hi - lo))
            opt = BIPOPCMAES(
                x0=0.5 * (lo + hi),
                sigma0=max(sigma0, 1e-8),
                lower=lo,
                upper=hi,
                niche_radius=0.15,
                seed=seed,
            )
            best_x, best_f, _ = opt.minimize(
                objective, budget, generation_logger=generation_logger,
                optimum=problem.optimum, target=target
            )
        elif name == "lm-cma-es":
            lam = max(8, min(40, 4 + int(3 * math.log(problem.dimension + 1))))
            sigma0 = 0.20 * float(np.min(hi - lo))
            niche = NicheArchive(lo, hi, radius=0.15, max_size=1000, pressure=0.8)
            opt = LMCMAES(
                x0=0.5 * (lo + hi),
                sigma0=max(sigma0, 1e-8),
                lam=lam,
                lower=lo,
                upper=hi,
                niche=niche,
                memory=min(20, max(4, problem.dimension // 10 + 4)),
                seed=seed,
                min_lam=max(6, lam // 3),
            )
            best_x, best_f, _ = opt.minimize(
                objective, budget, generation_logger=generation_logger,
                optimum=problem.optimum, target=target
            )
        elif name == "l-shade":
            max_pop = max(20, min(100, 4 * problem.dimension))
            min_pop = max(4, min(20, max_pop // 4))
            opt = LSHADE(
                lo,
                hi,
                max_pop=max_pop,
                min_pop=min_pop,
                H=20,
                niche_radius=0.15,
                seed=seed,
            )
            best_x, best_f, _ = opt.minimize(
                objective, budget, generation_logger=generation_logger,
                optimum=problem.optimum, target=target
            )
        else:
            raise ValueError("algorithm must be 'bipop-cma-es', 'lm-cma-es', or 'l-shade'")
    finally:
        generation_logger.close()
        logger.close()

    final_f = logger.best_f if np.isfinite(logger.best_f) else float(best_f)
    final_x = objective.best_x if objective.best_x is not None else np.asarray(best_x, dtype=float)
    return RunResult(
        algorithm=name,
        problem_id=problem.problem_id,
        run=run,
        evaluations=logger.evaluations,
        best_x=np.asarray(final_x, dtype=float),
        best_f=float(final_f),
        optimum=float(problem.optimum),
        final_error=float(abs(final_f - problem.optimum)),
        target_fe=logger.target_fe,
        log_path=log_path,
        generation_log_path=generation_log_path,
    )


def run_gnbg_suite(
    algorithms: tuple[str, ...] = ("bipop-cma-es", "lm-cma-es", "l-shade"),
    problem_ids: tuple[int, ...] = tuple(range(1, 25)),
    runs: int = 1,
    budget: int = 500_000,
    seed0: int = 0,
    log_dir: str = "gnbg_logs",
    summary_csv: str = "gnbg_summary.csv",
    target: float = TARGET_ERROR,
    instances_folder: Optional[str] = None,
) -> list[RunResult]:
    """Run the configured algorithms over official GNBG IDs and write a summary."""
    results: list[RunResult] = []
    for pid in problem_ids:
        for algorithm in algorithms:
            for run in range(runs):
                # Fresh GNBG wrapper per run prevents IOH state from leaking
                # evaluations/logger state between independent trials.
                problem = load_gnbg_problem(pid, instances_folder=instances_folder)
                result = solve_gnbg_problem(
                    algorithm=algorithm,
                    problem=problem,
                    budget=budget,
                    seed=seed0 + run,
                    run=run,
                    log_dir=log_dir,
                    target=target,
                )
                results.append(result)
    parent = os.path.dirname(os.path.abspath(summary_csv))
    os.makedirs(parent, exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "algorithm", "problem_id", "run", "evaluations",
                "best_f", "optimum", "final_error", "target_fe", "success",
                "evaluation_log", "generation_log"
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "algorithm": r.algorithm,
                "problem_id": r.problem_id,
                "run": r.run,
                "evaluations": r.evaluations,
                "best_f": f"{r.best_f:.17g}",
                "optimum": f"{r.optimum:.17g}",
                "final_error": f"{r.final_error:.17g}",
                "target_fe": "" if r.target_fe is None else r.target_fe,
                "success": int(r.final_error <= target),
                "evaluation_log": r.log_path or "",
                "generation_log": r.generation_log_path or "",
            })
    return results


# ------------------------- benchmark helpers -------------------------


def rastrigin(x: Array) -> float:
    x = np.asarray(x)
    return float(10 * x.size + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def ackley(x: Array) -> float:
    x = np.asarray(x)
    a, b, c = 20.0, 0.2, 2 * np.pi
    s1 = np.mean(x * x)
    s2 = np.mean(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1)) - np.exp(s2) + a + math.e)



def run_self_test(instances_folder: Optional[str] = None) -> None:
    """Load GNBG f1, test scalar objective output and the three optimizers.

    This test consumes no official benchmark FE budget because it uses the
    direct evaluator before creating any logged optimization run.
    """
    problem = load_gnbg_problem(1, instances_folder=instances_folder)
    if problem.optimum_position is None:
        raise RuntimeError("GNBG optimum position was not loaded")
    x = reflect_bounds(problem.optimum_position, problem.lower, problem.upper)
    value = problem.fun(x)
    if not isinstance(value, (float, np.floating)) or not np.isfinite(float(value)):
        raise RuntimeError(
            f"GNBG scalar objective test failed: type={type(value).__name__}, value={value!r}"
        )
    print(
        f"GNBG loader OK: f{problem.problem_id:02d}, dimension={problem.dimension}, "
        f"raw_optimum={problem.raw_optimum_value:.17g}, shifted_f(optimum_position)={float(value):.3e}"
    )

    lo, hi = problem.lower, problem.upper
    center = 0.5 * (lo + hi)
    test_budget = 32

    opt = BIPOPCMAES(
        x0=center, sigma0=max(0.05 * float(np.min(hi - lo)), 1e-8),
        lower=lo, upper=hi, niche_radius=0.15, seed=1,
    )
    bx, bf, _ = opt.minimize(problem.fun, test_budget)
    assert np.all((bx >= lo) & (bx <= hi)) and np.isfinite(bf)

    niche = NicheArchive(lo, hi, radius=0.15, max_size=100)
    opt = LMCMAES(
        x0=center, sigma0=max(0.05 * float(np.min(hi - lo)), 1e-8),
        lam=10, lower=lo, upper=hi, niche=niche, memory=min(8, max(4, problem.dimension // 10 + 2)),
        seed=1, min_lam=4,
    )
    bx, bf, _ = opt.minimize(problem.fun, test_budget)
    assert np.all((bx >= lo) & (bx <= hi)) and np.isfinite(bf)

    max_pop = max(8, min(20, 4 * problem.dimension))
    min_pop = max(4, min(8, max_pop // 2))
    opt = LSHADE(
        lo, hi, max_pop=max_pop, min_pop=min_pop, H=5,
        niche_radius=0.15, seed=1,
    )
    bx, bf, _ = opt.minimize(problem.fun, test_budget)
    assert np.all((bx >= lo) & (bx <= hi)) and np.isfinite(bf)

    print("Self-test OK: objective scalar conversion, reflective bounds, and all three optimizers.")


# ------------------------------ command line ------------------------------


def _parse_ids(spec: str) -> tuple[int, ...]:
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            lo_id, hi_id = int(a), int(b)
            if hi_id < lo_id:
                raise ValueError("function range must be ascending")
            out.extend(range(lo_id, hi_id + 1))
        else:
            out.append(int(token))
    if not out:
        raise ValueError("no function IDs supplied")
    return tuple(dict.fromkeys(out))


def _parse_algorithms(spec: str) -> tuple[str, ...]:
    aliases = {
        "all": ("bipop-cma-es", "lm-cma-es", "l-shade"),
        "bipop": ("bipop-cma-es",),
        "bipop-cma-es": ("bipop-cma-es",),
        "lm": ("lm-cma-es",),
        "lm-cma-es": ("lm-cma-es",),
        "lshade": ("l-shade",),
        "l-shade": ("l-shade",),
    }
    out: list[str] = []
    for token in spec.split(","):
        key = token.strip().lower()
        if key not in aliases:
            raise ValueError(f"unknown algorithm: {token}")
        out.extend(aliases[key])
    if not out:
        raise ValueError("no algorithms supplied")
    return tuple(dict.fromkeys(out))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Integrated GNBG baselines: niching BIPOP-CMA-ES, "
            "LM-CMA-ES, and L-SHADE."
        )
    )
    ap.add_argument("--algorithm", default="all", type=_parse_algorithms)
    ap.add_argument("--functions", default="1-24", type=_parse_ids)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--log-dir", default="gnbg_logs")
    ap.add_argument("--summary", default="gnbg_summary.csv")
    ap.add_argument("--target", type=float, default=TARGET_ERROR)
    ap.add_argument("--instances-folder", default=None,
                    help="Directory containing official GNBG fN.mat files; omit to use installed IOHGNBG data.")
    ap.add_argument("--self-test", action="store_true",
                    help="Run a small f1 compatibility/optimizer test and exit.")
    args = ap.parse_args()

    if args.runs < 1:
        ap.error("--runs must be >= 1")
    if args.budget < 1:
        ap.error("--budget must be >= 1")
    if args.target <= 0:
        ap.error("--target must be > 0")
    if args.self_test:
        run_self_test(instances_folder=args.instances_folder)
        return

    results = run_gnbg_suite(
        algorithms=args.algorithm,
        problem_ids=args.functions,
        runs=args.runs,
        budget=args.budget,
        seed0=args.seed0,
        log_dir=args.log_dir,
        summary_csv=args.summary,
        target=args.target,
        instances_folder=args.instances_folder,
    )
    for r in results:
        target_fe = "FAIL" if r.target_fe is None else str(r.target_fe)
        print(
            f"{r.algorithm:14s} f{r.problem_id:02d} run={r.run:02d} "
            f"FE={r.evaluations:7d} error={r.final_error:.3e} "
            f"target_FE={target_fe}"
        )


if __name__ == "__main__":
    main()
