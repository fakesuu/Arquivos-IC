"""
lm_ma_niche.py
----------------
Limited-memory CMA-ES with niching AND success-feedback-driven parameter
adaptation, for high-dimensional problems.

IMPLEMENTATION NOTE: this implements the LM-MA-ES formulation (Loshchilov,
Glasmachers & Beyer, 2017), the practical, well-specified member of the
limited-memory CMA-ES family: m << n direction vectors instead of a full
covariance matrix, O(m*n) per sample, step size controlled directly in
"z-space" so no matrix inversion is ever needed. Constants follow the
spirit of that paper rather than reproducing its exact tuned schedule --
treat them as a tunable starting point.

Two feedback-driven additions on top of the previous baseline:

1. ACTIVE direction-vector update. Each stored direction vector M_i
   previously only learned from the elite (best-mu) recombination signal
   z_w. It now also receives a *negative*-weighted contribution from the
   worst offspring in the generation (z_w_neg), damped by a conservative
   fixed safety factor `eta_active`. There is no closed-form
   positive-definiteness guarantee for this implicit limited-memory
   representation the way there is for full active-CMA-ES, so the
   negative contribution is deliberately under-weighted rather than
   using the full active-CMA-ES safeguard formula -- this is a heuristic
   safeguard, not a proven one. The mean update and the step-size
   cumulation path are left exactly as before (elite-only): only the
   curvature/memory vectors get the active treatment.
   The default eta_active=0.1 is not a paper value -- it was picked by a
   small empirical sweep (eta in {0, 0.1, 0.15, 0.2, 0.3, 0.5} on 10-D
   Rastrigin, 12 seeds each). 0.1 gave the best mean/median best-fitness
   in that test and 0.5 was clearly worse than no active update at all,
   so treat 0.1 as a reasonable starting point to re-tune per problem,
   not a validated constant.

2. Success-conditioned IPOP restarts. A generation counts as "successful"
   if it produced a new global best. An EMA of that signal now gates how
   aggressively the population size grows on restart: population size
   only escalates while recent restarts have actually been finding
   improvements; if the recent success rate drops, the schedule backs
   off toward the base population size instead of continuing to double.
"""
from __future__ import annotations
import numpy as np
from .niching_utils import clearing, default_niche_radius, reflect_into_bounds, export_gen_log_csv


class LMMAES:
    """Single-population limited-memory matrix-adaptation ES with an
    active (success/failure-weighted) direction-vector update."""

    def __init__(self, mean, sigma, dim=None, popsize=None, n_vectors=None,
                 active=True, eta_active=0.1):
        self.n = dim if dim is not None else len(mean)
        n = self.n
        self.mean = np.array(mean, dtype=float)
        self.sigma = float(sigma)
        self.lam = popsize or (4 + int(3 * np.log(n)))
        self.mu = max(1, self.lam // 2)
        self.m = n_vectors or max(4, 4 + int(3 * np.log(n)))
        self.active = active
        self.eta_active = eta_active  # conservative damping on negative feedback

        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / np.sum(w)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        self.c_d = np.array([1.0 / (1.5 ** i * n) for i in range(self.m)])
        self.c_c = np.array([self.lam / (4.0 * n * (4.0 ** i)) for i in range(self.m)])
        self.c_c = np.clip(self.c_c, 1.0 / n, 0.3)

        self.cs = float(np.clip(2 * self.lam / n, 1e-3, 0.9))
        self.damps = 1.0 + n / (2.0 * self.lam)

        # negative-half weights for the active memory update (unit-sum,
        # same rank-based shape as CMA-ES's raw weight formula)
        n_neg = self.lam - self.mu
        if n_neg > 0:
            neg_raw = np.log((self.lam + 1) / 2.0) - np.log(np.arange(self.mu + 1, self.lam + 1))
            self.neg_weights = neg_raw / np.sum(np.abs(neg_raw))  # sums to -1
        else:
            self.neg_weights = np.array([])

        self.M = np.zeros((self.m, n))
        self.ps = np.zeros(n)
        self.chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.counteval = 0
        self.stop_flag = False
        self.last_success_rate = 0.5

    def _transform(self, Z):
        Dm = Z.copy()
        for i in range(self.m):
            v = self.M[i]
            vv = np.dot(v, v)
            if vv < 1e-300:
                continue
            coeff = (np.sqrt(1 + self.c_d[i] * vv) - 1) / vv
            Dm = Dm + coeff * np.outer(Dm @ v, v)
        return Dm

    def ask(self):
        self.Z = np.random.randn(self.lam, self.n)
        self.Dm = self._transform(self.Z)
        X = self.mean[None, :] + self.sigma * self.Dm
        return X

    def tell(self, X, fitness):
        idx_full = np.argsort(fitness)
        z_all = self.Z[idx_full]
        z_pos = z_all[: self.mu]
        d_pos = self.Dm[idx_full][: self.mu]

        if hasattr(self, '_prev_best'):
            self.last_success_rate = float(np.mean(fitness < self._prev_best))
        self._prev_best = float(np.min(fitness))

        # elite-only recombination: drives mean and step-size, unchanged
        z_w = self.weights @ z_pos
        d_w = self.weights @ d_pos
        self.mean = self.mean + self.sigma * d_w

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z_w
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1))

        # active feedback signal for the memory/direction vectors only
        z_w_active = z_w
        if self.active and len(self.neg_weights) > 0:
            z_neg = z_all[self.mu:]
            z_w_neg = self.neg_weights @ z_neg
            z_w_active = z_w - self.eta_active * z_w_neg

        for i in range(self.m):
            self.M[i] = (1 - self.c_c[i]) * self.M[i] + \
                np.sqrt(self.c_c[i] * (2 - self.c_c[i]) * self.mueff) * z_w_active

        self.counteval += self.lam
        if self.sigma < 1e-14 or self.sigma > 1e14 or not np.all(np.isfinite(self.mean)):
            self.stop_flag = True


class NichedLMMAES:
    def __init__(self, objective, bounds, budget,
                 niche_radius=None, capacity=1, max_niches=10,
                 base_popsize=None, active=True, pop_reduction=True, seed=None):
        """pop_reduction : linearly shrink the population budget (max
        simultaneous niches, and the size any newly restarted niche is
        allowed) as the evaluation budget is consumed -- the same LPSR
        idea used in L-SHADE, applied here at niche-spawn granularity
        rather than by resizing a live LM-MA-ES instance's lambda
        mid-run (which would require re-deriving its internal learning
        rates and is not something this implementation attempts)."""
        if seed is not None:
            np.random.seed(seed)
        self.f = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = len(self.bounds)
        self.budget = budget
        self.capacity = capacity
        self.max_niches = max_niches
        self.active = active
        self.base_lam = base_popsize or (4 + int(3 * np.log(self.dim)))
        self.niche_radius = niche_radius or default_niche_radius(self.bounds, self.dim)

        self.pop_reduction = pop_reduction
        self.min_popsize = max(4, self.base_lam // 2)

        self.evals = 0
        self.best_x = None
        self.best_f = np.inf
        self.restart_count = 0

        # success-feedback bookkeeping for restart population sizing
        self.success_ema = 0.5
        self.success_ema_beta = 0.85

        self.niches: list[dict] = []
        self._spawn(self.base_lam)

        # per-generation log: [{'gen','evals','n_niches','best_f','restart_prob'}, ...]
        # NOTE: like BIPOP-CMA-ES, LM-MA-ES has a single sampling operator,
        # not a discrete operator pool -- 'restart_prob' logs the success
        # EMA that gates how aggressively restart population size grows
        # (see run()), the closest analog available here. It is not a
        # per-individual selection probability the way L-SHADE's is.
        self.gen_log: list[dict] = []

    def _clip(self, X):
        # reflective repair, not clamping -- see niching_utils.reflect_into_bounds
        return reflect_into_bounds(X, self.bounds)

    def _total_popsize_budget(self):
        """Linearly-reducing cap on total population across active
        niches -- same LPSR idea as in L-SHADE / BIPOP-CMA-ES."""
        if not self.pop_reduction:
            return None
        frac = min(1.0, self.evals / self.budget)
        init_budget = self.max_niches * self.base_lam
        return max(self.min_popsize, round(init_budget + frac * (self.min_popsize - init_budget)))

    def _current_max_niches(self):
        """Tapers the number of simultaneously active niches from
        max_niches down to 1 as the budget is consumed."""
        if not self.pop_reduction:
            return self.max_niches
        frac = min(1.0, self.evals / self.budget)
        return max(1, round(self.max_niches * (1 - frac) + 1 * frac))

    def _spawn(self, popsize):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        mean = lo + np.random.rand(self.dim) * (hi - lo)
        sigma = 0.3 * np.mean(hi - lo)
        es = LMMAES(mean, sigma, dim=self.dim, popsize=popsize, active=self.active)
        self.niches.append({'es': es})

    def _merge_redundant_niches(self):
        if len(self.niches) < 2:
            return
        means = np.array([nd['es'].mean for nd in self.niches])
        keep = [True] * len(self.niches)
        for i in range(len(self.niches)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(self.niches)):
                if not keep[j]:
                    continue
                if np.linalg.norm(means[i] - means[j]) < self.niche_radius:
                    if self.niches[i]['es'].sigma <= self.niches[j]['es'].sigma:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
        self.niches = [nd for nd, k in zip(self.niches, keep) if k]

    def run(self, verbose=False):
        gen = 0
        while self.evals < self.budget and self.niches:
            gen += 1
            all_X, offsets = [], []
            offset = 0
            for nd in self.niches:
                X = self._clip(nd['es'].ask())
                all_X.append(X)
                offsets.append((offset, offset + nd['es'].lam))
                offset += nd['es'].lam
            X_all = np.vstack(all_X)
            fit_all = np.asarray(self.f(X_all), dtype=float)
            self.evals += len(fit_all)

            _ = clearing(X_all, fit_all, self.niche_radius, self.capacity)

            i = np.argmin(fit_all)
            improved = fit_all[i] < self.best_f
            if improved:
                self.best_f, self.best_x = float(fit_all[i]), X_all[i].copy()
            self.success_ema = (self.success_ema_beta * self.success_ema
                                 + (1 - self.success_ema_beta) * (1.0 if improved else 0.0))

            finished = []
            for k, nd in enumerate(self.niches):
                lo, hi = offsets[k]
                nd['es'].tell(X_all[lo:hi], fit_all[lo:hi])
                if nd['es'].stop_flag:
                    finished.append(k)

            for k in sorted(finished, reverse=True):
                del self.niches[k]

            if finished and len(self.niches) < self._current_max_niches():
                # success-conditioned IPOP: only keep escalating population
                # size while restarts have recently been paying off
                if self.success_ema > 0.15:
                    self.restart_count = min(self.restart_count + 1, 6)
                else:
                    self.restart_count = max(self.restart_count - 1, 0)
                popsize = self.base_lam * (2 ** self.restart_count)
                budget_cap = self._total_popsize_budget()
                if budget_cap is not None:
                    current_total = sum(nd['es'].lam for nd in self.niches)
                    popsize = max(self.min_popsize, min(popsize, budget_cap - current_total))
                self._spawn(popsize)

            self._merge_redundant_niches()

            self.gen_log.append({
                'gen': gen,
                'evals': self.evals,
                'n_niches': len(self.niches),
                'total_pop': sum(nd['es'].lam for nd in self.niches),
                'best_f': self.best_f,
                'restart_prob': {'success_ema': self.success_ema},
            })

            if verbose and gen % 20 == 0:
                print(f"gen {gen:5d} evals {self.evals:7d} niches {len(self.niches):2d} "
                      f"best {self.best_f:.6g} succ_ema={self.success_ema:.2f}")

        return self.best_x, self.best_f, self.niches

    def export_log_csv(self, path: str) -> None:
        """Write the per-generation log (gen, evals, n_niches, best_f,
        restart_prob_success_ema) to a CSV file."""
        export_gen_log_csv(self.gen_log, path)
