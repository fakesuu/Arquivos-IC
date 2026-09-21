import numpy as np
import os
import csv
import iohgnbg

# ============================================================================
# Workaround: iohgnbg.gnbg_problem.GNBG.fitness returns a numpy array even
# for single-point evaluations. IOH's C-level wrapper expects a Python
# scalar and raises:
#     SystemError: ... returned a result with an exception set
#     TypeError: only length-1 arrays can be converted to Python scalars
#
# This monkeypatch wraps GNBG.fitness so that single-point evaluations
# return a Python float, and multi-point evaluations return a 1-D numpy
# array. The input shape is left untouched so the original broadcasting
# semantics (which expect a flat (dim,) vector) are preserved.
# ============================================================================
from iohgnbg.gnbg_problem import GNBG as _GNBG

if not getattr(_GNBG.fitness, "_is_dim_fixed", False):
    _original_gnbg_fitness = _GNBG.fitness

    def _gnbg_fitness_fixed(self, X):
        result = _original_gnbg_fitness(self, X)
        arr = np.asarray(result, dtype=float).ravel()
        if arr.size == 1:
            return float(arr[0])
        return arr

    _gnbg_fitness_fixed._is_dim_fixed = True
    _GNBG.fitness = _gnbg_fitness_fixed

# ============================================================================
# 0. Utility: Reflective Bounds
# ============================================================================
def reflect_bounds(x, lower, upper, max_reflections=10):
    x = np.array(x, copy=True)
    lower = np.broadcast_to(lower, x.shape)
    upper = np.broadcast_to(upper, x.shape)
    for _ in range(max_reflections):
        mask_low = x < lower
        mask_high = x > upper
        if not np.any(mask_low) and not np.any(mask_high):
            break
        x[mask_low] = 2 * lower[mask_low] - x[mask_low]
        x[mask_high] = 2 * upper[mask_high] - x[mask_high]
    return np.clip(x, lower, upper)


# ============================================================================
# 1. ActiveCMAES (full covariance) with population reduction
# ============================================================================
class ActiveCMAES:
    def __init__(self, func, dim, x0=None, sigma0=0.5, bounds=None,
                 max_evals=10000, seed=None, lambda_scale=1.0):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        if x0 is None:
            self.xmean = self.rng.uniform(bounds[0], bounds[1], size=dim) if bounds is not None else self.rng.randn(dim)
        else:
            self.xmean = np.array(x0)
        self.sigma = sigma0

        base_lam = 4 + int(3 * np.log(dim))
        self.lam_init = max(4, int(round(base_lam * lambda_scale)))
        self.lam_min = max(4, self.lam_init // 2)
        self.lam = self.lam_init
        self._update_strategy_parameters()

        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.count_evals = 0
        self.best_x = None
        self.best_f = np.inf
        self.logs = []

    def _update_strategy_parameters(self):
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.mu_neg = self.lam - self.mu
        neg_weights = -np.ones(self.mu_neg) / self.mu_neg
        self.weights_full = np.concatenate([self.weights, neg_weights])
        self.mueff_neg = 1.0 / np.sum(self.weights_full ** 2) - self.mueff
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.c1_neg = 2 / ((self.dim + 1.3) ** 2 + self.mueff_neg)
        self.cmu_neg = min(1 - self.c1_neg, 2 * (self.mueff_neg - 2 + 1 / self.mueff_neg) / ((self.dim + 2) ** 2 + self.mueff_neg))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

    def _reduce_population(self):
        progress = self.count_evals / self.max_evals
        new_lam = int(round(self.lam_init - (self.lam_init - self.lam_min) * progress))
        if new_lam != self.lam:
            self.lam = max(self.lam_min, new_lam)
            self._update_strategy_parameters()

    def sample(self):
        arz = self.rng.randn(self.lam, self.dim)
        ary = arz @ (self.B * self.D).T
        arx = self.xmean + self.sigma * ary
        if self.bounds is not None:
            arx = reflect_bounds(arx, self.bounds[0], self.bounds[1])
        return arx, ary, arz

    def update(self, arx, ary, arz, arf):
        idx = np.argsort(arf)
        arx, ary, arz, arf = arx[idx], ary[idx], arz[idx], arf[idx]
        if arf[0] < self.best_f:
            self.best_f = arf[0]
            self.best_x = arx[0].copy()

        parent_f = self.func(self.xmean)
        success = np.sum(arf < parent_f) / self.lam

        self.xmean = np.sum(self.weights_full[:, None] * arx, axis=0)
        ymean_pos = np.sum(self.weights[:, None] * ary[:self.mu], axis=0)
        ymean_neg = np.sum(self.weights_full[self.mu:, None] * ary[self.mu:], axis=0)
        zmean = np.sum(self.weights[:, None] * arz[:self.mu], axis=0)

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zmean
        denom = np.sqrt(max(1e-12, 1 - (1 - self.cs) ** (2 * self.count_evals / self.lam)))
        hsig = (np.linalg.norm(self.ps) / denom / np.sqrt(self.dim) * 1.4) < 1.0
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean_pos

        C_pos = (1 - self.c1 - self.cmu) * self.C + \
                self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
        for i in range(self.mu):
            C_pos += self.cmu * self.weights[i] * np.outer(ary[i], ary[i])

        C_neg = 0.0
        if self.mu_neg > 0:
            C_neg = self.c1_neg * np.outer(ymean_neg, ymean_neg)
            for i in range(self.mu, self.lam):
                C_neg += self.cmu_neg * self.weights_full[i] * np.outer(ary[i], ary[i])
        self.C = C_pos + C_neg

        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        self.C = (self.C + self.C.T) / 2
        try:
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 1e-12))
        except np.linalg.LinAlgError:
            pass

        self.count_evals += self.lam

        self.logs.append({
            'gen': len(self.logs),
            'evals': self.count_evals,
            'sigma': self.sigma,
            'success_rate': success,
            'c1': self.c1,
            'cmu': self.cmu,
            'lambda': self.lam,
            'condition': np.max(self.D) / np.min(self.D) if np.min(self.D) > 0 else np.inf
        })

        self._reduce_population()

    def run(self):
        while self.count_evals < self.max_evals:
            arx, ary, arz = self.sample()
            arf = np.array([self.func(x) for x in arx])
            self.update(arx, ary, arz, arf)
            if self.sigma < 1e-12:
                break
        return self.best_x, self.best_f, self.logs


# ============================================================================
# 2. BIPOP-CMA-ES wrapper
# ============================================================================
class BIPOP_CMAES:
    def __init__(self, func, dim, bounds, max_evals, seed=None):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_f = np.inf
        self.evals_used = 0
        self.all_logs = []

    def run(self):
        restart = 0
        sigma0 = 0.5
        while self.evals_used < self.max_evals and restart < 30:
            factor = 1 if restart % 2 == 0 else 2
            budget = min(self.max_evals - self.evals_used, int(100 * self.dim * (restart + 1)))
            if budget <= 0:
                break
            x0 = self.rng.uniform(self.bounds[0], self.bounds[1], size=self.dim) if self.bounds is not None else self.rng.randn(self.dim)
            cma = ActiveCMAES(self.func, self.dim, x0=x0, sigma0=sigma0, bounds=self.bounds,
                              max_evals=budget, seed=self.rng.randint(0, 1e6),
                              lambda_scale=factor)
            x, f, logs = cma.run()
            self.evals_used += budget
            if f < self.best_f:
                self.best_f = f
                self.best_x = x
            for log in logs:
                log['restart'] = restart
                self.all_logs.append(log)
            avg_success = np.mean([log['success_rate'] for log in logs]) if logs else 0.5
            if avg_success > 0.5:
                sigma0 = min(sigma0 * 2.0, 2.0)
            elif avg_success < 0.2:
                sigma0 = max(sigma0 * 0.5, 0.1)
            restart += 1
        return self.best_x, self.best_f, self.all_logs


# ============================================================================
# 3. ActiveLM-CMA-ES (diagonal covariance) with population reduction
# ============================================================================
class ActiveLM_CMAES:
    def __init__(self, func, dim, x0=None, sigma0=0.5, bounds=None, max_evals=10000, seed=None):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        if x0 is None:
            self.xmean = self.rng.uniform(bounds[0], bounds[1], size=dim) if bounds is not None else self.rng.randn(dim)
        else:
            self.xmean = np.array(x0)
        self.sigma = sigma0

        self.lam_init = 4 + int(3 * np.log(dim))
        self.lam_min = 4
        self.lam = self.lam_init
        self._update_strategy_parameters()

        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C_diag = np.ones(dim)
        self.count_evals = 0
        self.best_x = None
        self.best_f = np.inf
        self.logs = []

    def _update_strategy_parameters(self):
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.mu_neg = self.lam - self.mu
        neg_weights = -np.ones(self.mu_neg) / self.mu_neg
        self.weights_full = np.concatenate([self.weights, neg_weights])
        self.mueff_neg = 1.0 / np.sum(self.weights_full ** 2) - self.mueff
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

    def _reduce_population(self):
        progress = self.count_evals / self.max_evals
        new_lam = int(round(self.lam_init - (self.lam_init - self.lam_min) * progress))
        if new_lam != self.lam:
            self.lam = max(self.lam_min, new_lam)
            self._update_strategy_parameters()

    def sample(self):
        arz = self.rng.randn(self.lam, self.dim)
        ary = arz * np.sqrt(self.C_diag)
        arx = self.xmean + self.sigma * ary
        if self.bounds is not None:
            arx = reflect_bounds(arx, self.bounds[0], self.bounds[1])
        return arx, ary, arz

    def update(self, arx, ary, arz, arf):
        idx = np.argsort(arf)
        arx, ary, arz, arf = arx[idx], ary[idx], arz[idx], arf[idx]
        if arf[0] < self.best_f:
            self.best_f = arf[0]
            self.best_x = arx[0].copy()

        parent_f = self.func(self.xmean)
        success = np.sum(arf < parent_f) / self.lam

        self.xmean = np.sum(self.weights_full[:, None] * arx, axis=0)
        ymean_pos = np.sum(self.weights[:, None] * ary[:self.mu], axis=0)
        ymean_neg = np.sum(self.weights_full[self.mu:, None] * ary[self.mu:], axis=0)
        zmean = np.sum(self.weights[:, None] * arz[:self.mu], axis=0)

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zmean
        denom = np.sqrt(max(1e-12, 1 - (1 - self.cs) ** (2 * self.count_evals / self.lam)))
        hsig = (np.linalg.norm(self.ps) / denom / np.sqrt(self.dim) * 1.4) < 1.0
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean_pos

        if success > 0.5:
            self.c1 = min(0.9, self.c1 * 1.1)
            self.cmu = max(0.01, self.cmu * 0.9)
        elif success < 0.2:
            self.c1 = max(0.01, self.c1 * 0.9)
            self.cmu = min(0.9, self.cmu * 1.1)

        C_pos = (1 - self.c1 - self.cmu) * self.C_diag + \
                self.c1 * (self.pc ** 2 + (1 - hsig) * self.cc * (2 - self.cc) * self.C_diag)
        for i in range(self.mu):
            C_pos += self.cmu * self.weights[i] * (ary[i] ** 2)
        C_neg = 0.0
        if self.mu_neg > 0:
            C_neg = self.c1 * (ymean_neg ** 2)
            for i in range(self.mu, self.lam):
                C_neg += self.cmu * self.weights_full[i] * (ary[i] ** 2)
        self.C_diag = C_pos + C_neg
        self.C_diag = np.maximum(self.C_diag, 1e-12)

        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        self.count_evals += self.lam

        self.logs.append({
            'gen': len(self.logs),
            'evals': self.count_evals,
            'sigma': self.sigma,
            'success_rate': success,
            'c1': self.c1,
            'cmu': self.cmu,
            'lambda': self.lam,
            'min_var': np.min(self.C_diag),
            'max_var': np.max(self.C_diag)
        })

        self._reduce_population()

    def run(self):
        while self.count_evals < self.max_evals:
            arx, ary, arz = self.sample()
            arf = np.array([self.func(x) for x in arx])
            self.update(arx, ary, arz, arf)
            if self.sigma < 1e-12:
                break
        return self.best_x, self.best_f, self.logs


# ============================================================================
# 4. Improved L-SHADE
# ============================================================================
class Improved_L_SHADE:
    def __init__(self, func, dim, bounds, max_evals, pop_size=None, seed=None):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        self.N_init = pop_size if pop_size is not None else 100
        self.N = self.N_init
        self.N_min = 4
        self.F = 0.5
        self.CR = 0.5
        self.H = 5
        self.M_F = np.ones(self.H) * 0.5
        self.M_CR = np.ones(self.H) * 0.5
        self.archive = []
        self.archive_size = int(2.6 * self.N)
        self.p = 0.11
        self.best_x = None
        self.best_f = np.inf
        self.count_evals = 0
        self.pop = self._init_pop()
        self.fitness = np.array([self.func(x) for x in self.pop])
        self.count_evals += self.N
        self._update_best()
        self.logs = []

    def _init_pop(self):
        return self.rng.uniform(self.bounds[0], self.bounds[1], (self.N, self.dim))

    def _update_best(self):
        idx = np.argmin(self.fitness)
        if self.fitness[idx] < self.best_f:
            self.best_f = self.fitness[idx]
            self.best_x = self.pop[idx].copy()

    def _mutate(self, idx):
        pbest_size = max(1, int(self.N * self.p))
        sorted_idx = np.argsort(self.fitness)
        pbest = self.pop[sorted_idx[:pbest_size]]
        pbest_idx = self.rng.randint(0, pbest_size)
        xpbest = pbest[pbest_idx]
        xr1 = self.pop[self.rng.randint(self.N)]
        while np.array_equal(xr1, self.pop[idx]):
            xr1 = self.pop[self.rng.randint(self.N)]
        if len(self.archive) > 0 and self.rng.rand() < 0.5:
            xr2 = self.archive[self.rng.randint(len(self.archive))]
        else:
            xr2 = self.pop[self.rng.randint(self.N)]
            while np.array_equal(xr2, self.pop[idx]) or np.array_equal(xr2, xr1):
                xr2 = self.pop[self.rng.randint(self.N)]
        if self.rng.rand() < 0.5:
            F = self.rng.normal(self.F, 0.1)
        else:
            F = self.F + 0.1 * self.rng.standard_cauchy()
        F = np.clip(F, 0, 1)
        mutant = self.pop[idx] + F * (xpbest - self.pop[idx]) + F * (xr1 - xr2)
        mutant = reflect_bounds(mutant, self.bounds[0], self.bounds[1])
        return mutant, F

    def _crossover(self, target, mutant, CR):
        j_rand = self.rng.randint(self.dim)
        mask = self.rng.rand(self.dim) < CR
        mask[j_rand] = True
        trial = np.where(mask, mutant, target)
        trial = reflect_bounds(trial, self.bounds[0], self.bounds[1])
        return trial

    def _adapt_params(self, S_F, S_CR, improvements):
        if len(S_F) > 0:
            weights = improvements / np.sum(improvements) if np.sum(improvements) > 0 else np.ones(len(S_F)) / len(S_F)
            self.M_F[self.mem_idx] = np.sum(weights * S_F ** 2) / np.sum(weights * S_F) if np.sum(weights * S_F) > 0 else 0.5
            self.M_CR[self.mem_idx] = np.sum(weights * S_CR) if np.sum(weights) > 0 else 0.5
            self.mem_idx = (self.mem_idx + 1) % self.H

    def run(self):
        self.mem_idx = 0
        success_rates = []
        while self.count_evals < self.max_evals and self.N > self.N_min:
            S_F, S_CR, improvements = [], [], []
            offspring = []
            offspring_fitness = []
            generation_success = 0
            for i in range(self.N):
                old_f = self.fitness[i]
                mutant, F = self._mutate(i)
                if self.rng.rand() < 0.5:
                    CR = self.rng.normal(self.M_CR[self.mem_idx], 0.1)
                else:
                    CR = self.M_CR[self.mem_idx] + 0.1 * self.rng.standard_cauchy()
                CR = np.clip(CR, 0, 1)
                trial = self._crossover(self.pop[i], mutant, CR)
                ftrial = self.func(trial)
                self.count_evals += 1
                if ftrial < old_f:
                    offspring.append(trial)
                    offspring_fitness.append(ftrial)
                    S_F.append(F)
                    S_CR.append(CR)
                    improvements.append(old_f - ftrial)
                    generation_success += 1
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size:
                        del self.archive[self.rng.randint(len(self.archive))]
                else:
                    offspring.append(self.pop[i])
                    offspring_fitness.append(old_f)
                if self.count_evals >= self.max_evals:
                    break

            self.pop = np.array(offspring)
            self.fitness = np.array(offspring_fitness)
            self._update_best()
            success_rate = generation_success / self.N
            success_rates.append(success_rate)
            self._adapt_params(np.array(S_F), np.array(S_CR), np.array(improvements))

            if len(success_rates) >= 5:
                avg_sr = np.mean(success_rates[-5:])
                if avg_sr < 0.15:
                    self.p = min(0.25, self.p * 1.5)
                elif avg_sr > 0.3:
                    self.p = max(0.05, self.p * 0.8)

            self.logs.append({
                'gen': len(self.logs),
                'evals': self.count_evals,
                'F_mean': self.F,
                'CR_mean': np.mean(self.M_CR),
                'p': self.p,
                'pop_size': self.N,
                'success_rate': success_rate
            })

            self.N = max(self.N_min, int(round(self.N_init - (self.N_init - self.N_min) * self.count_evals / self.max_evals)))
            if self.N < len(self.pop):
                idx = np.argsort(self.fitness)[:self.N]
                self.pop = self.pop[idx]
                self.fitness = self.fitness[idx]
        return self.best_x, self.best_f, self.logs


# ============================================================================
# 5. Niching Wrapper
# ============================================================================
class NichingOptimizer:
    def __init__(self, optimizer_class, func, dim, bounds, max_evals_total,
                 num_niches, exclusion_radius, seed=None, **optimizer_kwargs):
        self.optimizer_class = optimizer_class
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals_total = max_evals_total
        self.num_niches = num_niches
        self.exclusion_radius = exclusion_radius
        self.optimizer_kwargs = optimizer_kwargs
        self.rng = np.random.RandomState(seed)
        self.found_optima = []
        self.total_evals = 0
        self.all_logs = []

    def run(self):
        niche_idx = 0
        while self.total_evals < self.max_evals_total and len(self.found_optima) < self.num_niches:
            remaining_niches = self.num_niches - len(self.found_optima)
            budget = (self.max_evals_total - self.total_evals) // remaining_niches
            if budget <= 0:
                break
            opt = self.optimizer_class(self.func, self.dim, bounds=self.bounds,
                                       max_evals=budget, seed=self.rng.randint(0, 1e6),
                                       **self.optimizer_kwargs)
            x, f, logs = opt.run()
            self.total_evals += budget
            too_close = False
            for prev_x, _ in self.found_optima:
                if np.linalg.norm(x - prev_x) < self.exclusion_radius:
                    too_close = True
                    break
            if not too_close:
                self.found_optima.append((x, f))
            for log in logs:
                log['niche'] = niche_idx
                self.all_logs.append(log)
            niche_idx += 1
        return self.found_optima, self.all_logs


# ============================================================================
# 6. GNBG Integration (Official iohgnbg package)
# ============================================================================
def get_gnbg_problem(index, dim):
    """
    Return (function, bounds, optimum, actual_dim) for a GNBG benchmark
    problem using the official iohgnbg package.

    Uses documented IOH attributes:
      - problem.bounds.lb / problem.bounds.ub   (RealBounds)
      - problem.optimum.y                        (Optimum)
      - problem.meta_data.n_variables            (problem dimension)
    """
    pkg_dir = os.path.dirname(iohgnbg.__file__)
    instances_folder = None
    for root, _, files in os.walk(pkg_dir):
        if any(fname.endswith(".mat") for fname in files):
            instances_folder = root
            break

    if instances_folder is None:
        raise RuntimeError(
            "Could not find GNBG instance files (.mat) inside the "
            "iohgnbg package installation."
        )

    problem = iohgnbg.get_problem(index, instances_folder)
    f = problem
    bounds = np.array([float(problem.bounds.lb[0]), float(problem.bounds.ub[0])])
    optimum = float(problem.optimum.y)
    actual_dim = int(problem.meta_data.n_variables)

    if actual_dim != dim:
        print(f"  [Info] GNBG instance f{index} has dimension "
              f"{actual_dim} (requested {dim}). Using {actual_dim}.")

    return f, bounds, optimum, actual_dim


# ============================================================================
# 7. CSV Logging Utilities (GECCO-compliant format)
# ============================================================================
def _ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log_run_summary(output_dir, algo_name, func_idx, run_idx, seed,
                    best_error, evals_to_threshold, success,
                    actual_dim, bounds, optimum, total_evals):
    """
    Append a per-run summary to <output_dir>/<algo_name>_summary.csv.

    Columns follow the GECCO competition convention:
      algorithm, function, run, seed, dimension, lower_bound, upper_bound,
      optimum, best_error, evals_to_threshold, success, total_evals
    """
    _ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, f"{algo_name}_summary.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow([
                "algorithm", "function", "run", "seed", "dimension",
                "lower_bound", "upper_bound", "optimum",
                "best_error", "evals_to_threshold", "success", "total_evals"
            ])
        writer.writerow([
            algo_name, func_idx, run_idx, seed, actual_dim,
            float(bounds[0]), float(bounds[1]), optimum,
            float(best_error),
            int(evals_to_threshold) if evals_to_threshold is not None else -1,
            int(bool(success)),
            int(total_evals)
        ])


def log_run_params(output_dir, algo_name, func_idx, run_idx, best_x):
    """
    Append the best-found parameter vector of a run to
    <output_dir>/<algo_name>_params.csv.

    Columns: algorithm, function, run, x0, x1, ..., x_{d-1}
    """
    _ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, f"{algo_name}_params.csv")
    file_exists = os.path.isfile(csv_path)

    if best_x is None:
        best_x = np.full(1, np.nan)
    best_x = np.asarray(best_x).ravel()

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            header = ["algorithm", "function", "run"] + [f"x{i}" for i in range(len(best_x))]
            writer.writerow(header)
        writer.writerow([algo_name, func_idx, run_idx] + [float(v) for v in best_x])


def log_generations(output_dir, algo_name, func_idx, run_idx, logs):
    """
    Write per-generation logs for a single run to
    <output_dir>/<algo_name>_generations_f{func_idx}_run{run_idx}.csv.

    The column set is the union of all keys present across the run's log
    dictionaries (CMA-ES and L-SHADE emit different operator fields).
    """
    if not logs:
        return
    _ensure_dir(output_dir)
    csv_path = os.path.join(
        output_dir,
        f"{algo_name}_generations_f{func_idx}_run{run_idx}.csv"
    )

    # Union of keys, preserving insertion order
    fieldnames = []
    for entry in logs:
        for k in entry.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in logs:
            writer.writerow(entry)


def log_combined_results(output_dir, algo_name, records):
    """
    Write a combined IOH-compatible CSV for a whole (algorithm, function)
    experiment. Columns follow the IOH data format:
      evaluation_counter, function_value, function_id, algorithm_id,
      problem_dimension, run_id
    See https://iohprofiler.github.io/IOHanalyzer/data/ for the spec.
    """
    if not records:
        return
    _ensure_dir(output_dir)
    csv_path = os.path.join(
        output_dir, f"{algo_name}_combined.csv"
    )
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow([
                "evaluation_counter", "function_value",
                "function_id", "algorithm_id",
                "problem_dimension", "run_id"
            ])
        writer.writerows(records)


def log_function_statistics(output_dir, algo_name, func_idx,
                            success_count, num_runs,
                            best_errors, evals_to_threshold_list,
                            max_evals_total):
    """
    Append aggregate statistics for one (algorithm, function) pair to
    <output_dir>/<algo_name>_statistics.csv.

    Columns:
      algorithm, function, runs, budget, success_count, success_rate,
      mean_best_error, std_best_error, min_best_error, max_best_error,
      median_evals, mean_evals, std_evals
    """
    _ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, f"{algo_name}_statistics.csv")
    file_exists = os.path.isfile(csv_path)

    best_errors = np.asarray(best_errors, dtype=float)
    if evals_to_threshold_list:
        ev = np.asarray(evals_to_threshold_list, dtype=float)
        median_evals = float(np.median(ev))
        mean_evals = float(np.mean(ev))
        std_evals = float(np.std(ev))
    else:
        median_evals = float("nan")
        mean_evals = float("nan")
        std_evals = float("nan")

    with open(csv_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow([
                "algorithm", "function", "runs", "budget",
                "success_count", "success_rate",
                "mean_best_error", "std_best_error",
                "min_best_error", "max_best_error",
                "median_evals", "mean_evals", "std_evals"
            ])
        writer.writerow([
            algo_name, func_idx, num_runs, max_evals_total,
            success_count, 100.0 * success_count / num_runs,
            float(np.mean(best_errors)),
            float(np.std(best_errors)),
            float(np.min(best_errors)),
            float(np.max(best_errors)),
            median_evals, mean_evals, std_evals
        ])


# ============================================================================
# 8. Main Execution: 31 Independent Runs per Algorithm & Function
# ============================================================================
if __name__ == "__main__":
    algo_classes = {
        'BIPOP-CMA-ES': BIPOP_CMAES,
        'LM-CMA-ES':    ActiveLM_CMAES,
        'L-SHADE':      Improved_L_SHADE,
    }

    dim = 30
    num_niches = 3
    exclusion_radius = 0.2
    error_threshold = 1e-8
    num_runs = 31
    base_seed = 42

    # Root folder for all CSV output
    output_root = os.path.join(os.getcwd(), "gnbg_results")
    _ensure_dir(output_root)

    function_indices = list(range(1, 25))

    for algo_name, algo_class in algo_classes.items():
        print(f"\n{'='*70}")
        print(f"=== {algo_name}: {num_runs} runs per function, dim={dim} ===")
        print(f"{'='*70}")

        algo_output_dir = os.path.join(output_root, algo_name)
        _ensure_dir(algo_output_dir)

        for func_idx in function_indices:
            if 1 <= func_idx <= 15:
                max_evals_total = 500_000
            elif 16 <= func_idx <= 24:
                max_evals_total = 1_000_000
            else:
                raise ValueError(f"Invalid GNBG function index: {func_idx}")

            print(f"\n--- Function {func_idx} (budget = {max_evals_total:,} evals) ---")

            success_count = 0
            best_errors = []
            evals_to_threshold_list = []
            combined_records = []

            for run_idx in range(num_runs):
                run_seed = base_seed + run_idx * 1000 + func_idx

                f, bounds, optimum, actual_dim = get_gnbg_problem(func_idx, dim)

                eval_count = [0]
                best_err = [np.inf]
                error_history = []

                def tracked_func(x, _eval_count=eval_count, _best_err=best_err,
                                 _history=error_history, _f=f):
                    _eval_count[0] += 1
                    fval = _f(x)
                    if fval < _best_err[0]:
                        _best_err[0] = fval
                        _history.append((_eval_count[0], fval))
                    return fval

                opt = NichingOptimizer(
                    algo_class, tracked_func, actual_dim, bounds,
                    max_evals_total, num_niches, exclusion_radius,
                    seed=run_seed
                )
                found_optima, logs = opt.run()

                if found_optima:
                    best_x, best_f = min(found_optima, key=lambda t: t[1])
                else:
                    best_x, best_f = None, np.inf
                best_errors.append(best_f)

                evals_to_threshold = None
                if best_f < error_threshold:
                    success_count += 1
                    for evals, err in error_history:
                        if err <= error_threshold:
                            evals_to_threshold = evals
                            evals_to_threshold_list.append(evals)
                            break

                # ---- CSV: per-run summary ----
                log_run_summary(
                    algo_output_dir, algo_name, func_idx, run_idx,
                    run_seed, best_f, evals_to_threshold,
                    best_f < error_threshold, actual_dim, bounds,
                    optimum, eval_count[0]
                )

                # ---- CSV: per-run best parameter vector ----
                log_run_params(algo_output_dir, algo_name, func_idx,
                               run_idx, best_x)

                # ---- CSV: per-generation operator probabilities ----
                log_generations(algo_output_dir, algo_name, func_idx,
                                run_idx, logs)

                # ---- Collect IOH-compatible records ----
                for evals, err in error_history:
                    combined_records.append([
                        int(evals), float(err), int(func_idx),
                        algo_name, int(actual_dim), int(run_idx)
                    ])

            # ---- CSV: aggregate statistics for this function ----
            log_function_statistics(
                algo_output_dir, algo_name, func_idx,
                success_count, num_runs, best_errors,
                evals_to_threshold_list, max_evals_total
            )

            # ---- CSV: combined IOH-compatible records ----
            log_combined_results(algo_output_dir, algo_name, combined_records)

            # ---- Console summary ----
            success_rate = 100.0 * success_count / num_runs
            avg_best_err = float(np.mean(best_errors))
            avg_evals = float(np.mean(evals_to_threshold_list)) if evals_to_threshold_list else float('nan')

            print(f"  Success Rate        : {success_rate:.1f}% ({success_count}/{num_runs})")
            print(f"  Average Best Error  : {avg_best_err:.4e}")
            if evals_to_threshold_list:
                print(f"  Avg. Evals to 1e-08 : {avg_evals:.0f} (over {len(evals_to_threshold_list)} runs)")
            else:
                print("  Avg. Evals to 1e-08 : N/A")

    print(f"\nAll CSV logs written to: {output_root}")