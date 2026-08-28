import numpy as np
from collections import defaultdict
import csv

# ============================================================================
# 0. Utility: Reflective Bounds
# ============================================================================
def reflect_bounds(x, lower, upper, max_reflections=10):
    """Reflect coordinates outside [lower, upper] back into the feasible region."""
    x = np.array(x, copy=True)
    lower = np.array(lower)
    upper = np.array(upper)
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
    def __init__(self, func, dim, x0=None, sigma0=0.5, bounds=None, max_evals=10000, seed=None):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.rng = np.random.RandomState(seed)
        if x0 is None:
            self.xmean = self.rng.uniform(bounds[0], bounds[1]) if bounds is not None else self.rng.randn(dim)
        else:
            self.xmean = np.array(x0)
        self.sigma = sigma0

        # Initial population size and schedule
        self.lam_init = 4 + int(3 * np.log(dim))
        self.lam_min = 4
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
        self.logs = []   # per-generation info

    def _update_strategy_parameters(self):
        """Recompute weights and learning rates based on current lambda."""
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        # Negative weights for active update
        self.mu_neg = max(1, self.lam // 4)
        neg_weights = -self.weights[-self.mu_neg:] / np.sum(np.abs(self.weights[-self.mu_neg:]))
        self.weights_full = np.concatenate([self.weights, neg_weights])
        self.mueff_neg = 1.0 / np.sum(self.weights_full ** 2) - self.mueff

        # Learning rates
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

        # Success rate relative to parent (xmean)
        parent_f = self.func(self.xmean)
        success = np.sum(arf < parent_f) / self.lam

        # Weighted recombination (positive and negative weights)
        self.xmean = np.sum(self.weights_full[:, None] * arx, axis=0)
        ymean_pos = np.sum(self.weights[:, None] * ary[:self.mu], axis=0)
        ymean_neg = np.sum(self.weights_full[self.mu:, None] * ary[self.mu:], axis=0)
        zmean = np.sum(self.weights[:, None] * arz[:self.mu], axis=0)

        # Step-size adaptation
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * zmean
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.count_evals / self.lam)) / np.sqrt(self.dim) * 1.4) < 1.0
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean_pos

        # Active covariance update
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

        # Log generation info
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

        # Reduce population for next generation
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
# 2. BIPOP-CMA-ES wrapper with restarts and sigma scaling
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
        self.all_logs = []   # aggregated logs from all restarts

    def run(self):
        restart = 0
        sigma0 = 0.5
        while self.evals_used < self.max_evals and restart < 30:
            factor = 1 if restart % 2 == 0 else 2
            budget = min(self.max_evals - self.evals_used, int(100 * self.dim * (restart + 1)))
            if budget <= 0:
                break
            x0 = self.rng.uniform(self.bounds[0], self.bounds[1]) if self.bounds is not None else self.rng.randn(self.dim)
            cma = ActiveCMAES(self.func, self.dim, x0=x0, sigma0=sigma0, bounds=self.bounds,
                              max_evals=budget, seed=self.rng.randint(0, 1e6))
            x, f, logs = cma.run()
            self.evals_used += budget
            if f < self.best_f:
                self.best_f = f
                self.best_x = x
            # Aggregate logs
            for log in logs:
                log['restart'] = restart
                self.all_logs.append(log)
            # Success-based sigma scaling for next restart
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
            self.xmean = self.rng.uniform(bounds[0], bounds[1]) if bounds is not None else self.rng.randn(dim)
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
        self.mu_neg = max(1, self.lam // 4)
        neg_weights = -self.weights[-self.mu_neg:] / np.sum(np.abs(self.weights[-self.mu_neg:]))
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
        hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.count_evals / self.lam)) / np.sqrt(self.dim) * 1.4) < 1.0
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean_pos

        # Success-based adjustment of learning rates
        if success > 0.5:
            self.c1 = min(0.9, self.c1 * 1.1)
            self.cmu = max(0.01, self.cmu * 0.9)
        elif success < 0.2:
            self.c1 = max(0.01, self.c1 * 0.9)
            self.cmu = min(0.9, self.cmu * 1.1)

        # Active diagonal update
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
# 4. Improved L-SHADE with population size reduction and logging
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
        F = self.rng.normal(self.F, 0.1) if self.rng.rand() < 0.5 else self.rng.cauchy(self.F, 0.1)
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
                CR = self.rng.normal(self.M_CR[self.mem_idx], 0.1) if self.rng.rand() < 0.5 else self.rng.cauchy(self.M_CR[self.mem_idx], 0.1)
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

            # Linear population size reduction
            self.N = max(self.N_min, int(round(self.N_init - (self.N_init - self.N_min) * self.count_evals / self.max_evals)))
            if self.N < len(self.pop):
                idx = np.argsort(self.fitness)[:self.N]
                self.pop = self.pop[idx]
                self.fitness = self.fitness[idx]
        return self.best_x, self.best_f, self.logs


# ============================================================================
# 5. Niching Wrapper (sequential optimizations with exclusion radius)
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
            # Tag and store logs
            for log in logs:
                log['niche'] = niche_idx
                self.all_logs.append(log)
            niche_idx += 1
        return self.found_optima, self.all_logs


# ============================================================================
# 6. Benchmark helper functions (GNBG optional)
# ============================================================================
def get_gnbg_problem(index, dim):
    """
    Placeholder for GNBG. Replace with actual GNBG calls if available.
    Returns function, bounds, optimum.
    """
    # Example: multimodal functions on [0,1]^D with global minimum 0
    if index == 1:  # Rastrigin-like
        def f(x):
            z = 10 * x - 5
            return 10 * len(x) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z), axis=0)
    elif index == 2:  # Griewank-like
        def f(x):
            z = 600 * x - 300
            return 1 + np.sum(z**2)/4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, len(x)+1))))
    elif index == 3:  # Ackley-like
        def f(x):
            z = 2 * x - 1
            return -20 * np.exp(-0.2 * np.sqrt(np.mean(z**2))) - np.exp(np.mean(np.cos(2*np.pi*z))) + 20 + np.e
    else:
        def f(x):
            return np.sum((x - 0.25)**2)
    return f, np.array([0.0, 1.0]), 0.0


# ============================================================================
# 7. Example Usage: Benchmark one algorithm on a single function
# ============================================================================
if __name__ == "__main__":
    # Configuration
    algo_classes = {
        'BIPOP-CMA-ES': BIPOP_CMAES,
        'LM-CMA-ES': ActiveLM_CMAES,
        'L-SHADE': Improved_L_SHADE,
    }
    dim = 10
    max_evals_total = 20000
    num_niches = 3
    exclusion_radius = 0.2
    error_threshold = 1e-8

    # Run each algorithm on a sample function (index 1) and log results
    for algo_name, algo_class in algo_classes.items():
        print(f"\n=== {algo_name} ===")
        f, bounds, optimum = get_gnbg_problem(1, dim)

        # Wrap function to track eval count and error history
        eval_count = 0
        best_err = np.inf
        error_history = []
        def tracked_func(x):
            nonlocal eval_count, best_err
            eval_count += 1
            fval = f(x)
            if fval < best_err:
                best_err = fval
                error_history.append((eval_count, fval))
            return fval

        # Run niching optimizer
        opt = NichingOptimizer(algo_class, tracked_func, dim, bounds,
                               max_evals_total, num_niches, exclusion_radius,
                               seed=42)
        found_optima, logs = opt.run()

        # Find best solution across niches
        if found_optima:
            best_x, best_f = min(found_optima, key=lambda t: t[1])
        else:
            best_x, best_f = None, np.inf

        # Find eval count when threshold reached
        evals_threshold = None
        for evals, err in error_history:
            if err <= error_threshold:
                evals_threshold = evals
                break

        print(f"Best error: {best_f:.2e}")
        print(f"Evals to reach {error_threshold}: {evals_threshold if evals_threshold else 'not reached'}")
        print(f"Total evals used: {eval_count}")
        print(f"Logs captured for {len(logs)} generations")

        # Save logs to CSV (optional)
        if logs:
            with open(f"logs_{algo_name}.csv", 'w', newline='') as f_csv:
                writer = csv.DictWriter(f_csv, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
            print(f"Logs saved to logs_{algo_name}.csv")