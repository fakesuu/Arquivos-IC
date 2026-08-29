"""
bipop_cma_niche.py
-------------------
BIPOP-CMA-ES with niching AND success-feedback-driven parameter adaptation.

`CMAES` now implements ACTIVE covariance matrix adaptation (Jastrebski &
Arnold, 2006): the rank-mu covariance update no longer only learns from
the best (elite) offspring -- it also learns, with a *negative* weight,
from the worst offspring in each generation. Directions that produced bad
samples are actively pushed down, not just ignored. This is the standard,
published way to make a single CMA-ES-family operator "listen to" both
its successes and its failures each generation, and it noticeably speeds
up adaptation on ill-conditioned landscapes.

Concretely:
    - offspring are ranked over the FULL population (not just the top
      half), giving a positive-weight elite half (as before, used for the
      mean shift and step-size path -- unchanged and still safe/standard)
      and a negative-weight "worst half",
    - the negative weights are scaled by a safety factor
      alpha = min(alpha_mu-, alpha_mueff-, alpha_posdef-), the standard
      three-way safeguard from the active-CMA-ES literature that keeps
      the (1 - c1 - cmu) + rank-one + rank-mu combination positive
      semi-definite in the typical case.
    - NOTE (honesty about scope): the published active-CMA-ES additionally
      rescales each negative sample by its individual Mahalanobis norm
      before adding it to the rank-mu term; this implementation omits
      that per-sample rescaling for simplicity. The alpha safeguard above
      is retained, and eigenvalues are still floored during decomposition
      as a belt-and-braces numerical safety net -- but this is a
      simplified active update, not a byte-for-byte reproduction of the
      paper.

The niching wrapper additionally tracks, per BIPOP regime ('large' vs
'small' population restarts), an exponential moving average of how often
that regime has recently produced a new global-best solution. When the
BIPOP budget-alternation rule is roughly tied, the regime with the better
recent success rate is preferred -- so restart-strategy selection itself
is now informed by which regime has actually been paying off, not only
by which one has used less evaluation budget.
"""
from __future__ import annotations
import numpy as np
from .niching_utils import clearing, default_niche_radius, reflect_into_bounds, export_gen_log_csv


class CMAES:
    """Single-population CMA-ES with active (success/failure-weighted)
    rank-mu covariance update."""

    def __init__(self, mean, sigma, dim=None, popsize=None, active=True):
        self.n = dim if dim is not None else len(mean)
        n = self.n
        self.mean = np.array(mean, dtype=float)
        self.sigma = float(sigma)
        self.lam = popsize or (4 + int(3 * np.log(n)))
        self.active = active

        # Full-population rank weights (Hansen's default formula extended
        # over the whole population, not just the elite half) -- this is
        # what naturally splits into a positive elite half and a negative
        # "worst half" for the active update.
        raw_w = np.log((self.lam + 1) / 2.0) - np.log(np.arange(1, self.lam + 1))
        pos_mask = raw_w > 0
        self.mu = int(np.sum(pos_mask))
        w_pos = raw_w[pos_mask] / np.sum(raw_w[pos_mask])
        self.weights = w_pos                       # sums to 1, elite half only
        self.mueff = 1.0 / np.sum(w_pos ** 2)

        neg_raw = raw_w[~pos_mask]
        if len(neg_raw) > 0:
            self.mueff_minus = np.sum(neg_raw) ** 2 / np.sum(neg_raw ** 2)
            self.w_neg_unit = neg_raw / np.sum(np.abs(neg_raw))  # sums to -1
        else:
            self.mueff_minus = 0.0
            self.w_neg_unit = np.array([])

        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                        2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0.0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs

        # active-update safeguard (Jastrebski & Arnold, 2006)
        if self.active and len(self.w_neg_unit) > 0 and self.cmu > 1e-12:
            alpha_mu_minus = 1 + self.c1 / self.cmu
            alpha_mueff_minus = 1 + 2 * self.mueff_minus / (self.mueff + 2)
            alpha_posdef_minus = (1 - self.c1 - self.cmu) / (n * self.cmu)
            self.alpha_active = max(0.0, min(alpha_mu_minus, alpha_mueff_minus, alpha_posdef_minus))
        else:
            self.alpha_active = 0.0

        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.C = np.eye(n)
        self.chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.counteval = 0
        self.eigeneval = 0
        self.stop_flag = False
        self.last_success_rate = 0.5  # fraction of offspring beating the previous best

    def ask(self):
        self.z = np.random.randn(self.lam, self.n)
        y = self.z @ (self.B * self.D).T
        self.y_last = y
        X = self.mean[None, :] + self.sigma * y
        return X

    def tell(self, X, fitness):
        idx_full = np.argsort(fitness)               # best first, full population
        y_all = (X[idx_full] - self.mean) / self.sigma
        y_pos = y_all[: self.mu]                      # elite half

        # --- success-rate feedback signal (reporting + niche-level use) ---
        if hasattr(self, '_prev_best') :
            self.last_success_rate = float(np.mean(fitness < self._prev_best))
        self._prev_best = float(np.min(fitness))

        # mean shift and step-size path: elite-only, standard & unchanged
        y_w = self.weights @ y_pos
        self.mean = self.mean + self.sigma * y_w

        invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + \
            np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (invsqrtC @ y_w)
        ps_norm = np.linalg.norm(self.ps)
        self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1))

        self.counteval += self.lam
        hsig = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * self.counteval / self.lam)) \
            < (1.4 + 2 / (self.n + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + \
            hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

        # --- rank-mu term: ACTIVE if enabled, i.e. informed by both the
        # elite (positive weight) and the worst (negative weight) offspring ---
        if self.active and self.alpha_active > 0 and len(self.w_neg_unit) > 0:
            y_neg = y_all[self.mu:]
            full_w = np.concatenate([self.weights, self.alpha_active * self.w_neg_unit])
            y_full = np.vstack([y_pos, y_neg])
            rank_mu_term = self.cmu * (y_full.T * full_w) @ y_full
        else:
            rank_mu_term = self.cmu * (y_pos.T * self.weights) @ y_pos

        self.C = (1 - self.c1 - self.cmu) * self.C \
            + self.c1 * (np.outer(self.pc, self.pc)
                         + (1 - hsig) * self.cc * (2 - self.cc) * self.C) \
            + rank_mu_term

        if self.counteval - self.eigeneval > self.lam / (self.c1 + self.cmu) / self.n / 10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            evals, evecs = np.linalg.eigh(self.C)
            evals = np.maximum(evals, 1e-20)  # numerical safety net
            self.D = np.sqrt(evals)
            self.B = evecs

        cond = np.max(self.D) / np.min(self.D)
        if self.sigma * np.max(self.D) < 1e-12 or cond > 1e7 or self.sigma < 1e-14:
            self.stop_flag = True


class NichedBIPOPCMAES:
    def __init__(self, objective, bounds, budget,
                 niche_radius=None, capacity=1, max_niches=10,
                 base_popsize=None, active=True, pop_reduction=True, seed=None):
        """
        objective : callable, X (N, dim) -> fitness (N,), MINIMIZED
        bounds    : (dim, 2) array of [lo, hi] per dimension
        budget    : total number of objective evaluations allowed
        active    : use active (success/failure-weighted) covariance update
        pop_reduction : linearly shrink the population budget (both the
            max number of simultaneous niches and the size any newly
            spawned niche is allowed) as the evaluation budget is
            consumed, mirroring L-SHADE's linear population-size
            reduction (LPSR). Also fixes an unbounded-growth issue in the
            BIPOP 'large' restart rule: without a cap, repeated large
            restarts double lambda every time (base_lam * 2^k) with no
            ceiling, so a long run with many restarts could balloon a
            single generation's cost late in the run when the budget is
            actually running out -- exactly the opposite of what you
            want. With pop_reduction on, that growth is capped by the
            shrinking budget below instead of growing unchecked.
        """
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

        # BIPOP bookkeeping
        self.evals_large = 0
        self.evals_small = 0
        self.large_restarts = 0

        # success-feedback bookkeeping per regime (EMA of "did this
        # regime's niches produce a new global best this generation")
        self.regime_success = {'large': 0.5, 'small': 0.5}
        self.success_ema_beta = 0.9

        self.niches: list[dict] = []
        self._spawn('large', self.base_lam)

        # per-generation log: [{'gen','evals','n_niches','best_f','regime_prob'}, ...]
        # NOTE: CMA-ES-family algorithms have a single sampling operator
        # (the Gaussian), not a discrete operator pool like L-SHADE's
        # ensemble -- 'regime_prob' logs the closest analog we have: the
        # success-EMA per BIPOP regime ('large' vs 'small' restarts) that
        # biases which restart strategy gets used next (see
        # _next_bipop_regime). It is not a per-individual selection
        # probability.
        self.gen_log: list[dict] = []

    def _clip(self, X):
        # despite the name (kept to minimize diff at call sites), this
        # repairs infeasible points by reflection, not clamping -- see
        # niching_utils.reflect_into_bounds for why that matters here.
        return reflect_into_bounds(X, self.bounds)

    def _total_popsize_budget(self):
        """Linearly-reducing cap on total population (summed lambda
        across all active niches). Mirrors L-SHADE's linear
        population-size reduction: starts with room for a full roster of
        max_niches base-sized niches, and tapers toward a single
        min-size niche as the evaluation budget is consumed."""
        if not self.pop_reduction:
            return None  # no cap
        frac = min(1.0, self.evals / self.budget)
        init_budget = self.max_niches * self.base_lam
        return max(self.min_popsize, round(init_budget + frac * (self.min_popsize - init_budget)))

    def _current_max_niches(self):
        """Number of simultaneously active niches allowed right now.
        Tapers from max_niches down to 1 as the budget is consumed, so
        the algorithm naturally narrows to exploiting its best basin(s)
        late in the run instead of still paying for wide parallel search."""
        if not self.pop_reduction:
            return self.max_niches
        frac = min(1.0, self.evals / self.budget)
        return max(1, round(self.max_niches * (1 - frac) + 1 * frac))

    def _spawn(self, regime: str, popsize: int):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        mean = lo + np.random.rand(self.dim) * (hi - lo)
        if regime == 'large':
            sigma = 0.3 * np.mean(hi - lo)
        else:
            sigma = 0.3 * np.mean(hi - lo) * (10 ** (-2 * np.random.rand()))
        cma = CMAES(mean, sigma, dim=self.dim, popsize=popsize, active=self.active)
        self.niches.append({'cma': cma, 'regime': regime})

    def _next_bipop_regime(self):
        """BIPOP restart rule, biased by recent success feedback.

        Base rule (as before): spend budget on whichever regime has used
        less of it so far. When the two regimes are roughly tied on
        budget, break the tie using which regime's success EMA is higher,
        instead of always defaulting to 'large'.
        """
        budget_gap = self.evals_large - self.evals_small
        roughly_tied = abs(budget_gap) < 0.15 * max(self.evals_large + self.evals_small, 1)

        if roughly_tied:
            go_large = self.regime_success['large'] >= self.regime_success['small']
        else:
            go_large = budget_gap <= 0

        if go_large or self.large_restarts == 0:
            self.large_restarts += 1
            popsize = self.base_lam * (2 ** self.large_restarts)
            return 'large', popsize
        else:
            lam_large = self.base_lam * (2 ** max(self.large_restarts, 1))
            u = np.random.rand() ** 2
            popsize = max(4, int(0.5 * lam_large * u))
            return 'small', popsize

    def _merge_redundant_niches(self):
        if len(self.niches) < 2:
            return
        means = np.array([nd['cma'].mean for nd in self.niches])
        keep = [True] * len(self.niches)
        for i in range(len(self.niches)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(self.niches)):
                if not keep[j]:
                    continue
                if np.linalg.norm(means[i] - means[j]) < self.niche_radius:
                    if self.niches[i]['cma'].sigma <= self.niches[j]['cma'].sigma:
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
                X = self._clip(nd['cma'].ask())
                all_X.append(X)
                offsets.append((offset, offset + nd['cma'].lam))
                offset += nd['cma'].lam
            X_all = np.vstack(all_X)
            fit_all = np.asarray(self.f(X_all), dtype=float)
            self.evals += len(fit_all)

            _ = clearing(X_all, fit_all, self.niche_radius, self.capacity)

            i = np.argmin(fit_all)
            improved = fit_all[i] < self.best_f
            if improved:
                self.best_f, self.best_x = float(fit_all[i]), X_all[i].copy()

            finished = []
            regimes_that_improved = set()
            for k, nd in enumerate(self.niches):
                lo, hi = offsets[k]
                Xk, fk = X_all[lo:hi], fit_all[lo:hi]
                nd['cma'].tell(Xk, fk)
                if nd['regime'] == 'large':
                    self.evals_large += hi - lo
                else:
                    self.evals_small += hi - lo
                if improved and np.min(fk) == self.best_f:
                    regimes_that_improved.add(nd['regime'])
                if nd['cma'].stop_flag:
                    finished.append(k)

            for regime in ('large', 'small'):
                success = 1.0 if regime in regimes_that_improved else 0.0
                self.regime_success[regime] = (self.success_ema_beta * self.regime_success[regime]
                                                + (1 - self.success_ema_beta) * success)

            for k in sorted(finished, reverse=True):
                del self.niches[k]

            if finished and len(self.niches) < self._current_max_niches():
                regime, popsize = self._next_bipop_regime()
                budget_cap = self._total_popsize_budget()
                if budget_cap is not None:
                    current_total = sum(nd['cma'].lam for nd in self.niches)
                    popsize = max(self.min_popsize, min(popsize, budget_cap - current_total))
                self._spawn(regime, popsize)

            self._merge_redundant_niches()

            self.gen_log.append({
                'gen': gen,
                'evals': self.evals,
                'n_niches': len(self.niches),
                'total_pop': sum(nd['cma'].lam for nd in self.niches),
                'best_f': self.best_f,
                'regime_prob': {'large': self.regime_success['large'],
                                 'small': self.regime_success['small']},
            })

            if verbose and gen % 20 == 0:
                print(f"gen {gen:5d} evals {self.evals:7d} niches {len(self.niches):2d} "
                      f"best {self.best_f:.6g} succ(L/S)={self.regime_success['large']:.2f}"
                      f"/{self.regime_success['small']:.2f}")

        return self.best_x, self.best_f, self.niches

    def export_log_csv(self, path: str) -> None:
        """Write the per-generation log (gen, evals, n_niches, best_f,
        regime_prob_large, regime_prob_small) to a CSV file."""
        export_gen_log_csv(self.gen_log, path)
