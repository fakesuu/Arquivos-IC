"""
lshade_niche.py
------------------
L-SHADE with niching AND success-feedback-driven operator adaptation.

Baseline L-SHADE already adapts CR and F via a success-history memory
(Tanabe & Fukunaga, 2014): each generation's successful (improving) CR/F
values are folded into a historical memory via a fitness-weighted Lehmer
mean. This file extends that idea one level up, to the MUTATION OPERATOR
ITSELF, following the SaDE / EPSDE family of "adaptive operator
selection" methods (Qin & Suganthan, 2009):

    - three structurally different mutation operators are pooled:
        * 'pbest1'     current-to-pbest/1 (+archive) -- exploitative,
                        the original L-SHADE operator
        * 'curr2rand1' current-to-rand/1 (simplified) -- balanced
        * 'rand1'      classic rand/1 -- explorative, no pbest bias,
                        useful for keeping distinct niches alive
    - each operator has its OWN success-history CR/F memory (some
      operators may simply want different step sizes than others),
    - every individual's operator is chosen by roulette-wheel sampling
      from a selection-probability vector,
    - every `learning_period` generations, the probability vector is
      recomputed from each operator's recent success rate
      S_k = successes_k / (successes_k + failures_k) over that window,
      normalized to sum to 1 (the standard SaDE update rule) -- so
      operators that have actually been producing improving trials get
      selected more often, and the ones that haven't get scaled back
      (never to exactly zero, so they can recover if the landscape
      changes).

Niching (from the previous version) is unchanged: replacement is decided
by CROWDING (a trial only competes with its nearest neighbour among a
random subset), and periodic clearing prunes near-duplicate individuals
after LPSR shrinks the population.
"""
from __future__ import annotations
import numpy as np
from .niching_utils import (clearing, speciate, nearest_crowding_partner,
                             default_niche_radius, reflect_into_bounds, export_gen_log_csv)

OPERATORS = ('pbest1', 'curr2rand1', 'rand1')


class NichedLSHADE:
    def __init__(self, objective, bounds, budget,
                 pop_init=None, pop_min=4, memory_size=6,
                 archive_rate=2.0, crowding_factor=3,
                 niche_radius=None, learning_period=20, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.f = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = len(self.bounds)
        self.budget = budget
        self.pop_min = pop_min
        self.pop_init = pop_init or max(pop_min, 18 * self.dim)
        self.NP = self.pop_init
        self.H = memory_size
        self.archive_rate = archive_rate
        self.CF = crowding_factor
        self.niche_radius = niche_radius or default_niche_radius(self.bounds, self.dim)
        self.LP = learning_period

        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self.pop = lo + np.random.rand(self.NP, self.dim) * (hi - lo)
        self.fit = np.asarray(self.f(self.pop), dtype=float)
        self.evals = self.NP

        self.archive = np.empty((0, self.dim))

        # per-operator success-history CR/F memory
        self.K = len(OPERATORS)
        self.M_CR = np.full((self.K, self.H), 0.5)
        self.M_F = np.full((self.K, self.H), 0.5)
        self.mem_idx = np.zeros(self.K, dtype=int)

        # adaptive operator selection (SaDE-style)
        self.op_prob = np.full(self.K, 1.0 / self.K)
        self.op_success = np.zeros(self.K)
        self.op_failure = np.zeros(self.K)
        self.gens_since_update = 0

        i0 = np.argmin(self.fit)
        self.best_x, self.best_f = self.pop[i0].copy(), float(self.fit[i0])

        # per-generation log: [{'gen','evals','NP','best_f','op_prob'}, ...]
        # op_prob is the real per-individual operator-selection probability
        # vector (one entry per operator in OPERATORS), updated every
        # `learning_period` generations from recent success rates.
        self.gen_log: list[dict] = []

    def _clip(self, X):
        # reflective repair, not clamping -- see niching_utils.reflect_into_bounds.
        # Mutation is the only place DE can go out of bounds (initial
        # population and crossover-with-parent are already feasible by
        # construction), so this is the single call site that matters.
        return reflect_into_bounds(X, self.bounds)

    def _sample_CR_F(self, op_idx):
        r = np.random.randint(0, self.H)
        cr = float(np.clip(np.random.normal(self.M_CR[op_idx, r], 0.1), 0, 1))
        while True:
            val = self.M_F[op_idx, r] + 0.1 * np.tan(np.pi * (np.random.rand() - 0.5))
            if val > 0:
                f = min(val, 1.0)
                break
        return cr, f, r

    def _update_memory(self, op_idx, S_CR, S_F, S_dfit):
        if len(S_CR) == 0:
            return
        w = S_dfit / np.sum(S_dfit)
        mean_CR = np.sum(w * S_CR)
        mean_F = np.sum(w * S_F ** 2) / np.sum(w * S_F)
        self.M_CR[op_idx, self.mem_idx[op_idx]] = mean_CR
        self.M_F[op_idx, self.mem_idx[op_idx]] = mean_F
        self.mem_idx[op_idx] = (self.mem_idx[op_idx] + 1) % self.H

    def _distinct(self, n_needed, exclude, pool_size):
        exclude = set(exclude)
        avail = [j for j in range(pool_size) if j not in exclude]
        if len(avail) >= n_needed:
            return list(np.random.choice(avail, n_needed, replace=False))
        # population too small (can happen near pop_min): allow repeats
        return list(np.random.choice(pool_size, n_needed, replace=True))

    def _mutate(self, i, op_idx, F, order_by_fit, p_count_i):
        union_pool = np.vstack([self.pop, self.archive]) if len(self.archive) else self.pop
        op = OPERATORS[op_idx]

        if op == 'pbest1':
            pbest_pool = order_by_fit[:p_count_i]
            pbest = self.pop[np.random.choice(pbest_pool)]
            r1, = self._distinct(1, {i}, self.NP)
            r2, = self._distinct(1, {i, r1}, len(union_pool))
            mutant = self.pop[i] + F * (pbest - self.pop[i]) + F * (self.pop[r1] - union_pool[r2])
        elif op == 'curr2rand1':
            r1, r2, r3 = self._distinct(3, {i}, self.NP)
            mutant = self.pop[i] + F * (self.pop[r1] - self.pop[i]) + F * (self.pop[r2] - self.pop[r3])
        else:  # 'rand1' -- fully explorative, no dependence on x_i
            r1, r2, r3 = self._distinct(3, {i}, self.NP)
            mutant = self.pop[r1] + F * (self.pop[r2] - self.pop[r3])

        return self._clip(mutant[None, :])[0]

    def _next_pop_size(self):
        frac = self.evals / self.budget
        return max(self.pop_min, round(self.pop_init + frac * (self.pop_min - self.pop_init)))

    def _shrink_population(self, new_NP):
        if new_NP >= self.NP:
            return
        order = np.argsort(self.fit)
        keep = order[:new_NP]
        self.pop = self.pop[keep]
        self.fit = self.fit[keep]
        self.NP = new_NP
        max_arc = int(self.NP * self.archive_rate)
        if len(self.archive) > max_arc:
            idx = np.random.choice(len(self.archive), max_arc, replace=False)
            self.archive = self.archive[idx]

    def _update_operator_probabilities(self):
        S = self.op_success / np.maximum(self.op_success + self.op_failure, 1e-12)
        S = S + 0.01  # keep every operator selectable (recovery epsilon)
        self.op_prob = S / np.sum(S)
        self.op_success[:] = 0
        self.op_failure[:] = 0

    def run(self, verbose=False, report_every=25):
        gen = 0
        p_min_frac = 2.0 / self.NP
        while self.evals < self.budget and self.NP > 2:
            gen += 1
            self.gens_since_update += 1
            order_by_fit = np.argsort(self.fit)
            p_frac = np.random.uniform(p_min_frac, 0.2, size=self.NP)
            p_count = np.maximum(2, (p_frac * self.NP).astype(int))

            op_choice = np.random.choice(self.K, size=self.NP, p=self.op_prob)
            CRs = np.empty(self.NP)
            Fs = np.empty(self.NP)
            mem_r = np.empty(self.NP, dtype=int)
            trials = np.empty_like(self.pop)

            for i in range(self.NP):
                op_idx = op_choice[i]
                cr, f, r = self._sample_CR_F(op_idx)
                CRs[i], Fs[i], mem_r[i] = cr, f, r

                mutant = self._mutate(i, op_idx, f, order_by_fit, p_count[i])
                jrand = np.random.randint(self.dim)
                mask = np.random.rand(self.dim) < cr
                mask[jrand] = True
                trials[i] = np.where(mask, mutant, self.pop[i])

            trial_fit = np.asarray(self.f(trials), dtype=float)
            self.evals += self.NP

            S_CR = [[] for _ in range(self.K)]
            S_F = [[] for _ in range(self.K)]
            S_dfit = [[] for _ in range(self.K)]
            new_archive = []

            for i in range(self.NP):
                cand = np.random.choice(self.NP, size=min(self.CF, self.NP), replace=False)
                if i not in cand:
                    cand = np.append(cand, i)
                target = nearest_crowding_partner(trials[i], self.pop, cand)
                op_idx = op_choice[i]

                if trial_fit[i] <= self.fit[target]:
                    if trial_fit[i] < self.fit[target]:
                        S_CR[op_idx].append(CRs[i]); S_F[op_idx].append(Fs[i])
                        S_dfit[op_idx].append(abs(self.fit[target] - trial_fit[i]))
                        new_archive.append(self.pop[target].copy())
                        self.op_success[op_idx] += 1
                    self.pop[target] = trials[i]
                    self.fit[target] = trial_fit[i]
                else:
                    self.op_failure[op_idx] += 1

            if new_archive:
                self.archive = np.vstack([self.archive, np.array(new_archive)]) if len(self.archive) \
                    else np.array(new_archive)

            for k in range(self.K):
                self._update_memory(k, np.array(S_CR[k]), np.array(S_F[k]), np.array(S_dfit[k]))

            if self.gens_since_update >= self.LP:
                self._update_operator_probabilities()
                self.gens_since_update = 0

            i0 = np.argmin(self.fit)
            if self.fit[i0] < self.best_f:
                self.best_f, self.best_x = float(self.fit[i0]), self.pop[i0].copy()

            new_NP = self._next_pop_size()
            self._shrink_population(new_NP)
            p_min_frac = 2.0 / max(self.NP, 3)

            self.gen_log.append({
                'gen': gen,
                'evals': self.evals,
                'NP': self.NP,
                'best_f': self.best_f,
                'op_prob': dict(zip(OPERATORS, self.op_prob.copy())),
            })

            if gen % report_every == 0:
                eff = clearing(self.pop, self.fit, self.niche_radius, capacity=1)
                dup = np.isinf(eff)
                if np.any(dup) and self.NP > self.pop_min + np.sum(dup):
                    lo, hi = self.bounds[:, 0], self.bounds[:, 1]
                    n_dup = int(np.sum(dup))
                    self.pop[dup] = lo + np.random.rand(n_dup, self.dim) * (hi - lo)
                    self.fit[dup] = self.f(self.pop[dup])
                    self.evals += n_dup

            if verbose and gen % report_every == 0:
                n_species = len(speciate(self.pop, self.fit, self.niche_radius))
                probs = " ".join(f"{op}:{p:.2f}" for op, p in zip(OPERATORS, self.op_prob))
                print(f"gen {gen:5d} evals {self.evals:7d} NP {self.NP:4d} "
                      f"niches~{n_species:3d} best {self.best_f:.6g} [{probs}]")

        return self.best_x, self.best_f, speciate(self.pop, self.fit, self.niche_radius)

    def export_log_csv(self, path: str) -> None:
        """Write the per-generation log (gen, evals, NP, best_f,
        op_prob_<operator name> for each operator) to a CSV file."""
        export_gen_log_csv(self.gen_log, path)
