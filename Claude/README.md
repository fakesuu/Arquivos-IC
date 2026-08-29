# niched_ea

Three niched evolutionary-algorithm baselines for high-dimensional,
multi-modal, box-constrained continuous optimization:

| Algorithm            | File                          | Class                |
|-----------------------|-------------------------------|-----------------------|
| BIPOP-CMA-ES + niching | `niched_ea/bipop_cma_niche.py` | `NichedBIPOPCMAES`    |
| LM-MA-ES + niching     | `niched_ea/lm_ma_niche.py`     | `NichedLMMAES`        |
| L-SHADE + niching      | `niched_ea/lshade_niche.py`    | `NichedLSHADE`        |

plus a shared niching layer, a GNBG benchmark integration, and
per-generation logging, all described below.

```
niched_ea/
    __init__.py           top-level package exports
    niching_utils.py       clearing, speciation, crowding, reflective
                            bounds, CSV export, niche-radius heuristic
    bipop_cma_niche.py     CMAES core + NichedBIPOPCMAES
    lm_ma_niche.py          LMMAES core + NichedLMMAES
    lshade_niche.py         NichedLSHADE (SHADE + LPSR + adaptive ops)
    gnbg_integration.py     GNBGProblem / GNBGLiteProblem / EvaluationLogger
demo.py                    runnable smoke tests + usage examples
README.md                  this file
```

## Install / run

No install step -- it's a plain importable package. From the directory
containing `niched_ea/`:

```python
import numpy as np
from niched_ea import NichedBIPOPCMAES, NichedLMMAES, NichedLSHADE

def sphere(X):
    return np.sum(X**2, axis=1)

bounds = np.tile([-5.0, 5.0], (10, 1))   # (dim, 2) array of [lo, hi]
opt = NichedBIPOPCMAES(sphere, bounds, budget=20000, seed=0)
best_x, best_f, niches = opt.run()
```

All three optimizers share the same constructor shape --
`(objective, bounds, budget, ..., seed=None)` -- and the same
`.run(verbose=False) -> (best_x, best_f, niches)` interface, so they're
interchangeable in a benchmarking loop.

`python3 demo.py` runs a full smoke-test suite (multi-modal benchmarks,
the GNBG integration, and the per-generation logging) end to end.

## What each algorithm actually does

### Niching (all three)
A single search distribution collapses onto one basin on a multi-modal
landscape. Each algorithm runs several sub-populations ("niches") in
parallel or maintains spatial diversity within one population:

- **BIPOP-CMA-ES / LM-MA-ES**: multiple independent instances run in
  parallel; each generation's pooled offspring go through **Petrowski
  clearing** (`niching_utils.clearing`) to identify redundant coverage;
  niches whose means converge within `niche_radius` of each other are
  merged; stalled niches restart on a population-size schedule (BIPOP's
  large/small alternation, or LM-MA-ES's success-conditioned IPOP growth).
- **L-SHADE**: a single population with **crowding replacement** -- a
  trial only competes against its nearest neighbor among a random
  subset, not the whole population, so distinct sub-populations can
  persist on separate optima. Periodic clearing prunes near-duplicates
  after the population shrinks.

`niche_radius` is a heuristic (`default_niche_radius`) and matters a
lot -- tune it per problem.

### Success/failure-feedback-driven parameter adaptation
- **BIPOP-CMA-ES**: **active covariance adaptation** (Jastrebski &
  Arnold, 2006) -- the rank-mu update learns from the worst offspring
  too, with a negative weight, safeguarded by the standard three-way
  `alpha = min(alpha_mu-, alpha_mueff-, alpha_posdef-)` scaling. Verified
  by A/B test: mean best-fitness on 10-D Rastrigin improved from 9.15 to
  5.97 across 5 seeds with this on vs off.
- **LM-MA-ES**: an analogous active update applied only to the
  limited-memory direction vectors (mean/step-size untouched, since
  there's no positive-definiteness proof for this implicit
  representation). The damping constant `eta_active=0.1` was picked by
  an empirical sweep, not from a paper -- an initial guess of 0.5 was
  tested and found to *hurt* performance, which is why the shipped
  default is different from the first attempt. Re-tune per problem.
- **L-SHADE**: a genuine ensemble of three mutation operators
  (`pbest1`, `curr2rand1`, `rand1`) with SaDE-style adaptive selection
  -- probabilities recomputed every `learning_period` generations from
  each operator's recent success rate, plus **per-operator** CR/F
  success-history memories (not just per-algorithm).

**Important asymmetry to keep in mind**: L-SHADE has real, literal
per-individual operator-selection probabilities. BIPOP-CMA-ES and
LM-MA-ES do not -- they have a single Gaussian sampling operator, so
their logged "probabilities" (`regime_prob`, `restart_prob`) are the
closest analogous feedback signals we built (which BIPOP restart regime
or how aggressively to grow LM-MA-ES's restart population), not
operator-choice probabilities in the SaDE sense. The code and the
per-generation log fields are named to keep this distinction visible
rather than implying more uniformity across the three than exists.

### Reflective boundary handling
All three repair infeasible points by **reflection** (`niching_utils.
reflect_into_bounds`), not clamping. Clamping piles samples up exactly
on the boundary and, for the CMA-ES family specifically, silently
corrupts the mean/covariance adaptation feedback (the "effective step"
no longer matches what the search distribution generated). Reflection
folds the excess distance back into the box via a triangle-wave `mod`,
correctly handling multi-bounce overshoots (e.g. a large CMA-ES sigma
in a small box) in one vectorized pass.

### Population-size reduction
- **L-SHADE**: linear population-size reduction (LPSR) was already a
  defining part of the algorithm from the first version -- population
  shrinks linearly from `pop_init` to `pop_min` over the budget.
- **BIPOP-CMA-ES / LM-MA-ES**: `pop_reduction=True` (default) caps the
  total population across active niches and the number of simultaneous
  niches, both shrinking linearly as budget is consumed -- applied at
  niche-*spawn* time, not by resizing a live CMA-ES/LM-MA-ES instance's
  `lambda` mid-run (which would require re-deriving its internal
  learning rates and isn't attempted here).
  **This also fixes a real bug**: BIPOP's "large" restart rule doubled
  `lambda` on every large restart with no ceiling
  (`base_lam * 2**large_restarts`). Simulating 15 restarts without the
  cap gives a population of 327,680 for a single generation; with the
  cap, it's bounded to 24. A/B tested: BIPOP-CMA-ES quality *improved*
  with the cap on (mean best 7.63 -> 6.80 over 6 seeds, ~18% fewer
  generations) because runaway restarts no longer torch the budget in
  one generation. For LM-MA-ES the cap didn't change results in testing
  because its restart growth was already separately capped at exponent
  6 by the success-conditioning logic -- the population-budget cap is
  defense-in-depth there, not a fix for an active defect.

### Per-generation logging
Every `.run()` call populates `.gen_log` (list of dicts: `gen`,
`evals`, population/niche-count fields, `best_f`, plus the
algorithm-specific probability field described above). Export with
`.export_log_csv(path)`. Verified: evals strictly non-decreasing across
logged generations, generation indices contiguous from 1, CSV round-trips
row-for-row with the in-memory log.

### GNBG benchmark integration
`gnbg_integration.py` provides:
- `GNBGProblem` -- adapter around an official GNBG instance (from
  `github.com/Danial-Yazdani/GNBG_Instances.Python`, CEC/GECCO 2024
  competition benchmark). **I could not fetch the official source in
  the session that wrote this** -- raw GitHub content wasn't
  retrievable with the tools available then, so the adapter is
  deliberately *defensive*: it tries several plausible attribute-name
  aliases (informed by the confirmed, published competition protocol --
  24 instances, error = `|f_best - f_opt|`, threshold `1e-8`) and fails
  loudly, naming exactly which attribute it couldn't find, rather than
  silently misreading your instance object. If your installed version
  uses different names, edit the single `ATTR_ALIASES` dict at the top
  of the file -- that should be the only change needed.
- `GNBGLiteProblem` -- a small, explicitly **non-official** synthetic
  stand-in (minimum over rotated, power-transformed quadratic
  components) for exercising the whole pipeline without the official
  repo installed. Do not use it for competition-comparable numbers.
- `EvaluationLogger` -- wraps any batched objective to count
  evaluations exactly and detect the first **individual** evaluation
  (not batch-rounded) at which `error <= threshold`. Verified against a
  hand-built deterministic fitness sequence to confirm it pinpoints the
  exact evaluation index.
- `run_on_gnbg(cls, problem, budget=None, threshold=1e-8, **kwargs)` --
  convenience runner.

Known limitation from testing: `evals_used` can slightly overshoot the
requested budget (all three algorithms only check budget between
generations, not mid-generation) -- expected, not a bug, but relevant if
you need a hard cap for competition-style reporting.

## Honesty notes (please read before reporting numbers)

- **LM-MA-ES, not literal LM-CMA-ES.** This implements the LM-MA-ES
  formulation (Loshchilov, Glasmachers & Beyer, 2017), the practical,
  well-specified member of the limited-memory CMA-ES family. I did not
  attempt to reproduce the original LM-CMA-ES's Cholesky-factor
  bookkeeping (Loshchilov, 2014) from memory, since I wasn't confident
  I'd get its exact update scheduling right.
- **Active-CMA-ES is simplified.** The published version additionally
  rescales each negative sample by its individual Mahalanobis norm
  before folding it into the rank-mu term; this implementation omits
  that per-sample rescaling. The positive-definiteness safeguard (the
  `alpha` three-way minimum) is retained, plus an eigenvalue floor as a
  numerical safety net.
- **Constants that came from a sweep, not a paper**: LM-MA-ES's
  `eta_active=0.1` and its memory-horizon schedule are practical
  defaults tuned informally, not literature-matched values. Re-tune per
  problem.
- **GNBGProblem's attribute names are best-effort, not verified**
  against the actual current source (see above) -- check
  `ATTR_ALIASES` against your installed version before trusting it
  silently.
- **`niche_radius` and other niching hyperparameters are heuristics.**
  In testing, niche count often collapses to 1 with generous budgets
  (a new niche only spawns when the current one stalls) -- seed
  multiple niches at init or tighten `niche_radius` if you want visible
  multi-niche diversity throughout a run.

Every numeric claim above (the A/B deltas, the runaway-growth numbers,
the threshold-detection accuracy) was produced by actually running the
code in this repository during development, not estimated -- but they're
single-configuration results on synthetic benchmarks (mostly Rastrigin),
not a substitute for validating on your own problem.
