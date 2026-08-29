"""
gnbg_integration.py
----------------------
Integration layer between the three niched EA baselines
(bipop_cma_niche.NichedBIPOPCMAES, lm_ma_niche.NichedLMMAES,
lshade_niche.NichedLSHADE) and the GNBG benchmark suite (Yazdani et al.,
GECCO/CEC 2024 competition on single-objective box-constrained numerical
optimization: 24 instances f1..f24).

HONESTY NOTE ON SCOPE: I was not able to fetch the official GNBG.py
source (github.com/Danial-Yazdani/GNBG_Instances.Python) in this session
-- raw GitHub file content wasn't retrievable with the tools available
here, so I could not verify the exact current attribute/method names of
the official instance object. `GNBGProblem` below is therefore a
DEFENSIVE ADAPTER: it tries several plausible attribute-name aliases
(informed by the confirmed, published competition protocol) and fails
loudly, naming exactly which attribute it couldn't find, rather than
silently doing the wrong thing. If your installed version uses different
names, edit the `ATTR_ALIASES` dict below -- that should be the only
thing that needs to change.

Confirmed protocol details (cross-checked across the GNBG papers and the
GECCO/CEC competition pages):
    - 24 problem instances (f1..f24): unimodal, single-component
      multimodal, and multi-component multimodal
    - dimensionality is 30 for (almost) all instances
    - competition budget: 500,000 FEs for f1-f15, 1,000,000 for f16-f24
    - error is measured as e = |f_best_so_far - f_optimum|
    - acceptance threshold: e <= 1e-8 counts as a "successful" run
    - reported metrics: average absolute error, average FEs-to-threshold
      (over successful runs), success rate across repeated runs
That is exactly what `EvaluationLogger` below computes.

Contents:
    - GNBGProblem      : adapter around an official GNBG instance object
    - GNBGLiteProblem   : NON-OFFICIAL synthetic stand-in, same interface
                           shape, for exercising this integration without
                           the official repo installed
    - EvaluationLogger  : wraps ANY batched objective to count
                           evaluations exactly, log a best-error curve,
                           and detect the first evaluation at which
                           error <= threshold
    - run_on_gnbg       : convenience runner wiring an optimizer class +
                           a problem + a logger together
"""
from __future__ import annotations
import numpy as np

# If your installed GNBG instance object uses different attribute names
# than the ones tried here, edit these lists (checked in order) -- this
# should be the only edit needed to match your actual GNBG version.
ATTR_ALIASES = {
    'dim':       ['Dimension', 'dim', 'n', 'D'],
    'lower':     ['MinCoordinate', 'lower_bound', 'lb', 'Lb', 'xmin'],
    'upper':     ['MaxCoordinate', 'upper_bound', 'ub', 'Ub', 'xmax'],
    'f_opt':     ['OptimumValue', 'fopt', 'f_opt', 'optimum_value'],
    'x_opt':     ['OptimumPosition', 'xopt', 'x_opt', 'optimum_position'],
    'max_evals': ['MaxEvals', 'max_evals', 'MaxFEs', 'budget'],
}


def _find_attr(obj, key):
    for name in ATTR_ALIASES[key]:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(
        f"Could not find a '{key}' attribute on the GNBG instance object "
        f"(tried {ATTR_ALIASES[key]}). Add the actual attribute name from "
        f"your installed GNBG version to ATTR_ALIASES['{key}'] in "
        f"gnbg_integration.py."
    )


class GNBGProblem:
    """
    Thin adapter around an official GNBG problem instance (one object per
    f1..f24 instance, however your installed GNBG package constructs or
    loads it from the corresponding f#.mat parameter file).

    Usage:
        gnbg_instance = ...  # from your installed GNBG_Instances.Python
        problem = GNBGProblem(gnbg_instance)
        best_x, best_f, logger = run_on_gnbg(NichedBIPOPCMAES, problem)

    Expects the wrapped instance to expose a dimensionality attribute,
    lower/upper bound attributes, and an optimum-value attribute (see
    ATTR_ALIASES). Evaluation is resolved, in order, as: the instance
    being directly callable, or a .fitness()/.evaluate() method -- tried
    batched first (X: (N,dim) -> (N,)), falling back to one row at a time
    if the batched call fails, since some GNBG ports evaluate a single
    solution per call.
    """

    def __init__(self, gnbg_instance, f_opt=None, bounds=None, dim=None):
        self._inst = gnbg_instance
        self.dim = dim if dim is not None else int(_find_attr(gnbg_instance, 'dim'))

        if bounds is not None:
            self.bounds = np.asarray(bounds, dtype=float)
        else:
            lo = np.broadcast_to(np.asarray(_find_attr(gnbg_instance, 'lower'), dtype=float),
                                  (self.dim,)).copy()
            hi = np.broadcast_to(np.asarray(_find_attr(gnbg_instance, 'upper'), dtype=float),
                                  (self.dim,)).copy()
            self.bounds = np.stack([lo, hi], axis=1)

        if f_opt is not None:
            self.f_opt = float(f_opt)
        else:
            self.f_opt = float(np.asarray(_find_attr(gnbg_instance, 'f_opt')).reshape(-1)[0])

        try:
            self.max_evals = int(np.asarray(_find_attr(gnbg_instance, 'max_evals')).reshape(-1)[0])
        except AttributeError:
            self.max_evals = None  # not all instances expose this; pass budget explicitly then

        self._eval_fn = self._resolve_eval_fn()

    def _resolve_eval_fn(self):
        inst = self._inst
        candidates = []
        if callable(inst):
            candidates.append(inst)
        for name in ('fitness', 'evaluate', 'Fitness', 'Evaluate'):
            if hasattr(inst, name):
                candidates.append(getattr(inst, name))
        if not candidates:
            raise AttributeError(
                "The GNBG instance object is not callable and has none of "
                "fitness()/evaluate()/Fitness()/Evaluate(). Add the correct "
                "method name in GNBGProblem._resolve_eval_fn()."
            )
        return candidates[0]

    def __call__(self, X):
        X = np.atleast_2d(X)
        try:
            out = np.asarray(self._eval_fn(X), dtype=float).reshape(-1)
            if out.shape[0] == X.shape[0]:
                return out
        except Exception:
            pass
        # per-row fallback for instance APIs that evaluate one solution per call
        return np.array([float(np.asarray(self._eval_fn(x)).reshape(-1)[0]) for x in X])


class GNBGLiteProblem:
    """
    NOT the official GNBG benchmark -- no official parameter files went
    into this. It's a small synthetic stand-in built around the same
    *structural idea* the GNBG papers describe (the objective is the
    minimum over k components, each a rotated, power-transformed
    quadratic bowl with its own center, depth, and conditioning), so you
    can exercise this integration (logging, threshold detection, all
    three algorithms) end-to-end without the official f1..f24 .mat files
    installed. Do not use this for competition-comparable numbers -- get
    the real instances from GNBG_Instances.Python for that.
    """

    def __init__(self, dim=10, n_components=5, bound=100.0, seed=None):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.bounds = np.tile([-bound, bound], (dim, 1)).astype(float)
        self.n_components = n_components

        self.centers = rng.uniform(-bound * 0.8, bound * 0.8, size=(n_components, dim))
        self.depths = np.concatenate([[0.0], rng.uniform(1.0, 500.0, size=n_components - 1)])
        self.rotations = [self._random_rotation(dim, rng) for _ in range(n_components)]
        self.scales = rng.uniform(0.5, 5.0, size=(n_components, dim))
        self.powers = rng.uniform(1.6, 2.4, size=n_components)

        opt_idx = int(np.argmin(self.depths))
        self.x_opt = self.centers[opt_idx].copy()
        self.f_opt = 0.0  # by construction: the deepest component has depth 0 at its own center
        self.max_evals = 500_000 if dim <= 30 else 1_000_000

    @staticmethod
    def _random_rotation(dim, rng):
        A = rng.normal(size=(dim, dim))
        Q, _ = np.linalg.qr(A)
        return Q

    def __call__(self, X):
        X = np.atleast_2d(X)
        vals = np.empty((X.shape[0], self.n_components))
        for k in range(self.n_components):
            d = (X - self.centers[k]) @ self.rotations[k]
            r = np.sum(self.scales[k] * np.abs(d) ** self.powers[k], axis=1)
            vals[:, k] = self.depths[k] + r
        return np.min(vals, axis=1)


class EvaluationLogger:
    """
    Wraps ANY batched objective (X: (N,dim) -> fitness: (N,)) to add,
    without changing its numerical behaviour:
        - exact evaluation counting,
        - a recorded (evals, best_error_so_far) curve, one point per
          objective call (i.e. per generation/batch, not per individual
          -- compact enough to plot directly),
        - detection of the first EVALUATION (not batch) at which
          error = |f_best - f_opt| <= threshold, matching the GNBG
          competition's acceptance criterion (default threshold 1e-8),
        - a .summary() reporting the three official GNBG metrics: final
          absolute error, FEs-to-threshold (None if never reached), and
          whether the threshold was reached at all ("success").

    Threshold-crossing is checked per INDIVIDUAL within each batch, not
    just once per batch: a niched algorithm may call the objective with
    the combined offspring of several niches at once, and the FE count
    at which the threshold is first crossed should be the count at that
    specific individual, not rounded up to the end of the whole batch.
    """

    def __init__(self, objective, f_opt=0.0, threshold=1e-8, max_evals=None):
        self._f = objective
        self.f_opt = float(f_opt)
        self.threshold = float(threshold)
        self.max_evals = max_evals

        self.evals = 0
        self.best_f = np.inf
        self.best_error = np.inf
        self.history = []          # [(evals_after_this_call, best_error_so_far), ...]
        self.fe_to_threshold = None
        self.success = False

    def __call__(self, X):
        X = np.atleast_2d(X)
        fit = np.asarray(self._f(X), dtype=float).reshape(-1)

        for f in fit:
            self.evals += 1
            if f < self.best_f:
                self.best_f = float(f)
                self.best_error = abs(self.best_f - self.f_opt)
            if not self.success and self.best_error <= self.threshold:
                self.success = True
                self.fe_to_threshold = self.evals

        self.history.append((self.evals, self.best_error))
        return fit

    def summary(self):
        return {
            'evals_used': self.evals,
            'best_f': self.best_f,
            'best_error': self.best_error,
            'success': self.success,
            'fe_to_threshold': self.fe_to_threshold,
            'threshold': self.threshold,
        }


def run_on_gnbg(optimizer_cls, problem, budget=None, threshold=1e-8, verbose=False, **kwargs):
    """
    Run one of the three niched optimizers (NichedBIPOPCMAES, NichedLMMAES,
    NichedLSHADE) on a GNBG-style problem (GNBGProblem or GNBGLiteProblem),
    with evaluation logging and threshold tracking wired in automatically.
    No changes are needed to the optimizer files themselves -- the logger
    is just a callable dropped in where they already expect `objective`.

    Returns (best_x, best_f, logger). logger.summary() has evaluations
    used and the FE count at which error <= threshold was first reached
    (None if it never was, over this run's budget).
    """
    budget = budget or getattr(problem, 'max_evals', None)
    if budget is None:
        raise ValueError("No budget given and the problem has no max_evals attribute.")

    logger = EvaluationLogger(problem, f_opt=problem.f_opt, threshold=threshold, max_evals=budget)
    opt = optimizer_cls(logger, problem.bounds, budget, seed=kwargs.pop('seed', None), **kwargs)
    best_x, best_f, niches = opt.run(verbose=verbose)
    return best_x, best_f, logger
