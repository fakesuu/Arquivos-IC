"""
niched_ea
=========
Three niched evolutionary-algorithm baselines for high-dimensional,
multi-modal continuous (box-constrained) optimization:

    - NichedBIPOPCMAES  (bipop_cma_niche.py)
    - NichedLMMAES       (lm_ma_niche.py)   -- limited-memory CMA-ES family
    - NichedLSHADE       (lshade_niche.py)

All three share:
    - a common niching layer (Petrowski clearing / crowding / speciation
      -- niching_utils.py)
    - reflective boundary handling (niching_utils.reflect_into_bounds)
    - success/failure-feedback-driven parameter adaptation (active
      covariance / direction-vector updates for the CMA-ES family,
      SaDE-style adaptive operator selection for L-SHADE)
    - a linearly-reducing population-size schedule (LPSR for L-SHADE;
      an analogous niche/population budget cap for the other two)
    - a per-generation log (`.gen_log`) and CSV export
      (`.export_log_csv(path)`)

Plus a GNBG benchmark integration layer (gnbg_integration.py) for
evaluation logging and 1e-8 error-threshold tracking.

See README.md for the full design notes, honesty caveats, and a usage
walkthrough -- read that before relying on any of this for reported
numbers, especially the caveats around GNBGProblem's attribute-name
guessing and the LM-MA-ES / active-CMA-ES simplifications.

Quick start:
    from niched_ea import NichedBIPOPCMAES, NichedLMMAES, NichedLSHADE
    import numpy as np

    def sphere(X):
        return np.sum(X**2, axis=1)

    bounds = np.tile([-5.0, 5.0], (10, 1))
    opt = NichedBIPOPCMAES(sphere, bounds, budget=20000, seed=0)
    best_x, best_f, niches = opt.run()
"""
from .niching_utils import (
    clearing,
    speciate,
    nearest_crowding_partner,
    reflect_into_bounds,
    default_niche_radius,
    export_gen_log_csv,
)
from .bipop_cma_niche import CMAES, NichedBIPOPCMAES
from .lm_ma_niche import LMMAES, NichedLMMAES
from .lshade_niche import NichedLSHADE, OPERATORS
from .gnbg_integration import (
    GNBGProblem,
    GNBGLiteProblem,
    EvaluationLogger,
    run_on_gnbg,
)

__all__ = [
    "NichedBIPOPCMAES", "CMAES",
    "NichedLMMAES", "LMMAES",
    "NichedLSHADE", "OPERATORS",
    "clearing", "speciate", "nearest_crowding_partner",
    "reflect_into_bounds", "default_niche_radius", "export_gen_log_csv",
    "GNBGProblem", "GNBGLiteProblem", "EvaluationLogger", "run_on_gnbg",
]

__version__ = "1.0.0"
