import numpy as np
from niched_ea import (NichedBIPOPCMAES, NichedLMMAES, NichedLSHADE,
                        GNBGLiteProblem, run_on_gnbg)


def rastrigin(X):
    A = 10.0
    n = X.shape[1]
    return A * n + np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=1)


def himmelblau_like_multimodal(X):
    # 2D-only sanity function with 4 known global optima, tiled/summed
    # across dims in pairs for a quick multi-modal smoke test.
    n = X.shape[1]
    total = np.zeros(X.shape[0])
    for k in range(0, n - 1, 2):
        x, y = X[:, k], X[:, k + 1]
        total += (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return total


def run_test(name, cls, dim, bounds_val, budget, func, **kwargs):
    print(f"\n=== {name} | dim={dim} budget={budget} ===")
    bounds = np.tile([-bounds_val, bounds_val], (dim, 1))
    opt = cls(func, bounds, budget, seed=0, **kwargs)
    best_x, best_f, niches = opt.run(verbose=True)
    print(f"-> best_f={best_f:.6g}  n_final_niches={len(niches)}")
    assert np.isfinite(best_f)
    return best_x, best_f, niches


if __name__ == "__main__":
    np.random.seed(0)

    # Rastrigin 10D: highly multi-modal, translation-separable
    run_test("BIPOP-CMA-ES (niched)", NichedBIPOPCMAES, 10, 5.12, 40000, rastrigin)
    run_test("LM-MA-ES (niched)", NichedLMMAES, 10, 5.12, 40000, rastrigin)
    run_test("L-SHADE (niched)", NichedLSHADE, 10, 5.12, 40000, rastrigin)

    # Quick smoke test on a higher-dimensional problem for LM-MA-ES
    # (the algorithm this baseline is meant to scale to)
    run_test("LM-MA-ES (niched, 200D)", NichedLMMAES, 200, 5.12, 60000, rastrigin)

    # 2D multi-modal (4 known optima) - check multiple niches get found
    run_test("BIPOP-CMA-ES (niched, 2D 4-modal)", NichedBIPOPCMAES, 2, 6.0, 8000,
              himmelblau_like_multimodal)

    print("\nAll smoke tests completed without error.")

    # --- GNBG-style integration: evaluation logging + threshold tracking ---
    # NOTE: GNBGLiteProblem is a small NON-OFFICIAL stand-in (see
    # gnbg_integration.py) used here only to demonstrate the wiring; for
    # competition-comparable numbers use GNBGProblem with the official
    # f1..f24 instances from GNBG_Instances.Python.
    print("\n=== GNBG-style integration demo (GNBGLiteProblem, non-official) ===")
    for name, cls in [("BIPOP-CMA-ES", NichedBIPOPCMAES),
                       ("LM-MA-ES", NichedLMMAES),
                       ("L-SHADE", NichedLSHADE)]:
        problem = GNBGLiteProblem(dim=8, n_components=4, bound=50.0, seed=1)
        best_x, best_f, logger = run_on_gnbg(cls, problem, budget=60000, threshold=1e-8, seed=0)
        s = logger.summary()
        status = f"reached 1e-8 at FE {s['fe_to_threshold']}" if s['success'] else "threshold not reached"
        print(f"{name:14s} evals={s['evals_used']:6d}  error={s['best_error']:.3e}  {status}")

    # --- per-generation logging + CSV export ---
    print("\n=== per-generation log (gen, evals, population, operator/regime "
          "probabilities) ===")
    bounds10 = np.tile([-5.12, 5.12], (10, 1))
    opt = NichedLSHADE(rastrigin, bounds10, 20000, seed=0)
    opt.run()
    print("L-SHADE gen_log sample row:", opt.gen_log[len(opt.gen_log) // 2])
    opt.export_log_csv("/tmp/lshade_gen_log.csv")
    print("  -> full log written to /tmp/lshade_gen_log.csv "
          f"({len(opt.gen_log)} generations)")
