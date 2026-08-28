Integrated GNBG Niching Baselines
=================================

Main module:
  gnbg_niching_baselines_final.py

Algorithms:
  1. BIPOP-CMA-ES (BIPOP-style restarts + full covariance CMA-ES)
  2. LM-CMA-ES (limited-memory covariance adaptation)
  3. L-SHADE (success-history DE with strategy portfolio)

Shared features:
  - reflective box constraints
  - decision-space niching / fitness sharing with elite archive
  - success-conditioned operator adaptation
  - population-size reduction
  - evaluation-level CSV logging
  - generation-level CSV logging with operator probabilities/counts
  - target-fe tracking for |best_f - f*| <= 1e-8
  - GNBG / IOHGNBG adapter
  - exact evaluation-budget enforcement

Installation:
  pip install numpy iohgnbg

Example:
  python run_gnbg_final.py --algorithm all --functions 1-24 --runs 30 \
      --budget 500000 --log-dir gnbg_logs --summary gnbg_summary.csv

The evaluation log contains one row per FE. The generation log contains one row
per optimizer generation, including population size, operator probabilities,
operator counts, best value, error, and target status.
