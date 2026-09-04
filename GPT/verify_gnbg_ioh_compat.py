"""Verify the GNBG/IOH scalar-return compatibility used by the final runner.

Run from the same Python/Spyder environment in which iohgnbg and ioh are installed:
    %runfile 'C:/Users/Julio Cesar/Documents/GitHub/Arquivos-IC/GPT/verify_gnbg_ioh_compat.py' --wdir
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gnbg_niching_baselines_final as ea

try:
    iohgnbg = importlib.import_module("iohgnbg")
    ioh = importlib.import_module("ioh")
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Install the official GNBG package with:\n"
        "    python -m pip install iohgnbg\n"
        "and run this check again."
    ) from exc

print("Python:", sys.version.split()[0])
print("NumPy:", __import__("numpy").__version__)
print("iohgnbg:", getattr(iohgnbg, "__version__", "unknown"))
print("ioh:", getattr(ioh, "__version__", "unknown"))

# load_gnbg_problem applies the compatibility patch before calling the official
# iohgnbg.get_problem(...) factory.
p = ea.load_gnbg_problem(1)
x = 0.5 * (p.lower + p.upper)
y = p.fun(x)

print("GNBG problem:", p.problem_id)
print("dimension:", p.dimension)
print("objective type:", type(y).__name__)
print("objective value:", y)

if not isinstance(y, float):
    raise SystemExit(
        "Compatibility check failed: objective is not a Python float. "
        f"Got {type(y).__name__}."
    )

print("OK: the objective callback returns a Python float for a 1-D point.")
