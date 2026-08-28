#!/usr/bin/env python3
"""Run the niching BIPOP-CMA-ES, LM-CMA-ES, and L-SHADE baselines on GNBG.

Examples
--------
python run_gnbg.py --algorithm all --functions 1-24 --runs 30 --budget 500000
python run_gnbg.py --algorithm l-shade --functions 1,7,16 --runs 5 --budget 500000
"""
from __future__ import annotations

import argparse

from niching_ea_baselines import run_gnbg_suite


def parse_functions(spec: str) -> tuple[int, ...]:
    out: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    if not out:
        raise argparse.ArgumentTypeError("no function IDs supplied")
    return tuple(dict.fromkeys(out))


def parse_algorithms(spec: str) -> tuple[str, ...]:
    if spec.strip().lower() == "all":
        return ("bipop-cma-es", "lm-cma-es", "l-shade")
    aliases = {
        "bipop": "bipop-cma-es",
        "bipop-cma-es": "bipop-cma-es",
        "lm": "lm-cma-es",
        "lm-cma-es": "lm-cma-es",
        "l-shade": "l-shade",
        "lshade": "l-shade",
    }
    out = []
    for token in spec.split(","):
        key = token.strip().lower()
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"unknown algorithm: {token}")
        out.append(aliases[key])
    return tuple(dict.fromkeys(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithm", default="all", type=parse_algorithms)
    ap.add_argument("--functions", default="1-24", type=parse_functions)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--log-dir", default="gnbg_logs")
    ap.add_argument("--summary", default="gnbg_summary.csv")
    ap.add_argument("--target", type=float, default=1e-8)
    args = ap.parse_args()

    results = run_gnbg_suite(
        algorithms=args.algorithm,
        problem_ids=args.functions,
        runs=args.runs,
        budget=args.budget,
        seed0=args.seed0,
        log_dir=args.log_dir,
        summary_csv=args.summary,
        target=args.target,
    )

    for r in results:
        target = "FAIL" if r.target_fe is None else str(r.target_fe)
        print(
            f"{r.algorithm:14s} f{r.problem_id:02d} run={r.run:02d} "
            f"FE={r.evaluations:7d} error={r.final_error:.3e} target_FE={target}"
        )


if __name__ == "__main__":
    main()
