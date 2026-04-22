"""Launch the 5-seed sweep for D1 and F6.

Usage:
    python scripts/run_multi_seed.py [--seeds N] [--d1 | --f6 | --both]

Produces:
    results/multi_seed_d1.json   — 5-seed TFIM scaling at n ∈ {4, 6, 8}
    results/multi_seed_f6.json   — 5-seed Nash-vs-SA on 3 topologies
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pqcgraphs.experiments import exp_d1_tfim_scaling, exp_f6_vs_sa_dqas
from pqcgraphs.game.multi_seed import run_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds to sweep (default 5).")
    ap.add_argument("--d1", action="store_true")
    ap.add_argument("--f6", action="store_true")
    ap.add_argument("--both", action="store_true",
                    help="Run D1 followed by F6 (default when no flag).")
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    run_d1 = args.d1 or args.both or (not args.f6)
    run_f6 = args.f6 or args.both or (not args.d1)

    if run_d1:
        print(f"=== D1 TFIM scaling, {args.seeds} seeds ===")
        run_seeds(
            exp_d1_tfim_scaling.run,
            seeds=seeds,
            kwargs=dict(n_values=(4, 6, 8), n_iters=15,
                        population_size=3, theta_gd_steps=40),
            out_path=Path("results/multi_seed_d1.json"),
        )

    if run_f6:
        print(f"=== F6 Nash-vs-SA head-to-head, {args.seeds} seeds ===")
        run_seeds(
            exp_f6_vs_sa_dqas.run,
            seeds=seeds,
            kwargs=dict(n_iters=15, population_size=3),
            out_path=Path("results/multi_seed_f6.json"),
        )


if __name__ == "__main__":
    main()
