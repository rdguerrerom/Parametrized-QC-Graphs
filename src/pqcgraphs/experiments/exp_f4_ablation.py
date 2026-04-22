"""F4: Player-ablation on MaxCut K_4.

Run the full four-player Nash game, plus four reduced-roster runs that
each remove one player by setting its weight to 0. This isolates which
players' payoffs are load-bearing and which are redundant. Same task
(MaxCut K_4, grid_2x2 topology, |+>^4 warm start), same budget, same
seed.

Reports for each roster:
  - raw (f1, f2, f3, f4) at convergence
  - final Phi
  - structural metrics (n_ops, depth, n_params)
  - magic, eff_dim at output
  - post-hoc delta_Nash measured under the FULL four-player game
    (so we can see, e.g., that the "no anti-BP" ablation still appears as
    a high-gap state when judged by the full player set).

Primary manuscript claim supported: the full four-player set is
non-redundant — no single-player ablation reproduces the Nash equilibrium.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

import networkx as nx

from ..baselines import posthoc_nash_gap
from ..dag import grid_2d
from ..dag.initial_states import plus_layer
from ..game import PQCNashGame, make_default_players, potential
from ..objectives import maxcut_hamiltonian


ROSTERS: Dict[str, dict] = {
    "full_four_player": dict(w_anti_bp=0.3, w_anti_sim=0.2,
                             w_performance=1.0, w_hardware=0.2),
    "no_anti_bp":       dict(w_anti_bp=0.0, w_anti_sim=0.2,
                             w_performance=1.0, w_hardware=0.2),
    "no_anti_sim":      dict(w_anti_bp=0.3, w_anti_sim=0.0,
                             w_performance=1.0, w_hardware=0.2),
    "no_performance":   dict(w_anti_bp=0.3, w_anti_sim=0.2,
                             w_performance=0.0, w_hardware=0.2),
    "no_hardware":      dict(w_anti_bp=0.3, w_anti_sim=0.2,
                             w_performance=1.0, w_hardware=0.0),
}


def _full_metrics(dag, reference_players, topology) -> dict:
    raw = {p.name: float(p.raw(dag)) for p in reference_players}
    gap = posthoc_nash_gap(reference_players, dag, topology, rng_seed=0)
    return {
        "n_ops": dag.n_ops,
        "depth": dag.depth(),
        "n_params": dag.n_params,
        "gate_counts": dag.gate_counts(),
        "f1_eff_dim_ratio": raw["anti_bp"],
        "f2_magic_per_qubit": raw["anti_sim"],
        "f3_performance": raw["performance"],
        "f4_hardware_cost": raw["hardware"],
        "phi_full_game": float(potential(reference_players, dag)),
        "delta_nash_full_game": float(gap.delta_nash),
    }


def run(
    n_iters: int = 15,
    population_size: int = 3,
    seed: int = 0,
    out_path: Path = Path("results/exp_f4_ablation.json"),
) -> dict:
    n_qubits = 4
    G = nx.complete_graph(n_qubits)
    H = maxcut_hamiltonian(G)
    topo = grid_2d(2, 2)

    # Reference roster used to *judge* every ablation's output — always the
    # full four-player game.
    reference_players = make_default_players(
        hamiltonian=H, topology=topo, **ROSTERS["full_four_player"],
        minimize_performance=False,
    )

    results_by_roster = {}
    total_start = time.perf_counter()

    for roster_name, weights in ROSTERS.items():
        t0 = time.perf_counter()
        roster_players = make_default_players(
            hamiltonian=H, topology=topo, **weights,
            minimize_performance=False,
        )
        game = PQCNashGame(
            n_qubits=n_qubits, players=roster_players, topology=topo,
            initial_dag_factory=lambda n=n_qubits: plus_layer(n),
            population_size=population_size,
            temperature=1.0, cooling_rate=0.9,
            seed=seed,
        )
        best = game.run(n_iters, verbose=False)
        dt = time.perf_counter() - t0
        m = _full_metrics(best, reference_players, topo)
        m["wall_time_s"] = dt
        m["weights"] = weights
        results_by_roster[roster_name] = m

    result = {
        "name": "F4_player_ablation",
        "task": "MaxCut_K4",
        "topology": "grid_2x2",
        "n_qubits": n_qubits,
        "n_iters": n_iters,
        "reference_roster": "full_four_player",
        "rosters": results_by_roster,
        "total_wall_s": time.perf_counter() - total_start,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(f"F4 done ({r['total_wall_s']:.1f}s):")
    print(f"  Reference roster: {r['reference_roster']}")
    for name, m in r["rosters"].items():
        print(
            f"  {name:22s} maxcut={m['f3_performance']:+.3f}  "
            f"magic={m['f2_magic_per_qubit']:.3f}  "
            f"eff_dim={m['f1_eff_dim_ratio']:.2f}  "
            f"ops={m['n_ops']} depth={m['depth']}  "
            f"Phi_full={m['phi_full_game']:+.3f}  "
            f"delta_Nash_full={m['delta_nash_full_game']:.3e}"
        )
