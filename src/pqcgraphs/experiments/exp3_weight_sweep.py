"""E3: Sweep over (w_anti_bp, w_anti_sim) to trace the BP/simulability Pareto frontier.

We fix the task (MaxCut on K_4), the topology (Rydberg all-to-all, 4 qubits),
and the performance + hardware weights. We sweep w_anti_bp ∈ {0, 0.3, 1.0, 3.0}
and w_anti_sim ∈ {0, 0.3, 1.0, 3.0} and record, for each corner:
  - final Nash potential
  - per-player raw values (effective QFIM-rank ratio, stabilizer Rényi
    entropy per qubit, maxcut expectation, hardware cost)
  - the architecture (gate counts, depth, n_params)

These points trace the Pareto frontier between barren-plateau avoidance
(high eff-dim) and classical simulability (low magic). Tier1.md §1 names
this the core methodological contribution.

Writes `results/exp3_weight_sweep.json`.
"""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import networkx as nx

from ..dag import rydberg_all_to_all
from ..dag.initial_states import plus_layer
from ..game import PQCNashGame, make_default_players, potential
from ..objectives import maxcut_hamiltonian


def run(
    n_qubits: int = 4,
    n_iters: int = 12,
    population_size: int = 3,
    w_anti_bp_values=(0.0, 0.3, 1.0, 3.0),
    w_anti_sim_values=(0.0, 0.3, 1.0, 3.0),
    seed: int = 0,
    out_path: Path = Path("results/exp3_weight_sweep.json"),
) -> dict:
    topo = rydberg_all_to_all(n_qubits)
    G = nx.complete_graph(n_qubits)
    H = maxcut_hamiltonian(G)

    rows = []
    total_start = time.perf_counter()
    for w_bp, w_sim in product(w_anti_bp_values, w_anti_sim_values):
        t0 = time.perf_counter()
        players = make_default_players(
            hamiltonian=H,
            topology=topo,
            w_anti_bp=w_bp,
            w_anti_sim=w_sim,
            w_performance=1.0,
            w_hardware=0.1,
            minimize_performance=False,  # MaxCut: maximise <H>
        )
        game = PQCNashGame(
            n_qubits=n_qubits, players=players, topology=topo,
            initial_dag_factory=lambda: plus_layer(n_qubits),
            population_size=population_size, temperature=1.0, cooling_rate=0.9,
            seed=seed,
        )
        best = game.run(n_iters, verbose=False)
        dt = time.perf_counter() - t0

        f3 = next(p.scorer for p in players if p.name == "performance")
        # f3 for MaxCut returns +<H>, so this IS the cut expectation directly.
        maxcut_val = float(f3(best))
        f1 = next(p for p in players if p.name == "anti_bp").raw(best)
        f2 = next(p for p in players if p.name == "anti_sim").raw(best)
        f4 = next(p for p in players if p.name == "hardware").raw(best)

        rows.append({
            "w_anti_bp": w_bp,
            "w_anti_sim": w_sim,
            "final_potential": float(potential(players, best)),
            "maxcut_value": maxcut_val,
            "qfim_eff_dim_ratio": float(f1),
            "magic_per_qubit": float(f2),
            "hardware_cost": float(f4),
            "n_ops": best.n_ops,
            "depth": best.depth(),
            "n_params": best.n_params,
            "gate_counts": best.gate_counts(),
            "wall_time_s": dt,
        })

    result = {
        "name": "E3_weight_sweep",
        "task": "MaxCut_K4",
        "topology": "rydberg_4q_all2all",
        "n_qubits": n_qubits,
        "n_iters": n_iters,
        "rows": rows,
        "total_wall_s": time.perf_counter() - total_start,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(f"E3 done: {len(r['rows'])} weight corners "
          f"(total {r['total_wall_s']:.1f}s)")
    for row in r["rows"]:
        print(f"  w_bp={row['w_anti_bp']} w_sim={row['w_anti_sim']}: "
              f"maxcut={row['maxcut_value']:.3f}  "
              f"eff_dim={row['qfim_eff_dim_ratio']:.2f}  "
              f"magic={row['magic_per_qubit']:.3f}  "
              f"depth={row['depth']}")
