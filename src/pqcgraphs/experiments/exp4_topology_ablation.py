"""E4: Topology ablation — identical task/weights, three hardware backends.

Tier1.md §1: "the IBM heavy-hex, Rydberg all-to-all, and superconducting
grid cases all become objective reconfiguration problems" under the Nash
framework. Here we verify the practical claim: running the SAME four-player
game on the SAME task but swapping the Topology should produce qualitatively
different best circuits (fewer swaps on native-friendly backends; more on
restrictive ones).

Task: MaxCut on K_4 (4 qubits, 6 edges).
Hardware backends:
  A) heavy_hex_4q_subset    — degree-3 sparse graph (IBM heavy-hex excerpt)
  B) grid_2x2               — 4 qubits on a 2×2 grid (4 edges)
  C) rydberg_all_to_all_4   — fully connected

For each, run the Nash game with identical weights. Record:
  - best circuit structure (gate counts, depth)
  - f4 hardware cost (depth + non-native + connectivity penalties)
  - maxcut expectation achieved

Writes `results/exp4_topology_ablation.json`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx

from ..dag import Topology, grid_2d, heavy_hex, rydberg_all_to_all
from ..dag.initial_states import plus_layer
from ..game import PQCNashGame, make_default_players, potential
from ..objectives import maxcut_hamiltonian


def _subgraph_first_k(topo: Topology, k: int) -> Topology:
    pairs = frozenset((u, v) for (u, v) in topo.pairs if u < k and v < k)
    return Topology(topo.tag, k, pairs)


def run(
    n_iters: int = 12,
    population_size: int = 3,
    seed: int = 0,
    out_path: Path = Path("results/exp4_topology_ablation.json"),
) -> dict:
    n_qubits = 4
    G = nx.complete_graph(n_qubits)
    H = maxcut_hamiltonian(G)

    topologies = {
        "heavy_hex_4q_subset": _subgraph_first_k(heavy_hex(7), n_qubits),
        "grid_2x2": grid_2d(2, 2),
        "rydberg_all_to_all_4": rydberg_all_to_all(n_qubits),
    }

    rows = []
    total_start = time.perf_counter()
    for name, topo in topologies.items():
        t0 = time.perf_counter()
        players = make_default_players(
            hamiltonian=H,
            topology=topo,
            w_anti_bp=0.3,
            w_anti_sim=0.2,
            w_performance=1.0,
            w_hardware=0.2,  # slightly stronger hardware push for this ablation
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
        f4 = next(p for p in players if p.name == "hardware").raw(best)

        rows.append({
            "topology": name,
            "allowed_pair_count": len(topo.pairs),
            "maxcut_value": maxcut_val,
            "hardware_cost": float(f4),
            "n_ops": best.n_ops,
            "depth": best.depth(),
            "n_params": best.n_params,
            "gate_counts": best.gate_counts(),
            "final_potential": float(potential(players, best)),
            "wall_time_s": dt,
        })

    result = {
        "name": "E4_topology_ablation",
        "task": "MaxCut_K4",
        "n_qubits": n_qubits,
        "rows": rows,
        "total_wall_s": time.perf_counter() - total_start,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(f"E4 done ({r['total_wall_s']:.1f}s):")
    for row in r["rows"]:
        print(
            f"  {row['topology']:25s} "
            f"pairs={row['allowed_pair_count']:2d}  "
            f"maxcut={row['maxcut_value']:.3f}  "
            f"hw_cost={row['hardware_cost']:.2f}  "
            f"depth={row['depth']}"
        )
