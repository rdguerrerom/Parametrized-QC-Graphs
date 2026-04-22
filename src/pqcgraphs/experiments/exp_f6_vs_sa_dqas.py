"""F6: Head-to-head — four-player Nash vs single-objective SA baseline.

Task: MaxCut K_4 on three hardware topologies (heavy-hex subset, 2x2 grid,
Rydberg all-to-all). Same initial state (|+>^4), same iteration budget,
same move set, same cooling schedule, same seed. Compares:

    Nash (4 players: anti-BP + anti-sim + performance + hardware)
        vs.
    SingleObjectiveSA (1 player: performance, + hardware regulariser)

Headline metrics (per topology, per algorithm):
  - MaxCut expectation <H> at the best DAG
  - Circuit complexity: n_ops, depth, n_params
  - Magic (stabilizer Renyi entropy per qubit)
  - Eff-dim ratio (QFIM rank / n_params)
  - Wall-clock and total candidate evaluations
  - Final delta_Nash of each output, measured against the SAME four-player
    game — so we see not just who performs better, but whether the
    baseline's output is structurally stable under the BP/sim pressures.

This is the F6 experiment of docs/manuscript_evidence_plan.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx

from ..baselines import SingleObjectiveSA, posthoc_nash_gap
from ..dag import Topology, grid_2d, heavy_hex, rydberg_all_to_all
from ..dag.initial_states import plus_layer
from ..game import PQCNashGame, make_default_players, potential
from ..game.players import NashPlayer
from ..objectives import maxcut_hamiltonian


def _subgraph_first_k(topo: Topology, k: int) -> Topology:
    pairs = frozenset((u, v) for (u, v) in topo.pairs if u < k and v < k)
    return Topology(topo.tag, k, pairs)


def _metrics(dag, players, topology) -> dict:
    """Unified metrics block for any CircuitDAG, independent of which
    algorithm produced it."""
    # Per-player raw values (f1, f2, f3, f4)
    raw = {p.name: float(p.raw(dag)) for p in players}
    gap_report = posthoc_nash_gap(players, dag, topology, rng_seed=0)
    return {
        "n_ops": dag.n_ops,
        "depth": dag.depth(),
        "n_params": dag.n_params,
        "gate_counts": dag.gate_counts(),
        "f1_eff_dim_ratio": raw["anti_bp"],
        "f2_magic_per_qubit": raw["anti_sim"],
        "f3_performance": raw["performance"],
        "f4_hardware_cost": raw["hardware"],
        "phi_nash": float(potential(players, dag)),
        "delta_nash": float(gap_report.delta_nash),
        "delta_nash_per_player": dict(gap_report.per_player),
    }


def run(
    n_iters: int = 15,
    population_size: int = 3,
    seed: int = 0,
    out_path: Path = Path("results/exp_f6_vs_sa_dqas.json"),
) -> dict:
    n_qubits = 4
    G = nx.complete_graph(n_qubits)
    H = maxcut_hamiltonian(G)

    topologies = {
        "heavy_hex_4q_subset": _subgraph_first_k(heavy_hex(7), n_qubits),
        "grid_2x2": grid_2d(2, 2),
        "rydberg_all_to_all_4": rydberg_all_to_all(n_qubits),
    }

    # Weights shared by both algorithms (the SA baseline uses only the
    # performance and hardware weights; the Nash game uses all four).
    W = dict(
        w_anti_bp=0.3,
        w_anti_sim=0.2,
        w_performance=1.0,
        w_hardware=0.2,
    )

    rows = []
    total_start = time.perf_counter()

    for topo_name, topo in topologies.items():
        players = make_default_players(
            hamiltonian=H, topology=topo, **W,
            minimize_performance=False,  # MaxCut: maximise <H>
        )
        # Reuse the same scorers for the baseline so per-call JIT caches
        # are shared between algorithms.
        perf_scorer = next(p.scorer for p in players if p.name == "performance")
        hw_scorer = next(p.scorer for p in players if p.name == "hardware")

        # ----- Nash -----
        t0 = time.perf_counter()
        game = PQCNashGame(
            n_qubits=n_qubits, players=players, topology=topo,
            initial_dag_factory=lambda n=n_qubits: plus_layer(n),
            population_size=population_size,
            temperature=1.0, cooling_rate=0.9,
            seed=seed,
            # Inner θ-GD: same Hamiltonian, same sign convention, same
            # number of steps as the SA baseline. This is what makes the
            # comparison a test of "structural search ability", not
            # "who happens to random-walk θ better".
            theta_gd_hamiltonian=H,
            theta_gd_minimize=False,   # MaxCut: maximise <H>
            theta_gd_steps=50,
            theta_gd_lr=0.1,
        )
        best_nash = game.run(n_iters, verbose=False)
        t_nash = time.perf_counter() - t0
        # For Nash, count evaluations heuristically: per iter per pop member
        # there are |candidates| evaluations of each of the four scorers.
        # We instead report the snapshot count; exact matching is via iters.
        nash_metrics = _metrics(best_nash, players, topo)

        # ----- Single-objective SA -----
        t0 = time.perf_counter()
        sa = SingleObjectiveSA(
            n_qubits=n_qubits,
            performance_scorer=perf_scorer,
            hardware_scorer=hw_scorer,
            topology=topo,
            initial_dag_factory=lambda n=n_qubits: plus_layer(n),
            w_performance=W["w_performance"],
            w_hardware=W["w_hardware"],
            population_size=population_size,
            temperature=1.0, cooling_rate=0.9,
            seed=seed,
            theta_gd_hamiltonian=H,
            theta_gd_minimize=False,
            theta_gd_steps=50,
            theta_gd_lr=0.1,
        )
        best_sa = sa.run(n_iters, verbose=False)
        t_sa = time.perf_counter() - t0
        sa_metrics = _metrics(best_sa, players, topo)

        rows.append({
            "topology": topo_name,
            "n_iters": n_iters,
            "nash": {**nash_metrics, "wall_time_s": t_nash},
            "sa_baseline": {
                **sa_metrics,
                "wall_time_s": t_sa,
                "total_evaluations": sa.total_evaluations,
            },
        })

    result = {
        "name": "F6_nash_vs_sa_dqas_matched_budget",
        "task": "MaxCut_K4",
        "n_qubits": n_qubits,
        "shared_weights": W,
        "shared_iters": n_iters,
        "rows": rows,
        "total_wall_s": time.perf_counter() - total_start,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(f"F6 done ({r['total_wall_s']:.1f}s):")
    for row in r["rows"]:
        n = row["nash"]
        s = row["sa_baseline"]
        print(
            f"\n  TOPOLOGY: {row['topology']}"
            f"\n    Nash   : maxcut={n['f3_performance']:.3f}  "
            f"magic={n['f2_magic_per_qubit']:.3f}  "
            f"eff_dim={n['f1_eff_dim_ratio']:.2f}  "
            f"ops={n['n_ops']} depth={n['depth']} params={n['n_params']}  "
            f"delta_Nash={n['delta_nash']:.3e}  "
            f"({n['wall_time_s']:.1f}s)"
            f"\n    SA-only: maxcut={s['f3_performance']:.3f}  "
            f"magic={s['f2_magic_per_qubit']:.3f}  "
            f"eff_dim={s['f1_eff_dim_ratio']:.2f}  "
            f"ops={s['n_ops']} depth={s['depth']} params={s['n_params']}  "
            f"delta_Nash={s['delta_nash']:.3e}  "
            f"({s['wall_time_s']:.1f}s)"
        )
