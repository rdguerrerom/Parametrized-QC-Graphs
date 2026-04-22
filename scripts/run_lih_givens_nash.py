"""Nash structure-search on the Givens-doubles LiH seed.

Seed: 58-gate Givens-doubles (HF + 2 paired DoubleExcitations). Adam reaches
E_GS with this ansatz in <1s. The scientific test: can Nash's structural
moves (remove / retype / rewire / perturb-θ) COMPRESS the circuit further
while preserving the correlation recovered?

Compare against two baselines:
  (A) Adam on the same 58-gate ansatz (already run → exp_c3_lih_givens_adam.json)
  (B) HEA-random + θ-only (existing exp_c3_lih_vqe.json, BP-trapped)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pqcgraphs.dag import Topology, heavy_hex, CircuitDAG
from pqcgraphs.dag.chem_ansatze import lih_givens_doubles_seed
from pqcgraphs.game import PQCNashGame, make_default_players, potential
from pqcgraphs.objectives import lih_sto3g_hamiltonian, lih_sto3g_reference_energies

ROOT = Path(__file__).resolve().parent.parent


def _sub(topo: Topology, k: int) -> Topology:
    return Topology(
        topo.tag, k,
        frozenset((u, v) for (u, v) in topo.pairs if u < k and v < k),
    )


def main() -> None:
    n_q = 6
    topo = _sub(heavy_hex(7), n_q)
    H = lih_sto3g_hamiltonian()
    refs = lih_sto3g_reference_energies()
    E_HF = refs["E_HF"]; E_GS = refs["E_ground_active"]
    print(f"[refs] E_HF={E_HF:.5f}  E_GS={E_GS:.5f}", flush=True)

    # Weights: heavy pressure on performance (recover correlation) +
    # reasonable hardware/anti-sim so Nash is incentivised to compress.
    players = make_default_players(
        hamiltonian=H, topology=topo,
        w_anti_bp=0.3, w_anti_sim=0.1, w_performance=5.0, w_hardware=0.08,
    )

    seed_dag = lih_givens_doubles_seed()
    print(f"[seed] n_ops={seed_dag.n_ops}  n_params={seed_dag.n_params}  "
          f"depth={seed_dag.depth()}", flush=True)

    t0 = time.perf_counter()
    game = PQCNashGame(
        n_qubits=n_q, players=players, topology=topo,
        initial_dag_factory=lih_givens_doubles_seed,
        population_size=2, temperature=0.5, cooling_rate=0.9,
        theta_gd_hamiltonian=H, theta_gd_minimize=True,
        theta_gd_steps=80, theta_gd_lr=0.05,
        seed=0,
    )
    best = game.run(20, verbose=True)
    wall = time.perf_counter() - t0

    f3 = next(p.scorer for p in players if p.name == "performance")
    E_final = -float(f3(best))
    corr = (E_HF - E_final) / (E_HF - E_GS)
    print(f"[final] E={E_final:.6f}  corr={100*corr:.1f}%  "
          f"ops={best.n_ops} depth={best.depth()} params={best.n_params}  "
          f"wall={wall:.1f}s", flush=True)

    def _snap(s):
        return {
            "iteration": s.iteration,
            "potential": s.potential,
            "n_ops": s.n_ops, "depth": s.depth, "n_params": s.n_params,
            "nash_gap": s.nash_gap, "temperature": s.temperature,
            "elapsed_s": s.elapsed_s,
            "per_player_raw": {k: (v["raw"] if isinstance(v, dict) else v)
                                for k, v in s.per_player.items() if k != "Phi"},
        }

    out = {
        "name": "C3_lih_givens_nash",
        "description": (
            "Nash structure-search seeded from the 58-gate Givens-doubles "
            "ansatz. All moves enabled (append/remove/retype/rewire/"
            "perturb-θ) + θ-GD inner loop. Tests whether Nash compresses "
            "the ansatz while preserving correlation energy."
        ),
        "n_qubits": n_q,
        "seed_n_ops": int(seed_dag.n_ops),
        "seed_n_params": int(seed_dag.n_params),
        "seed_depth": int(seed_dag.depth()),
        "final_n_ops": int(best.n_ops),
        "final_depth": int(best.depth()),
        "final_n_params": int(best.n_params),
        "final_energy_Ha": float(E_final),
        "final_potential": float(potential(players, best)),
        "fraction_correlation_recovered": float(corr),
        "references": {
            "HF_energy_Ha": E_HF,
            "active_ground_energy_Ha": E_GS,
            "FCI_full_energy_Ha": refs["E_FCI_full"],
        },
        "weights": {
            "w_anti_bp": 0.3, "w_anti_sim": 0.1,
            "w_performance": 5.0, "w_hardware": 0.08,
        },
        "n_iters": 20,
        "theta_gd_steps": 80,
        "wall_time_s": float(wall),
        "trajectory": [_snap(s) for s in game.history],
    }
    out_path = ROOT / "results" / "exp_c3_lih_givens_nash.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
