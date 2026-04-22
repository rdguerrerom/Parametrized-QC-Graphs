"""E2: VQE H₂ on IBM heavy-hex (4 qubits) — Nash vs hardware-efficient baseline.

Two runs on the same (H₂, topology, objective weights) setup:
  (A) Nash-equilibrium search from scratch with the four-player potential.
  (B) A "hardware-efficient ansatz" (HEA) baseline — fixed structure, only
       θ-optimization via the same Nash engine restricted to perturb moves.

We compare:
  - Final energy ⟨H_H₂⟩ vs the published FCI value (−1.1373 Ha) and HF
    (−1.1167 Ha).
  - Final Phi (Nash potential).
  - Per-player contributions at the converged state.
  - Circuit complexity: n_ops, depth, n_params.

Writes `results/exp2_h2_heavy_hex.json` with full histories for both runs.

Subset of heavy_hex(7) used here — we project onto the first 4 qubits by
keeping edges whose endpoints are both < 4. This yields a connected 4-qubit
subgraph matching the JW wiring of H₂.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

import numpy as np

from ..dag import CircuitDAG, Topology, heavy_hex
from ..game import NashSnapshot, PQCNashGame, make_default_players, potential
from ..game.moves import perturb_theta_candidates
from ..objectives import h2_sto3g_hamiltonian
from ..objectives.hardware import make_f4_hardware


def _subgraph_first_k(topo: Topology, k: int) -> Topology:
    """Restrict a Topology to the first k qubits (keep only edges inside)."""
    pairs = frozenset((u, v) for (u, v) in topo.pairs if u < k and v < k)
    return Topology(topo.tag, k, pairs)


def _build_hea(n_qubits: int, n_layers: int, topo: Topology, seed: int = 0) -> CircuitDAG:
    """Hardware-efficient ansatz: alternating ry + rzz entangling layers."""
    rng = np.random.default_rng(seed)
    dag = CircuitDAG(n_qubits)
    pairs = sorted(topo.pairs)
    for _layer in range(n_layers):
        for q in range(n_qubits):
            dag.append_gate("ry", (q,), theta=float(rng.uniform(0.0, 2 * np.pi)))
        for (u, v) in pairs:
            dag.append_gate("rzz", (u, v), theta=float(rng.uniform(0.0, 2 * np.pi)))
    return dag


class _ParamOnlyGame(PQCNashGame):
    """PQCNashGame variant where only θ-perturb moves are generated.

    This is the controlled baseline: same Nash engine, same potential, same
    objectives, but structure is frozen.
    """

    def _candidates(self, dag):
        return perturb_theta_candidates(dag, rng=self.rng, max_candidates=10)

    def step(self, iteration: int):
        import time as _t

        import numpy as _np
        from ..game.nash_gap import compute_nash_gap
        from ..game.players import evaluate_all

        t0 = _t.perf_counter()
        for i, dag in enumerate(self.population):
            current_phi = self._potential(dag)
            cands = self._candidates(dag)
            if not cands:
                continue
            scored = []
            for name, cand in cands:
                try:
                    phi = self._potential(cand)
                except Exception:
                    continue
                scored.append((phi, name, cand))
            if not scored:
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            best_phi, _, best_cand = scored[0]
            if self._accept(current_phi, best_phi):
                self.population[i] = best_cand
                if best_phi > self.best_potential:
                    self.best_potential = best_phi
                    self.best_dag = best_cand.copy()

        phis = [self._potential(d) for d in self.population]
        leader_idx = int(_np.argmax(phis))
        leader = self.population[leader_idx]
        gap_cands = self._candidates(leader)
        gap_report = compute_nash_gap(self.players, leader, gap_cands)
        snap = NashSnapshot(
            iteration=iteration,
            potential=float(phis[leader_idx]),
            per_player=evaluate_all(self.players, leader),
            n_ops=leader.n_ops,
            depth=leader.depth(),
            n_params=leader.n_params,
            nash_gap=gap_report.delta_nash,
            temperature=self.temperature,
            elapsed_s=_t.perf_counter() - t0,
        )
        self.history.append(snap)
        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
        return snap


def run(
    n_iters: int = 15,
    population_size: int = 3,
    seed: int = 0,
    out_path: Path = Path("results/exp2_h2_heavy_hex.json"),
) -> dict:
    n_qubits = 4
    topo = _subgraph_first_k(heavy_hex(7), n_qubits)
    H = h2_sto3g_hamiltonian()
    players = make_default_players(
        hamiltonian=H,
        topology=topo,
        w_anti_bp=0.3,
        w_anti_sim=0.2,
        w_performance=1.0,
        w_hardware=0.1,
    )

    # --- Run A: free Nash search
    t0 = time.perf_counter()
    game_nash = PQCNashGame(
        n_qubits=n_qubits, players=players, topology=topo,
        population_size=population_size, temperature=1.0, cooling_rate=0.9,
        seed=seed,
    )
    best_nash = game_nash.run(n_iters, verbose=False)
    t_nash = time.perf_counter() - t0

    # f3 returns -<H>, so extract the actual energy
    f3 = next(p.scorer for p in players if p.name == "performance")
    energy_nash = -float(f3(best_nash))

    # --- Run B: HEA baseline, only θ-opt
    t0 = time.perf_counter()
    hea_factory = lambda: _build_hea(n_qubits, n_layers=2, topo=topo, seed=seed)
    game_hea = _ParamOnlyGame(
        n_qubits=n_qubits, players=players, topology=topo,
        initial_dag_factory=hea_factory,
        population_size=population_size, temperature=1.0, cooling_rate=0.9,
        seed=seed,
    )
    best_hea = game_hea.run(n_iters, verbose=False)
    t_hea = time.perf_counter() - t0

    energy_hea = -float(f3(best_hea))

    def _snap_to_dict(s: NashSnapshot) -> dict:
        return {
            "iteration": s.iteration,
            "potential": s.potential,
            "n_ops": s.n_ops,
            "depth": s.depth,
            "n_params": s.n_params,
            "nash_gap": s.nash_gap,
            "temperature": s.temperature,
            "elapsed_s": s.elapsed_s,
            "per_player": s.per_player,
        }

    result = {
        "name": "E2_h2_heavy_hex",
        "topology": "heavy_hex_4q_subset",
        "n_qubits": n_qubits,
        "hamiltonian": "H2_STO3G_0.7414A",
        "references": {
            "HF_energy_Ha":  -1.1167593073964224,
            "FCI_energy_Ha": -1.1372838344885023,  # standard published value
        },
        "runs": {
            "nash_free": {
                "final_energy_Ha": energy_nash,
                "final_potential": float(potential(players, best_nash)),
                "n_ops": best_nash.n_ops,
                "depth": best_nash.depth(),
                "n_params": best_nash.n_params,
                "wall_time_s": t_nash,
                "history": [_snap_to_dict(s) for s in game_nash.history],
            },
            "hea_baseline": {
                "final_energy_Ha": energy_hea,
                "final_potential": float(potential(players, best_hea)),
                "n_ops": best_hea.n_ops,
                "depth": best_hea.depth(),
                "n_params": best_hea.n_params,
                "wall_time_s": t_hea,
                "history": [_snap_to_dict(s) for s in game_hea.history],
            },
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(
        f"Nash: E = {r['runs']['nash_free']['final_energy_Ha']:.5f} Ha "
        f"(FCI -1.1373, HF -1.1168); ops={r['runs']['nash_free']['n_ops']}, "
        f"depth={r['runs']['nash_free']['depth']}"
    )
    print(
        f"HEA : E = {r['runs']['hea_baseline']['final_energy_Ha']:.5f} Ha; "
        f"ops={r['runs']['hea_baseline']['n_ops']}, depth={r['runs']['hea_baseline']['depth']}"
    )
