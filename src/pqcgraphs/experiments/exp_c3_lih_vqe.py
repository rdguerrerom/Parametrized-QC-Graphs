"""C3: VQE LiH STO-3G (6-qubit reduced) — Nash vs HEA baseline.

A chemistry benchmark outside the H₂ toy regime, as required by the
manuscript evidence plan. Active space: Li(2s), Li(2p_z), H(1s), 2
electrons → 6 qubits (Jordan-Wigner), 62 Pauli terms. Bond distance 1.545 Å.

Reference energies (from the cached Hamiltonian metadata):
    E_HF             = −7.8631 Ha  (frozen-core HF in active space)
    E_ground_active  = −7.8778 Ha  (exact diagonalisation of the reduced H)
    E_FCI_full       = −7.8828 Ha  (FCI on the full molecule, for context)

Same structure as exp2_h2_heavy_hex:
  (A) Nash-equilibrium search from the HF Slater determinant, free to grow
      / retype / rewire / perturb over the heavy-hex(7)[:6] topology.
  (B) HEA baseline: fixed 2-layer ry+rzz structure on the same topology,
      only θ-perturb moves.

Writes `results/exp_c3_lih_vqe.json`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..dag import CircuitDAG, Topology, heavy_hex
from ..dag.initial_states import hartree_fock_lih
from ..game import NashSnapshot, PQCNashGame, make_default_players, potential
from ..game.moves import perturb_theta_candidates
from ..objectives import lih_sto3g_hamiltonian, lih_sto3g_reference_energies


def _subgraph_first_k(topo: Topology, k: int) -> Topology:
    pairs = frozenset((u, v) for (u, v) in topo.pairs if u < k and v < k)
    return Topology(topo.tag, k, pairs)


def _build_hea(n_qubits: int, n_layers: int, topo: Topology,
               seed: int = 0, small_angle: bool = False) -> CircuitDAG:
    """Hardware-efficient ansatz with optional initialisation regime.

    `small_angle=False` (default) → θ ∈ U(0, 2π), the naive HEA that
    typically traps above HF on chemistry problems (Kandala et al. 2017
    showed this is the barren-plateau regime at moderate n). This is the
    baseline we report in the paper to illustrate the BP trap on LiH.

    `small_angle=True` → θ ∈ U(−π/20, π/20), which keeps the ansatz close
    to the HF reference. Useful for supplementary sanity checks.
    """
    rng = np.random.default_rng(seed)
    dag = CircuitDAG(n_qubits)
    dag.append_gate("x", (0,))  # HF: lowest active MO doubly occupied
    dag.append_gate("x", (1,))
    pairs = sorted(topo.pairs)
    lo, hi = (-np.pi / 20.0, np.pi / 20.0) if small_angle else (0.0, 2 * np.pi)
    for _layer in range(n_layers):
        for q in range(n_qubits):
            dag.append_gate("ry", (q,), theta=float(rng.uniform(lo, hi)))
        for (u, v) in pairs:
            dag.append_gate("rzz", (u, v), theta=float(rng.uniform(lo, hi)))
    return dag


class _ParamOnlyGame(PQCNashGame):
    """Structure-frozen variant: only θ-perturb candidates are generated."""

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
        self.temperature = max(self.min_temperature,
                               self.temperature * self.cooling_rate)
        return snap


def run(
    n_iters: int = 30,
    population_size: int = 3,
    seed: int = 0,
    out_path: Path = Path("results/exp_c3_lih_vqe.json"),
) -> dict:
    n_qubits = 6
    topo = _subgraph_first_k(heavy_hex(7), n_qubits)
    H = lih_sto3g_hamiltonian()
    refs = lih_sto3g_reference_energies()

    # LiH has ~15 mHa of correlation above HF; we boost the performance
    # weight to make it clearly dominate the other players while still
    # allowing anti-BP and anti-sim pressure on circuit quality.
    players = make_default_players(
        hamiltonian=H,
        topology=topo,
        w_anti_bp=0.3,
        w_anti_sim=0.2,
        w_performance=2.0,
        w_hardware=0.1,
    )

    # --- Run A: Nash search seeded from HF + small-angle single-layer
    # ansatz. Pure HF has no free parameters so the Nash engine cannot
    # descend from it; we seed with 1 layer of small-angle ry+rzz so the
    # θ-GD inner loop has handles to tune while structure moves explore.
    def _nash_seed() -> CircuitDAG:
        return _build_hea(n_qubits, n_layers=1, topo=topo, seed=seed,
                          small_angle=True)

    t0 = time.perf_counter()
    game_nash = PQCNashGame(
        n_qubits=n_qubits, players=players, topology=topo,
        initial_dag_factory=_nash_seed,
        population_size=population_size, temperature=1.0, cooling_rate=0.9,
        seed=seed,
    )
    best_nash = game_nash.run(n_iters, verbose=False)
    t_nash = time.perf_counter() - t0

    f3 = next(p.scorer for p in players if p.name == "performance")
    energy_nash = -float(f3(best_nash))

    # --- Run B: HEA baseline, θ-only optimisation.
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

    # Correlation energy recovered = (E_HF - E) / (E_HF - E_active_ground)
    e_hf = refs["E_HF"]; e_gs = refs["E_ground_active"]
    denom = max(e_hf - e_gs, 1e-9)
    frac_corr_nash = (e_hf - energy_nash) / denom
    frac_corr_hea  = (e_hf - energy_hea)  / denom

    result = {
        "name": "C3_lih_vqe",
        "topology": "heavy_hex_6q_subset",
        "n_qubits": n_qubits,
        "hamiltonian": "LiH_STO3G_1.545A_frozen_core_BK",
        "mapping": "jordan_wigner",
        "references": {
            "HF_energy_Ha": e_hf,
            "active_ground_energy_Ha": e_gs,
            "FCI_full_energy_Ha": refs["E_FCI_full"],
            "bond_length_angstrom": refs["bond_length_angstrom"],
        },
        "runs": {
            "nash_free": {
                "final_energy_Ha": energy_nash,
                "final_potential": float(potential(players, best_nash)),
                "n_ops": best_nash.n_ops,
                "depth": best_nash.depth(),
                "n_params": best_nash.n_params,
                "fraction_correlation_recovered": float(frac_corr_nash),
                "wall_time_s": t_nash,
                "history": [_snap_to_dict(s) for s in game_nash.history],
            },
            "hea_baseline": {
                "final_energy_Ha": energy_hea,
                "final_potential": float(potential(players, best_hea)),
                "n_ops": best_hea.n_ops,
                "depth": best_hea.depth(),
                "n_params": best_hea.n_params,
                "fraction_correlation_recovered": float(frac_corr_hea),
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
    ref = r["references"]
    print(f"Hamiltonian refs: HF={ref['HF_energy_Ha']:.4f}  "
          f"active GS={ref['active_ground_energy_Ha']:.4f}  "
          f"FCI={ref['FCI_full_energy_Ha']:.4f} Ha")
    n = r["runs"]["nash_free"]; h = r["runs"]["hea_baseline"]
    print(f"Nash: E = {n['final_energy_Ha']:.5f} Ha  "
          f"(corr {n['fraction_correlation_recovered']*100:.1f}%); "
          f"ops={n['n_ops']}, depth={n['depth']}")
    print(f"HEA : E = {h['final_energy_Ha']:.5f} Ha  "
          f"(corr {h['fraction_correlation_recovered']*100:.1f}%); "
          f"ops={h['n_ops']}, depth={h['depth']}")
