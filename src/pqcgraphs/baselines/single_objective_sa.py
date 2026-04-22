"""Single-objective simulated-annealing QAS baseline (F6).

This module implements a representative from the "simulated-annealing over
structure + θ" family of Quantum Architecture Search methods, intended for
head-to-head comparison against the four-player Nash engine at matched
evaluation budget. It is faithful to the common core of SA-DQAS
(arXiv:2406.08882), QuantumDARTS (He et al., ICML 2023), and greedy
RL-QAS (Sogabe et al., arXiv:2402.13754), all of which share:

  - A discrete architecture-search outer loop (add/remove/retype gates).
  - A scalar objective equal to a weighted combination of task performance
    and implementation cost (depth, CNOT count, or similar hardware proxy).
  - No explicit barren-plateau-avoidance term and no classical-simulability
    (magic) term.
  - No δ_Nash convergence certificate — the output is just the circuit at
    the end of the budget, with no stability guarantee.

This baseline lets us report (at matched budget):
  1. Final task performance (cut value, energy).
  2. Circuit complexity (n_ops, depth, n_params).
  3. Post-hoc δ_Nash of the baseline's output — demonstrating that the
     SA-only circuit is NOT a Nash equilibrium of the four-player game.
  4. Post-hoc magic and eff-dim — demonstrating that the SA-only circuit
     sits at an unprincipled point in the BP/simulability plane.

To match the Nash engine exactly, this baseline reuses the *same* move
generators from `game.moves` (so the explored architecture space is
identical) and the *same* simulated-annealing cooling schedule. The only
difference is the potential function.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..dag.circuit_dag import CircuitDAG
from ..dag.topology import Topology
from ..game.moves import bounded_candidates
from ..game.nash_gap import NashGapReport, compute_nash_gap
from ..game.players import NashPlayer


@dataclass
class SASnapshot:
    iteration: int
    potential: float        # SA objective at leader
    performance_raw: float  # raw f3 (e.g. cut value or -energy)
    hardware_raw: float     # raw f4 penalty
    n_ops: int
    depth: int
    n_params: int
    temperature: float
    elapsed_s: float


class SingleObjectiveSA:
    """SA over the same DAG move set with a single-objective performance+hardware potential.

    Parameters mirror `PQCNashGame` so that a head-to-head run at matched
    budget is a pure algorithm swap.

    Objective
    ---------
        Phi_SA(dag) = w_performance * f3(dag) - w_hardware * f4(dag)

    No f1 / f2 terms, no Nash-gap closure. The user is responsible for
    passing the same weights used in the Nash run for the performance and
    hardware players, so the two algorithms only differ in the presence of
    the trainability (f1) and simulability (f2) players.
    """

    def __init__(
        self,
        n_qubits: int,
        *,
        performance_scorer: Callable[[CircuitDAG], float],
        hardware_scorer: Callable[[CircuitDAG], float],
        topology: Topology,
        initial_dag_factory: Optional[Callable[[], CircuitDAG]] = None,
        w_performance: float = 1.0,
        w_hardware: float = 0.2,
        population_size: int = 4,
        temperature: float = 2.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.1,
        seed: int = 0,
        candidate_budget: Optional[Dict[str, int]] = None,
        theta_gd_hamiltonian=None,
        theta_gd_minimize: bool = True,
        theta_gd_steps: int = 50,
        theta_gd_lr: float = 0.1,
    ) -> None:
        if topology.n_qubits != n_qubits:
            raise ValueError(
                f"Topology n_qubits={topology.n_qubits} != algorithm n_qubits={n_qubits}"
            )
        self.n_qubits = n_qubits
        self.performance_scorer = performance_scorer
        self.hardware_scorer = hardware_scorer
        self.w_performance = float(w_performance)
        self.w_hardware = float(w_hardware)
        self.topology = topology
        self.rng = random.Random(seed)

        if initial_dag_factory is None:
            initial_dag_factory = lambda: CircuitDAG(n_qubits)
        self.population: List[CircuitDAG] = [
            initial_dag_factory() for _ in range(population_size)
        ]
        self.best_dag = self.population[0].copy()
        self.best_potential = float("-inf")

        self.temperature = float(temperature)
        self.cooling_rate = float(cooling_rate)
        self.min_temperature = float(min_temperature)

        from ..game.pqc_nash_game import PQCNashGame
        self.candidate_budget = dict(PQCNashGame.DEFAULT_CANDIDATE_BUDGET)
        if candidate_budget is not None:
            self.candidate_budget.update(candidate_budget)

        # Same inner-θ-GD plumbing as PQCNashGame — applied to the winning
        # candidate per population-member update, for apples-to-apples
        # comparison.
        self._theta_gd_ham = theta_gd_hamiltonian
        self._theta_gd_minimize = bool(theta_gd_minimize)
        self._theta_gd_steps = int(theta_gd_steps)
        self._theta_gd_lr = float(theta_gd_lr)

        self.history: List[SASnapshot] = []
        # For matched-budget accounting, count every candidate evaluation.
        self.total_evaluations: int = 0

    def _maybe_theta_gd(self, dag: CircuitDAG) -> CircuitDAG:
        if self._theta_gd_ham is None or dag.n_params == 0:
            return dag
        from ..gpu.theta_optimizer import optimize_theta
        return optimize_theta(
            dag, self._theta_gd_ham,
            minimize=self._theta_gd_minimize,
            n_steps=self._theta_gd_steps,
            lr=self._theta_gd_lr,
        )

    def _potential(self, dag: CircuitDAG) -> float:
        perf = float(self.performance_scorer(dag))
        hw = float(self.hardware_scorer(dag))
        return self.w_performance * perf - self.w_hardware * hw

    def _accept(self, current: float, proposed: float) -> bool:
        if proposed >= current:
            return True
        return self.rng.random() < np.exp(
            (proposed - current) / max(self.temperature, 1e-9)
        )

    def step(self, iteration: int) -> SASnapshot:
        t0 = time.perf_counter()
        for i, dag in enumerate(self.population):
            current_phi = self._potential(dag)
            cands = bounded_candidates(
                dag, self.topology, rng=self.rng, **self.candidate_budget
            )
            if not cands:
                continue
            scored = []
            for name, cand in cands:
                try:
                    phi = self._potential(cand)
                    self.total_evaluations += 1
                except Exception:  # noqa: BLE001
                    continue
                scored.append((phi, name, cand))
            if not scored:
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            best_phi, _, best_cand = scored[0]

            # Inner θ-GD refine before acceptance — matches PQCNashGame.
            best_cand = self._maybe_theta_gd(best_cand)
            best_phi = self._potential(best_cand)

            if self._accept(current_phi, best_phi):
                self.population[i] = best_cand
                if best_phi > self.best_potential:
                    self.best_potential = best_phi
                    self.best_dag = best_cand.copy()

        phis = [self._potential(d) for d in self.population]
        leader_idx = int(np.argmax(phis))
        leader = self.population[leader_idx]

        snap = SASnapshot(
            iteration=iteration,
            potential=float(phis[leader_idx]),
            performance_raw=float(self.performance_scorer(leader)),
            hardware_raw=float(self.hardware_scorer(leader)),
            n_ops=leader.n_ops,
            depth=leader.depth(),
            n_params=leader.n_params,
            temperature=self.temperature,
            elapsed_s=time.perf_counter() - t0,
        )
        self.history.append(snap)
        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
        return snap

    def run(self, max_iterations: int, *, verbose: bool = False) -> CircuitDAG:
        for it in range(max_iterations):
            self.step(it)
        return self.best_dag


def posthoc_nash_gap(
    players: Sequence[NashPlayer],
    dag: CircuitDAG,
    topology: Topology,
    *,
    rng_seed: int = 0,
) -> NashGapReport:
    """Compute the δ_Nash of an arbitrary DAG under the four-player game.

    Use this to measure how far a baseline's output is from a Nash
    equilibrium of the full four-player game. A high gap demonstrates the
    baseline's output is not structurally stable under the BP / magic
    pressures that Nash optimises for.
    """
    rng = random.Random(rng_seed)
    candidates = bounded_candidates(dag, topology, rng=rng)
    return compute_nash_gap(players, dag, candidates)


__all__ = ["SingleObjectiveSA", "SASnapshot", "posthoc_nash_gap"]
