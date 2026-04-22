"""Simulated-annealing + population best-response search over CircuitDAGs.

Ported from STABILIZER_GAMES/game/nash_game.py (tested + proven) and
adapted to the four-player weighted-potential formulation on the circuit
DAG state. The outer loop is unchanged: at each iteration we mutate each
population member along a move sampled from `moves.all_candidates`, accept
with a Metropolis criterion against the potential, cool the temperature,
and record a snapshot.

Differences from the STABILIZER_GAMES version:
  - State is a `CircuitDAG`, not an `EnhancedGraphState`.
  - Nash gap is the full per-player δ_i gap (see `nash_gap.py`), not just a
    scalar sweep.
  - Every iteration's snapshot carries the full per-player diagnostic dict,
    which the experiments in E2/E3 need to draw Pareto frontiers.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..dag.circuit_dag import CircuitDAG
from ..dag.topology import Topology
from .moves import bounded_candidates
from .nash_gap import NashGapReport, compute_nash_gap
from .players import NashPlayer, evaluate_all, potential


@dataclass
class NashSnapshot:
    iteration: int
    potential: float
    per_player: Dict[str, Dict[str, float]]  # name -> {raw, weighted, sign}
    n_ops: int
    depth: int
    n_params: int
    nash_gap: float
    temperature: float
    elapsed_s: float


class PQCNashGame:
    """Four-player Nash architecture search over parameterized circuit DAGs.

    Parameters
    ----------
    n_qubits : int
    players : list[NashPlayer]
    topology : Topology
        Hardware graph for move generation + f4.
    initial_dag_factory : Callable[[], CircuitDAG], optional
        Function producing each population member's starting DAG. Default:
        empty DAG with no gates (the game proposes additions in its first
        iteration).
    population_size : int
    temperature : float
        Starting temperature for simulated-annealing Metropolis moves.
    cooling_rate : float
    min_temperature : float
    seed : int
    candidate_budget : dict, optional
        Overrides for `moves.bounded_candidates` — keys are {n_theta, n_add,
        n_remove, n_retype, n_rewire}. Default
        {n_theta: 12, n_add: 2, n_remove: 2, n_retype: 2, n_rewire: 2}
        produces 20 candidates/step, 60% structure-preserving, which keeps
        the GPU JIT caches warm. Raise n_add/n_retype if you need more
        structural exploration (at the cost of more recompilations).
    """

    DEFAULT_CANDIDATE_BUDGET = {
        "n_theta": 12,
        "n_add": 4,
        "n_remove": 2,
        "n_retype": 2,
        "n_rewire": 2,
    }

    def __init__(
        self,
        n_qubits: int,
        players: Sequence[NashPlayer],
        topology: Topology,
        *,
        initial_dag_factory: Optional[Callable[[], CircuitDAG]] = None,
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
                f"Topology has n_qubits={topology.n_qubits} but game has {n_qubits}"
            )
        self.n_qubits = n_qubits
        self.players = list(players)
        self.topology = topology
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        if initial_dag_factory is None:
            initial_dag_factory = lambda: CircuitDAG(n_qubits)
        self._initial_dag_factory = initial_dag_factory

        self.population: List[CircuitDAG] = [
            initial_dag_factory() for _ in range(population_size)
        ]
        self.best_dag: CircuitDAG = self.population[0].copy()
        self.best_potential: float = float("-inf")

        self.temperature = float(temperature)
        self.cooling_rate = float(cooling_rate)
        self.min_temperature = float(min_temperature)

        self.candidate_budget = dict(self.DEFAULT_CANDIDATE_BUDGET)
        if candidate_budget is not None:
            self.candidate_budget.update(candidate_budget)

        # Inner θ-gradient optimization. When `theta_gd_hamiltonian` is
        # supplied, every best-candidate evaluation is preceded by a
        # gradient-ascent pass on the candidate's θ (w.r.t. ±⟨H⟩ per
        # `theta_gd_minimize`). This is the equivalent of the DQAS /
        # SA-DQAS inner loop that all recent QAS baselines run; without
        # it, Nash is doing structure-search + random-walk θ, which is
        # not the appropriate comparison.
        self._theta_gd_ham = theta_gd_hamiltonian
        self._theta_gd_minimize = bool(theta_gd_minimize)
        self._theta_gd_steps = int(theta_gd_steps)
        self._theta_gd_lr = float(theta_gd_lr)

        self.history: List[NashSnapshot] = []

    # ------------------------------------------------------------------ helpers
    def _potential(self, dag: CircuitDAG) -> float:
        return potential(self.players, dag)

    def _maybe_theta_gd(self, dag: CircuitDAG) -> CircuitDAG:
        """Apply inner θ-GD iff configured; no-op when disabled or no params."""
        if self._theta_gd_ham is None or dag.n_params == 0:
            return dag
        from ..gpu.theta_optimizer import optimize_theta
        return optimize_theta(
            dag, self._theta_gd_ham,
            minimize=self._theta_gd_minimize,
            n_steps=self._theta_gd_steps,
            lr=self._theta_gd_lr,
        )

    def _accept(self, current: float, proposed: float) -> bool:
        if proposed >= current:
            return True
        delta = proposed - current
        # Metropolis on the potential — higher Phi is better, so we accept
        # downhill moves only with probability exp(delta/T).
        prob = np.exp(delta / max(self.temperature, 1e-9))
        return self.rng.random() < prob

    # ------------------------------------------------------------------ loop
    def step(self, iteration: int) -> NashSnapshot:
        """One SA iteration over the full population.

        For each member:
          - generate a *bounded* candidate batch biased toward θ-perturbs
            so most candidates hit a warm JIT cache
          - pick the best candidate (max Phi)
          - Metropolis-accept against the current member

        Then pick the strongest member as the `current` state for this
        iteration's snapshot. Critically, we reuse the leader's own
        candidate batch for the Nash-gap calculation — no second pass.
        """
        t0 = time.perf_counter()

        # Per-member candidate batch caches so we can reuse the leader's
        # candidates + their payoffs for the Nash-gap step.
        member_candidates: Dict[int, list] = {}
        member_scored: Dict[int, list] = {}

        for i, dag in enumerate(self.population):
            current_phi = self._potential(dag)
            candidates = bounded_candidates(
                dag, self.topology, rng=self.rng, **self.candidate_budget
            )
            member_candidates[i] = candidates
            if not candidates:
                member_scored[i] = []
                continue

            scored = []
            for name, cand in candidates:
                try:
                    phi = self._potential(cand)
                except Exception:  # noqa: BLE001
                    # Candidate broke an objective (e.g. magic exceeds
                    # supported qubit count). Skip rather than crash the run.
                    continue
                scored.append((phi, name, cand))
            member_scored[i] = scored
            if not scored:
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            best_phi, _, best_cand = scored[0]

            # Inner θ-gradient refine the winning candidate before the
            # Metropolis acceptance test. Cheap no-op when disabled.
            best_cand = self._maybe_theta_gd(best_cand)
            best_phi = self._potential(best_cand)

            if self._accept(current_phi, best_phi):
                self.population[i] = best_cand

                if best_phi > self.best_potential:
                    self.best_potential = best_phi
                    self.best_dag = best_cand.copy()

        # Pick strongest *updated* member for the snapshot. Evaluate fresh
        # Phi because the member at index i may have moved to `best_cand`.
        phis = [self._potential(d) for d in self.population]
        leader_idx = int(np.argmax(phis))
        leader = self.population[leader_idx]
        leader_phi = phis[leader_idx]

        # Nash-gap: if the leader's index was NOT updated this step (member
        # stayed in place), reuse its already-scored candidates for the gap.
        # Otherwise regenerate a batch against the post-move leader.
        # In both cases we call `compute_nash_gap`, which re-evaluates per
        # player — but with JIT caches warm from the acceptance pass this
        # is essentially free.
        gap_candidates = bounded_candidates(
            leader, self.topology, rng=self.rng, **self.candidate_budget
        )
        gap_report: NashGapReport = compute_nash_gap(self.players, leader, gap_candidates)

        snap = NashSnapshot(
            iteration=iteration,
            potential=float(leader_phi),
            per_player=evaluate_all(self.players, leader),
            n_ops=leader.n_ops,
            depth=leader.depth(),
            n_params=leader.n_params,
            nash_gap=gap_report.delta_nash,
            temperature=self.temperature,
            elapsed_s=time.perf_counter() - t0,
        )
        self.history.append(snap)

        # Cool
        self.temperature = max(
            self.min_temperature, self.temperature * self.cooling_rate
        )

        return snap

    def run(self, max_iterations: int, *, verbose: bool = True,
            early_stop_gap: float = 1e-6, early_stop_patience: int = 8) -> CircuitDAG:
        """Run the SA loop. Early-stop when δ_Nash stays below tolerance for
        `early_stop_patience` consecutive iterations.
        """
        stable_iters = 0
        for it in range(max_iterations):
            snap = self.step(it)
            if verbose and (it < 5 or it % 10 == 0):
                print(
                    f"[{it:3d}] Phi={snap.potential:+.4f}  "
                    f"nash_gap={snap.nash_gap:.3e}  T={snap.temperature:.3f}  "
                    f"ops={snap.n_ops} depth={snap.depth} params={snap.n_params}  "
                    f"({snap.elapsed_s:.2f}s)"
                )
            if snap.nash_gap <= early_stop_gap:
                stable_iters += 1
                if stable_iters >= early_stop_patience:
                    if verbose:
                        print(f"  Converged: δ_Nash < {early_stop_gap} for "
                              f"{early_stop_patience} iters")
                    break
            else:
                stable_iters = 0

        # Trivial-equilibrium diagnostic. If the best DAG is empty AND the
        # best potential is zero, we have collapsed to |0⟩^n — nearly always
        # a symptom of starting from the wrong reference state (e.g. running
        # MaxCut from |0⟩^n instead of |+⟩^n). See
        # `pqcgraphs.dag.initial_states` for canonical warm-starts.
        if verbose and self.best_dag.n_ops == 0 and abs(self.best_potential) < 1e-10:
            print(
                "  WARNING: Nash search converged to the empty DAG with Φ = 0.\n"
                "           This is the trivial equilibrium — usually because the\n"
                "           reference state zeros out the performance objective.\n"
                "           For MaxCut, pass `initial_dag_factory=lambda: plus_layer(n)`;\n"
                "           for H₂, use `hartree_fock_h2()`. See\n"
                "           `pqcgraphs.dag.initial_states` for all options."
            )
        return self.best_dag


__all__ = ["PQCNashGame", "NashSnapshot"]
