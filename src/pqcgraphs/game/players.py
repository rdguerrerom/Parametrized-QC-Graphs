"""Four Nash players packaged as `NashPlayer` records.

Each player wraps a scorer (DAG -> float) with metadata (name, weight, sign).
The Nash potential is

    Phi(dag) = sum_i  sign_i * weight_i * score_i(dag)

with sign_i = +1 for reward players (f1, f2, f3) and -1 for the penalty
player (f4, hardware). This keeps the potential convention unified and the
Nash-gap calculation symmetric: a deviation improves the potential iff it
improves at least one weighted contribution relative to all others.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import networkx as nx

from ..dag.circuit_dag import CircuitDAG
from ..dag.topology import Topology
from ..objectives import (
    PauliSumOperator,
    f1_anti_bp,
    f2_anti_sim,
    make_f3_performance,
    make_f4_hardware,
)


@dataclass
class NashPlayer:
    name: str
    weight: float
    scorer: Callable[[CircuitDAG], float]
    sign: int  # +1 reward, -1 penalty

    def payoff(self, dag: CircuitDAG) -> float:
        """Weighted signed contribution to Phi for this player."""
        return self.sign * self.weight * float(self.scorer(dag))

    def raw(self, dag: CircuitDAG) -> float:
        return float(self.scorer(dag))


def make_default_players(
    *,
    hamiltonian: PauliSumOperator,
    topology: Topology,
    w_anti_bp: float = 1.0,
    w_anti_sim: float = 1.0,
    w_performance: float = 1.0,
    w_hardware: float = 0.2,
    hardware_sub_weights: tuple = (1.0, 2.0, 5.0),
    minimize_performance: bool = True,
) -> List[NashPlayer]:
    """Build the canonical four-player roster from Tier1.md sec. 1.

    Parameters
    ----------
    hamiltonian, topology : the task / hardware definitions.
    w_* : relative weights for each objective.
    hardware_sub_weights : (depth, non_native, connectivity) sub-weights.
    minimize_performance : task semantics for f3.
        True  — ground-state problems (H2, spin chains, etc.). f3 = -<H>.
        False — maximisation problems (MaxCut, QAOA). f3 = +<H>.
        Mismatching this against the task is the most common misconfiguration
        and produces the trivial "tear down the circuit to get Φ = 0" failure
        mode. Prefer the convenience factories `make_f3_h2` and
        `make_f3_maxcut` (in `objectives.performance`) when possible — they
        set the flag for you.
    """
    depth_w, non_native_w, conn_w = hardware_sub_weights
    f3 = make_f3_performance(hamiltonian, minimize=minimize_performance)
    f4 = make_f4_hardware(topology, depth_w, non_native_w, conn_w)
    return [
        NashPlayer("anti_bp",      w_anti_bp,      f1_anti_bp, sign=+1),
        NashPlayer("anti_sim",     w_anti_sim,     f2_anti_sim, sign=+1),
        NashPlayer("performance",  w_performance,  f3,         sign=+1),
        NashPlayer("hardware",     w_hardware,     f4,         sign=-1),
    ]


def potential(players: List[NashPlayer], dag: CircuitDAG) -> float:
    """Total Nash potential Phi(dag) = sum_i sign_i * weight_i * score_i(dag)."""
    return sum(p.payoff(dag) for p in players)


def evaluate_all(players: List[NashPlayer], dag: CircuitDAG) -> dict:
    """Per-player diagnostic dict (raw and weighted contributions)."""
    out = {}
    for p in players:
        raw = p.raw(dag)
        contrib = p.sign * p.weight * raw
        out[p.name] = {"raw": raw, "weighted": contrib, "sign": p.sign}
    out["Phi"] = sum(v["weighted"] for v in out.values())
    return out


__all__ = [
    "NashPlayer",
    "make_default_players",
    "potential",
    "evaluate_all",
]
