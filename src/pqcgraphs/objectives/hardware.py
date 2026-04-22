"""f4: hardware objective.

Per Tier1.md, P_4 (the hardware player) penalises:
  - circuit depth (as a proxy for decoherence / runtime)
  - non-native gate count (compilation into native gates multiplies depth)
  - connectivity violations (gates acting on pairs outside the coupling map)

The score is a *penalty* that the potential maximiser subtracts (Phi has
`-w_4 f_4`). f4 is therefore always >= 0 and smaller is better.

Pure Python, no GPU needed — depth and gate counts are structural.
"""
from __future__ import annotations

from typing import Callable

from ..dag.circuit_dag import CircuitDAG
from ..dag.gate_types import is_native, gate_spec
from ..dag.topology import Topology


def make_f4_hardware(
    topology: Topology,
    depth_weight: float = 1.0,
    non_native_weight: float = 2.0,
    connectivity_weight: float = 5.0,
) -> Callable[[CircuitDAG], float]:
    """Return a closure  dag -> depth_w*depth + nn_w*nn_count + conn_w*conn_viol.

    Weights default to (1, 2, 5) — connectivity violations are the hardest
    penalty because the router can't satisfy them without inserting swaps
    (which further inflate depth).
    """
    if not isinstance(topology, Topology):
        raise TypeError(
            f"make_f4_hardware requires a Topology, got {type(topology)}"
        )

    def f4(dag: CircuitDAG) -> float:
        depth = dag.depth()
        non_native_count = 0
        connectivity_violations = 0

        for op in dag.ops():
            if not is_native(op.gate_name, topology.tag):
                non_native_count += 1
            spec = gate_spec(op.gate_name)
            if spec.arity == 2:
                u, v = op.qubits[0], op.qubits[1]
                if not topology.allows(u, v):
                    connectivity_violations += 1

        score = (
            depth_weight * float(depth)
            + non_native_weight * float(non_native_count)
            + connectivity_weight * float(connectivity_violations)
        )
        return score

    return f4


__all__ = ["make_f4_hardware"]
