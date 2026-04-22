"""CircuitDAG: discrete state for Tier-1 Nash architecture search.

Generalizes graph-state adjacency matrices (see docs/circuit-dag-generalization.md)
into directed, gate-typed, parameterized DAGs.
"""
from .circuit_dag import CircuitDAG, from_graph_state, from_graph_state_parameterized
from .gate_types import GateSpec, GATE_SPECS, gate_spec, is_native
from . import initial_states
from .lowering import make_state_fn, to_tensorcircuit
from .node import Node, WireEdge
from .topology import Topology, heavy_hex, grid_2d, rydberg_all_to_all

__all__ = [
    "CircuitDAG",
    "from_graph_state",
    "from_graph_state_parameterized",
    "GateSpec",
    "GATE_SPECS",
    "gate_spec",
    "is_native",
    "make_state_fn",
    "to_tensorcircuit",
    "Node",
    "WireEdge",
    "Topology",
    "heavy_hex",
    "grid_2d",
    "rydberg_all_to_all",
    "initial_states",
]
