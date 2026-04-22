"""Node types for the CircuitDAG.

Three node flavours — input, op, output — each identified by a monotonic
integer id assigned by the owning CircuitDAG. Op nodes carry their gate
name (looked up in `gate_types.GATE_SPECS`), the ordered tuple of qubits
they act on, and (if parametric) an index into the DAG's flat parameter
vector θ.

Keeping a single `Node` dataclass with a `kind` tag simplifies iteration
and networkx interop; the alternative (separate classes) buys nothing
since nodes are immutable records once placed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class Node:
    """A single DAG node.

    node_id: unique int inside the owning CircuitDAG.
    kind: one of "input", "op", "output".
    qubits: qubits this node is attached to. For input/output, a 1-tuple
            naming the wire. For op, the ordered arguments to the gate.
    gate_name: gate-type key (only for kind == "op").
    param_idx: index into the flat θ vector (only for parametric op nodes).
    """
    node_id: int
    kind: str
    qubits: Tuple[int, ...]
    gate_name: Optional[str] = None
    param_idx: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in ("input", "op", "output"):
            raise ValueError(f"kind must be input/op/output, got {self.kind!r}")
        if self.kind in ("input", "output"):
            if len(self.qubits) != 1:
                raise ValueError(
                    f"{self.kind} node has qubits={self.qubits}; expected 1-tuple"
                )
            if self.gate_name is not None or self.param_idx is not None:
                raise ValueError(f"{self.kind} node must not carry gate metadata")
        else:  # op
            if self.gate_name is None:
                raise ValueError("op node requires gate_name")
            if len(self.qubits) == 0:
                raise ValueError("op node must act on ≥1 qubit")


@dataclass(frozen=True)
class WireEdge:
    """A directed edge labelled by the qubit (wire) flowing across it.

    src, dst: source/destination node ids in the owning CircuitDAG.
    qubit: which wire this edge represents. An op node with arity a has
           exactly a incoming and a outgoing wire edges, one per qubit in
           `Node.qubits`.
    """
    src: int
    dst: int
    qubit: int
