"""Parameterized-circuit DAG.

CircuitDAG is the discrete state in the Tier-1 Nash game. It stores:

  - n_qubits wire identifiers (0..n-1)
  - A topologically-ordered list of op nodes with gate + qubit args + param idx
  - A flat parameter vector θ shared by all parametric op nodes
  - (For internal iteration) per-wire last-node tracking, rebuilt on mutation

Mutation primitives — `append_gate`, `insert_gate`, `remove_op`, `retype_op`,
`rewire_op` — correspond 1-to-1 to the Tier-1 move set. The DAG is ALWAYS
kept consistent: every op node has exactly `spec.arity` incoming wire edges
(from the previous op on each of its qubits, or from the input node) and one
outgoing edge per qubit (to the next op on that qubit, or to the output
node).

`copy()` returns a deep structural copy suitable for a candidate move in the
Nash best-response loop.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from .gate_types import GateSpec, gate_spec
from .node import Node, WireEdge


class CircuitDAG:
    """Directed acyclic graph representation of a parameterized quantum circuit."""

    def __init__(self, n_qubits: int) -> None:
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        self.n_qubits: int = n_qubits

        self._next_id: int = 0
        self._nodes: Dict[int, Node] = {}
        self._edges: List[WireEdge] = []
        self._op_order: List[int] = []  # topological order of op node ids
        self._thetas: List[float] = []

        # input/output node id per wire
        self._input_id: List[int] = []
        self._output_id: List[int] = []
        # last op node currently driving each wire (input id if no op yet)
        self._last_on_wire: List[int] = []

        for q in range(n_qubits):
            in_id = self._fresh_id()
            out_id = self._fresh_id()
            self._nodes[in_id] = Node(in_id, "input", (q,))
            self._nodes[out_id] = Node(out_id, "output", (q,))
            self._edges.append(WireEdge(in_id, out_id, q))
            self._input_id.append(in_id)
            self._output_id.append(out_id)
            self._last_on_wire.append(in_id)

    # ------------------------------------------------------------------ basics
    def _fresh_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    @property
    def n_params(self) -> int:
        return len(self._thetas)

    @property
    def thetas(self) -> np.ndarray:
        return np.asarray(self._thetas, dtype=np.float64)

    @thetas.setter
    def thetas(self, values: Sequence[float]) -> None:
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.shape[0] != len(self._thetas):
            raise ValueError(
                f"θ has {arr.shape[0]} entries but DAG expects {len(self._thetas)}"
            )
        self._thetas = arr.tolist()

    @property
    def n_ops(self) -> int:
        return len(self._op_order)

    @property
    def op_ids(self) -> Tuple[int, ...]:
        return tuple(self._op_order)

    def node(self, node_id: int) -> Node:
        return self._nodes[node_id]

    def ops(self) -> List[Node]:
        """Op nodes in topological (execution) order."""
        return [self._nodes[i] for i in self._op_order]

    def depth(self) -> int:
        """Longest path in the DAG measured in op nodes."""
        if not self._op_order:
            return 0
        # Each op's depth = 1 + max(depth of op-predecessors on any wire)
        dep: Dict[int, int] = {}
        for op_id in self._op_order:
            preds = [e.src for e in self._edges if e.dst == op_id]
            pred_ops = [p for p in preds if self._nodes[p].kind == "op"]
            dep[op_id] = 1 + (max(dep[p] for p in pred_ops) if pred_ops else 0)
        return max(dep.values())

    def edges_on_wire(self, qubit: int) -> List[WireEdge]:
        return [e for e in self._edges if e.qubit == qubit]

    def gate_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in self.ops():
            counts[n.gate_name] = counts.get(n.gate_name, 0) + 1
        return counts

    # ------------------------------------------------------------------ mutation
    def append_gate(
        self,
        gate_name: str,
        qubits: Sequence[int],
        theta: Optional[float] = None,
    ) -> int:
        """Add a gate at the end of the current circuit.

        Returns the new op node id. Raises if gate arity or parametricity
        is inconsistent with the request.
        """
        spec = gate_spec(gate_name)
        qt = tuple(int(q) for q in qubits)
        self._check_qubits(spec, qt)

        param_idx: Optional[int] = None
        if spec.is_parametric:
            if theta is None:
                raise ValueError(f"Gate {gate_name!r} is parametric; theta required")
            param_idx = len(self._thetas)
            self._thetas.append(float(theta))
        elif theta is not None:
            raise ValueError(f"Gate {gate_name!r} is non-parametric; theta forbidden")

        op_id = self._fresh_id()
        self._nodes[op_id] = Node(op_id, "op", qt, gate_name=gate_name, param_idx=param_idx)
        self._op_order.append(op_id)

        # Redirect each wire this op touches: last_on_wire[q] --q--> op --q--> output
        for q in qt:
            prev = self._last_on_wire[q]
            out = self._output_id[q]
            # remove edge (prev -> out) if it exists
            self._edges = [e for e in self._edges if not (e.src == prev and e.dst == out and e.qubit == q)]
            # insert edges (prev -> op) and (op -> out)
            self._edges.append(WireEdge(prev, op_id, q))
            self._edges.append(WireEdge(op_id, out, q))
            self._last_on_wire[q] = op_id
        return op_id

    def remove_op(self, op_id: int) -> None:
        """Remove an op node and splice its wires back together.

        Also releases the op's θ slot if parametric (by compacting θ and
        reindexing any higher-indexed op).
        """
        node = self._nodes[op_id]
        if node.kind != "op":
            raise ValueError(f"node {op_id} is not an op")

        # Per-wire splice: for each qubit, find predecessor/successor along that
        # wire and join them directly.
        for q in node.qubits:
            pred_edge = next(e for e in self._edges if e.dst == op_id and e.qubit == q)
            succ_edge = next(e for e in self._edges if e.src == op_id and e.qubit == q)
            self._edges.remove(pred_edge)
            self._edges.remove(succ_edge)
            self._edges.append(WireEdge(pred_edge.src, succ_edge.dst, q))
            if self._last_on_wire[q] == op_id:
                self._last_on_wire[q] = pred_edge.src

        self._op_order.remove(op_id)
        del self._nodes[op_id]

        if node.param_idx is not None:
            idx = node.param_idx
            del self._thetas[idx]
            # decrement param_idx for any op whose index was above `idx`
            for other_id in self._op_order:
                other = self._nodes[other_id]
                if other.param_idx is not None and other.param_idx > idx:
                    self._nodes[other_id] = Node(
                        other.node_id, "op", other.qubits,
                        gate_name=other.gate_name, param_idx=other.param_idx - 1,
                    )

    def retype_op(self, op_id: int, new_gate: str, new_theta: Optional[float] = None) -> None:
        """Swap a gate's type, preserving its wire placement where arity matches."""
        node = self._nodes[op_id]
        if node.kind != "op":
            raise ValueError(f"node {op_id} is not an op")
        new_spec = gate_spec(new_gate)
        if new_spec.arity != len(node.qubits):
            raise ValueError(
                f"retype: new gate {new_gate!r} has arity {new_spec.arity}, "
                f"op currently acts on {len(node.qubits)} qubits"
            )

        # Handle parameter slot transitions.
        old_idx = node.param_idx
        new_idx: Optional[int] = None
        if old_idx is not None and new_spec.is_parametric:
            # in-place parameter replacement
            if new_theta is not None:
                self._thetas[old_idx] = float(new_theta)
            new_idx = old_idx
        elif old_idx is not None and not new_spec.is_parametric:
            # drop parameter
            del self._thetas[old_idx]
            for other_id in self._op_order:
                if other_id == op_id:
                    continue
                other = self._nodes[other_id]
                if other.param_idx is not None and other.param_idx > old_idx:
                    self._nodes[other_id] = Node(
                        other.node_id, "op", other.qubits,
                        gate_name=other.gate_name, param_idx=other.param_idx - 1,
                    )
        elif old_idx is None and new_spec.is_parametric:
            if new_theta is None:
                raise ValueError(f"retype → parametric {new_gate!r} requires new_theta")
            new_idx = len(self._thetas)
            self._thetas.append(float(new_theta))

        self._nodes[op_id] = Node(
            op_id, "op", node.qubits, gate_name=new_gate, param_idx=new_idx
        )

    def rewire_op(self, op_id: int, new_qubits: Sequence[int]) -> None:
        """Move an op node onto a different set of qubits (arity preserved)."""
        node = self._nodes[op_id]
        if node.kind != "op":
            raise ValueError(f"node {op_id} is not an op")
        nq = tuple(int(q) for q in new_qubits)
        spec = gate_spec(node.gate_name)
        self._check_qubits(spec, nq)
        if set(nq) == set(node.qubits) and nq == node.qubits:
            return  # no-op

        # Easiest: remove and re-append preserving gate + theta (rewire destroys
        # the op's temporal position, consistent with "move-to-end" semantics).
        theta = self._thetas[node.param_idx] if node.param_idx is not None else None
        self.remove_op(op_id)
        self.append_gate(node.gate_name, nq, theta=theta)

    def perturb_theta(self, param_idx: int, delta: float) -> None:
        if not (0 <= param_idx < len(self._thetas)):
            raise IndexError(f"param_idx {param_idx} out of range")
        self._thetas[param_idx] = float(self._thetas[param_idx] + delta)

    # ------------------------------------------------------------------ utils
    def _check_qubits(self, spec: GateSpec, qt: Tuple[int, ...]) -> None:
        if len(qt) != spec.arity:
            raise ValueError(
                f"Gate {spec.name!r} has arity {spec.arity} but got qubits={qt}"
            )
        if len(set(qt)) != len(qt):
            raise ValueError(f"Duplicate qubits in {qt}")
        for q in qt:
            if not (0 <= q < self.n_qubits):
                raise ValueError(f"qubit {q} out of range [0, {self.n_qubits})")

    def copy(self) -> "CircuitDAG":
        """Deep structural copy. Needed for Nash candidate moves."""
        dup = CircuitDAG.__new__(CircuitDAG)
        dup.n_qubits = self.n_qubits
        dup._next_id = self._next_id
        dup._nodes = dict(self._nodes)  # Node is frozen/immutable
        dup._edges = list(self._edges)
        dup._op_order = list(self._op_order)
        dup._thetas = list(self._thetas)
        dup._input_id = list(self._input_id)
        dup._output_id = list(self._output_id)
        dup._last_on_wire = list(self._last_on_wire)
        return dup

    def to_networkx(self) -> nx.DiGraph:
        """Return a networkx DiGraph mirror for visualization / analysis."""
        g = nx.DiGraph()
        for n in self._nodes.values():
            g.add_node(n.node_id, kind=n.kind, qubits=n.qubits,
                       gate=n.gate_name, param_idx=n.param_idx)
        for e in self._edges:
            g.add_edge(e.src, e.dst, qubit=e.qubit)
        return g

    def __repr__(self) -> str:
        return (
            f"CircuitDAG(n_qubits={self.n_qubits}, n_ops={self.n_ops}, "
            f"depth={self.depth()}, n_params={self.n_params})"
        )


def from_graph_state(graph: nx.Graph, n_qubits: Optional[int] = None) -> CircuitDAG:
    """Embedding ι: G → D_G from docs/circuit-dag-generalization.md §3.

    Constructs the canonical graph-state preparation circuit:
      (Hadamard layer) ∘ ∏_{(i,j)∈E} CZ_ij applied to |0⟩^n.

    CZ ordering follows `graph.edges()`; because CZs commute, all orderings
    produce the same state (but not the same DAG).
    """
    if n_qubits is None:
        n_qubits = max(graph.nodes()) + 1 if graph.number_of_nodes() else 0
    dag = CircuitDAG(n_qubits)
    for q in range(n_qubits):
        dag.append_gate("h", (q,))
    for u, v in graph.edges():
        dag.append_gate("cz", (u, v))
    return dag


def from_graph_state_parameterized(graph: nx.Graph, theta_init: float = 0.0,
                                   n_qubits: Optional[int] = None) -> CircuitDAG:
    """Parameterized graph-state circuit U_G(θ) = ∏_{(i,j)∈E} exp(-i θ_ij Z_iZ_j/2) · H^⊗n.

    Matches the abelian-bound setup of Stabilizer-Rank-from-DLA.md §1.1: one
    rzz gate per edge with independent parameter, following a Hadamard layer.
    """
    if n_qubits is None:
        n_qubits = max(graph.nodes()) + 1 if graph.number_of_nodes() else 0
    dag = CircuitDAG(n_qubits)
    for q in range(n_qubits):
        dag.append_gate("h", (q,))
    for u, v in graph.edges():
        dag.append_gate("rzz", (u, v), theta=theta_init)
    return dag
