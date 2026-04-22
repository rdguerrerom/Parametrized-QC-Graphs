"""CircuitDAG → tensorcircuit.Circuit lowering.

Single source of truth for translating the discrete DAG state into a
TensorCircuit quantum circuit. All GPU objectives (f1, f2, f3) consume the
output of `to_tensorcircuit` — never build their own circuit independently —
so any change to gate semantics or ordering is centralised here.

The lowering is **structure-static / parameters-dynamic**: given a fixed DAG
topology, `make_state_fn(dag)` returns a JAX-jittable function
`state_fn(theta_vec) -> amplitude_vector` whose only runtime dependency is
the θ array. This is the key JIT handle: re-jitting per Nash iteration
would be catastrophic, so moves that change structure invalidate a cached
`state_fn` and we rebuild lazily.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np

from .circuit_dag import CircuitDAG


def to_tensorcircuit(dag: CircuitDAG):
    """Build a concrete `tc.Circuit` for the current DAG state and θ values.

    Use this when you just want the circuit once (e.g. state-vector dump).
    For repeated evaluation with varying θ, prefer `make_state_fn`.
    """
    import tensorcircuit as tc

    c = tc.Circuit(dag.n_qubits)
    thetas = dag.thetas
    for op in dag.ops():
        _apply_op(c, op.gate_name, op.qubits, thetas, op.param_idx)
    return c


def make_state_fn(dag: CircuitDAG) -> Callable:
    """Return a Python function theta_vec -> state_vector for the current DAG topology.

    The returned function is JAX-traceable: callers may wrap it with
    `jax.jit` / `jax.grad` / `jax.hessian`. Structural changes to the DAG
    require a fresh call to `make_state_fn`.
    """
    import tensorcircuit as tc

    # Capture the structural schedule at build time so JAX tracing sees only
    # θ as a variable (gate types, qubit args, param indices are Python ints).
    schedule: List[Tuple[str, Tuple[int, ...], int]] = []
    for op in dag.ops():
        schedule.append((op.gate_name, op.qubits, -1 if op.param_idx is None else op.param_idx))
    n_qubits = dag.n_qubits

    def state_fn(theta_vec):
        c = tc.Circuit(n_qubits)
        for gate_name, qubits, pidx in schedule:
            _apply_op(c, gate_name, qubits, theta_vec, pidx)
        return c.state()

    return state_fn


def _apply_op(circuit, gate_name: str, qubits: Tuple[int, ...], theta_vec, param_idx: int) -> None:
    """Dispatch a single op onto a tensorcircuit.Circuit in place.

    `param_idx = -1` means non-parametric; otherwise index into `theta_vec`.
    `theta_vec` can be an ndarray, list, or JAX traced tensor — all work.
    """
    # Non-parametric single-qubit
    if gate_name == "h":
        circuit.h(qubits[0]); return
    if gate_name == "s":
        circuit.s(qubits[0]); return
    if gate_name == "sd":
        circuit.sd(qubits[0]); return
    if gate_name == "x":
        circuit.x(qubits[0]); return
    if gate_name == "y":
        circuit.y(qubits[0]); return
    if gate_name == "z":
        circuit.z(qubits[0]); return
    if gate_name == "t":
        circuit.t(qubits[0]); return
    if gate_name == "td":
        circuit.td(qubits[0]); return

    # Parametric single-qubit
    if gate_name == "rx":
        circuit.rx(qubits[0], theta=theta_vec[param_idx]); return
    if gate_name == "ry":
        circuit.ry(qubits[0], theta=theta_vec[param_idx]); return
    if gate_name == "rz":
        circuit.rz(qubits[0], theta=theta_vec[param_idx]); return

    # Two-qubit Clifford
    if gate_name == "cz":
        circuit.cz(qubits[0], qubits[1]); return
    if gate_name == "cnot":
        circuit.cnot(qubits[0], qubits[1]); return

    # Two-qubit parametric
    if gate_name == "rzz":
        circuit.rzz(qubits[0], qubits[1], theta=theta_vec[param_idx]); return
    if gate_name == "rxx":
        circuit.rxx(qubits[0], qubits[1], theta=theta_vec[param_idx]); return
    if gate_name == "ryy":
        circuit.ryy(qubits[0], qubits[1], theta=theta_vec[param_idx]); return

    raise ValueError(f"Lowering has no rule for gate {gate_name!r}")
