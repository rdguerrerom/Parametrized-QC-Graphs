"""f3: task performance objective.

Given a Hamiltonian H (as a PauliSumOperator) and a task flag `minimize`:

    f3(dag) = - <psi(theta) | H | psi(theta)>   if minimize=True   (ground-state VQE)
    f3(dag) = + <psi(theta) | H | psi(theta)>   if minimize=False  (MaxCut / QAOA)

Phi is maximised, so in both cases a better task performance ⇒ larger f3.

Why the flag is necessary
-------------------------
For ground-state VQE the "loss" to be minimised is the energy ⟨H⟩, so the
payoff for player P3 is −⟨H⟩. For MaxCut the Hamiltonian is constructed
such that ⟨H⟩ = cut_value, and the task is to *maximise* the cut — the
payoff is therefore +⟨H⟩. Using the wrong sign causes the Nash search to
drive the circuit toward zero-cut states (e.g. by unrolling a QAOA warm
start back to |0⟩^n).

Internally we:
  1. Call `make_state_fn(dag)` once per structural identity of the DAG,
     cached so that repeated evaluations at varying theta hit a warm JIT.
  2. Compose with PauliSumOperator.expectation (itself JIT-compiled).
  3. Return a plain float.

Cache key
---------
A DAG's *structural* identity is the tuple of (gate_name, qubit_tuple,
param_idx) triples in execution order plus (n_qubits, n_params). This is
identical to the key used by `qfim_effdim`, so re-entry with the same
topology reuses the JIT trace.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Tuple

import networkx as nx

import jax
import jax.numpy as jnp

from ..dag.circuit_dag import CircuitDAG
from ..dag.lowering import _apply_op
from ..gpu import tc_backend  # noqa: F401
from ..gpu.hamiltonians import (
    PauliSumOperator,
    h2_sto3g_hamiltonian,
    maxcut_hamiltonian,
)


def _dag_structure_key(dag: CircuitDAG) -> Tuple:
    sched = tuple(
        (op.gate_name, op.qubits, -1 if op.param_idx is None else op.param_idx)
        for op in dag.ops()
    )
    return (dag.n_qubits, dag.n_params, sched)


@lru_cache(maxsize=512)
def _cached_energy_fn(structure_key, hamiltonian_id: int, minimize: bool) -> Callable:
    """Compiled theta -> ±<psi|H|psi> closure, sign determined by `minimize`.

    `hamiltonian_id` is only part of the key for cache disambiguation; the
    actual H is re-looked-up through the registry below.
    """
    n_qubits, n_params, sched = structure_key
    import tensorcircuit as tc

    ham = _HAM_REGISTRY[hamiltonian_id]

    # Build expectation fn for this H by re-using its JIT cache.
    # PauliSumOperator.expectation already JIT-compiles; here we inline the
    # state-fn + expectation into one traced function so JAX sees the chain.
    from ..gpu.hamiltonians import _expectation_fn, _pauli_string_to_digits

    pstr_digits = tuple(_pauli_string_to_digits(s) for s in ham.pauli_strings)
    coeffs = tuple(float(c) for c in ham.coeffs)
    expec = _expectation_fn(pstr_digits, coeffs, ham.n_qubits)
    sign = -1.0 if minimize else 1.0

    def state_fn(theta_vec):
        c = tc.Circuit(n_qubits)
        for gate_name, qubits, pidx in sched:
            _apply_op(c, gate_name, qubits, theta_vec, pidx)
        return c.state()

    def signed_energy(theta_vec):
        psi = state_fn(theta_vec)
        return sign * expec(psi)

    if n_params == 0:
        # No theta: JIT the zero-arity closure by binding an empty array.
        jitted = jax.jit(signed_energy)

        def fn_no_params(theta_vec):
            return jitted(theta_vec)

        return fn_no_params

    return jax.jit(signed_energy)


# Registry mapping id(PauliSumOperator) -> operator object. We use id() as
# the cache-key slot so that the lru_cache above never hashes the ndarray
# inside the operator (which would be a pain).
_HAM_REGISTRY: dict = {}


def _register_hamiltonian(ham: PauliSumOperator) -> int:
    hid = id(ham)
    _HAM_REGISTRY[hid] = ham
    return hid


def make_f3_performance(
    hamiltonian: PauliSumOperator,
    *,
    minimize: bool = True,
) -> Callable[[CircuitDAG], float]:
    """Factory: closes over a Hamiltonian, returns a DAG -> float scorer.

    Parameters
    ----------
    hamiltonian : PauliSumOperator
    minimize : bool, default True
        If True, f3 = -<H> (ground-state VQE convention; higher payoff
        means lower energy). If False, f3 = +<H> (MaxCut / QAOA convention;
        higher payoff means higher cut value).

    The closure caches JIT'd compositions per (structural identity, sign)
    of the DAG; the same DAG structure re-evaluated at different theta is a
    hot path. The DAG's qubit count must match the Hamiltonian's n_qubits.
    """
    if not isinstance(hamiltonian, PauliSumOperator):
        raise TypeError(
            f"make_f3_performance requires PauliSumOperator, got {type(hamiltonian)}"
        )
    hid = _register_hamiltonian(hamiltonian)
    expected_n = hamiltonian.n_qubits

    def f3(dag: CircuitDAG) -> float:
        if dag.n_qubits != expected_n:
            raise ValueError(
                f"Hamiltonian has n_qubits={expected_n} but DAG has "
                f"n_qubits={dag.n_qubits}"
            )
        key = _dag_structure_key(dag)
        fn = _cached_energy_fn(key, hid, minimize)
        theta = jnp.asarray(dag.thetas, dtype=jnp.float64)
        return float(fn(theta))

    return f3


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------

def make_f3_h2() -> Callable[[CircuitDAG], float]:
    """f3 for the H2 STO-3G (4-qubit JW) Hamiltonian at R = 0.7414 A.

    Ground-state task: minimize=True so f3 = -<H>. Higher f3 corresponds
    to lower electronic energy.
    """
    return make_f3_performance(h2_sto3g_hamiltonian(), minimize=True)


def make_f3_maxcut(graph: nx.Graph) -> Callable[[CircuitDAG], float]:
    """f3 for a MaxCut Hamiltonian built from `graph`.

    Maximization task: minimize=False so f3 = +<H>. Higher f3 corresponds
    to larger expected cut value.
    """
    return make_f3_performance(maxcut_hamiltonian(graph), minimize=False)


__all__ = [
    "make_f3_performance",
    "make_f3_h2",
    "make_f3_maxcut",
]
