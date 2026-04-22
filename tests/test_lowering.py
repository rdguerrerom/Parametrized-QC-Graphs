"""Verification that CircuitDAG lowering produces the intended quantum state."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("tensorcircuit")
pytest.importorskip("jax")

import tensorcircuit as tc

tc.set_backend("jax")
tc.set_dtype("complex128")

import jax.numpy as jnp  # noqa: E402

from pqcgraphs.dag import CircuitDAG, from_graph_state, make_state_fn, to_tensorcircuit  # noqa: E402


def test_empty_dag_is_product_zero():
    d = CircuitDAG(3)
    psi = to_tensorcircuit(d).state()
    expected = np.zeros(2 ** 3, dtype=np.complex128)
    expected[0] = 1.0
    np.testing.assert_allclose(np.asarray(psi), expected, atol=1e-12)


def test_hadamard_layer_produces_plus_state():
    n = 4
    d = CircuitDAG(n)
    for q in range(n):
        d.append_gate("h", (q,))
    psi = np.asarray(to_tensorcircuit(d).state())
    expected = np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=np.complex128)
    np.testing.assert_allclose(psi, expected, atol=1e-12)


def test_graph_state_embedding_matches_direct_construction():
    """ι: G → D_G should produce the same state as hand-built CZ circuit."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    dag = from_graph_state(G, n_qubits=3)
    psi_dag = np.asarray(to_tensorcircuit(dag).state())

    c = tc.Circuit(3)
    for q in range(3):
        c.h(q)
    for u, v in G.edges():
        c.cz(u, v)
    psi_ref = np.asarray(c.state())

    np.testing.assert_allclose(psi_dag, psi_ref, atol=1e-12)


def test_make_state_fn_reflects_theta_changes():
    """state_fn(θ) must actually depend on θ — not on the DAG's stored θ at build time."""
    d = CircuitDAG(1)
    d.append_gate("rx", (0,), theta=0.0)
    fn = make_state_fn(d)

    psi0 = np.asarray(fn(jnp.asarray([0.0])))
    psi1 = np.asarray(fn(jnp.asarray([np.pi])))

    # Rx(0) = I  → |0>
    np.testing.assert_allclose(psi0, np.array([1.0, 0.0], dtype=np.complex128), atol=1e-12)
    # Rx(π) = -i X  → -i |1>
    np.testing.assert_allclose(np.abs(psi1) ** 2, np.array([0.0, 1.0]), atol=1e-12)


def test_make_state_fn_is_jit_compilable():
    """The lowering must produce a pure function that jax.jit can compile."""
    import jax

    d = CircuitDAG(2)
    d.append_gate("h", (0,))
    d.append_gate("rx", (1,), theta=0.0)
    d.append_gate("cz", (0, 1))
    d.append_gate("rzz", (0, 1), theta=0.0)
    fn = make_state_fn(d)
    jfn = jax.jit(fn)

    theta = jnp.array([0.7, 1.2])
    psi = np.asarray(jfn(theta))
    assert psi.shape == (4,)
    np.testing.assert_allclose(np.vdot(psi, psi).real, 1.0, atol=1e-10)


def test_make_state_fn_differentiable():
    """f1 requires autodiff w.r.t. θ. Smoke-test jax.grad composition."""
    import jax

    d = CircuitDAG(1)
    d.append_gate("rx", (0,), theta=0.0)
    fn = make_state_fn(d)

    def overlap_with_zero(theta):
        psi = fn(theta)
        return jnp.abs(psi[0]) ** 2

    g = jax.grad(overlap_with_zero)(jnp.array([0.3]))
    # d/dθ |<0|Rx(θ)|0>|^2 = d/dθ cos²(θ/2) = -sin(θ/2)cos(θ/2) = -sin(θ)/2
    expected = -np.sin(0.3) / 2
    np.testing.assert_allclose(float(g[0]), expected, atol=1e-8)
