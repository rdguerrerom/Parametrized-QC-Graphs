"""Structural / algebraic unit tests for CircuitDAG.

These tests treat the DAG as a pure data structure — they do NOT import
tensorcircuit so they run fast in any environment. Tensor-network tests
that depend on the lowering live in test_lowering.py.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from pqcgraphs.dag import (
    CircuitDAG,
    from_graph_state,
    from_graph_state_parameterized,
    gate_spec,
    is_native,
)


def test_construct_empty():
    d = CircuitDAG(4)
    assert d.n_qubits == 4
    assert d.n_ops == 0
    assert d.n_params == 0
    assert d.depth() == 0


def test_append_non_parametric_gate_rejects_theta():
    d = CircuitDAG(2)
    with pytest.raises(ValueError, match="non-parametric"):
        d.append_gate("h", (0,), theta=0.5)


def test_append_parametric_gate_requires_theta():
    d = CircuitDAG(2)
    with pytest.raises(ValueError, match="parametric"):
        d.append_gate("rx", (0,))


def test_append_checks_arity():
    d = CircuitDAG(3)
    with pytest.raises(ValueError, match="arity"):
        d.append_gate("cz", (0,))
    with pytest.raises(ValueError, match="arity"):
        d.append_gate("h", (0, 1))


def test_append_rejects_duplicate_qubits():
    d = CircuitDAG(3)
    with pytest.raises(ValueError, match="Duplicate"):
        d.append_gate("cz", (1, 1))


def test_append_rejects_out_of_range():
    d = CircuitDAG(2)
    with pytest.raises(ValueError, match="out of range"):
        d.append_gate("h", (2,))


def test_theta_bookkeeping():
    d = CircuitDAG(2)
    d.append_gate("rx", (0,), theta=0.1)
    d.append_gate("h", (1,))
    d.append_gate("rzz", (0, 1), theta=0.2)
    d.append_gate("ry", (1,), theta=0.3)
    assert d.n_params == 3
    np.testing.assert_allclose(d.thetas, [0.1, 0.2, 0.3])


def test_depth_linear_chain():
    d = CircuitDAG(1)
    for _ in range(5):
        d.append_gate("h", (0,))
    assert d.depth() == 5


def test_depth_parallel_gates():
    d = CircuitDAG(4)
    d.append_gate("h", (0,))
    d.append_gate("h", (1,))
    d.append_gate("h", (2,))
    d.append_gate("h", (3,))
    # All four ops lie on disjoint wires → depth = 1
    assert d.depth() == 1


def test_depth_entangling_chain():
    d = CircuitDAG(3)
    d.append_gate("h", (0,))
    d.append_gate("h", (1,))
    d.append_gate("h", (2,))
    d.append_gate("cz", (0, 1))
    d.append_gate("cz", (1, 2))
    # Each CZ depends on the h on its qubits; cz(1,2) depends on cz(0,1) via wire 1
    assert d.depth() == 3


def test_remove_op_splices_wire():
    d = CircuitDAG(2)
    g0 = d.append_gate("h", (0,))
    g1 = d.append_gate("rx", (0,), theta=0.4)
    g2 = d.append_gate("h", (0,))
    d.remove_op(g1)
    assert d.n_ops == 2
    assert d.n_params == 0
    # Wire 0 must still be well-formed: input -> g0 -> g2 -> output
    edges_w0 = d.edges_on_wire(0)
    srcs = sorted(e.src for e in edges_w0)
    dsts = sorted(e.dst for e in edges_w0)
    assert g0 in srcs and g2 in dsts


def test_remove_op_reindexes_params():
    d = CircuitDAG(2)
    d.append_gate("rx", (0,), theta=1.0)  # param 0
    d.append_gate("ry", (1,), theta=2.0)  # param 1
    d.append_gate("rz", (0,), theta=3.0)  # param 2
    # Remove the middle parametric op
    d.remove_op(d.op_ids[1])
    np.testing.assert_allclose(d.thetas, [1.0, 3.0])
    # rz should now live at param_idx = 1
    rz_node = d.ops()[-1]
    assert rz_node.gate_name == "rz"
    assert rz_node.param_idx == 1


def test_retype_preserves_arity_changes_parametricity():
    d = CircuitDAG(2)
    op = d.append_gate("cz", (0, 1))
    d.retype_op(op, "rzz", new_theta=0.9)
    assert d.ops()[0].gate_name == "rzz"
    assert d.n_params == 1
    np.testing.assert_allclose(d.thetas, [0.9])

    # Flip back: rzz -> cz should drop the parameter
    d.retype_op(op, "cz")
    assert d.n_params == 0


def test_retype_rejects_arity_mismatch():
    d = CircuitDAG(2)
    op = d.append_gate("cz", (0, 1))
    with pytest.raises(ValueError, match="arity"):
        d.retype_op(op, "h")


def test_rewire_op_moves_wires():
    d = CircuitDAG(3)
    op = d.append_gate("cz", (0, 1))
    d.append_gate("h", (2,))
    d.rewire_op(op, (1, 2))
    # Find the cz op and confirm its qubits
    cz_nodes = [n for n in d.ops() if n.gate_name == "cz"]
    assert len(cz_nodes) == 1
    assert cz_nodes[0].qubits == (1, 2)


def test_perturb_theta():
    d = CircuitDAG(1)
    d.append_gate("rx", (0,), theta=1.0)
    d.perturb_theta(0, 0.25)
    np.testing.assert_allclose(d.thetas, [1.25])


def test_copy_is_deep():
    d = CircuitDAG(2)
    d.append_gate("h", (0,))
    d.append_gate("rx", (1,), theta=0.5)
    dd = d.copy()
    dd.append_gate("h", (1,))
    dd.perturb_theta(0, 1.0)
    assert d.n_ops == 2 and dd.n_ops == 3
    np.testing.assert_allclose(d.thetas, [0.5])
    np.testing.assert_allclose(dd.thetas, [1.5])


def test_from_graph_state_embedding_matches_adjacency():
    """docs/circuit-dag-generalization.md §3: one CZ per edge + H layer."""
    G = nx.erdos_renyi_graph(5, 0.5, seed=7)
    d = from_graph_state(G)
    counts = d.gate_counts()
    assert counts.get("h", 0) == 5
    assert counts.get("cz", 0) == G.number_of_edges()


def test_from_graph_state_parameterized_matches_dla_setup():
    """Stabilizer-Rank-from-DLA.md §1.1: U_G(θ) = ∏ exp(-iθ Z_iZ_j/2) · H^⊗n."""
    G = nx.cycle_graph(6)
    d = from_graph_state_parameterized(G, theta_init=0.0)
    assert d.gate_counts() == {"h": 6, "rzz": 6}
    assert d.n_params == 6


def test_gate_spec_is_native():
    assert is_native("cz", "heavy_hex")
    assert is_native("cz", "grid")
    assert not is_native("cz", "rydberg")
    assert is_native("rxx", "rydberg")
    # Single-qubit gates are native everywhere
    assert is_native("h", "heavy_hex")
    assert is_native("rx", "rydberg")


def test_gate_spec_unknown_raises():
    import pytest as _pt
    with _pt.raises(ValueError, match="Unknown gate"):
        gate_spec("xyz")
