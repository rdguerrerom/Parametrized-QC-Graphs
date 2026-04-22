"""Tier-1 GPU objectives: end-to-end correctness tests.

All tests below consume the single `CircuitDAG -> tc.Circuit` lowering
pipeline; no test builds its own circuit. Tolerance choices are called out
in each test; if a test fails at its stated tolerance the correct fix is
to investigate (likely JAX complex-autodiff convention) rather than
loosening tolerance.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import math

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("tensorcircuit")
pytest.importorskip("jax")

import jax.numpy as jnp
import tensorcircuit as tc  # noqa: E402

from pqcgraphs.dag import (  # noqa: E402
    CircuitDAG,
    from_graph_state,
    from_graph_state_parameterized,
    heavy_hex,
)
from pqcgraphs.gpu import (  # noqa: E402
    check_gpu,
    qfim,
    qfim_spectrum,
    effective_dimension,
    stabilizer_renyi_entropy,
    nonstabilizerness_m2,
    h2_sto3g_hamiltonian,
    maxcut_hamiltonian,
    compute_dla_dimension,
    graph_state_dla_generators,
)
from pqcgraphs.objectives import (  # noqa: E402
    f1_anti_bp,
    f2_anti_sim,
    make_f3_performance,
    make_f3_maxcut,
    make_f4_hardware,
)


# --------------------------------------------------------------------------
# GPU sanity
# --------------------------------------------------------------------------

def test_check_gpu_runs():
    info = check_gpu()
    assert info["backend"] == "jax"
    assert info["dtype"] == "complex128"
    assert "devices" in info


# --------------------------------------------------------------------------
# 1) QFIM rank == |E| for parameterised graph state (abelian bound)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "graph_builder,n",
    [
        (lambda n: nx.path_graph(n), 3),
        (lambda n: nx.path_graph(n), 4),
        (lambda n: nx.cycle_graph(n), 3),
        (lambda n: nx.cycle_graph(n), 4),
        (lambda n: nx.complete_graph(n), 3),
        (lambda n: nx.complete_graph(n), 4),
    ],
)
def test_qfim_rank_matches_edge_count(graph_builder, n):
    """For exp(-i theta ZZ/2) on a graph state, QFIM is diagonal of rank |E|."""
    G = graph_builder(n)
    dag = from_graph_state_parameterized(G, theta_init=0.3, n_qubits=n)
    # Perturb to a generic theta to avoid accidental degeneracies.
    rng = np.random.default_rng(0)
    dag.thetas = rng.uniform(0.1, 1.3, size=dag.n_params)

    r = effective_dimension(dag)
    assert r == G.number_of_edges(), (
        f"QFIM rank {r} != |E| {G.number_of_edges()} for {graph_builder.__name__}({n})"
    )


# --------------------------------------------------------------------------
# 2) Magic is zero on a stabilizer state (cycle(4) graph state)
# --------------------------------------------------------------------------

def test_magic_zero_on_stabilizer_state():
    G = nx.cycle_graph(4)
    dag = from_graph_state(G, n_qubits=4)
    m2 = stabilizer_renyi_entropy(dag)
    assert m2 < 1e-10, f"M_2 on cycle-4 graph state should be ~0, got {m2}"


# --------------------------------------------------------------------------
# 3) Magic > 0 after a T gate
# --------------------------------------------------------------------------

def test_magic_positive_after_t_gate():
    G = nx.cycle_graph(4)
    dag = from_graph_state(G, n_qubits=4)
    dag.append_gate("t", (0,))
    m2 = stabilizer_renyi_entropy(dag)
    assert m2 > 1e-3, f"M_2 after a T gate on qubit 0 should be > 0, got {m2}"


# --------------------------------------------------------------------------
# 4) H2 Hartree-Fock energy
# --------------------------------------------------------------------------

def test_h2_hartree_fock_energy():
    """HF state (qubits 0 and 1 occupied in JW) is the 2-electron H2 HF solution.

    TC state convention: qubit 0 = MSB, so X_0 X_1 |0000> lives at state
    index 12 (0b1100). Under this state <Z_0>=<Z_1>=-1 (occupied) and
    <Z_2>=<Z_3>=+1 (unoccupied), and X/Y-containing terms vanish.

    The expected value below is a direct closed-form sum of the tabulated
    coefficients evaluated on <Z> = [-1,-1,+1,+1]. It is consistent with
    the published H2/STO-3G HF energy at R = 0.7414 A, ~ -1.1167 Ha (see
    O'Malley et al., PRX 6, 031007 (2016) and Kandala et al., Nature 549,
    242 (2017)).
    """
    H = h2_sto3g_hamiltonian()
    # Build the HF state (qubits 0 and 1 occupied)
    dag = CircuitDAG(4)
    dag.append_gate("x", (0,))
    dag.append_gate("x", (1,))
    # Lower and evaluate the expectation:
    from pqcgraphs.dag import make_state_fn
    fn = make_state_fn(dag)
    psi = fn(jnp.asarray(dag.thetas, dtype=jnp.float64))
    e_hf = H.expectation(psi)

    # Closed-form from our tabulated coefficients (see hamiltonians.py),
    # evaluated on |0011>_TC (qubit 0 = MSB convention).
    expected = -1.1167593073964224
    assert abs(e_hf - expected) < 1e-10, (
        f"H2 HF energy {e_hf} vs expected {expected}"
    )


def test_h2_bond_length_guard():
    with pytest.raises(ValueError):
        h2_sto3g_hamiltonian(0.80)


# --------------------------------------------------------------------------
# 5) MaxCut expectation / upper bound on K_3
# --------------------------------------------------------------------------

def test_maxcut_k3_upper_bound():
    """For K_3 the optimal cut has size 2; H_MaxCut max eigenvalue = 2.

    On |+>^3 every edge contributes (1 - <Z_i Z_j>)/2 = 1/2 (since
    <+|Z|+>=0 and the Z_iZ_j expectation is also 0 for a product state),
    so <H_MaxCut> = 3/2.
    """
    G = nx.complete_graph(3)
    H = maxcut_hamiltonian(G)

    # Max cut value == 2 (split one vertex from the other two).
    assert abs(H.max_eigenvalue() - 2.0) < 1e-10

    # |+>^3 expectation = 3/2.
    dag = CircuitDAG(3)
    for q in range(3):
        dag.append_gate("h", (q,))
    from pqcgraphs.dag import make_state_fn
    fn = make_state_fn(dag)
    psi = fn(jnp.asarray(dag.thetas, dtype=jnp.float64))
    e = H.expectation(psi)
    assert abs(e - 1.5) < 1e-10, f"<+^3|H_MaxCut|+^3> should be 3/2, got {e}"


# --------------------------------------------------------------------------
# 6) DLA dimension for parameterised graph state == |E|
# --------------------------------------------------------------------------

def test_dla_dim_graph_state_path4():
    G = nx.path_graph(4)
    gens = graph_state_dla_generators(G, 4)
    dim = compute_dla_dimension(gens)
    assert dim == G.number_of_edges() == 3


def test_dla_dim_graph_state_cycle5():
    G = nx.cycle_graph(5)
    gens = graph_state_dla_generators(G, 5)
    dim = compute_dla_dimension(gens)
    assert dim == G.number_of_edges() == 5


# --------------------------------------------------------------------------
# 7) f4 hardware penalty == depth on a native CZ-only heavy_hex circuit
# --------------------------------------------------------------------------

def test_f4_native_circuit_heavy_hex():
    topo = heavy_hex(7)
    dag = CircuitDAG(7)
    # Build a CZ circuit over heavy-hex edges. All CZs are native there, and
    # all pairs respect the coupling map by construction.
    for (u, v) in topo.allowed_pairs():
        dag.append_gate("cz", (u, v))

    f4 = make_f4_hardware(topo, depth_weight=1.0,
                          non_native_weight=2.0,
                          connectivity_weight=5.0)
    score = f4(dag)
    expected_depth = dag.depth()
    # No non-native gates, no connectivity violations => f4 == depth.
    assert abs(score - expected_depth) < 1e-12, (
        f"f4 on pure-CZ heavy-hex circuit should equal depth={expected_depth}, got {score}"
    )


# --------------------------------------------------------------------------
# 8) f4 penalises rxx on heavy_hex (non-native)
# --------------------------------------------------------------------------

def test_f4_rxx_penalised_on_heavy_hex():
    topo = heavy_hex(7)
    dag = CircuitDAG(7)
    # Use an edge allowed by the coupling map to isolate the non-native hit.
    (u, v) = next(iter(topo.allowed_pairs()))
    dag.append_gate("rxx", (u, v), theta=0.3)

    f4 = make_f4_hardware(topo, depth_weight=1.0,
                          non_native_weight=2.0,
                          connectivity_weight=5.0)
    score = f4(dag)
    # depth=1 + non_native=1 (rxx) + no connectivity violation (edge allowed)
    expected = 1.0 * 1 + 2.0 * 1 + 5.0 * 0
    assert abs(score - expected) < 1e-12, (
        f"f4 should be {expected} (depth + non-native penalty), got {score}"
    )


# --------------------------------------------------------------------------
# Extra: f1 and f2 sanity on small DAGs
# --------------------------------------------------------------------------

def test_f1_anti_bp_nonzero_on_parameterised_path():
    G = nx.path_graph(4)
    dag = from_graph_state_parameterized(G, theta_init=0.3, n_qubits=4)
    rng = np.random.default_rng(1)
    dag.thetas = rng.uniform(0.1, 1.3, size=dag.n_params)
    v = f1_anti_bp(dag)
    # Expect 3/3 = 1.0 because QFIM has full rank |E| on a generic theta.
    assert abs(v - 1.0) < 1e-12


def test_f2_anti_sim_zero_on_stabilizer():
    G = nx.cycle_graph(4)
    dag = from_graph_state(G, n_qubits=4)
    assert f2_anti_sim(dag) < 1e-10


def test_f2_refuses_n_gt_12():
    """With chunked symplectic magic, the exact enumeration runs up to
    n=12. Beyond that (n=13) we explicitly refuse rather than silently
    estimating."""
    dag = CircuitDAG(13)
    with pytest.raises(NotImplementedError):
        f2_anti_sim(dag)


def test_f3_h2_hf_energy_matches_expectation():
    """f3 = -<psi|H|psi>; on HF state the loss is E_HF so f3 = -E_HF."""
    from pqcgraphs.objectives import make_f3_h2

    dag = CircuitDAG(4)
    dag.append_gate("x", (0,))
    dag.append_gate("x", (1,))
    f3 = make_f3_h2()
    v = f3(dag)
    assert abs(v - 1.1167593073964224) < 1e-9
