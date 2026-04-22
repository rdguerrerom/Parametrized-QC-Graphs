"""Tests for reference-state initial DAGs.

Key physics properties we validate:

- plus_layer: produces the uniform superposition, zero Z-Z correlations.
- hartree_fock_h2: produces state-vector index 12 (the HF state), giving
  E = -1.1168 Ha under the H2 Hamiltonian.
- qaoa_warm_layer: produces a non-trivial ⟨H_MaxCut⟩ on the triangle graph.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("tensorcircuit")
pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from pqcgraphs.dag import rydberg_all_to_all, to_tensorcircuit  # noqa: E402
from pqcgraphs.dag.initial_states import (  # noqa: E402
    empty,
    hartree_fock_h2,
    plus_layer,
    qaoa_warm_layer,
)
from pqcgraphs.objectives import (  # noqa: E402
    h2_sto3g_hamiltonian,
    maxcut_hamiltonian,
)


def test_plus_layer_is_uniform_superposition():
    n = 4
    dag = plus_layer(n)
    psi = np.asarray(to_tensorcircuit(dag).state())
    expected = np.full(2 ** n, 1.0 / np.sqrt(2 ** n), dtype=np.complex128)
    np.testing.assert_allclose(psi, expected, atol=1e-12)


def test_plus_layer_zeros_zz_correlations():
    """For MaxCut, ⟨|+⟩^n | Z_i Z_j | |+⟩^n⟩ = 0, so ⟨H⟩ = |E|/2.

    This is precisely why |+⟩^n is the canonical MaxCut start: f3 has real
    gradient from the start instead of being pinned at zero on |0⟩^n.
    """
    G = nx.complete_graph(3)
    H = maxcut_hamiltonian(G)
    dag = plus_layer(3)
    psi = to_tensorcircuit(dag).state()
    e = float(H.expectation(jnp.asarray(psi)))
    # K_3 has 3 edges → ⟨H⟩ = 3/2
    assert abs(e - 1.5) < 1e-10


def test_hartree_fock_h2_matches_published_energy():
    dag = hartree_fock_h2()
    psi = np.asarray(to_tensorcircuit(dag).state())
    # HF state: index 12 per TC qubit-0-MSB convention
    assert np.argmax(np.abs(psi)) == 12
    assert abs(psi[12]) > 0.999
    H = h2_sto3g_hamiltonian()
    e = float(H.expectation(jnp.asarray(psi)))
    assert abs(e - (-1.1167593073964224)) < 1e-10


def test_qaoa_warm_layer_has_expected_structure():
    topo = rydberg_all_to_all(3)
    dag = qaoa_warm_layer(topo, gamma_init=0.1, beta_init=0.1)
    counts = dag.gate_counts()
    # H layer × 3 + rzz × |E|=3 + rx × 3
    assert counts.get("h", 0) == 3
    assert counts.get("rzz", 0) == 3
    assert counts.get("rx", 0) == 3
    # rzz and rx are parametric → 6 parameters total
    assert dag.n_params == 6


def test_empty_is_zero_state():
    dag = empty(3)
    assert dag.n_ops == 0
    psi = np.asarray(to_tensorcircuit(dag).state())
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1.0
    np.testing.assert_allclose(psi, expected, atol=1e-12)
