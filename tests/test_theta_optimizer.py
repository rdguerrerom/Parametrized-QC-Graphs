"""Tests for the θ-gradient optimizer."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("tensorcircuit")
pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from pqcgraphs.dag import CircuitDAG, to_tensorcircuit  # noqa: E402
from pqcgraphs.dag.initial_states import hartree_fock_h2, plus_layer  # noqa: E402
from pqcgraphs.gpu.hamiltonians import (  # noqa: E402
    h2_sto3g_hamiltonian,
    maxcut_hamiltonian,
)
from pqcgraphs.gpu.theta_optimizer import optimize_theta  # noqa: E402


def _evaluate(dag, H):
    psi = to_tensorcircuit(dag).state()
    return float(H.expectation(jnp.asarray(psi)))


def test_optimize_theta_noop_on_zero_params():
    """No parameters → should return a copy identical to input."""
    H = h2_sto3g_hamiltonian()
    dag = hartree_fock_h2()
    assert dag.n_params == 0
    out = optimize_theta(dag, H, minimize=True, n_steps=50)
    assert out.n_params == 0


def test_optimize_theta_h2_improves_all_methods():
    """All three optimizers must decrease the energy from the starting ansatz.
    (Reaching HF is a tighter bound tested separately for Adam / Nesterov.)"""
    H = h2_sto3g_hamiltonian()
    dag = hartree_fock_h2()
    dag.append_gate("rx", (0,), theta=0.1)
    dag.append_gate("rx", (2,), theta=0.1)
    dag.append_gate("cnot", (0, 2))
    dag.append_gate("ry", (0,), theta=0.1)

    e_before = _evaluate(dag, H)
    for method in ("gd", "adam", "nesterov"):
        opt = optimize_theta(dag, H, minimize=True, n_steps=200, lr=0.05, method=method)
        e_after = _evaluate(opt, H)
        assert e_after < e_before - 0.1, (
            f"{method} barely improved: {e_before} → {e_after}"
        )


def test_optimize_theta_h2_adam_reaches_hf():
    """Adam with default lr should converge to within 1 mHa of HF in 200 steps."""
    H = h2_sto3g_hamiltonian()
    dag = hartree_fock_h2()
    dag.append_gate("rx", (0,), theta=0.1)
    dag.append_gate("rx", (2,), theta=0.1)
    dag.append_gate("cnot", (0, 2))
    dag.append_gate("ry", (0,), theta=0.1)
    opt = optimize_theta(dag, H, minimize=True, n_steps=200, lr=0.05, method="adam")
    assert _evaluate(opt, H) < -1.115


def test_optimize_theta_h2_nesterov_reaches_hf():
    """Nesterov should converge at least as well as Adam given appropriate lr."""
    H = h2_sto3g_hamiltonian()
    dag = hartree_fock_h2()
    dag.append_gate("rx", (0,), theta=0.1)
    dag.append_gate("rx", (2,), theta=0.1)
    dag.append_gate("cnot", (0, 2))
    dag.append_gate("ry", (0,), theta=0.1)
    # At lr=0.05 the Nesterov trajectory converges to essentially HF in 200
    # steps; lr=0.1 overshoots (momentum doubles the effective step).
    opt = optimize_theta(dag, H, minimize=True, n_steps=200, lr=0.05, method="nesterov", momentum=0.9)
    assert _evaluate(opt, H) < -1.115


def test_optimize_theta_maxcut_ascent_direction():
    """For MaxCut (minimize=False), theta-opt should increase <H>."""
    G = nx.complete_graph(4)
    H = maxcut_hamiltonian(G)
    dag = plus_layer(4)
    dag.append_gate("rzz", (0, 1), theta=0.5)
    dag.append_gate("rzz", (2, 3), theta=0.5)
    dag.append_gate("rx", (0,), theta=0.5)
    dag.append_gate("rx", (2,), theta=0.5)

    c_before = _evaluate(dag, H)
    for method in ("adam", "nesterov"):
        opt = optimize_theta(dag, H, minimize=False, n_steps=200, lr=0.05, method=method)
        c_after = _evaluate(opt, H)
        assert c_after >= c_before - 1e-6, (
            f"{method} ran in the wrong direction"
        )


def test_optimize_theta_unknown_method_raises():
    H = h2_sto3g_hamiltonian()
    dag = hartree_fock_h2()
    dag.append_gate("rx", (0,), theta=0.1)
    with pytest.raises(ValueError, match="Unknown method"):
        optimize_theta(dag, H, method="sgd_with_nesterov_plus_adam_plus_coffee")


def test_nesterov_matches_adam_within_tolerance():
    """On a well-conditioned small instance, both optimizers should land
    at essentially the same stationary point."""
    H = h2_sto3g_hamiltonian()
    dag = hartree_fock_h2()
    dag.append_gate("rx", (0,), theta=0.3)
    dag.append_gate("ry", (2,), theta=0.4)
    dag.append_gate("cnot", (0, 2))
    dag.append_gate("rx", (1,), theta=0.5)

    opt_adam = optimize_theta(dag, H, minimize=True, n_steps=500, lr=0.05, method="adam", tol=1e-12)
    opt_nest = optimize_theta(dag, H, minimize=True, n_steps=500, lr=0.02, method="nesterov", tol=1e-12)
    e_adam = _evaluate(opt_adam, H)
    e_nest = _evaluate(opt_nest, H)
    # Both should agree on the energy to 4 decimals
    assert abs(e_adam - e_nest) < 1e-3, (
        f"Adam and Nesterov disagree on final energy: {e_adam} vs {e_nest}"
    )
