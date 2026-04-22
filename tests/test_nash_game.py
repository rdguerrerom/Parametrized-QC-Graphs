"""Integration tests for the Nash game on small circuits."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("tensorcircuit")
pytest.importorskip("jax")

from pqcgraphs.dag import CircuitDAG, rydberg_all_to_all
from pqcgraphs.game import (
    PQCNashGame,
    compute_nash_gap,
    evaluate_all,
    make_default_players,
    potential,
)
from pqcgraphs.game.moves import all_candidates
from pqcgraphs.objectives import maxcut_hamiltonian


def _triangle_setup():
    G = nx.complete_graph(3)
    H = maxcut_hamiltonian(G)
    topo = rydberg_all_to_all(3)
    players = make_default_players(
        hamiltonian=H,
        topology=topo,
        w_anti_bp=0.3,
        w_anti_sim=0.2,
        w_performance=1.0,
        w_hardware=0.1,
    )
    return H, topo, players


def test_evaluate_all_reports_every_player():
    _, _, players = _triangle_setup()
    dag = CircuitDAG(3)
    report = evaluate_all(players, dag)
    assert set(report.keys()) == {"anti_bp", "anti_sim", "performance", "hardware", "Phi"}
    for name in ("anti_bp", "anti_sim", "performance", "hardware"):
        assert set(report[name].keys()) == {"raw", "weighted", "sign"}


def test_potential_matches_sum_of_payoffs():
    _, _, players = _triangle_setup()
    dag = CircuitDAG(3)
    dag.append_gate("h", (0,))
    dag.append_gate("rzz", (0, 1), theta=0.5)
    phi = potential(players, dag)
    expected = sum(p.payoff(dag) for p in players)
    assert abs(phi - expected) < 1e-12


def test_nash_gap_is_nonnegative():
    _, topo, players = _triangle_setup()
    dag = CircuitDAG(3)
    dag.append_gate("ry", (0,), theta=0.1)
    cands = all_candidates(dag, topo)
    report = compute_nash_gap(players, dag, cands)
    assert report.delta_nash >= 0.0
    for v in report.per_player.values():
        assert v >= 0.0


def test_game_runs_and_converges_on_improving_moves():
    """3 iterations: the potential on the leader must be non-decreasing or
    the best-so-far must strictly improve."""
    _, topo, players = _triangle_setup()
    game = PQCNashGame(
        n_qubits=3,
        players=players,
        topology=topo,
        population_size=2,
        temperature=0.3,  # low T -> mostly greedy
        cooling_rate=0.9,
        seed=42,
    )
    best_phis = []
    for it in range(3):
        game.step(it)
        best_phis.append(game.best_potential)
    # best-so-far is monotonic non-decreasing by construction
    for a, b in zip(best_phis, best_phis[1:]):
        assert b >= a - 1e-10


def test_snapshot_has_expected_fields():
    _, topo, players = _triangle_setup()
    game = PQCNashGame(
        n_qubits=3, players=players, topology=topo,
        population_size=1, seed=0,
    )
    game.step(0)
    snap = game.history[0]
    assert snap.iteration == 0
    assert snap.n_params >= 0
    assert snap.n_ops >= 0
    assert snap.nash_gap >= 0.0
    assert set(snap.per_player.keys()) == {"anti_bp", "anti_sim", "performance", "hardware", "Phi"}
