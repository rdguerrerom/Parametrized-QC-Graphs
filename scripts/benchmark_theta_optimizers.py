"""Benchmark vanilla-GD / Adam / Nesterov on realistic quantum objectives.

For each (task, ansatz, method) we record (iterations used, wall time,
final objective, gap to the best-achieved objective) so we can read off
the dominant optimizer in each regime.

Run with:
    conda run -n pytorch-env python scripts/benchmark_theta_optimizers.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

import jax.numpy as jnp

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from pqcgraphs.dag import CircuitDAG, to_tensorcircuit
from pqcgraphs.dag.initial_states import hartree_fock_h2, plus_layer
from pqcgraphs.gpu.hamiltonians import (
    h2_sto3g_hamiltonian,
    maxcut_hamiltonian,
    tfim_hamiltonian,
)
from pqcgraphs.gpu.theta_optimizer import optimize_theta, _cached_energy_grad_fn, _register, _dag_structure_key


def evaluate_sign_aware(dag, hamiltonian, minimize):
    """Return the scalar f = (±)<H> consistent with optimize_theta's sign."""
    psi = to_tensorcircuit(dag).state()
    E = float(hamiltonian.expectation(jnp.asarray(psi)))
    return -E if minimize else E


CASES = [
    {
        "name": "H2_STO3G_4q_6param_ansatz",
        "minimize": True,
        "n_steps": 300,
        "lrs": {"gd": 0.05, "adam": 0.05, "nesterov": 0.05},
        "build": lambda: _h2_ansatz(),
        "hamiltonian": lambda: h2_sto3g_hamiltonian(),
    },
    {
        "name": "MaxCut_K4_8param_ansatz",
        "minimize": False,
        "n_steps": 300,
        "lrs": {"gd": 0.05, "adam": 0.05, "nesterov": 0.05},
        "build": lambda: _maxcut_ansatz(4),
        "hamiltonian": lambda: maxcut_hamiltonian(nx.complete_graph(4)),
    },
    {
        "name": "TFIM_n6_g1_QAOA_p2",
        "minimize": True,
        "n_steps": 300,
        "lrs": {"gd": 0.03, "adam": 0.05, "nesterov": 0.03},
        "build": lambda: _qaoa_p_ansatz(6, p=2),
        "hamiltonian": lambda: tfim_hamiltonian(6, J=1.0, h=1.0),
    },
    {
        "name": "TFIM_n8_g1_QAOA_p2",
        "minimize": True,
        "n_steps": 300,
        "lrs": {"gd": 0.03, "adam": 0.05, "nesterov": 0.03},
        "build": lambda: _qaoa_p_ansatz(8, p=2),
        "hamiltonian": lambda: tfim_hamiltonian(8, J=1.0, h=1.0),
    },
]


def _h2_ansatz():
    dag = hartree_fock_h2()
    # 6-param UCC-like excitation (simple and parametric)
    for q in range(4):
        dag.append_gate("ry", (q,), theta=0.1)
    dag.append_gate("cnot", (0, 1))
    dag.append_gate("cnot", (2, 3))
    dag.append_gate("ry", (1,), theta=0.1)
    dag.append_gate("ry", (3,), theta=0.1)
    return dag


def _maxcut_ansatz(n):
    dag = plus_layer(n)
    # 1-layer QAOA on the complete graph
    for i in range(n):
        for j in range(i + 1, n):
            dag.append_gate("rzz", (i, j), theta=0.1)
    for q in range(n):
        dag.append_gate("rx", (q,), theta=0.1)
    return dag


def _qaoa_p_ansatz(n, p):
    dag = plus_layer(n)
    for _ in range(p):
        # problem layer (line interactions)
        for i in range(n - 1):
            dag.append_gate("rzz", (i, i + 1), theta=0.2)
        # mixer
        for q in range(n):
            dag.append_gate("rx", (q,), theta=0.2)
    return dag


def run_case(case):
    dag = case["build"]()
    H = case["hamiltonian"]()
    minimize = case["minimize"]
    n_steps = case["n_steps"]
    lrs = case["lrs"]

    base_theta = dag.thetas.copy()
    f_start = evaluate_sign_aware(dag, H, minimize)
    results = {}

    for method in ("gd", "adam", "nesterov"):
        dag.thetas = base_theta.copy()  # reset to same start
        t0 = time.perf_counter()
        opt = optimize_theta(
            dag, H,
            minimize=minimize,
            n_steps=n_steps,
            lr=lrs[method],
            tol=1e-12,   # effectively disable early exit so all methods run to cap
            method=method,
            momentum=0.9,
        )
        dt = time.perf_counter() - t0
        f_end = evaluate_sign_aware(opt, H, minimize)
        results[method] = {
            "f_final": f_end,
            "wall_s": dt,
            "lr": lrs[method],
        }

    # Pick best f_final as the reference "gold"
    gold_f = max(r["f_final"] for r in results.values())
    for m, r in results.items():
        r["gap_to_best"] = gold_f - r["f_final"]

    return {
        "case": case["name"],
        "minimize": minimize,
        "f_start": f_start,
        "n_params": dag.n_params,
        "n_steps_cap": n_steps,
        "results": results,
    }


def main():
    out = {"cases": []}
    for c in CASES:
        print(f"Running {c['name']} ...")
        r = run_case(c)
        out["cases"].append(r)
        print(f"  n_params={r['n_params']} start={r['f_start']:+.4f}")
        for m, rr in r["results"].items():
            print(f"    {m:10s} f_final={rr['f_final']:+.6f}  "
                  f"gap_to_best={rr['gap_to_best']:.2e}  "
                  f"wall={rr['wall_s']:.2f}s")
        print()

    out_path = Path("results/benchmark_theta_optimizers.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
