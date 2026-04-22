"""D1: Nash scaling on the TFIM ground state at the critical point g=1.

For n ∈ {4, 6, 8, 10, 12} and a fixed hardware topology (1D nearest-neighbour
line), run the four-player Nash game with an inner θ-GD loop. Record:

  - Final ground-state energy E vs the exact reference E0(n)
  - δ_Nash at convergence
  - Per-player Nash-gap decomposition
  - Circuit complexity (n_ops, depth, n_params)
  - Wall-clock per iteration

This is the core scaling experiment demanded by PRL reviewers. By running
at the critical point we stress-test the anti-BP player: standard HEA-style
ansatze are known to hit barren plateaus quickly on critical TFIM
(Holmes et al. 2022; Cerezo et al. 2025).

Exact ground-state energies at g=1 (OBC), computed by dense
diagonalisation and used as references in the JSON:
  n=4  : -4.7587704831436
  n=6  : -7.2962298105588
  n=8  : -9.8379514474594
  n=10 : -12.3814899996548
  n=12 : -14.9259711099087
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx

from ..dag import Topology
from ..dag.initial_states import qaoa_warm_layer
from ..game import PQCNashGame, make_default_players, potential
from ..objectives import tfim_hamiltonian


# Reference exact ground-state energies for TFIM at g=J=h=1, OBC,
# computed via dense diagonalisation. Checked independently of the Nash
# pipeline. Used only to report relative error in the JSON output.
TFIM_EXACT_E0 = {
    4:  -4.7587704831436,
    6:  -7.2962298105588,
    8:  -9.8379514474594,
    10: -12.3814899996548,
    12: -14.9259711099087,
}


def _line_topology(n: int) -> Topology:
    """1D open-boundary line topology (qubit i is connected to qubit i+1)."""
    pairs = frozenset((i, i + 1) for i in range(n - 1))
    return Topology("line", n, pairs)


def run(
    n_values=(4, 6, 8),
    n_iters: int = 15,
    population_size: int = 3,
    seed: int = 0,
    theta_gd_steps: int = 40,
    out_path: Path = Path("results/exp_d1_tfim_scaling.json"),
) -> dict:
    # NOTE: n=10 is technically achievable on the RTX 4060 after the chunked
    # symplectic magic upgrade, but the JAX JIT cache grows to ~10 GB during
    # a 15-iteration Nash run (each new structure triggers fresh QFIM +
    # magic + performance compilations), which exceeds the 8 GB VRAM and
    # spills to host swap. Limiting to n ≤ 8 lets the full sweep finish in
    # ~15 min cleanly. Extending to n=10 requires either (a) tighter cache
    # eviction in scorer lru_caches or (b) JAX XLA cache size limits.
    # NOTE: n=12 is technically supported by the chunked symplectic magic
    # kernel (it completes at ~10 s per magic call after warm-up) but with
    # 20 iterations × population 3 × ~20 candidates per step, a full D1 run
    # at n=12 takes > 1 h per run just in magic. Keeping n_values ≤ 10 lets
    # the whole sweep finish in ~15 min on the RTX 4060 and is sufficient
    # for the PRL scaling claim (Ragone 2024 gradient-variance law is
    # already visible at n=4..10). To include n=12, pass it explicitly.
    rows = []
    total_start = time.perf_counter()

    # JIT caches for the scorers accumulate compiled functions per structural
    # identity; over a sweep of n-values they balloon (one scorer compile at
    # n=8 can be ~50 MB of XLA bytecode, and ~150 structures per Nash run
    # means ~7.5 GB of cache by the time we reach n=10). Clear between n
    # so each run starts with a cold but isolated JIT.
    def _clear_caches():
        import jax
        from ..gpu import qfim_effdim, theta_optimizer
        from ..objectives import performance as _perf
        jax.clear_caches()
        try:
            qfim_effdim._cached_qfim_fn.cache_clear()
        except AttributeError:
            pass
        try:
            theta_optimizer._cached_energy_grad_fn.cache_clear()
        except AttributeError:
            pass
        try:
            _perf._cached_energy_fn.cache_clear()
        except AttributeError:
            pass
        import gc
        gc.collect()

    for n in n_values:
        _clear_caches()
        # Magic (f2) is exact for n ≤ 12 after the chunked vmap upgrade;
        # for n > 12 we drop f2 entirely rather than degrade precision.
        supports_magic = (n <= 12)

        topo = _line_topology(n)
        H = tfim_hamiltonian(n, J=1.0, h=1.0, pbc=False)

        players = make_default_players(
            hamiltonian=H,
            topology=topo,
            w_anti_bp=0.3,
            w_anti_sim=0.2 if supports_magic else 0.0,
            w_performance=1.0,
            w_hardware=0.1,
            minimize_performance=True,  # VQE ground state: min <H>
        )

        t0 = time.perf_counter()
        try:
            game = PQCNashGame(
                n_qubits=n, players=players, topology=topo,
                # QAOA p=1 warm start: H^n + rzz per edge + rx per qubit.
                # Gives a parameterized ansatz with 2n-1 free θ from iter 0
                # so θ-GD can push energy immediately and Nash's structural
                # moves refine from a non-trivial baseline rather than |+⟩ⁿ
                # where rx/rzz moves cannot lower ⟨H_TFIM⟩.
                initial_dag_factory=(lambda tt=topo: qaoa_warm_layer(tt, gamma_init=0.1, beta_init=0.1)),
                population_size=population_size,
                temperature=1.0, cooling_rate=0.9,
                seed=seed,
                theta_gd_hamiltonian=H,
                theta_gd_minimize=True,
                theta_gd_steps=theta_gd_steps,
                theta_gd_lr=0.05,
                # At n ≥ 8 the structural moves dominate wall time because
                # each unique structure triggers a fresh JIT trace across
                # QFIM + magic + performance (~2-3 s each at n=8). Bias the
                # candidate budget toward θ-perturbs so most candidates
                # reuse the warm JIT cache.
                candidate_budget=dict(
                    n_theta=16, n_add=2, n_remove=1, n_retype=1, n_rewire=1
                ),
            )
            best = game.run(n_iters, verbose=False)
            err = None
        except Exception as exc:  # noqa: BLE001
            best = None
            err = f"{type(exc).__name__}: {exc}"

        dt = time.perf_counter() - t0

        if best is None:
            rows.append({
                "n": n,
                "supports_magic": supports_magic,
                "error": err,
                "wall_time_s": dt,
            })
            continue

        # Final metrics
        f3 = next(p.scorer for p in players if p.name == "performance")
        E_found = -float(f3(best))  # f3 = -<H>, so E = -f3
        E0 = TFIM_EXACT_E0.get(n)
        rel_err = abs(E_found - E0) / abs(E0) if E0 is not None else None

        last = game.history[-1]
        rows.append({
            "n": n,
            "supports_magic": supports_magic,
            "n_iters": n_iters,
            "E_nash": E_found,
            "E_exact": E0,
            "rel_error_vs_exact": rel_err,
            "final_potential": float(potential(players, best)),
            "delta_nash_final": last.nash_gap,
            "per_player": last.per_player,
            "n_ops": best.n_ops,
            "depth": best.depth(),
            "n_params": best.n_params,
            "gate_counts": best.gate_counts(),
            "wall_time_s": dt,
            "mean_iter_s": dt / max(n_iters, 1),
        })

    result = {
        "name": "D1_tfim_scaling",
        "task": "TFIM_g1_OBC",
        "n_values": list(n_values),
        "topology_family": "line (nearest-neighbour)",
        "seed": seed,
        "rows": rows,
        "total_wall_s": time.perf_counter() - total_start,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(f"D1 done ({r['total_wall_s']:.1f}s):")
    for row in r["rows"]:
        if "error" in row:
            print(f"  n={row['n']:2d}: ERROR {row['error']}")
            continue
        print(
            f"  n={row['n']:2d}  "
            f"E={row['E_nash']:+.4f}  E0={row['E_exact']:+.4f}  "
            f"rel_err={row['rel_error_vs_exact']:.3e}  "
            f"δ_Nash={row['delta_nash_final']:.3e}  "
            f"ops={row['n_ops']} depth={row['depth']} params={row['n_params']}  "
            f"({row['wall_time_s']:.1f}s, {row['mean_iter_s']:.2f}s/iter)"
        )
