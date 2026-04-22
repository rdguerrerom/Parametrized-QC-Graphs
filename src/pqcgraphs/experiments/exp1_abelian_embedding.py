"""E1: Abelian graph-state embedding → QFIM rank = DLA dim = |E|.

Validates the embedding ι: G → D_G (docs/circuit-dag-generalization.md §3)
against the theorem in Stabilizer-Rank-from-DLA.md §1.2 (Theorem 1, abelian
bound). For the parameterized graph-state circuit
  U_G(θ) = ∏_{(i,j) ∈ E} exp(-i θ_ij Z_i Z_j / 2) · H^⊗n
the DLA is abelian with dim = |E|, and at a generic θ the QFIM rank should
also equal |E|.

We iterate over four graph families (path, cycle, complete, Erdős–Rényi) at
multiple n and confirm rank(QFIM) == |E| == compute_dla_dimension(generators).

Writes `results/exp1_abelian.json`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np

from ..dag import from_graph_state_parameterized
from ..gpu.dla_jax import compute_dla_dimension, graph_state_dla_generators
from ..gpu.qfim_effdim import effective_dimension, qfim


def run(
    n_values=(3, 4, 5, 6),
    er_prob: float = 0.5,
    n_er_samples: int = 3,
    seed: int = 0,
    out_path: Path = Path("results/exp1_abelian.json"),
) -> dict:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    def case(graph: nx.Graph, n: int, family: str, sample: int = 0) -> None:
        dag = from_graph_state_parameterized(graph, theta_init=0.0, n_qubits=n)
        # generic random θ in (0.1, 1.5) to stay away from Clifford angles
        dag.thetas = rng.uniform(0.1, 1.5, size=dag.n_params)

        t0 = time.perf_counter()
        qfim_rank = effective_dimension(dag)
        t_qfim = time.perf_counter() - t0

        t0 = time.perf_counter()
        gens = graph_state_dla_generators(graph, n)
        dla_dim = compute_dla_dimension(gens)
        t_dla = time.perf_counter() - t0

        E = graph.number_of_edges()
        rows.append({
            "family": family,
            "n": n,
            "sample": sample,
            "E": E,
            "qfim_rank": int(qfim_rank),
            "dla_dim": int(dla_dim),
            "abelian_bound_holds": bool(qfim_rank <= E and dla_dim <= E),
            "bound_tight": bool(qfim_rank == E and dla_dim == E),
            "t_qfim_s": t_qfim,
            "t_dla_s": t_dla,
        })

    for n in n_values:
        case(nx.path_graph(n), n, "path")
        case(nx.cycle_graph(n), n, "cycle")
        case(nx.complete_graph(n), n, "complete")
        for s in range(n_er_samples):
            G = nx.erdos_renyi_graph(n, er_prob, seed=seed + 100 * n + s)
            if G.number_of_edges() == 0:
                G.add_edge(0, min(n - 1, 1))
            case(G, n, "erdos_renyi_p0.5", sample=s)

    all_hold = all(r["abelian_bound_holds"] for r in rows)
    all_tight = all(r["bound_tight"] for r in rows)

    result = {
        "name": "E1_abelian_embedding",
        "rows": rows,
        "all_bound_holds": all_hold,
        "all_bound_tight": all_tight,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    r = run()
    print(f"E1: {len(r['rows'])} cases; bound holds={r['all_bound_holds']}; "
          f"all tight={r['all_bound_tight']}")
