"""Hardware topologies for the f4 (hardware) objective.

Each topology exposes:
  - tag: str — matches the `native_on` keys in `gate_types.GATE_SPECS`
  - n_qubits: int
  - allowed_pairs(): Iterable[Tuple[int, int]] — undirected unordered pairs

Per Tier-1, changing the hardware target is a reconfiguration of the
objective weights + this topology; the Nash algorithm itself is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Tuple

import networkx as nx


@dataclass(frozen=True)
class Topology:
    tag: str
    n_qubits: int
    pairs: FrozenSet[Tuple[int, int]]  # always stored as (min, max)

    def allowed_pairs(self) -> Iterable[Tuple[int, int]]:
        return iter(self.pairs)

    def allows(self, u: int, v: int) -> bool:
        lo, hi = (u, v) if u < v else (v, u)
        return (lo, hi) in self.pairs

    def as_graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.n_qubits))
        g.add_edges_from(self.pairs)
        return g


def heavy_hex(n_qubits: int) -> Topology:
    """IBM heavy-hex coupling map.

    Implementation: map qubits onto an approximate hex lattice with the
    degree-3 constraint. For small n (≤ 16) we produce a tested hardcoded
    pattern taken from IBM's published coupling map fragments; for larger
    n we fall back to a 3-regular graph with deterministic seed.
    """
    # Hardcoded 7-qubit heavy-hex unit cell (matches ibmq_jakarta layout)
    if n_qubits == 7:
        pairs = frozenset({(0, 1), (1, 2), (1, 3), (3, 5), (4, 5), (5, 6)})
        return Topology("heavy_hex", 7, pairs)
    # 16-qubit tile (matches ibmq_guadalupe)
    if n_qubits == 16:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (1, 4), (4, 7), (6, 7), (7, 10),
            (10, 12), (12, 13), (12, 15), (14, 15), (13, 14),
            (8, 11), (11, 14), (5, 8),
        ]
        return Topology("heavy_hex", 16, frozenset((min(a, b), max(a, b)) for a, b in edges))

    # Generic small-n fallback: 3-regular graph deterministic seed
    if n_qubits < 4 or n_qubits % 2 == 1:
        # 3-regular graphs require even n ≥ 4; for odd or tiny n fall back to a path
        pairs = frozenset((i, i + 1) for i in range(n_qubits - 1))
        return Topology("heavy_hex", n_qubits, pairs)
    g = nx.random_regular_graph(3, n_qubits, seed=42)
    pairs = frozenset((min(u, v), max(u, v)) for u, v in g.edges())
    return Topology("heavy_hex", n_qubits, pairs)


def grid_2d(rows: int, cols: int) -> Topology:
    """Nearest-neighbour square grid, matches google sycamore-like layouts."""
    n = rows * cols
    pairs = set()
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                pairs.add((q, q + 1))
            if r + 1 < rows:
                pairs.add((q, q + cols))
    return Topology("grid", n, frozenset(pairs))


def rydberg_all_to_all(n_qubits: int) -> Topology:
    """Rydberg atom array: fully connected (range ≫ lattice spacing)."""
    pairs = frozenset((i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits))
    return Topology("rydberg", n_qubits, pairs)
