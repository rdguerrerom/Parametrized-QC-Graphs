"""Dynamical Lie algebra dimension via F_2 symplectic closure.

This is a **CPU-side numpy** computation — despite the "_jax" filename
(kept for naming symmetry with the other gpu/ modules). The F_2 symplectic
closure is an integer-arithmetic fixed-point algorithm that does not
benefit from a GPU.

Role in the project:
  - Oracle for experiment E1 (abelian bound): for a graph state with
    generators {Z_i Z_j : (i,j) in E}, the DLA is abelian and has
    dimension |E|. This must match `qfim_effdim.effective_dimension` at
    generic theta.
  - Fallback / diagnostic for f1 when the QFIM rank is numerically
    ambiguous (e.g. near a symmetry point of the parameter space).

Algorithm: port of `~/Research/Science/Projects/STABILIZER_GAMES/DLA/
Stabilizer-Rank-from-DLA.md` sec. 2.2. Represent each Pauli operator
(modulo phase) as a symplectic vector v in F_2^{2n}:  v = (x | z),
x_i = 1 iff the operator has X or Y on qubit i,  z_i = 1 iff Z or Y.
The commutator [iP, iQ] is nonzero iff <v_P, v_Q>_symp = 1, and in
that case the resulting Pauli is v_P XOR v_Q.

Closure loop: BFS over the current basis; for every pair compute the
symplectic commutator and add new Pauli vectors until a fixed point is
reached. Worst-case |basis| <= 4^n - 1, so we cap at n <= 12 (4^12 ~ 16M
symplectic vectors would already be uncomfortable on 8 GB).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import networkx as nx


_MAX_N_DENSE = 12


# ---------------------------------------------------------------------------
# Symplectic representation
# ---------------------------------------------------------------------------

def pauli_string_to_symplectic(pstr: str) -> np.ndarray:
    """Convert a Pauli string 'IXYZ' to its F_2 symplectic vector (x|z).

    Returns a uint8 array of shape (2n,). Phase is discarded (irrelevant for
    the commutator structure).
    """
    n = len(pstr)
    v = np.zeros(2 * n, dtype=np.uint8)
    for i, c in enumerate(pstr):
        if c == "I":
            continue
        if c == "X":
            v[i] = 1
        elif c == "Y":
            v[i] = 1
            v[n + i] = 1
        elif c == "Z":
            v[n + i] = 1
        else:
            raise ValueError(f"Bad Pauli char {c!r} (expected I/X/Y/Z)")
    return v


def _symplectic_inner_product(v1: np.ndarray, v2: np.ndarray, n: int) -> int:
    """<v1, v2>_symp mod 2. Zero iff the two Paulis commute."""
    return int((np.dot(v1[:n], v2[n:]) + np.dot(v1[n:], v2[:n])) % 2)


def _commutator_symplectic(v1: np.ndarray, v2: np.ndarray, n: int):
    """Return the symplectic vector of the commutator, or None if they commute.

    [iP, iQ] = -[P, Q]. If P, Q anti-commute then PQ is Pauli (up to phase)
    with symplectic vector v_P XOR v_Q.
    """
    if _symplectic_inner_product(v1, v2, n) == 0:
        return None
    return ((v1 + v2) % 2).astype(np.uint8)


# ---------------------------------------------------------------------------
# DLA dimension
# ---------------------------------------------------------------------------

def compute_dla_dimension(generators: List[str]) -> int:
    """dim(DLA) = cardinality of the symplectic closure of the generators.

    Input: list of Pauli strings, all of the same length n, e.g.
      ["ZZII", "IZZI", "IIZZ"] for a 4-qubit linear rzz ansatz.
    Output: integer dim(g) >= 1 (always contains at least the generators,
    assuming they are nontrivial / distinct up to phase).

    For n <= 12 uses dense closure. Above 12 raises NotImplementedError.
    """
    if not generators:
        return 0
    n = len(generators[0])
    if not all(len(g) == n for g in generators):
        raise ValueError("All generators must have the same Pauli-string length")
    if n > _MAX_N_DENSE:
        raise NotImplementedError(
            f"compute_dla_dimension: dense F_2 closure supports n <= {_MAX_N_DENSE}. "
            f"Got n={n}. For larger n, switch to a sparse/symmetry-aware solver "
            "(not yet implemented)."
        )

    basis: List[np.ndarray] = []
    seen: set = set()

    def add(v: np.ndarray) -> bool:
        # Skip the identity (trivial element in the Lie algebra for nontrivial
        # generators). An all-zero vector represents +-I, which contributes
        # nothing to the Lie closure.
        if not np.any(v):
            return False
        key = bytes(v.tobytes())
        if key in seen:
            return False
        seen.add(key)
        basis.append(v)
        return True

    for g in generators:
        add(pauli_string_to_symplectic(g))

    # BFS: iterate until the next pair scan produces no new element. We pair
    # the newest element with every earlier one, which is sufficient for
    # commutator closure (commutator is symmetric up to sign).
    i = 0
    while i < len(basis):
        # Pair basis[i] with basis[0..len-1] (both directions trivially symmetric).
        cur = basis[i]
        # Snapshot length since we mutate during iteration.
        L = len(basis)
        for j in range(L):
            if i == j:
                continue
            c = _commutator_symplectic(cur, basis[j], n)
            if c is not None:
                add(c)
        i += 1

    return len(basis)


def graph_state_dla_generators(graph: nx.Graph, n_qubits: int) -> List[str]:
    """Generators for the parameterised graph-state ansatz: one Z_iZ_j per edge.

    Matches `from_graph_state_parameterized`: the parametric layer is
    exp(-i theta_e Z_i Z_j / 2) on every edge e = (i, j), so the DLA
    generators are {Z_i Z_j : (i, j) in E}. Abelian (all pairs of ZZ
    commute), so dim(DLA) = |E|.
    """
    gens: List[str] = []
    for u, v in graph.edges():
        if not (0 <= u < n_qubits and 0 <= v < n_qubits):
            raise ValueError(
                f"Edge ({u},{v}) has a qubit out of [0,{n_qubits})"
            )
        s = ["I"] * n_qubits
        s[u] = "Z"
        s[v] = "Z"
        gens.append("".join(s))
    return gens


__all__ = [
    "pauli_string_to_symplectic",
    "compute_dla_dimension",
    "graph_state_dla_generators",
]
