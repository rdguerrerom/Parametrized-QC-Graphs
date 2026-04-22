"""Real task Hamiltonians used by the f3 (performance) objective.

Precision policy: complex128 for Pauli operators; expectation returned as
real float.

Two Hamiltonian families in v1:

1) H2 / STO-3G at bond length 0.7414 Angstrom, 4-qubit Jordan-Wigner mapping.
   Hardcoded 15-term Pauli decomposition from the canonical table (Seeley,
   Richard, Love 2012; used in O'Malley et al., PRX 6, 031007 (2016) and
   Kandala et al., Nature 549, 242 (2017)). Nuclear repulsion energy at
   R = 0.7414 A is absorbed into the identity coefficient, so a forward
   simulation of the Hartree-Fock state (qubits 0 and 1 occupied per
   Jordan-Wigner on the two lowest spin-orbitals) yields the HF energy
   ~ -1.1167 Ha. TensorCircuit uses qubit 0 = MSB ordering, so the HF
   state lives at state-vector index 12 (bitstring 1100): <Z_0>=<Z_1>=-1,
   <Z_2>=<Z_3>=+1, and all X/Y-containing terms vanish.

   v1 only exposes the tabulated minimum-energy geometry. Other bond
   lengths require refitting the electronic integrals (one-body h_pq and
   two-body h_pqrs) and re-doing the JW transform; that is explicitly out
   of scope and `h2_sto3g_hamiltonian` raises ValueError for any other R.

2) MaxCut on a weighted graph G=(V,E,w):
      H = sum_{(i,j) in E} w_ij * (I - Z_i Z_j) / 2
   Unweighted graphs use w_ij = 1 for every edge. Ground state energy
   equals the max-cut value; max eigenvalue of H is the optimal cut.

`PauliSumOperator` is a tiny internal dataclass that stores
(pauli_strings, coeffs) and exposes a JITed `expectation(state)` that
computes sum_k c_k <psi|P_k|psi>.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import networkx as nx

from . import tc_backend  # noqa: F401

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# PauliSumOperator
# ---------------------------------------------------------------------------

# Single-qubit Paulis in canonical I/X/Y/Z ordering as complex128.
_SINGLE_PAULIS_NP = np.asarray(
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[0.0, -1j], [1j, 0.0]],
        [[1.0, 0.0], [0.0, -1.0]],
    ],
    dtype=np.complex128,
)
_CHAR_TO_IDX = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def _pauli_string_to_digits(pstr: str) -> Tuple[int, ...]:
    """Convert 'IXYZ' -> (0, 1, 2, 3)."""
    try:
        return tuple(_CHAR_TO_IDX[c] for c in pstr)
    except KeyError as exc:
        raise ValueError(f"Bad Pauli string {pstr!r}; letters must be in IXYZ") from exc


@dataclass
class PauliSumOperator:
    """Sum_k coeffs[k] * pauli_strings[k], with a JIT-compiled expectation.

    All Pauli strings share the same length == n_qubits. Coefficients are
    real floats (the Hamiltonians in v1 are Hermitian with real coefficients;
    complex coefficients would still work but we type as float64 for clarity).
    """
    pauli_strings: List[str]
    coeffs: np.ndarray       # shape (K,), float64
    n_qubits: int

    def __post_init__(self) -> None:
        if len(self.pauli_strings) != self.coeffs.shape[0]:
            raise ValueError(
                f"pauli_strings has {len(self.pauli_strings)} entries but "
                f"coeffs has {self.coeffs.shape[0]}"
            )
        for s in self.pauli_strings:
            if len(s) != self.n_qubits:
                raise ValueError(
                    f"Pauli string {s!r} length {len(s)} != n_qubits={self.n_qubits}"
                )

    def matrix(self) -> np.ndarray:
        """Dense matrix form under the TensorCircuit basis convention.

        TC's `Circuit.state()` indexes basis states with qubit 0 as the MSB
        (qubit n-1 as the LSB). To match, we build each Pauli tensor as
        sigma[s[0]] kron sigma[s[1]] kron ... kron sigma[s[n-1]]. Eigenvalues
        are permutation-invariant so this only matters for callers that
        compare matrix elements or bases to TC state vectors directly.
        """
        D = 1 << self.n_qubits
        M = np.zeros((D, D), dtype=np.complex128)
        for s, c in zip(self.pauli_strings, self.coeffs):
            Ms = np.array([[1.0]], dtype=np.complex128)
            for q in range(self.n_qubits):
                ch = s[q]
                Ms = np.kron(Ms, _SINGLE_PAULIS_NP[_CHAR_TO_IDX[ch]])
            M += c * Ms
        return M

    def max_eigenvalue(self) -> float:
        """Largest eigenvalue of the dense matrix form (small n only)."""
        M = self.matrix()
        w = np.linalg.eigvalsh(M)
        return float(w[-1])

    def expectation(self, state) -> float:
        """<psi|H|psi>, JIT-compiled and returned as a real float.

        `state` must be a JAX array of shape (2**n_qubits,) in complex128.
        """
        fn = _expectation_fn(
            tuple(_pauli_string_to_digits(s) for s in self.pauli_strings),
            tuple(float(c) for c in self.coeffs),
            self.n_qubits,
        )
        return float(fn(state))


@lru_cache(maxsize=64)
def _expectation_fn(pstr_digits_tuple: Tuple[Tuple[int, ...], ...],
                    coeffs_tuple: Tuple[float, ...],
                    n: int):
    """JIT-compiled  psi -> sum_k c_k <psi|P_k|psi>  (real).

    Cached per (Hamiltonian structure) so that repeated calls on the same
    operator hit a warm compilation.
    """
    pstr_arr = jnp.asarray(pstr_digits_tuple, dtype=jnp.int32)     # (K, n)
    coeffs_arr = jnp.asarray(coeffs_tuple, dtype=jnp.float64)      # (K,)
    single_paulis = jnp.asarray(_SINGLE_PAULIS_NP, dtype=jnp.complex128)

    def single_expectation(psi_vec, pstr):
        # Reshape psi to (2,)*n for per-qubit Pauli contraction.
        psi_tensor = jnp.reshape(psi_vec, (2,) * n) if n > 0 else psi_vec
        for q in range(n):
            sigma = single_paulis[pstr[q]]
            psi_tensor = jnp.tensordot(sigma, psi_tensor, axes=([1], [q]))
            psi_tensor = jnp.moveaxis(psi_tensor, 0, q)
        psi_after = jnp.reshape(psi_tensor, (1 << n,))
        return jnp.real(jnp.vdot(psi_vec, psi_after))

    vmapped = jax.vmap(single_expectation, in_axes=(None, 0))

    def expectation_fn(psi):
        xi = vmapped(psi, pstr_arr)          # (K,) real
        return jnp.sum(coeffs_arr * xi)

    return jax.jit(expectation_fn)


# ---------------------------------------------------------------------------
# H2 STO-3G hardcoded Hamiltonian (4-qubit JW)
# ---------------------------------------------------------------------------
# Canonical coefficients (Seeley-Richard-Love 2012; matches OpenFermion's
# published `h2_sto-3g_singlet_0.7414.hdf5`). Nuclear repulsion included in
# the identity term. String indexing: position 0 == qubit 0 (little-endian).

_H2_PAULIS_0_7414 = [
    # (string, coefficient)
    ("IIII",  -0.09706626816762845),
    ("ZIII",   0.17141282644776884),   # Z on qubit 0
    ("IZII",   0.17141282644776884),   # Z on qubit 1
    ("IIZI",  -0.22343153690813466),   # Z on qubit 2
    ("IIIZ",  -0.22343153690813466),   # Z on qubit 3
    ("ZZII",   0.16868898170361207),
    ("ZIZI",   0.12062523483390391),
    ("ZIIZ",   0.16592785033770346),
    ("IZZI",   0.16592785033770346),
    ("IZIZ",   0.12062523483390391),
    ("IIZZ",   0.17441287612261583),
    ("YYYY",  -0.04530261550379927),
    ("XXYY",   0.04530261550379927),
    ("YYXX",   0.04530261550379927),
    ("XXXX",  -0.04530261550379927),
]


def h2_sto3g_hamiltonian(bond_length_angstrom: float = 0.7414) -> PauliSumOperator:
    """4-qubit JW H2 Hamiltonian at R = 0.7414 A (tabulated minimum).

    v1 only supports R = 0.7414 A. Any other value -> ValueError (we refuse
    to interpolate between tabulated coefficients because the electronic
    integrals are non-linear in R).
    """
    # Exact float match to avoid ambiguity. Users who pass the literal
    # 0.7414 get the tabulated Hamiltonian.
    if abs(bond_length_angstrom - 0.7414) > 1e-12:
        raise ValueError(
            f"h2_sto3g_hamiltonian only supports bond_length_angstrom=0.7414 "
            f"in v1 (tabulated minimum). Got {bond_length_angstrom}. "
            "Computing H2 at other bond lengths requires refitting the "
            "electronic integrals, which is out of scope here."
        )
    pstrings = [p for p, _ in _H2_PAULIS_0_7414]
    coeffs = np.asarray([c for _, c in _H2_PAULIS_0_7414], dtype=np.float64)
    return PauliSumOperator(pauli_strings=pstrings, coeffs=coeffs, n_qubits=4)


# ---------------------------------------------------------------------------
# LiH STO-3G (6-qubit reduced)
# ---------------------------------------------------------------------------

# Cache path for the precomputed LiH Pauli sum. Generated once by
# `scripts/generate_lih_hamiltonian.py` using openfermion + openfermionpyscf;
# at runtime we only need the cached JSON so the core codebase has no
# openfermion/pyscf import.
_LIH_CACHE_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "hamiltonians" / "lih_sto3g_6q.json"
)


@lru_cache(maxsize=1)
def _load_lih_cache() -> dict:
    import json as _json
    with open(_LIH_CACHE_PATH) as f:
        return _json.load(f)


def lih_sto3g_hamiltonian() -> PauliSumOperator:
    """6-qubit LiH / STO-3G Hamiltonian at bond R = 1.545 A.

    Reduction: Jordan-Wigner with the Li 1s² core frozen and the Li 2p_x,
    2p_y virtual orbitals frozen (non-bonding for linear LiH along z).
    Active space is Li(2s), Li(2p_z), H(1s), containing 2 electrons in 6
    spin-orbitals (= 6 qubits). Nuclear repulsion + frozen-core energy is
    absorbed into the identity term; the ground-state eigenvalue of this
    operator is −7.8778 Ha (within 5 mHa of the full-molecule FCI, which
    is the correlation carried by the frozen 2p_{x,y} virtuals).

    Source: generated via `scripts/generate_lih_hamiltonian.py` (requires
    openfermion + pyscf); coefficients cached in
    `data/hamiltonians/lih_sto3g_6q.json` so runtime code has no chemistry
    dependency.
    """
    data = _load_lih_cache()
    pstrings = [t[0] for t in data["terms"]]
    coeffs = np.asarray([t[1] for t in data["terms"]], dtype=np.float64)
    return PauliSumOperator(pauli_strings=pstrings, coeffs=coeffs,
                            n_qubits=int(data["n_qubits"]))


def lih_sto3g_reference_energies() -> dict:
    """Return {'E_HF', 'E_FCI_full', 'E_ground_active'} for LiH at R=1.545 A."""
    data = _load_lih_cache()
    return {
        "E_HF": float(data["e_hf"]),
        "E_FCI_full": float(data["e_fci"]),
        "E_ground_active": float(data["e_active_ground"]),
        "bond_length_angstrom": float(data["bond_length_angstrom"]),
    }


# ---------------------------------------------------------------------------
# MaxCut Hamiltonian
# ---------------------------------------------------------------------------

def maxcut_hamiltonian(graph: nx.Graph) -> PauliSumOperator:
    """H = sum_{(i,j) in E} w_ij * (I - Z_i Z_j) / 2.

    Undirected `networkx.Graph`; optional per-edge "weight" attribute used
    if present, defaulting to 1.0 otherwise. Qubit labels must be the
    integers 0..n-1 (contiguous).
    """
    nodes = list(graph.nodes())
    if nodes:
        # Validate that qubit labels are 0..n-1
        expected = set(range(len(nodes)))
        if set(nodes) != expected:
            raise ValueError(
                f"maxcut_hamiltonian requires contiguous integer qubit labels "
                f"0..{len(nodes) - 1}; got {sorted(nodes)}"
            )
    n_qubits = len(nodes)

    pstrings: List[str] = []
    coeffs: List[float] = []

    identity = "I" * n_qubits
    total_I_coeff = 0.0

    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        # + w/2 * I
        total_I_coeff += 0.5 * w
        # - w/2 * Z_u Z_v
        zstr = ["I"] * n_qubits
        zstr[u] = "Z"
        zstr[v] = "Z"
        pstrings.append("".join(zstr))
        coeffs.append(-0.5 * w)

    if total_I_coeff != 0.0 or not pstrings:
        pstrings.insert(0, identity)
        coeffs.insert(0, total_I_coeff)

    return PauliSumOperator(
        pauli_strings=pstrings,
        coeffs=np.asarray(coeffs, dtype=np.float64),
        n_qubits=n_qubits,
    )


# ---------------------------------------------------------------------------
# Transverse-field Ising model (1D, open boundary conditions)
# ---------------------------------------------------------------------------

def tfim_hamiltonian(
    n_qubits: int,
    *,
    J: float = 1.0,
    h: float = 1.0,
    pbc: bool = False,
) -> PauliSumOperator:
    """1D transverse-field Ising Hamiltonian.

        H = -J * sum_{<i,j>} Z_i Z_j - h * sum_i X_i

    With J = h = 1.0 the model sits at its quantum critical point g ≡ h/J = 1;
    this is the canonical scaling benchmark because the ground state has
    entanglement ~ log(n) and most ansatz families hit their BP regime in
    the vicinity. `pbc=True` adds the Z_{n-1} Z_0 wraparound term.

    Ground-state energies used for benchmarking at g=1 (OBC), computed via
    exact diagonalisation of this implementation:
      n = 4  → E0 = -4.7587704831
      n = 6  → E0 = -7.2962298106
      n = 8  → E0 = -9.8379514475
      n = 10 → E0 = -12.3814899997
      n = 12 → E0 = -14.9259711099
    """
    if n_qubits < 2:
        raise ValueError(f"TFIM needs n_qubits >= 2, got {n_qubits}")

    pstrings: List[str] = []
    coeffs: List[float] = []

    # -J * Z_i Z_{i+1} (OBC), plus Z_{n-1} Z_0 if pbc
    def pair(i: int, j: int) -> str:
        s = ["I"] * n_qubits
        s[i] = "Z"
        s[j] = "Z"
        return "".join(s)

    for i in range(n_qubits - 1):
        pstrings.append(pair(i, i + 1))
        coeffs.append(-float(J))
    if pbc:
        pstrings.append(pair(n_qubits - 1, 0))
        coeffs.append(-float(J))

    # -h * X_i
    for i in range(n_qubits):
        s = ["I"] * n_qubits
        s[i] = "X"
        pstrings.append("".join(s))
        coeffs.append(-float(h))

    return PauliSumOperator(
        pauli_strings=pstrings,
        coeffs=np.asarray(coeffs, dtype=np.float64),
        n_qubits=n_qubits,
    )


__all__ = [
    "PauliSumOperator",
    "h2_sto3g_hamiltonian",
    "maxcut_hamiltonian",
    "tfim_hamiltonian",
]
