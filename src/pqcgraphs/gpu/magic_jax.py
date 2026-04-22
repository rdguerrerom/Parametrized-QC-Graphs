"""Stabilizer Renyi entropy M_2 — nonstabilizerness / magic content.

Precision policy: this file uses complex128.

Definition (Leone, Oliviero, Hamma, PRL 128, 050402 (2022)):

    M_2(|psi>) = -log_2  ( 1/2^n  *  sum_{P in Pauli_n}  <psi|P|psi>^4 )

Equivalently, writing xi_P := <psi|P|psi> (real because P is Hermitian and
|psi> is a pure state, though in complex128 we take the real part explicitly
to avoid tiny imaginary drift):

    Xi_4(|psi>) = (1/2^n) * sum_P xi_P^4
    M_2 = -log_2 Xi_4

Properties used for correctness tests:
  - Stabilizer states: xi_P in {-1, 0, +1}, exactly 2^n of them are nonzero
    (they form the stabilizer group up to sign), so sum_P xi_P^4 = 2^n and
    Xi_4 = 1, hence M_2 = 0.
  - Any non-stabilizer state: Xi_4 < 1 and M_2 > 0.

Implementation: enumerate all 4^n Pauli strings and compute xi_P in one
vectorized JIT pass via Kronecker expansion over the (4, 2, 2) single-qubit
Pauli tensor {I, X, Y, Z}. For n <= 10 (4^10 ≈ 1.05 M Paulis), this is fast
and exact on an RTX 4060. For n > 10 we refuse explicitly rather than
silently falling back to the Monte-Carlo estimator.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

from . import tc_backend  # noqa: F401

import jax
import jax.numpy as jnp

from ..dag.circuit_dag import CircuitDAG
from ..dag.lowering import make_state_fn


# Absolute ceiling for the exact enumeration. 4^12 = 16_777_216 Paulis; with
# chunked vmap the intermediate footprint stays bounded at B * 2^n bytes,
# so the 8 GB GDDR6 of an RTX 4060 fits all the way to n=12. Beyond n=12
# the 4^n enumeration itself becomes prohibitively slow (O(4^n · n)) even
# chunked; we refuse rather than degrade to MC silently.
_MAX_N_EXACT = 12


# Chunk size for the vmapped expectation kernel. Each chunk allocates a
# (chunk_size, 2^n) complex128 intermediate on the GPU, so memory scales
# as 16 * chunk_size * 2^n bytes. The right chunk size balances GPU
# parallelism (bigger is better) against memory headroom.
# - n ≤ 8  : full enumeration fits in one shot (no chunking).
# - n = 10 : chunk 65536 → (65536 × 1024 × 16) ≈ 1 GB intermediate.
# - n = 12 : chunk 65536 → (65536 × 4096 × 16) ≈ 4 GB intermediate.
# At n=12 the 4 GB peak is the RTX 4060's practical ceiling once JAX's
# other caches are counted; bumping the chunk size further risks OOM.
_PAULI_CHUNK_SIZE = 65536


# Single-qubit Paulis in the canonical 0/1/2/3 = I/X/Y/Z ordering.
_SINGLE_PAULIS = jnp.asarray(
    [
        [[1.0, 0.0], [0.0, 1.0]],        # I
        [[0.0, 1.0], [1.0, 0.0]],        # X
        [[0.0, -1j], [1j, 0.0]],         # Y
        [[1.0, 0.0], [0.0, -1.0]],       # Z
    ],
    dtype=jnp.complex128,
)


@lru_cache(maxsize=8)
def _pauli_index_table(n: int) -> jnp.ndarray:
    """All 4^n Pauli index strings as int array of shape (4^n, n).

    Row k holds the base-4 digits of k in order (digit 0 = qubit 0).
    Kept for tests / debugging — the fast expectation kernel below uses
    the symplectic (x_mask, z_mask, y_count) triple instead.
    """
    if n == 0:
        return jnp.zeros((1, 0), dtype=jnp.int32)
    k = np.arange(4 ** n, dtype=np.int64)
    digits = np.empty((4 ** n, n), dtype=np.int32)
    for q in range(n):
        digits[:, q] = (k >> (2 * q)) & 0b11
    return jnp.asarray(digits)


@lru_cache(maxsize=8)
def _pauli_symplectic_table(n: int) -> jnp.ndarray:
    """(4^n, 3) int table: (x_mask, z_mask, y_count) for every Pauli string.

    For each Pauli index k (base-4, digit 0 = qubit 0), each single-qubit
    Pauli has encoding I=0, X=1, Y=2, Z=3. The symplectic decomposition is

        x_bit = ((index + 1) >> 1) & 1   # 1 iff X or Y
        z_bit = (index >> 1) & 1         # 1 iff Y or Z
        y_bit = x_bit & z_bit            # 1 iff Y

    Assembled over all qubits, (x_mask, z_mask) fully determines the
    action P|b⟩ = i^(#Y) · (-1)^⟨b,z_mask⟩ · |b ⊕ x_mask⟩, which is the
    fast-path used by the expectation kernel.
    """
    k = np.arange(4 ** n, dtype=np.int64)
    digits = np.empty((4 ** n, n), dtype=np.int64)
    for q in range(n):
        digits[:, q] = (k >> (2 * q)) & 0b11
    x_bit = ((digits + 1) >> 1) & 1      # (4^n, n)
    z_bit = (digits >> 1) & 1
    y_bit = x_bit & z_bit
    # Pack the n-bit masks into a single int64 per row (n ≤ 62 for safety).
    weights = (1 << np.arange(n)).astype(np.int64)        # (n,)
    x_mask = (x_bit * weights[None, :]).sum(axis=1)       # (4^n,)
    z_mask = (z_bit * weights[None, :]).sum(axis=1)
    y_count = y_bit.sum(axis=1)                           # (4^n,)
    table = np.stack([x_mask, z_mask, y_count], axis=1).astype(np.int64)
    return jnp.asarray(table)


@lru_cache(maxsize=8)
def _basis_indices(n: int) -> jnp.ndarray:
    """Return [0, 1, ..., 2^n - 1] as int64 — indices into the state vector."""
    return jnp.arange(1 << n, dtype=jnp.int64)


@lru_cache(maxsize=8)
def _popcount_parity(n: int):
    """jitted int64 -> int64 popcount parity (0 or 1). For length-n bitmasks."""
    # JAX's `jnp.bit_count` is available in recent versions.
    if hasattr(jnp, "bitwise_count"):
        count_fn = jnp.bitwise_count
    elif hasattr(jnp, "popcount"):
        count_fn = jnp.popcount
    else:
        def _popcount_fallback(x):
            # SWAR 64-bit popcount
            x = x - ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
            return (x * 0x0101010101010101) >> 56
        count_fn = _popcount_fallback
    return count_fn


@lru_cache(maxsize=8)
def _expectation_kernel(n: int):
    """Fast kernel: (psi, sympl_table) → (batch,) real expectations.

    Uses the closed-form P|b⟩ = i^(#Y) · (-1)^⟨b, z_mask⟩ · |b ⊕ x_mask⟩
    derived from the symplectic decomposition of Pauli strings. For each
    Pauli (x_mask, z_mask, y_count):

        <ψ|P|ψ> = i^y · Σ_b conj(ψ[b]) · (-1)^⟨b, z_mask⟩ · ψ[b ⊕ x_mask]

    Since ⟨ψ|P|ψ⟩ is real (P Hermitian, |ψ⟩ pure), the i^y factor is
    either ±1 (y even) or ±i (y odd, in which case the sum must be
    purely imaginary and we take .imag with the correct sign).

    Complexity per Pauli: O(2^n) instead of O(n · 2^n) — an `n ≈ 10x`
    speed-up over the tensordot-per-qubit version. JAX compiles the
    whole vmapped contraction into a single GPU kernel.
    """
    popcnt = _popcount_parity(n)
    basis = _basis_indices(n)    # (D,) int64

    def single_expectation(psi_vec, row):
        # row: (3,) int64 = [x_mask, z_mask, y_count]
        x_mask = row[0]
        z_mask = row[1]
        y_count = row[2]
        # Permuted state: ψ[b ⊕ x_mask] for every b
        permuted_idx = jnp.bitwise_xor(basis, x_mask)
        psi_shift = psi_vec[permuted_idx]
        # Sign (-1)^⟨b, z_mask⟩ via popcount parity of (b AND z_mask).
        # jnp.bitwise_count returns uint8, which silently wraps modulo 256
        # under the (1 - 2·parity) arithmetic below; cast to int64 FIRST.
        parity = jnp.asarray(popcnt(jnp.bitwise_and(basis, z_mask)) & 1, dtype=jnp.int64)
        signs = 1 - 2 * parity                           # int64 ±1
        inner = jnp.sum(jnp.conjugate(psi_vec) * psi_shift * signs.astype(psi_vec.dtype))
        # i^y_count has pattern: y%4 → {1, i, -1, -i}. We only need the
        # real part of i^y · inner; split by y%4.
        y_mod = y_count & 3
        #   y%4 == 0 : real(inner)
        #   y%4 == 1 : real(i · inner) = -imag(inner)
        #   y%4 == 2 : real(-inner) = -real(inner)
        #   y%4 == 3 : real(-i · inner) = imag(inner)
        real_part = jnp.real(inner)
        imag_part = jnp.imag(inner)
        result = jnp.where(
            y_mod == 0, real_part,
            jnp.where(
                y_mod == 1, -imag_part,
                jnp.where(y_mod == 2, -real_part, imag_part),
            ),
        )
        return result

    vmapped = jax.vmap(single_expectation, in_axes=(None, 0))
    return jax.jit(vmapped)


def nonstabilizerness_m2(state: jax.Array, n_qubits: int) -> float:
    """M_2 of an already-computed (D=2**n,) state vector, in bits.

    Uses a chunked vmap over Pauli strings so that the peak intermediate
    allocation is O(chunk_size · 2^n) instead of O(4^n · 2^n). This keeps
    n=10..12 within the 8 GB VRAM envelope of an RTX 4060.
    """
    if n_qubits > _MAX_N_EXACT:
        raise NotImplementedError(
            f"magic_jax.nonstabilizerness_m2 supports n_qubits <= {_MAX_N_EXACT} "
            f"(exact 4^n enumeration). Got n_qubits={n_qubits}. Beyond the "
            "limit an MC estimator is required, which v1 deliberately does "
            "not provide (silent inaccuracy would poison the Nash objective)."
        )
    if state.shape != (1 << n_qubits,):
        raise ValueError(
            f"state shape {state.shape} inconsistent with n_qubits={n_qubits} "
            f"(expected ({1 << n_qubits},))"
        )

    n_paulis = 4 ** n_qubits
    kernel = _expectation_kernel(n_qubits)

    # Chunk the 4^n Pauli enumeration so peak memory stays bounded. At
    # small n the full table fits in one shot; above that we stream
    # chunks through the symplectic kernel.
    chunk = _PAULI_CHUNK_SIZE
    if n_paulis <= chunk:
        sympl = _pauli_symplectic_table(n_qubits)
        xi = kernel(state, sympl)
        xi4_sum = jnp.sum(xi ** 4)
    else:
        xi4_sum = jnp.zeros((), dtype=jnp.float64)
        full_sympl = _pauli_symplectic_table(n_qubits)
        n_chunks = (n_paulis + chunk - 1) // chunk
        for c in range(n_chunks):
            lo = c * chunk
            hi = min(lo + chunk, n_paulis)
            xi_chunk = kernel(state, full_sympl[lo:hi])
            xi4_sum = xi4_sum + jnp.sum(xi_chunk ** 4)

    Xi4 = xi4_sum / (1 << n_qubits)
    # Floor Xi4 to avoid log(0) / log(negative-due-to-fp). Theory: Xi4 in (0, 1].
    Xi4 = jnp.clip(Xi4, min=1e-300)
    m2 = -jnp.log2(Xi4)
    m2 = jnp.maximum(m2, 0.0)
    return float(m2)


def stabilizer_renyi_entropy(dag: CircuitDAG) -> float:
    """M_2(|psi(theta)>) for the state produced by the DAG at dag.thetas.

    Returns bits (log base 2). See module docstring for the definition.
    Exact enumeration is used; chunked so it stays in 8 GB VRAM up to
    `_MAX_N_EXACT` = 12 qubits.
    """
    n = dag.n_qubits
    if n > _MAX_N_EXACT:
        raise NotImplementedError(
            f"stabilizer_renyi_entropy supports n_qubits <= {_MAX_N_EXACT} "
            f"(exact 4^n enumeration). DAG has n_qubits={n}."
        )

    state_fn = make_state_fn(dag)
    theta = jnp.asarray(dag.thetas, dtype=jnp.float64)
    psi = state_fn(theta)
    return nonstabilizerness_m2(psi, n)


__all__ = [
    "stabilizer_renyi_entropy",
    "nonstabilizerness_m2",
]
