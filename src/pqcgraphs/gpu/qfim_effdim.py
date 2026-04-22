"""Pure-state QFIM via automatic differentiation of the state-vector Jacobian.

Precision policy: this file uses complex128 (set globally in `tc_backend`).

For a parameterised pure state |psi(theta)>, the Quantum Fisher Information
Matrix (real form) is

    F_ij(theta) = 4 * Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]

We compute it with a *single* application of `jax.jacrev` (or `jax.jacfwd`
when n_params <= 2**n_qubits, which is the forward-mode efficient regime)
to the structure-static lowering `make_state_fn(dag)`. The resulting
Jacobian J has shape (2**n_qubits, n_params) in complex128; from there
F is a single einsum. No parameter-shift rule; no O(n^2) evaluations.

Why jacrev by default?
----------------------
TensorCircuit's circuit evaluation is built as a (long) chain of vjp-friendly
ops. For vector-valued output of dimension 2^n_qubits vs n_params (typically
n_params << 2^n_qubits for the circuits we care about), jacfwd (forward mode)
is algorithmically cheaper: one tangent pass per input param. We switch based
on the ratio `n_params vs 2^n_qubits`, with jacrev as the safe default for
very wide states.

References
----------
- Fubini-Study metric and pure-state QFIM: Liu, Yuan, Lu, Wang,
  "Quantum Fisher information matrix and multiparameter estimation",
  J. Phys. A 53, 023001 (2020); also papers/QuantumFisherInformation.pdf s.2.4.
- TensorCircuit QFIM tutorial:
  https://tensorcircuit.readthedocs.io/en/latest/whitepaper/6-6-advanced-automatic-differentiation.html#Quantum-Fisher-Information
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Tuple

import numpy as np

from . import tc_backend  # noqa: F401  (ensures backend is configured first)

import jax
import jax.numpy as jnp

from ..dag.circuit_dag import CircuitDAG
from ..dag.lowering import make_state_fn


# ---------------------------------------------------------------------------
# Structural JIT cache
# ---------------------------------------------------------------------------

def _dag_structure_key(dag: CircuitDAG) -> Tuple:
    """Hash-key identifying the circuit *structure* (NOT the theta values).

    Two DAGs with the same structural key produce the same `state_fn` up to
    parameter values, so their compiled QFIM function can be reused.
    """
    sched = tuple(
        (op.gate_name, op.qubits, -1 if op.param_idx is None else op.param_idx)
        for op in dag.ops()
    )
    return (dag.n_qubits, dag.n_params, sched)


# Keyed on structure; caches the pure-JAX QFIM function for that structure.
# LRU is bounded to keep memory tame on long Nash runs.
@lru_cache(maxsize=512)
def _cached_qfim_fn(structure_key) -> Callable:
    """Build the compiled theta -> F(theta) function for a given structure.

    We re-create the state_fn from the structure key here (NOT passed in)
    so the closure sees only Python ints/strings, enabling safe lru_cache.
    """
    n_qubits, n_params, sched = structure_key

    # Rebuild a lightweight state_fn mirroring `make_state_fn` but keyed
    # only off the schedule tuple (so there's no DAG-object identity issue).
    import tensorcircuit as tc
    from ..dag.lowering import _apply_op  # internal gate dispatcher

    def state_fn(theta_vec):
        c = tc.Circuit(n_qubits)
        for gate_name, qubits, pidx in sched:
            _apply_op(c, gate_name, qubits, theta_vec, pidx)
        return c.state()

    # Choose fwd vs rev based on shape. jacfwd is more efficient when
    # n_params <= state_dim = 2**n_qubits; otherwise jacrev wins.
    state_dim = 1 << n_qubits
    if n_params == 0:
        # Degenerate: QFIM is a 0x0 matrix. Return a function that yields that.
        def qfim_fn_zero(theta_vec):
            return jnp.zeros((0, 0), dtype=jnp.float64)
        return jax.jit(qfim_fn_zero)

    # `jacrev` would be cheaper when n_params > state_dim, but it requires
    # real-valued outputs (or holomorphic=True). Our state function returns
    # a complex vector, and although it IS holomorphic, declaring so forces
    # every downstream differentiation to be holomorphic which breaks the
    # existing real-θ paths. Using `jacfwd` everywhere costs O(P·D) which
    # is fine for n_qubits ≤ ~8 and avoids the complex-output bug. Revisit
    # with jax.vjp if we ever need n_qubits > 8 with many parameters.
    jac_fn = jax.jacfwd(state_fn)

    def qfim_fn(theta_vec):
        psi = state_fn(theta_vec)                 # (D,) complex
        jac = jac_fn(theta_vec)                   # (D, P) complex
        # <d_i psi | d_j psi>  = sum_k conj(J[k,i]) * J[k,j]   => Jᴴ J
        gram = jnp.conjugate(jac).T @ jac         # (P, P) complex
        # <d_i psi | psi>  = sum_k conj(J[k,i]) * psi[k]
        o = jnp.conjugate(jac).T @ psi            # (P,)   complex
        # Classical correction: o_i * conj(o_j)
        corr = jnp.outer(o, jnp.conjugate(o))     # (P, P) complex
        F = 4.0 * jnp.real(gram - corr)           # (P, P) real
        # Symmetrise to kill asymmetric numerical drift (F is Hermitian real
        # => symmetric).
        F = 0.5 * (F + F.T)
        return F

    return jax.jit(qfim_fn)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def qfim(dag: CircuitDAG) -> jax.Array:
    """Compute the QFIM of the state parameterized by `dag` at dag.thetas.

    Returns a real (n_params, n_params) array on the JAX device.
    """
    key = _dag_structure_key(dag)
    fn = _cached_qfim_fn(key)
    theta = jnp.asarray(dag.thetas, dtype=jnp.float64)
    return fn(theta)


def qfim_spectrum(dag: CircuitDAG) -> jnp.ndarray:
    """Sorted-descending eigenvalues of the QFIM.

    Uses Hermitian eigensolver (F is symmetric positive semi-definite up to
    numerical noise; we clip the tiny negatives produced by float roundoff).
    """
    F = qfim(dag)
    # F is real symmetric; eigvalsh gives ascending real eigenvalues.
    w = jnp.linalg.eigvalsh(F)
    w = jnp.clip(w, min=0.0)   # numerical clean-up
    return jnp.flip(w)


def effective_dimension(dag: CircuitDAG, tol_rel: float = 1e-10) -> int:
    """Number of QFIM eigenvalues above `tol_rel * max_eigenvalue`.

    This is the rank of the QFIM at the current theta, which equals the
    dimension of the tangent space of the parameterised state manifold at
    that point. Used by f1 (anti-BP): higher rank -> more informative
    gradients / fewer BP-like flat directions.
    """
    if dag.n_params == 0:
        return 0
    w = np.asarray(qfim_spectrum(dag))
    if w.size == 0:
        return 0
    w_max = float(w[0])
    if w_max <= 0.0:
        return 0
    thresh = tol_rel * w_max
    return int(np.sum(w > thresh))


__all__ = ["qfim", "qfim_spectrum", "effective_dimension"]
