"""GPU / JAX-accelerated numerical primitives backing the Tier-1 objectives.

All modules in this package assume the JAX backend with complex128 dtype,
as configured by `tc_backend`. Import order matters once: `tc_backend` must
be imported before any direct use of `tensorcircuit` or `jax` elsewhere.
"""
from . import tc_backend  # noqa: F401  (configures tc + jax exactly once)
from .tc_backend import check_gpu
from .qfim_effdim import qfim, qfim_spectrum, effective_dimension
from .magic_jax import stabilizer_renyi_entropy, nonstabilizerness_m2
from .hamiltonians import (
    PauliSumOperator,
    h2_sto3g_hamiltonian,
    maxcut_hamiltonian,
)
from .dla_jax import (
    pauli_string_to_symplectic,
    compute_dla_dimension,
    graph_state_dla_generators,
)

__all__ = [
    "check_gpu",
    "qfim",
    "qfim_spectrum",
    "effective_dimension",
    "stabilizer_renyi_entropy",
    "nonstabilizerness_m2",
    "PauliSumOperator",
    "h2_sto3g_hamiltonian",
    "maxcut_hamiltonian",
    "pauli_string_to_symplectic",
    "compute_dla_dimension",
    "graph_state_dla_generators",
]
