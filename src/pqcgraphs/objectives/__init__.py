"""Four Nash objectives for the Tier-1 PQC architecture search.

Re-exports the four per-player scorers plus the Hamiltonian dataclass used
by f3. All scoring functions act on a `CircuitDAG` instance and return a
float; f3 and f4 come from factories (closing over a Hamiltonian / target
Topology respectively).
"""
from .anti_bp import f1_anti_bp
from .anti_sim import f2_anti_sim
from .performance import make_f3_performance, make_f3_h2, make_f3_maxcut
from .hardware import make_f4_hardware
from ..gpu.hamiltonians import (
    PauliSumOperator,
    h2_sto3g_hamiltonian,
    lih_sto3g_hamiltonian,
    lih_sto3g_reference_energies,
    maxcut_hamiltonian,
    tfim_hamiltonian,
)

__all__ = [
    "f1_anti_bp",
    "f2_anti_sim",
    "make_f3_performance",
    "make_f3_h2",
    "make_f3_maxcut",
    "make_f4_hardware",
    "PauliSumOperator",
    "h2_sto3g_hamiltonian",
    "lih_sto3g_hamiltonian",
    "lih_sto3g_reference_energies",
    "maxcut_hamiltonian",
    "tfim_hamiltonian",
]
