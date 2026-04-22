"""
Quantum structures for stabilizer codes and graph states.

This package provides the core quantum mechanical representations needed
for quantum error correction code discovery and analysis.

These modules have NO heavy dependencies (no Qiskit) - only NumPy and NetworkX.
For circuit generation, see the circuit package.
"""
from .pauli import PauliOperator
from .graph_state import EnhancedGraphState
from .stabilizer_group import StabilizerGroup, graph_state_to_stabilizer_group
from .time_evolution import YoshidaIntegrator, HamiltonianBuilder

__all__ = [
    'PauliOperator',
    'EnhancedGraphState',
    'StabilizerGroup',
    'graph_state_to_stabilizer_group',
    'YoshidaIntegrator',
    'HamiltonianBuilder',
]
