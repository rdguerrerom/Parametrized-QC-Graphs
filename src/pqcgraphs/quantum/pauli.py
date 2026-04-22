"""
Pauli operator representation for quantum error correction codes.

This module provides the PauliOperator class for representing and manipulating
Pauli operators with proper phase tracking.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass(frozen=True)
class PauliOperator:
    """
    Pauli operator with proper phase tracking.

    Represents a tensor product of single-qubit Pauli operators (I, X, Y, Z)
    with a global phase factor.

    Phase represents power of i: 0→1, 1→i, 2→-1, 3→-i

    Attributes:
        n_qubits: Number of qubits
        operators: Tuple of Pauli operators ('I', 'X', 'Y', 'Z')
        phase: Global phase as power of i (0, 1, 2, or 3)

    Examples:
        >>> p1 = PauliOperator.from_string("XYZ")
        >>> p2 = PauliOperator.from_string("ZZZ")
        >>> p3 = p1 * p2
        >>> print(p3)
    """
    n_qubits: int
    operators: Tuple[str, ...]
    phase: int = 0

    def __post_init__(self):
        """Validate the Pauli operator."""
        if len(self.operators) != self.n_qubits:
            raise ValueError(
                f"Length mismatch: n_qubits={self.n_qubits} but "
                f"operators has length {len(self.operators)}"
            )
        for op in self.operators:
            if op not in ['I', 'X', 'Y', 'Z']:
                raise ValueError(f"Invalid Pauli operator: {op}")
        # Normalize phase to [0, 1, 2, 3]
        object.__setattr__(self, 'phase', self.phase % 4)

    @classmethod
    def from_string(cls, pauli_string: str, phase: int = 0) -> 'PauliOperator':
        """
        Create a PauliOperator from a string representation.

        Args:
            pauli_string: String of Pauli operators (e.g., "XYZII")
            phase: Global phase (default: 0)

        Returns:
            PauliOperator instance

        Examples:
            >>> op = PauliOperator.from_string("XYZ")
            >>> op.n_qubits
            3
        """
        return cls(len(pauli_string), tuple(pauli_string.upper()), phase)

    def __str__(self) -> str:
        """String representation with phase."""
        phase_str = ['', 'i', '-', '-i'][self.phase]
        op_str = ''.join(self.operators)
        return f"{phase_str}{op_str}" if phase_str else op_str

    def __mul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """
        Multiply two Pauli operators with correct phase tracking.

        Args:
            other: Another PauliOperator

        Returns:
            Product of the two operators

        Raises:
            ValueError: If operators have different number of qubits

        Examples:
            >>> p1 = PauliOperator.from_string("XY")
            >>> p2 = PauliOperator.from_string("YZ")
            >>> p3 = p1 * p2
        """
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"Dimension mismatch: {self.n_qubits} vs {other.n_qubits}"
            )

        new_ops = []
        phase = (self.phase + other.phase) % 4

        # Pauli multiplication table with phase factors
        # Format: (result_operator, phase_shift)
        pauli_mult = {
            ('I', 'I'): ('I', 0), ('I', 'X'): ('X', 0),
            ('I', 'Y'): ('Y', 0), ('I', 'Z'): ('Z', 0),
            ('X', 'I'): ('X', 0), ('X', 'X'): ('I', 0),
            ('X', 'Y'): ('Z', 1), ('X', 'Z'): ('Y', 3),
            ('Y', 'I'): ('Y', 0), ('Y', 'X'): ('Z', 3),
            ('Y', 'Y'): ('I', 0), ('Y', 'Z'): ('X', 1),
            ('Z', 'I'): ('Z', 0), ('Z', 'X'): ('Y', 1),
            ('Z', 'Y'): ('X', 3), ('Z', 'Z'): ('I', 0),
        }

        for i in range(self.n_qubits):
            result_op, phase_shift = pauli_mult[(self.operators[i], other.operators[i])]
            new_ops.append(result_op)
            phase = (phase + phase_shift) % 4

        return PauliOperator(self.n_qubits, tuple(new_ops), phase)

    def commutes_with(self, other: 'PauliOperator') -> bool:
        """
        Check if this operator commutes with another.

        Two Pauli operators commute if and only if they anti-commute
        at an even number of positions.

        Args:
            other: Another PauliOperator

        Returns:
            True if operators commute, False otherwise

        Examples:
            >>> p1 = PauliOperator.from_string("XYZ")
            >>> p2 = PauliOperator.from_string("XII")
            >>> p1.commutes_with(p2)
            True
        """
        if self.n_qubits != other.n_qubits:
            return False

        # Count positions where operators anti-commute
        # Two different non-identity Paulis anti-commute
        anti_commute_count = sum(
            1 for i in range(self.n_qubits)
            if self.operators[i] != 'I'
            and other.operators[i] != 'I'
            and self.operators[i] != other.operators[i]
        )

        # Operators commute if anti_commute_count is even
        return anti_commute_count % 2 == 0

    def weight(self) -> int:
        """
        Return the weight of the Pauli operator.

        Weight is the number of non-identity operators.

        Returns:
            Number of non-identity Pauli operators

        Examples:
            >>> PauliOperator.from_string("XYZI").weight()
            3
        """
        return sum(1 for op in self.operators if op != 'I')

    def eigenvalue(self) -> Union[int, complex]:
        """
        Return the eigenvalue corresponding to the phase.

        Returns:
            Eigenvalue: 1, i, -1, or -i

        Examples:
            >>> PauliOperator.from_string("X", phase=0).eigenvalue()
            1
            >>> PauliOperator.from_string("X", phase=1).eigenvalue()
            1j
        """
        return [1, 1j, -1, -1j][self.phase]

    @classmethod
    def from_x_z_bits(cls, x_bits, z_bits, phase: int = 0) -> 'PauliOperator':
        """
        Create PauliOperator from binary symplectic representation.

        Args:
            x_bits: Boolean array indicating X components
            z_bits: Boolean array indicating Z components
            phase: Global phase (default: 0)

        Returns:
            PauliOperator instance
        """
        import numpy as np
        x_bits = np.asarray(x_bits, dtype=bool)
        z_bits = np.asarray(z_bits, dtype=bool)

        n = len(x_bits)
        operators = []

        for i in range(n):
            if x_bits[i] and z_bits[i]:
                operators.append('Y')
            elif x_bits[i]:
                operators.append('X')
            elif z_bits[i]:
                operators.append('Z')
            else:
                operators.append('I')

        return cls(n, tuple(operators), phase)

    @property
    def x(self):
        """Return X component as boolean array."""
        import numpy as np
        return np.array([op in ['X', 'Y'] for op in self.operators], dtype=bool)

    @property
    def z(self):
        """Return Z component as boolean array."""
        import numpy as np
        return np.array([op in ['Z', 'Y'] for op in self.operators], dtype=bool)

    @property
    def n(self):
        """Alias for n_qubits."""
        return self.n_qubits

    def to_matrix(self):
        """
        Convert to full matrix representation.

        Returns:
            Complex numpy array (2^n × 2^n)
        """
        import numpy as np

        # Single qubit Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        pauli_matrices = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

        # Start with first qubit
        result = pauli_matrices[self.operators[0]]

        # Tensor product with remaining qubits
        for op in self.operators[1:]:
            result = np.kron(result, pauli_matrices[op])

        # Apply global phase
        phase_factor = self.eigenvalue()
        return phase_factor * result

    def to_sparse_matrix(self):
        """
        Convert to sparse matrix representation.

        Returns:
            Sparse CSR matrix
        """
        import scipy.sparse as sp
        import numpy as np

        # For now, convert dense to sparse
        # TODO: Implement direct sparse construction for efficiency
        dense = self.to_matrix()
        return sp.csr_matrix(dense)

    def to_string(self) -> str:
        """Convert to string representation."""
        return ''.join(self.operators)

    def __matmul__(self, other: 'PauliOperator') -> 'PauliOperator':
        """Matrix multiplication operator (@)."""
        return self * other
