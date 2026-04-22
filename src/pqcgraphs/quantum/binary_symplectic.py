"""
Binary symplectic linear algebra for stabilizer codes.

This module provides GF(2) (binary field) linear algebra with symplectic
inner product, essential for stabilizer code analysis and encoder synthesis.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class StandardForm:
    """Result of symplectic Gaussian elimination."""
    h_x: np.ndarray  # X part of check matrix (n-k) × n
    h_z: np.ndarray  # Z part of check matrix (n-k) × n
    row_ops: List[Tuple[str, int, int]]  # Record of operations applied
    rank: int


class BinarySymplecticMatrix:
    """
    Binary matrix with symplectic structure for stabilizer codes.

    Represents check matrix H = [H_X | H_Z] where H ∈ Z_2^{(n-k) × 2n}.
    The symplectic inner product determines Pauli commutation relations.

    Attributes:
        data: Binary matrix (n-k) × 2n
        n_qubits: Number of qubits
        n_checks: Number of stabilizer generators (n-k)
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize from binary matrix.

        Args:
            data: Binary matrix of shape (n_checks, 2*n_qubits)
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D array")

        if data.shape[1] % 2 != 0:
            raise ValueError("Number of columns must be even (2n for n qubits)")

        self.data = data.astype(bool)
        self.n_checks = data.shape[0]
        self.n_qubits = data.shape[1] // 2

    @classmethod
    def from_x_z_blocks(cls, h_x: np.ndarray, h_z: np.ndarray) -> 'BinarySymplecticMatrix':
        """
        Construct from separate X and Z blocks.

        Args:
            h_x: X component (n-k) × n
            h_z: Z component (n-k) × n

        Returns:
            BinarySymplecticMatrix with data = [h_x | h_z]
        """
        if h_x.shape != h_z.shape:
            raise ValueError("H_X and H_Z must have same shape")

        data = np.hstack([h_x, h_z])
        return cls(data)

    def get_x_block(self) -> np.ndarray:
        """Extract X component of check matrix."""
        return self.data[:, :self.n_qubits].copy()

    def get_z_block(self) -> np.ndarray:
        """Extract Z component of check matrix."""
        return self.data[:, self.n_qubits:].copy()

    def get_row(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get row i as (x_bits, z_bits)."""
        return self.data[i, :self.n_qubits], self.data[i, self.n_qubits:]

    def symplectic_inner_product(self, row_i: int, row_j: int) -> bool:
        """
        Compute symplectic inner product between two rows.

        For rows r_i = (a_i | b_i) and r_j = (a_j | b_j):
            <r_i, r_j>_s = a_i · b_j + a_j · b_i (mod 2)

        This equals 0 if Paulis commute, 1 if they anticommute.

        Args:
            row_i: First row index
            row_j: Second row index

        Returns:
            True if anticommute (inner product = 1), False if commute (inner product = 0)
        """
        a_i, b_i = self.get_row(row_i)
        a_j, b_j = self.get_row(row_j)

        # Compute dot products as integers to avoid numpy boolean issues
        dot1 = np.sum(a_i.astype(int) * b_j.astype(int))
        dot2 = np.sum(a_j.astype(int) * b_i.astype(int))

        return (dot1 + dot2) % 2 == 1

    def commutes_with(self, row_i: int, row_j: int) -> bool:
        """Check if two rows represent commuting Pauli operators."""
        return not self.symplectic_inner_product(row_i, row_j)

    def validate_commutativity(self) -> bool:
        """
        Verify all rows mutually commute (valid stabilizer group).

        Returns:
            True if all pairs commute
        """
        for i in range(self.n_checks):
            for j in range(i + 1, self.n_checks):
                if self.symplectic_inner_product(i, j):
                    # Non-zero symplectic inner product means they anticommute
                    return False
        return True

    def row_add(self, target: int, source: int) -> None:
        """
        Add source row to target row (mod 2).

        Args:
            target: Row to modify
            source: Row to add
        """
        self.data[target] = (self.data[target].astype(int) +
                            self.data[source].astype(int)) % 2 == 1

    def row_swap(self, i: int, j: int) -> None:
        """Swap rows i and j."""
        self.data[[i, j]] = self.data[[j, i]]

    def gaussian_eliminate(self, preserve_symplectic: bool = True) -> 'BinarySymplecticMatrix':
        """
        Perform Gaussian elimination over GF(2).

        Args:
            preserve_symplectic: If True, maintain symplectic structure

        Returns:
            New matrix in row echelon form
        """
        result = BinarySymplecticMatrix(self.data.copy())

        current_row = 0

        for col in range(result.data.shape[1]):
            # Find pivot
            pivot_row = None
            for row in range(current_row, result.n_checks):
                if result.data[row, col]:
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            # Swap pivot to current position
            if pivot_row != current_row:
                result.row_swap(current_row, pivot_row)

            # Eliminate below
            for row in range(current_row + 1, result.n_checks):
                if result.data[row, col]:
                    result.row_add(row, current_row)

            current_row += 1
            if current_row >= result.n_checks:
                break

        return result

    def to_standard_form(self) -> StandardForm:
        """
        Convert check matrix to standard form via symplectic Gaussian elimination.

        Standard form attempts to make H_X have full row rank with identity-like structure
        while preserving the symplectic structure (commutativity).

        Returns:
            StandardForm with separated H_X and H_Z blocks
        """
        result = BinarySymplecticMatrix(self.data.copy())
        row_ops = []

        # First pass: row reduce X block
        current_row = 0
        pivot_cols = []

        for col in range(result.n_qubits):
            # Find pivot in X block
            pivot_row = None
            for row in range(current_row, result.n_checks):
                if result.data[row, col]:  # X block pivot
                    pivot_row = row
                    break

            if pivot_row is None:
                continue

            # Swap to current position
            if pivot_row != current_row:
                result.row_swap(current_row, pivot_row)
                row_ops.append(('swap', current_row, pivot_row))

            pivot_cols.append(col)

            # Eliminate in X block (both above and below for reduced form)
            for row in range(result.n_checks):
                if row != current_row and result.data[row, col]:
                    result.row_add(row, current_row)
                    row_ops.append(('add', row, current_row))

            current_row += 1
            if current_row >= result.n_checks:
                break

        rank = len(pivot_cols)

        return StandardForm(
            h_x=result.get_x_block(),
            h_z=result.get_z_block(),
            row_ops=row_ops,
            rank=rank
        )

    def rank(self) -> int:
        """Compute rank of the check matrix over GF(2)."""
        reduced = self.gaussian_eliminate(preserve_symplectic=False)

        # Count non-zero rows
        rank = 0
        for row in range(reduced.n_checks):
            if np.any(reduced.data[row]):
                rank += 1

        return rank

    def kernel(self) -> np.ndarray:
        """
        Find kernel (null space) of check matrix over GF(2).

        Returns vectors v such that H·v = 0 (mod 2).
        These represent operators commuting with all stabilizers.

        Returns:
            Array of kernel vectors, shape (kernel_dim, 2n)
        """
        # Reduce to standard form
        reduced = self.gaussian_eliminate(preserve_symplectic=False)

        # Find free variables
        pivot_cols = []
        current_row = 0

        for col in range(reduced.data.shape[1]):
            if current_row >= reduced.n_checks:
                break

            if reduced.data[current_row, col]:
                pivot_cols.append(col)
                current_row += 1

        pivot_cols = set(pivot_cols)
        free_cols = [c for c in range(reduced.data.shape[1]) if c not in pivot_cols]

        # Build kernel basis
        kernel_vectors = []

        for free_col in free_cols:
            # Create vector with 1 in free position
            v = np.zeros(reduced.data.shape[1], dtype=bool)
            v[free_col] = True

            # Back-substitute to satisfy H·v = 0
            for row in range(min(current_row, reduced.n_checks) - 1, -1, -1):
                # Find pivot column for this row
                pivot = None
                for col in range(reduced.data.shape[1]):
                    if reduced.data[row, col]:
                        pivot = col
                        break

                if pivot is not None:
                    # Set v[pivot] to satisfy equation
                    val = False
                    for col in range(pivot + 1, reduced.data.shape[1]):
                        if reduced.data[row, col]:
                            val = (val != v[col])  # XOR
                    v[pivot] = val

            kernel_vectors.append(v)

        if not kernel_vectors:
            return np.zeros((0, reduced.data.shape[1]), dtype=bool)

        return np.array(kernel_vectors)

    def __repr__(self) -> str:
        """String representation showing X and Z blocks."""
        h_x = self.get_x_block()
        h_z = self.get_z_block()

        lines = ["BinarySymplecticMatrix:"]
        lines.append(f"  Shape: ({self.n_checks} checks, {self.n_qubits} qubits)")
        lines.append(f"  H_X | H_Z:")

        for i in range(min(self.n_checks, 10)):  # Show first 10 rows
            x_str = ''.join('1' if x else '0' for x in h_x[i])
            z_str = ''.join('1' if z else '0' for z in h_z[i])
            lines.append(f"    {x_str} | {z_str}")

        if self.n_checks > 10:
            lines.append(f"    ... ({self.n_checks - 10} more rows)")

        return '\n'.join(lines)


def check_symplectic_orthogonality(
    v1: np.ndarray,
    v2: np.ndarray
) -> bool:
    """
    Check if two symplectic vectors are orthogonal.

    Args:
        v1: First vector (2n bits as x|z)
        v2: Second vector (2n bits as x|z)

    Returns:
        True if <v1, v2>_s = 0 (commute)
    """
    if len(v1) != len(v2) or len(v1) % 2 != 0:
        raise ValueError("Vectors must have even equal length")

    n = len(v1) // 2
    a1, b1 = v1[:n], v1[n:]
    a2, b2 = v2[:n], v2[n:]

    return (np.dot(a1.astype(int), b2.astype(int)) +
            np.dot(a2.astype(int), b1.astype(int))) % 2 == 0


def symplectic_gram_schmidt(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """
    Symplectic orthogonalization (not true Gram-Schmidt, but symplectic analog).

    Produces a set of mutually commuting Pauli operators from input set.

    Args:
        vectors: List of binary symplectic vectors

    Returns:
        Orthogonalized vectors (may be fewer if dependencies exist)
    """
    if not vectors:
        return []

    result = []

    for v in vectors:
        # Check if v is symplectic-orthogonal to all results so far
        is_orthogonal = all(check_symplectic_orthogonality(v, r) for r in result)

        if is_orthogonal:
            result.append(v.copy())
        else:
            # Try to make orthogonal by adding existing vectors (mod 2)
            v_modified = v.copy()
            for r in result:
                if not check_symplectic_orthogonality(v_modified, r):
                    v_modified = (v_modified.astype(int) + r.astype(int)) % 2 == 1

            # Check if modified version works
            if all(check_symplectic_orthogonality(v_modified, r) for r in result):
                if np.any(v_modified):  # Non-zero
                    result.append(v_modified)

    return result
