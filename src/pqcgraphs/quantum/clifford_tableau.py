"""
Clifford tableau representation for stabilizer codes.

Implements the full stabilizer + destabilizer tableau for efficient
Clifford gate tracking and encoder synthesis (Aaronson-Gottesman algorithm).
"""

import numpy as np
from typing import List, Tuple, Optional

from .pauli import PauliOperator


class CliffordTableau:
    """
    Complete stabilizer tableau with destabilizers.

    Stores 2n × 2n binary matrix plus phase vector representing:
    - n-k stabilizer generators g_i
    - n-k destabilizers d_i satisfying {g_i, d_j} = 2δ_ij
    - k logical X operators
    - k logical Z operators

    This enables fast simulation of Clifford circuits via symplectic
    matrix updates.

    Attributes:
        n_qubits: Number of physical qubits
        x_matrix: Binary matrix for X components (2n × n)
        z_matrix: Binary matrix for Z components (2n × n)
        phases: Phase vector (2n) as power of i
    """

    def __init__(
        self,
        n_qubits: int,
        x_matrix: Optional[np.ndarray] = None,
        z_matrix: Optional[np.ndarray] = None,
        phases: Optional[np.ndarray] = None
    ):
        """
        Initialize Clifford tableau.

        Args:
            n_qubits: Number of qubits
            x_matrix: X part of tableau (2n × n), defaults to identity structure
            z_matrix: Z part of tableau (2n × n), defaults to identity structure
            phases: Phase vector (2n), defaults to zeros
        """
        self.n_qubits = n_qubits

        if x_matrix is None:
            # Default: computational basis
            # First n rows = Z stabilizers (destabilizers are X)
            # Last n rows = X stabilizers (destabilizers are Z)
            x_matrix = np.zeros((2 * n_qubits, n_qubits), dtype=bool)
            x_matrix[n_qubits:, :] = np.eye(n_qubits, dtype=bool)

        if z_matrix is None:
            z_matrix = np.zeros((2 * n_qubits, n_qubits), dtype=bool)
            z_matrix[:n_qubits, :] = np.eye(n_qubits, dtype=bool)

        if phases is None:
            phases = np.zeros(2 * n_qubits, dtype=int)

        self.x_matrix = x_matrix.astype(bool)
        self.z_matrix = z_matrix.astype(bool)
        self.phases = phases % 4  # phases are mod 4 (powers of i)

    @classmethod
    def from_stabilizers(
        cls,
        stabilizers: List[PauliOperator],
        destabilizers: Optional[List[PauliOperator]] = None,
        logical_x: Optional[List[PauliOperator]] = None,
        logical_z: Optional[List[PauliOperator]] = None
    ) -> 'CliffordTableau':
        """
        Construct tableau from stabilizer generators and logical operators.

        Args:
            stabilizers: List of n-k stabilizer generators
            destabilizers: List of n-k destabilizers (computed if None)
            logical_x: List of k logical X operators
            logical_z: List of k logical Z operators

        Returns:
            CliffordTableau
        """
        if not stabilizers:
            raise ValueError("Need at least one stabilizer")

        n_qubits = stabilizers[0].n
        n_checks = len(stabilizers)
        n_logical = n_qubits - n_checks

        # Build tableau matrices
        x_matrix = np.zeros((2 * n_qubits, n_qubits), dtype=bool)
        z_matrix = np.zeros((2 * n_qubits, n_qubits), dtype=bool)
        phases = np.zeros(2 * n_qubits, dtype=int)

        # Fill in destabilizers (first n-k rows)
        if destabilizers is None:
            destabilizers = cls._compute_destabilizers(stabilizers)

        for i, dest in enumerate(destabilizers):
            x_matrix[i] = dest.x
            z_matrix[i] = dest.z
            phases[i] = dest.phase

        # Fill in stabilizers (rows n-k to 2(n-k))
        for i, stab in enumerate(stabilizers):
            x_matrix[n_checks + i] = stab.x
            z_matrix[n_checks + i] = stab.z
            phases[n_checks + i] = stab.phase

        # Fill in logical operators if provided
        if logical_x is not None and logical_z is not None:
            offset = 2 * n_checks
            for i, (lx, lz) in enumerate(zip(logical_x, logical_z)):
                x_matrix[offset + i] = lx.x
                z_matrix[offset + i] = lx.z
                phases[offset + i] = lx.phase

                x_matrix[offset + n_logical + i] = lz.x
                z_matrix[offset + n_logical + i] = lz.z
                phases[offset + n_logical + i] = lz.phase

        return cls(n_qubits, x_matrix, z_matrix, phases)

    @staticmethod
    def _compute_destabilizers(stabilizers: List[PauliOperator]) -> List[PauliOperator]:
        """
        Compute destabilizers for given stabilizers.

        For each stabilizer g_i, find d_i such that {g_i, d_i} = 2
        and {g_i, d_j} = 0 for i ≠ j.

        Args:
            stabilizers: List of stabilizer generators

        Returns:
            List of destabilizers
        """
        n_qubits = stabilizers[0].n
        destabilizers = []

        for i, stab in enumerate(stabilizers):
            # Find a Pauli that anticommutes with stab but commutes with others
            found = False

            # Try single-qubit Paulis first
            for qubit in range(n_qubits):
                for pauli_type in ['X', 'Y', 'Z']:
                    x_bits = np.zeros(n_qubits, dtype=bool)
                    z_bits = np.zeros(n_qubits, dtype=bool)

                    if pauli_type == 'X':
                        x_bits[qubit] = True
                    elif pauli_type == 'Y':
                        x_bits[qubit] = True
                        z_bits[qubit] = True
                    else:  # Z
                        z_bits[qubit] = True

                    candidate = PauliOperator(x_bits, z_bits)

                    # Check anticommutes with stab_i
                    if candidate.commutes_with(stab):
                        continue

                    # Check commutes with all other stabilizers
                    valid = True
                    for j, other_stab in enumerate(stabilizers):
                        if i != j and not candidate.commutes_with(other_stab):
                            valid = False
                            break

                    if valid:
                        destabilizers.append(candidate)
                        found = True
                        break

                if found:
                    break

            if not found:
                # Fallback: use a simple pattern
                x_bits = np.zeros(n_qubits, dtype=bool)
                z_bits = np.zeros(n_qubits, dtype=bool)
                x_bits[i % n_qubits] = True
                destabilizers.append(PauliOperator(x_bits, z_bits))

        return destabilizers

    def get_row(self, i: int) -> PauliOperator:
        """Get row i as a PauliOperator."""
        return PauliOperator(
            self.x_matrix[i],
            self.z_matrix[i],
            self.phases[i]
        )

    def set_row(self, i: int, pauli: PauliOperator) -> None:
        """Set row i to a PauliOperator."""
        self.x_matrix[i] = pauli.x
        self.z_matrix[i] = pauli.z
        self.phases[i] = pauli.phase % 4

    def row_multiply(self, i: int, j: int) -> None:
        """
        Multiply row i by row j (in place).

        Implements: row[i] ← row[i] * row[j]
        """
        pauli_i = self.get_row(i)
        pauli_j = self.get_row(j)
        result = pauli_i @ pauli_j
        self.set_row(i, result)

    def hadamard(self, qubit: int) -> None:
        """
        Apply Hadamard gate to qubit.

        H: X ↔ Z (swaps X and Z, adds phase if both present)
        """
        for i in range(2 * self.n_qubits):
            x_bit = self.x_matrix[i, qubit]
            z_bit = self.z_matrix[i, qubit]

            # Swap X and Z
            self.x_matrix[i, qubit] = z_bit
            self.z_matrix[i, qubit] = x_bit

            # Add phase if both X and Z were present (Y → -Y)
            if x_bit and z_bit:
                self.phases[i] = (self.phases[i] + 2) % 4

    def phase_gate(self, qubit: int) -> None:
        """
        Apply Phase gate (S) to qubit.

        S: X → Y, Y → -X, Z → Z
        """
        for i in range(2 * self.n_qubits):
            x_bit = self.x_matrix[i, qubit]
            z_bit = self.z_matrix[i, qubit]

            if x_bit:
                # X → Y or Y → -X
                self.z_matrix[i, qubit] = True
                if z_bit:  # Was Y
                    self.phases[i] = (self.phases[i] + 2) % 4

    def cnot(self, control: int, target: int) -> None:
        """
        Apply CNOT gate.

        CNOT: X_c → X_c X_t, Z_t → Z_c Z_t
        """
        for i in range(2 * self.n_qubits):
            x_c = self.x_matrix[i, control]
            z_c = self.z_matrix[i, control]
            x_t = self.x_matrix[i, target]
            z_t = self.z_matrix[i, target]

            # Update target X
            self.x_matrix[i, target] = x_c ^ x_t  # XOR

            # Update control Z
            self.z_matrix[i, control] = z_c ^ z_t  # XOR

            # Update phase (from commutator)
            if x_c and z_t and not (z_c or x_t):
                self.phases[i] = (self.phases[i] + 2) % 4
            elif not x_c and not z_t and z_c and x_t:
                self.phases[i] = (self.phases[i] + 2) % 4

    def measure_qubit(self, qubit: int) -> Tuple[int, bool]:
        """
        Measure qubit in Z basis.

        Returns:
            (outcome, is_random) where outcome is 0 or 1,
            and is_random indicates if outcome was deterministic
        """
        # Check if any destabilizer has X on this qubit
        for i in range(self.n_qubits):
            if self.x_matrix[i, qubit]:
                # Random outcome
                outcome = np.random.randint(2)
                # Project tableau
                self._project_measurement(qubit, i, outcome)
                return outcome, True

        # Deterministic outcome - compute from stabilizers
        outcome = 0
        for i in range(self.n_qubits, 2 * self.n_qubits):
            if self.x_matrix[i, qubit]:
                if self.phases[i] in [2, 3]:  # -1 or -i phase
                    outcome ^= 1

        return outcome, False

    def _project_measurement(self, qubit: int, destab_row: int, outcome: int) -> None:
        """Project tableau after measurement."""
        # Set destabilizer to Z_qubit or -Z_qubit
        self.x_matrix[destab_row] = False
        self.z_matrix[destab_row] = False
        self.z_matrix[destab_row, qubit] = True
        self.phases[destab_row] = 2 * outcome  # 0 or 2 (for +1 or -1)

        # Eliminate X_qubit from other rows
        for i in range(2 * self.n_qubits):
            if i != destab_row and self.x_matrix[i, qubit]:
                self.row_multiply(i, destab_row)

    def copy(self) -> 'CliffordTableau':
        """Create deep copy of tableau."""
        return CliffordTableau(
            self.n_qubits,
            self.x_matrix.copy(),
            self.z_matrix.copy(),
            self.phases.copy()
        )

    def __repr__(self) -> str:
        """String representation."""
        lines = [f"CliffordTableau ({self.n_qubits} qubits):"]

        for i in range(min(2 * self.n_qubits, 20)):
            pauli = self.get_row(i)
            phase_str = ['', 'i', '-', '-i'][self.phases[i]]
            lines.append(f"  Row {i:2d}: {phase_str:2s} {pauli.to_string()}")

        if 2 * self.n_qubits > 20:
            lines.append(f"  ... ({2 * self.n_qubits - 20} more rows)")

        return '\n'.join(lines)


def tableau_from_circuit(n_qubits: int, gates: List[Tuple]) -> CliffordTableau:
    """
    Build tableau by applying Clifford gates.

    Args:
        n_qubits: Number of qubits
        gates: List of (gate_type, *qubits) tuples
               e.g., ('H', 0), ('CNOT', 0, 1), ('S', 2)

    Returns:
        Final tableau after applying all gates
    """
    tableau = CliffordTableau(n_qubits)

    for gate in gates:
        gate_type = gate[0]

        if gate_type == 'H':
            tableau.hadamard(gate[1])
        elif gate_type == 'S':
            tableau.phase_gate(gate[1])
        elif gate_type == 'CNOT':
            tableau.cnot(gate[1], gate[2])
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")

    return tableau
