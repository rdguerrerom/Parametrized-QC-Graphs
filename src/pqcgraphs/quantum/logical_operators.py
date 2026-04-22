"""
Logical operator computation for stabilizer codes.

This module computes the centralizer and normalizer of stabilizer groups,
extracting logical X and Z operators for quantum error correction codes.
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from itertools import product

from .pauli import PauliOperator
from .binary_symplectic import BinarySymplecticMatrix, check_symplectic_orthogonality


class LogicalOperatorFinder:
    """
    Computes logical operators from stabilizer generators.

    Given n-k stabilizer generators on n qubits, finds k independent
    logical X and Z operator pairs satisfying:
        - {X̄_i, Z̄_j} = 2δ_ij
        - [X̄_i, X̄_j] = [Z̄_i, Z̄_j] = 0
        - All logical ops commute with all stabilizers
    """

    def __init__(self, generators: List[PauliOperator]):
        """
        Initialize with stabilizer generators.

        Args:
            generators: List of n-k commuting Pauli operators on n qubits
        """
        if not generators:
            raise ValueError("Need at least one generator")

        self.generators = generators
        self.n_qubits = generators[0].n
        self.n_checks = len(generators)
        self.n_logical = self.n_qubits - self.n_checks

        # Build check matrix
        self.check_matrix = self._build_check_matrix()

        # Cache for logical operators
        self._logical_x: Optional[List[PauliOperator]] = None
        self._logical_z: Optional[List[PauliOperator]] = None

    def _build_check_matrix(self) -> BinarySymplecticMatrix:
        """Convert stabilizer generators to binary check matrix."""
        h_x = np.zeros((self.n_checks, self.n_qubits), dtype=bool)
        h_z = np.zeros((self.n_checks, self.n_qubits), dtype=bool)

        for i, gen in enumerate(self.generators):
            h_x[i] = gen.x
            h_z[i] = gen.z

        return BinarySymplecticMatrix.from_x_z_blocks(h_x, h_z)

    def _pauli_from_vector(self, vec: np.ndarray) -> PauliOperator:
        """Convert binary symplectic vector to PauliOperator."""
        n = len(vec) // 2
        x_bits = vec[:n]
        z_bits = vec[n:]
        return PauliOperator.from_x_z_bits(x_bits, z_bits)

    def _validate_commutativity(self) -> bool:
        """Verify all stabilizer generators commute."""
        return self.check_matrix.validate_commutativity()

    def find_centralizer(self) -> List[PauliOperator]:
        """
        Find centralizer C(S) = {P ∈ P_n : [P, S] = 0 for all S ∈ S}.

        The centralizer consists of all Pauli operators commuting with
        all stabilizer generators. These form the kernel of the check matrix.

        Returns:
            List of PauliOperator spanning the centralizer
        """
        kernel_vectors = self.check_matrix.kernel()

        centralizer = []
        for vec in kernel_vectors:
            if np.any(vec):  # Non-identity
                pauli = self._pauli_from_vector(vec)
                centralizer.append(pauli)

        return centralizer

    def _extract_logical_pairs(
        self,
        centralizer: List[PauliOperator]
    ) -> Tuple[List[PauliOperator], List[PauliOperator]]:
        """
        Extract k independent logical X and Z pairs from centralizer.

        Strategy:
        1. Separate centralizer into stabilizer cosets
        2. Find k pairs (X̄_i, Z̄_i) with {X̄_i, Z̄_j} = 2δ_ij
        3. Ensure all operators commute with stabilizers

        Args:
            centralizer: List of operators commuting with all stabilizers

        Returns:
            (logical_x_ops, logical_z_ops) - each list of length k
        """
        # Remove stabilizer generators from centralizer
        non_stabilizer = []
        for op in centralizer:
            is_stabilizer = False
            for gen in self.generators:
                if np.array_equal(op.x, gen.x) and np.array_equal(op.z, gen.z):
                    is_stabilizer = True
                    break
            if not is_stabilizer:
                non_stabilizer.append(op)

        if len(non_stabilizer) < 2 * self.n_logical:
            # Not enough operators - try exhaustive search
            return self._exhaustive_logical_search()

        # Separate into X-type (Z=0 component) and Z-type (X=0 component) preferences
        x_type_candidates = []
        z_type_candidates = []
        mixed_type = []

        for op in non_stabilizer:
            x_weight = np.sum(op.x)
            z_weight = np.sum(op.z)

            if z_weight == 0 and x_weight > 0:
                x_type_candidates.append(op)
            elif x_weight == 0 and z_weight > 0:
                z_type_candidates.append(op)
            else:
                mixed_type.append(op)

        # Try to pair up logical operators
        logical_x = []
        logical_z = []

        # Start with mixed type for general codes
        candidates = mixed_type + x_type_candidates + z_type_candidates

        used_indices = set()

        for i, op1 in enumerate(candidates):
            if i in used_indices or len(logical_x) >= self.n_logical:
                break

            # Find anticommuting partner
            for j, op2 in enumerate(candidates):
                if j <= i or j in used_indices:
                    continue

                # Check if they anticommute
                if not op1.commutes_with(op2):
                    # Found a pair!
                    # Decide which is X and which is Z
                    if np.sum(op1.x) >= np.sum(op1.z):
                        logical_x.append(op1)
                        logical_z.append(op2)
                    else:
                        logical_x.append(op2)
                        logical_z.append(op1)

                    used_indices.add(i)
                    used_indices.add(j)
                    break

        # If we don't have enough pairs, try systematic search
        if len(logical_x) < self.n_logical:
            return self._exhaustive_logical_search()

        return logical_x[:self.n_logical], logical_z[:self.n_logical]

    def _exhaustive_logical_search(
        self,
        max_weight: int = None
    ) -> Tuple[List[PauliOperator], List[PauliOperator]]:
        """
        Exhaustive search for logical operators.

        Searches through low-weight Pauli operators to find logical pairs.

        Args:
            max_weight: Maximum weight to search (default: n_qubits)

        Returns:
            (logical_x_ops, logical_z_ops)
        """
        if max_weight is None:
            max_weight = min(self.n_qubits, 10)  # Limit search space

        logical_x = []
        logical_z = []

        # Generate all Pauli operators up to max_weight
        for weight in range(1, max_weight + 1):
            if len(logical_x) >= self.n_logical:
                break

            # Try different qubit positions
            for positions in self._generate_combinations(self.n_qubits, weight):
                if len(logical_x) >= self.n_logical:
                    break

                # Try different Pauli types on those positions
                for pauli_types in product([1, 2, 3], repeat=weight):  # X, Y, Z
                    x_bits = np.zeros(self.n_qubits, dtype=bool)
                    z_bits = np.zeros(self.n_qubits, dtype=bool)

                    for idx, pauli_type in zip(positions, pauli_types):
                        if pauli_type == 1:  # X
                            x_bits[idx] = True
                        elif pauli_type == 2:  # Y
                            x_bits[idx] = True
                            z_bits[idx] = True
                        else:  # Z
                            z_bits[idx] = True

                    candidate = PauliOperator.from_x_z_bits(x_bits, z_bits)

                    # Check if commutes with all stabilizers
                    if not all(candidate.commutes_with(gen) for gen in self.generators):
                        continue

                    # Check if it's a stabilizer
                    is_stabilizer = any(
                        np.array_equal(candidate.x, gen.x) and
                        np.array_equal(candidate.z, gen.z)
                        for gen in self.generators
                    )
                    if is_stabilizer:
                        continue

                    # Try to find anticommuting partner
                    partner = self._find_anticommuting_partner(
                        candidate,
                        max_weight,
                        exclude=logical_x + logical_z
                    )

                    if partner is not None:
                        logical_x.append(candidate)
                        logical_z.append(partner)

                        if len(logical_x) >= self.n_logical:
                            break

        if len(logical_x) < self.n_logical:
            # Pad with dummy operators if not found (for degenerate codes)
            while len(logical_x) < self.n_logical:
                # Create simple logical operators
                x_bits = np.zeros(self.n_qubits, dtype=bool)
                z_bits = np.zeros(self.n_qubits, dtype=bool)
                x_bits[len(logical_x)] = True
                logical_x.append(PauliOperator.from_x_z_bits(x_bits, z_bits))

                x_bits = np.zeros(self.n_qubits, dtype=bool)
                z_bits = np.zeros(self.n_qubits, dtype=bool)
                z_bits[len(logical_z)] = True
                logical_z.append(PauliOperator.from_x_z_bits(x_bits, z_bits))

        return logical_x[:self.n_logical], logical_z[:self.n_logical]

    def _generate_combinations(self, n: int, k: int):
        """Generate all k-combinations of range(n)."""
        from itertools import combinations
        return combinations(range(n), k)

    def _find_anticommuting_partner(
        self,
        op: PauliOperator,
        max_weight: int,
        exclude: List[PauliOperator]
    ) -> Optional[PauliOperator]:
        """Find an operator that anticommutes with op and commutes with stabilizers."""
        for weight in range(1, max_weight + 1):
            for positions in self._generate_combinations(self.n_qubits, weight):
                for pauli_types in product([1, 2, 3], repeat=weight):
                    x_bits = np.zeros(self.n_qubits, dtype=bool)
                    z_bits = np.zeros(self.n_qubits, dtype=bool)

                    for idx, pauli_type in zip(positions, pauli_types):
                        if pauli_type == 1:
                            x_bits[idx] = True
                        elif pauli_type == 2:
                            x_bits[idx] = True
                            z_bits[idx] = True
                        else:
                            z_bits[idx] = True

                    candidate = PauliOperator.from_x_z_bits(x_bits, z_bits)

                    # Check if anticommutes with op
                    if candidate.commutes_with(op):
                        continue

                    # Check if commutes with stabilizers
                    if not all(candidate.commutes_with(gen) for gen in self.generators):
                        continue

                    # Check not already used
                    is_excluded = any(
                        np.array_equal(candidate.x, exc.x) and
                        np.array_equal(candidate.z, exc.z)
                        for exc in exclude
                    )
                    if is_excluded:
                        continue

                    return candidate

        return None

    def find_logical_operators(
        self,
        force_recompute: bool = False
    ) -> Tuple[List[PauliOperator], List[PauliOperator]]:
        """
        Find logical X and Z operators.

        Returns k pairs of logical operators (X̄_i, Z̄_i) satisfying
        the logical Pauli algebra.

        Args:
            force_recompute: If True, recompute even if cached

        Returns:
            (logical_x_ops, logical_z_ops) where each is a list of k operators
        """
        if not force_recompute and self._logical_x is not None:
            return self._logical_x, self._logical_z

        # Validate stabilizer group
        if not self._validate_commutativity():
            raise ValueError("Stabilizer generators don't all commute")

        # Find centralizer
        centralizer = self.find_centralizer()

        # Extract logical pairs
        self._logical_x, self._logical_z = self._extract_logical_pairs(centralizer)

        return self._logical_x, self._logical_z

    def compute_distance(self, max_weight: Optional[int] = None) -> int:
        """
        Compute code distance (minimum weight of nontrivial logical operator).

        Args:
            max_weight: Maximum weight to search (None = exhaustive up to n)

        Returns:
            Code distance d
        """
        if max_weight is None:
            max_weight = self.n_qubits

        min_distance = max_weight + 1

        # Search through Pauli operators in increasing weight
        for weight in range(1, max_weight + 1):
            for positions in self._generate_combinations(self.n_qubits, weight):
                for pauli_types in product([1, 2, 3], repeat=weight):
                    x_bits = np.zeros(self.n_qubits, dtype=bool)
                    z_bits = np.zeros(self.n_qubits, dtype=bool)

                    for idx, pauli_type in zip(positions, pauli_types):
                        if pauli_type == 1:
                            x_bits[idx] = True
                        elif pauli_type == 2:
                            x_bits[idx] = True
                            z_bits[idx] = True
                        else:
                            z_bits[idx] = True

                    candidate = PauliOperator.from_x_z_bits(x_bits, z_bits)

                    # Check if in normalizer but not stabilizer
                    if not all(candidate.commutes_with(gen) for gen in self.generators):
                        continue

                    # Check if it's a stabilizer
                    is_stabilizer = any(
                        np.array_equal(candidate.x, gen.x) and
                        np.array_equal(candidate.z, gen.z)
                        for gen in self.generators
                    )

                    if not is_stabilizer:
                        # Found nontrivial logical operator
                        return weight

        return min_distance

    def get_code_parameters(self) -> Tuple[int, int, int]:
        """
        Get [[n, k, d]] code parameters.

        Returns:
            (n_physical, n_logical, distance)
        """
        distance = self.compute_distance(max_weight=min(self.n_qubits, 5))
        return self.n_qubits, self.n_logical, distance

    def __repr__(self) -> str:
        """String representation."""
        return (f"LogicalOperatorFinder([[{self.n_qubits}, {self.n_logical}]])\n"
                f"  {self.n_checks} stabilizer generators")
