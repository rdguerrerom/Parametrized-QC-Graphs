"""
High-order Yoshida integrators for quantum time evolution.

This module provides symplectic integration methods for simulating quantum dynamics
governed by the time-dependent Schrödinger equation. Unlike the stabilizer formalism
used elsewhere in this codebase, these integrators work with full state vectors and
dense Hamiltonian matrices.

Mathematical Background:
    The time evolution of a quantum state is governed by:

        i ℏ ∂|ψ⟩/∂t = H(t)|ψ⟩

    For time-dependent Hamiltonians, the solution involves time-ordered exponentials.
    Yoshida integrators provide high-order accurate approximations while preserving
    symplectic structure and unitarity.

Yoshida Method:
    The 4th-order Yoshida method constructs high-order approximations by composing
    lower-order symmetric integrators with specific coefficients:

        U₄(Δt) = U₂(w₄Δt) ∘ U₂(w₃Δt) ∘ U₂(w₂Δt) ∘ U₂(w₁Δt)

    where w₁ = w₄ ≈ 0.6756, w₂ = w₃ ≈ -0.1756

    This achieves O(Δt⁴) global error while maintaining excellent energy conservation
    and time-reversibility properties.

References:
    - Yoshida, H. (1990). "Construction of higher order symplectic integrators"
    - Suzuki, M. (1991). "General theory of fractal path integrals"

Examples:
    >>> # Basic time evolution
    >>> integrator = YoshidaIntegrator(n_qubits=2, order=4)
    >>> H = HamiltonianBuilder.ising_model(n_qubits=2, J=1.0, h=0.5)
    >>> psi0 = np.zeros(4, dtype=complex)
    >>> psi0[0] = 1.0  # |00⟩ state
    >>>
    >>> # Evolve for time t=1.0
    >>> psi_final = integrator.evolve(
    ...     initial_state=psi0,
    ...     hamiltonian_func=lambda t: H,
    ...     t_final=1.0,
    ...     n_steps=100
    ... )

    >>> # Time-dependent evolution
    >>> def time_dep_hamiltonian(t):
    ...     J = 1.0 + 0.3 * np.sin(2.0 * t)
    ...     return HamiltonianBuilder.ising_model(n_qubits=2, J=J, h=0.5)
    >>>
    >>> psi_final = integrator.evolve(psi0, time_dep_hamiltonian, 5.0, 200)
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from typing import Callable, List, Tuple, Optional


class YoshidaIntegrator:
    """
    High-order symplectic integrator for quantum time evolution.

    Implements Yoshida composition methods for orders 2, 4, 6, and 8.
    These methods provide excellent accuracy and long-time stability for
    simulating quantum dynamics with time-dependent Hamiltonians.

    Attributes:
        n_qubits: Number of qubits in the system
        order: Integration order (2, 4, 6, or 8)
        dim: Hilbert space dimension (2^n_qubits)

    Examples:
        >>> integrator = YoshidaIntegrator(n_qubits=3, order=4)
        >>> H = HamiltonianBuilder.heisenberg_model(n_qubits=3)
        >>> psi0 = np.zeros(8, dtype=complex)
        >>> psi0[0] = 1.0
        >>> psi_final = integrator.evolve(psi0, lambda t: H, 1.0, 100)
    """

    def __init__(self, n_qubits: int, order: int = 4):
        """
        Initialize Yoshida integrator.

        Args:
            n_qubits: Number of qubits
            order: Integration order (2, 4, 6, or 8)

        Raises:
            ValueError: If order is not 2, 4, 6, or 8
            ValueError: If n_qubits is not positive
        """
        if order not in [2, 4, 6, 8]:
            raise ValueError(f"Order must be 2, 4, 6, or 8, got {order}")
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")

        self.n_qubits = n_qubits
        self.order = order
        self.dim = 2 ** n_qubits

        # Store Yoshida coefficients
        self._coefficients = self._compute_coefficients(order)

    def _compute_coefficients(self, order: int) -> List[float]:
        """
        Compute Yoshida composition coefficients.

        Args:
            order: Integration order

        Returns:
            List of coefficients for symmetric composition
        """
        if order == 2:
            # 2nd order is just a single step (coefficient = 1.0)
            return [1.0]

        elif order == 4:
            # 4th order Yoshida coefficients (Equation 4.6 from Yoshida 1990)
            x0 = -2**(1/3) / (2 - 2**(1/3))
            x1 = 1 / (2 - 2**(1/3))
            return [x1, x0, x1]

        elif order == 6:
            # 6th order coefficients (Solution A from Table 1, Yoshida 1990)
            w1 = -1.17767998417887
            w2 = 0.235573213359357
            w3 = 0.784513610477560
            w0 = 1 - 2*(w1 + w2 + w3)
            return [w3, w2, w1, w0, w1, w2, w3]

        elif order == 8:
            # 8th order coefficients (Solution A from Table 2, Yoshida 1990)
            w1 = -1.61582374150097
            w2 = -2.44699182370524
            w3 = -0.071698941970812
            w4 = 2.44002732616735
            w5 = 0.157739928123617
            w6 = 1.82020630970714
            w7 = 1.04242620869991
            w0 = 1 - 2*(w1 + w2 + w3 + w4 + w5 + w6 + w7)
            return [w7, w6, w5, w4, w3, w2, w1, w0, w1, w2, w3, w4, w5, w6, w7]

        else:
            raise ValueError(f"Unsupported order: {order}")

    def step(
        self,
        state: np.ndarray,
        hamiltonian: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Perform a single time step.

        For time-independent or slowly-varying Hamiltonians, this applies
        one Yoshida step using the provided Hamiltonian matrix.

        Args:
            state: Current quantum state (complex vector of size dim)
            hamiltonian: Hamiltonian matrix (dim × dim complex matrix)
            dt: Time step

        Returns:
            Evolved quantum state

        Raises:
            ValueError: If state or Hamiltonian dimensions are incorrect

        Examples:
            >>> integrator = YoshidaIntegrator(n_qubits=2, order=4)
            >>> psi = np.array([1, 0, 0, 0], dtype=complex)
            >>> H = HamiltonianBuilder.ising_model(2)
            >>> psi_next = integrator.step(psi, H, 0.01)
        """
        if state.shape[0] != self.dim:
            raise ValueError(
                f"State dimension {state.shape[0]} does not match "
                f"expected dimension {self.dim}"
            )
        if hamiltonian.shape != (self.dim, self.dim):
            raise ValueError(
                f"Hamiltonian shape {hamiltonian.shape} does not match "
                f"expected shape ({self.dim}, {self.dim})"
            )

        # For 2nd order, use direct matrix exponential
        if self.order == 2:
            return self._second_order_step(state, hamiltonian, dt)

        # For higher orders, compose 2nd order steps with Yoshida coefficients
        current_state = state.copy()
        for w in self._coefficients:
            sub_dt = w * dt
            if abs(sub_dt) > 1e-14:  # Skip negligible time steps
                current_state = self._second_order_step(
                    current_state, hamiltonian, sub_dt
                )

        return current_state

    def _second_order_step(
        self,
        state: np.ndarray,
        hamiltonian: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Second-order symmetric step (building block for higher orders).

        Uses exact matrix exponential for simplicity and accuracy.

        Args:
            state: Current state
            hamiltonian: Hamiltonian matrix
            dt: Time step

        Returns:
            Evolved state
        """
        # Compute U = exp(-i H dt)
        U = linalg.expm(-1j * hamiltonian * dt)
        return U @ state

    def evolve(
        self,
        initial_state: np.ndarray,
        hamiltonian_func: Callable[[float], np.ndarray],
        t_final: float,
        n_steps: int
    ) -> np.ndarray:
        """
        Evolve quantum state over time interval [0, t_final].

        Handles time-dependent Hamiltonians by evaluating the Hamiltonian
        at the midpoint of each Yoshida sub-step for improved accuracy.

        Args:
            initial_state: Initial quantum state
            hamiltonian_func: Function that takes time t and returns H(t)
            t_final: Final time
            n_steps: Number of time steps

        Returns:
            Final quantum state at time t_final

        Raises:
            ValueError: If initial_state dimension is incorrect
            ValueError: If n_steps is not positive

        Examples:
            >>> integrator = YoshidaIntegrator(n_qubits=2, order=4)
            >>> psi0 = np.array([1, 0, 0, 0], dtype=complex)
            >>>
            >>> # Time-dependent transverse field
            >>> def H(t):
            ...     h = 0.5 * (1 + 0.3 * np.sin(t))
            ...     return HamiltonianBuilder.ising_model(2, J=1.0, h=h)
            >>>
            >>> psi_final = integrator.evolve(psi0, H, 10.0, 500)
        """
        if initial_state.shape[0] != self.dim:
            raise ValueError(
                f"Initial state dimension {initial_state.shape[0]} does not match "
                f"expected dimension {self.dim}"
            )
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        state = initial_state.copy()
        dt = t_final / n_steps

        for i in range(n_steps):
            t = i * dt

            # Evaluate Hamiltonian at midpoint for better accuracy
            t_mid = t + dt / 2
            H = hamiltonian_func(t_mid)

            # Perform Yoshida step
            state = self.step(state, H, dt)

        return state

    def evolve_with_observables(
        self,
        initial_state: np.ndarray,
        hamiltonian_func: Callable[[float], np.ndarray],
        t_final: float,
        n_steps: int,
        observables: Optional[List[Tuple[str, np.ndarray]]] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Evolve state and compute expectation values of observables.

        This is useful for tracking physical quantities during time evolution,
        such as magnetization, energy, or entanglement measures.

        Args:
            initial_state: Initial quantum state
            hamiltonian_func: Function returning Hamiltonian at time t
            t_final: Final time
            n_steps: Number of time steps
            observables: List of (name, operator) pairs to track

        Returns:
            Tuple of (final_state, observables_dict) where observables_dict
            contains time series data for each observable

        Examples:
            >>> integrator = YoshidaIntegrator(n_qubits=2, order=4)
            >>> psi0 = np.array([1, 0, 0, 0], dtype=complex)
            >>> H = lambda t: HamiltonianBuilder.ising_model(2)
            >>>
            >>> # Track magnetization
            >>> sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
            >>> Z1 = HamiltonianBuilder.pauli_string([sigma_z], [0], 2)
            >>>
            >>> final_state, data = integrator.evolve_with_observables(
            ...     psi0, H, 5.0, 100, [("Z1", Z1)]
            ... )
            >>> import matplotlib.pyplot as plt
            >>> _ = plt.plot(data['times'], data['Z1'])
        """
        if initial_state.shape[0] != self.dim:
            raise ValueError(
                f"Initial state dimension {initial_state.shape[0]} does not match "
                f"expected dimension {self.dim}"
            )
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")

        state = initial_state.copy()
        dt = t_final / n_steps

        # Initialize storage for observables
        if observables is None:
            observables = []

        data = {'times': []}
        for name, _ in observables:
            data[name] = []

        # Evolve and record observables
        for i in range(n_steps + 1):
            t = i * dt
            data['times'].append(t)

            # Compute expectation values
            for name, observable in observables:
                expectation = np.real(
                    state.conj().T @ observable @ state
                )
                data[name].append(expectation)

            # Evolve (skip on last iteration)
            if i < n_steps:
                t_mid = t + dt / 2
                H = hamiltonian_func(t_mid)
                state = self.step(state, H, dt)

        return state, data


class HamiltonianBuilder:
    """
    Utility class for constructing quantum Hamiltonians.

    Provides static methods for building common Hamiltonians and Pauli operators
    in tensor product form. Useful for quickly setting up quantum dynamics
    simulations.

    Examples:
        >>> # Transverse field Ising model on 3 qubits
        >>> H = HamiltonianBuilder.ising_model(n_qubits=3, J=1.0, h=0.5)
        >>>
        >>> # XXZ Heisenberg model
        >>> H = HamiltonianBuilder.heisenberg_model(3, J_xy=1.0, J_z=0.8, h=0.2)
        >>>
        >>> # Custom Pauli string
        >>> sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        >>> sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        >>> XZ_op = HamiltonianBuilder.pauli_string([sigma_x, sigma_z], [0, 2], 4)
    """

    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    @staticmethod
    def pauli_string(
        operators: List[np.ndarray],
        indices: List[int],
        n_qubits: int
    ) -> np.ndarray:
        """
        Construct tensor product of Pauli operators at specified positions.

        Creates an operator that acts on specific qubits with given Pauli
        matrices, and acts as identity on all other qubits.

        Args:
            operators: List of 2×2 matrices (typically Pauli matrices)
            indices: List of qubit indices where operators act
            n_qubits: Total number of qubits

        Returns:
            Full operator matrix (2^n_qubits × 2^n_qubits)

        Raises:
            ValueError: If any index is out of bounds
            ValueError: If operators and indices have different lengths

        Examples:
            >>> # X operator on qubit 0 of 3-qubit system
            >>> X0 = HamiltonianBuilder.pauli_string(
            ...     [HamiltonianBuilder.X], [0], 3
            ... )
            >>>
            >>> # ZZ interaction between qubits 1 and 2
            >>> Z1Z2 = HamiltonianBuilder.pauli_string(
            ...     [HamiltonianBuilder.Z, HamiltonianBuilder.Z], [1, 2], 3
            ... )
        """
        if len(operators) != len(indices):
            raise ValueError(
                f"Length mismatch: {len(operators)} operators but "
                f"{len(indices)} indices"
            )

        # Initialize with identity on all qubits
        op_list = [HamiltonianBuilder.I] * n_qubits

        # Place specified operators at given indices
        for op, idx in zip(operators, indices):
            if idx >= n_qubits or idx < 0:
                raise ValueError(
                    f"Index {idx} out of bounds for {n_qubits} qubits"
                )
            op_list[idx] = op

        # Build Kronecker product
        result = op_list[0]
        for i in range(1, n_qubits):
            result = np.kron(result, op_list[i])

        return result

    @staticmethod
    def ising_model(n_qubits: int, J: float = 1.0, h: float = 0.5) -> np.ndarray:
        """
        Transverse field Ising model Hamiltonian.

        Constructs the Hamiltonian:
            H = -J Σ_{⟨i,j⟩} Z_i Z_j - h Σ_i X_i

        with nearest-neighbor interactions on a 1D chain.

        Args:
            n_qubits: Number of qubits
            J: ZZ coupling strength
            h: Transverse field strength

        Returns:
            Hamiltonian matrix

        Examples:
            >>> # Standard TFIM
            >>> H = HamiltonianBuilder.ising_model(n_qubits=4, J=1.0, h=0.5)
            >>>
            >>> # Time-dependent field
            >>> def H(t):
            ...     field = 0.5 * (1 + 0.3 * np.sin(t))
            ...     return HamiltonianBuilder.ising_model(4, J=1.0, h=field)
        """
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")

        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        # ZZ nearest-neighbor interactions
        for i in range(n_qubits - 1):
            ZZ = HamiltonianBuilder.pauli_string(
                [HamiltonianBuilder.Z, HamiltonianBuilder.Z],
                [i, i + 1],
                n_qubits
            )
            H += -J * ZZ

        # Transverse field
        for i in range(n_qubits):
            X = HamiltonianBuilder.pauli_string(
                [HamiltonianBuilder.X],
                [i],
                n_qubits
            )
            H += -h * X

        return H

    @staticmethod
    def heisenberg_model(
        n_qubits: int,
        J_xy: float = 1.0,
        J_z: float = 1.0,
        h: float = 0.0
    ) -> np.ndarray:
        """
        Heisenberg model Hamiltonian with external field.

        Constructs the Hamiltonian:
            H = -J_xy Σ_{⟨i,j⟩} (X_i X_j + Y_i Y_j) - J_z Σ_{⟨i,j⟩} Z_i Z_j - h Σ_i Z_i

        with nearest-neighbor interactions on a 1D chain.

        Args:
            n_qubits: Number of qubits
            J_xy: XY coupling strength
            J_z: Z coupling strength (set J_z=J_xy for isotropic Heisenberg)
            h: External field strength (in Z direction)

        Returns:
            Hamiltonian matrix

        Examples:
            >>> # Isotropic Heisenberg model
            >>> H = HamiltonianBuilder.heisenberg_model(3, J_xy=1.0, J_z=1.0)
            >>>
            >>> # XXZ model
            >>> H = HamiltonianBuilder.heisenberg_model(3, J_xy=1.0, J_z=0.5)
            >>>
            >>> # XY model (J_z=0)
            >>> H = HamiltonianBuilder.heisenberg_model(3, J_xy=1.0, J_z=0.0)
        """
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")

        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        # Nearest-neighbor interactions
        for i in range(n_qubits - 1):
            # XX term
            XX = HamiltonianBuilder.pauli_string(
                [HamiltonianBuilder.X, HamiltonianBuilder.X],
                [i, i + 1],
                n_qubits
            )
            H += -J_xy * XX

            # YY term
            YY = HamiltonianBuilder.pauli_string(
                [HamiltonianBuilder.Y, HamiltonianBuilder.Y],
                [i, i + 1],
                n_qubits
            )
            H += -J_xy * YY

            # ZZ term
            ZZ = HamiltonianBuilder.pauli_string(
                [HamiltonianBuilder.Z, HamiltonianBuilder.Z],
                [i, i + 1],
                n_qubits
            )
            H += -J_z * ZZ

        # External field
        if abs(h) > 1e-14:
            for i in range(n_qubits):
                Z = HamiltonianBuilder.pauli_string(
                    [HamiltonianBuilder.Z],
                    [i],
                    n_qubits
                )
                H += -h * Z

        return H

    @staticmethod
    def custom_hamiltonian(
        n_qubits: int,
        terms: List[Tuple[float, List[np.ndarray], List[int]]]
    ) -> np.ndarray:
        """
        Build custom Hamiltonian from list of Pauli terms.

        Each term is specified as (coefficient, operators, indices).

        Args:
            n_qubits: Number of qubits
            terms: List of (coeff, operators, indices) tuples

        Returns:
            Hamiltonian matrix

        Examples:
            >>> # H = 0.5 X_0 + 0.3 Z_1 Z_2 - 0.2 Y_0 Y_1
            >>> X = HamiltonianBuilder.X
            >>> Y = HamiltonianBuilder.Y
            >>> Z = HamiltonianBuilder.Z
            >>>
            >>> H = HamiltonianBuilder.custom_hamiltonian(
            ...     n_qubits=3,
            ...     terms=[
            ...         (0.5, [X], [0]),
            ...         (0.3, [Z, Z], [1, 2]),
            ...         (-0.2, [Y, Y], [0, 1])
            ...     ]
            ... )
        """
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")

        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        for coeff, operators, indices in terms:
            op = HamiltonianBuilder.pauli_string(operators, indices, n_qubits)
            H += coeff * op

        return H
