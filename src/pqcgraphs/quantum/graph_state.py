"""
Graph state representation for quantum error correction codes.

This module provides the EnhancedGraphState class for representing graph states
with comprehensive code parameter calculation and structural analysis.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import random
import itertools
from typing import List, Tuple, Dict, Optional, Any
from .pauli import PauliOperator


class EnhancedGraphState:
    """
    Enhanced graph state with improved distance calculation.

    A graph state is a quantum state associated with an undirected graph G = (V, E).
    For each vertex v, there is a stabilizer generator K_v = X_v ∏_{u∈N(v)} Z_u,
    where N(v) is the set of neighbors of v.

    Attributes:
        n_vertices: Number of vertices in the graph
        graph: NetworkX graph representing the state
        stabilizers: List of PauliOperator stabilizer generators

    Examples:
        >>> state = EnhancedGraphState(5)
        >>> state.add_edge(0, 1)
        >>> state.add_edge(1, 2)
        >>> n, k, d = state.code_parameters()
    """

    def __init__(self, n_vertices: int):
        """
        Initialize graph state with n_vertices isolated qubits.

        Args:
            n_vertices: Number of qubits/vertices
        """
        self.n_vertices = n_vertices
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n_vertices))
        self.stabilizers = []
        self._distance_cache = {}
        self._update_stabilizers()

    def add_edge(self, u: int, v: int):
        """
        Add an edge to the graph.

        Args:
            u: First vertex index
            v: Second vertex index

        Raises:
            ValueError: If vertices are out of range or equal
        """
        if u >= self.n_vertices or v >= self.n_vertices:
            raise ValueError(f"Vertices must be < {self.n_vertices}")
        if u == v:
            raise ValueError("No self-loops allowed")
        self.graph.add_edge(u, v)
        self._distance_cache.clear()
        self._update_stabilizers()

    def remove_edge(self, u: int, v: int):
        """Remove an edge from the graph."""
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            self._distance_cache.clear()
            self._update_stabilizers()

    def _update_stabilizers(self):
        """Generate stabilizers K_v = X_v ∏_{u∈N(v)} Z_u"""
        self.stabilizers = []
        for vertex in self.graph.nodes():
            ops = ['I'] * self.n_vertices
            ops[vertex] = 'X'
            for neighbor in self.graph.neighbors(vertex):
                ops[neighbor] = 'Z'
            self.stabilizers.append(PauliOperator(self.n_vertices, tuple(ops), 0))

    def _compute_stabilizer_rank(self) -> int:
        """
        Compute the rank of the stabilizer group using binary representation.

        This determines the actual number of independent stabilizers,
        which is used to calculate k = n - rank.

        Returns:
            Rank of the stabilizer group over GF(2)
        """
        if not self.stabilizers:
            return 0

        binary_matrix = []
        for stab in self.stabilizers:
            row = []
            for op in stab.operators:
                if op == 'I':
                    row.extend([0, 0])
                elif op == 'X':
                    row.extend([1, 0])
                elif op == 'Z':
                    row.extend([0, 1])
                elif op == 'Y':
                    row.extend([1, 1])
            binary_matrix.append(row)

        # Compute rank over GF(2) using Gaussian elimination
        matrix = np.array(binary_matrix, dtype=int)
        rank = 0
        rows, cols = matrix.shape

        for col in range(min(rows, cols)):
            for row in range(rank, rows):
                if matrix[row, col] == 1:
                    if row != rank:
                        matrix[[row, rank]] = matrix[[rank, row]]
                    for r in range(rows):
                        if r != rank and matrix[r, col] == 1:
                            matrix[r] = (matrix[r] + matrix[rank]) % 2
                    rank += 1
                    break
        return rank

    def _find_logical_operators(self) -> Dict[str, List[PauliOperator]]:
        """
        Find logical X and Z operators for the code.

        These are operators that commute with all stabilizers but aren't
        in the stabilizer group.

        Returns:
            Dictionary with keys 'X' and 'Z' containing lists of logical operators
        """
        n, k, d = self.code_parameters()

        if k == 0:
            return {'X': [], 'Z': []}

        logical_ops = {'X': [], 'Z': []}

        # This is a simplified search - full implementation would use
        # more sophisticated methods (e.g., symplectic Gaussian elimination)
        max_weight = min(5, self.n_vertices)

        for weight in range(1, max_weight + 1):
            if len(logical_ops['X']) >= k and len(logical_ops['Z']) >= k:
                break

            # Try all Pauli operators of given weight
            for positions in itertools.combinations(range(self.n_vertices), weight):
                for pauli_types in itertools.product(['X', 'Y', 'Z'], repeat=weight):
                    ops = ['I'] * self.n_vertices
                    for pos, p_type in zip(positions, pauli_types):
                        ops[pos] = p_type

                    test_op = PauliOperator(self.n_vertices, tuple(ops), phase=0)

                    # Check if commutes with all stabilizers
                    commutes_with_all = all(test_op.commutes_with(s) for s in self.stabilizers)

                    if commutes_with_all:
                        # Check if it's not in the stabilizer group
                        is_stabilizer = any(
                            test_op.operators == s.operators for s in self.stabilizers
                        )

                        if not is_stabilizer:
                            # Classify as X or Z type based on dominant operator
                            x_count = sum(1 for op in ops if op == 'X')
                            z_count = sum(1 for op in ops if op == 'Z')

                            if x_count > z_count and len(logical_ops['X']) < k:
                                logical_ops['X'].append(test_op)
                            elif z_count >= x_count and len(logical_ops['Z']) < k:
                                logical_ops['Z'].append(test_op)

        self._logical_operators = logical_ops
        return logical_ops

    def code_parameters(self) -> Tuple[int, int, int]:
        """
        Return [n, k, d] code parameters.

        Returns:
            Tuple of (n, k, d) where:
            - n: number of physical qubits
            - k: number of logical qubits (n - rank of stabilizer group)
            - d: code distance (minimum weight of non-trivial logical operator)
        """
        n = self.n_vertices
        rank = self._compute_stabilizer_rank()
        k = n - rank
        d = self._enhanced_distance_estimation()
        return (n, k, d)

    def _stabilizers_to_binary(self) -> np.ndarray:
        """
        Convert stabilizers to binary symplectic form.

        Returns:
            Binary matrix where each row is a stabilizer in symplectic form
        """
        if not self.stabilizers:
            return np.zeros((0, 2 * self.n_vertices), dtype=int)

        binary_matrix = []
        for stab in self.stabilizers:
            row = []
            for op in stab.operators:
                if op == 'I':
                    row.extend([0, 0])
                elif op == 'X':
                    row.extend([1, 0])
                elif op == 'Z':
                    row.extend([0, 1])
                elif op == 'Y':
                    row.extend([1, 1])
            binary_matrix.append(row)

        return np.array(binary_matrix, dtype=int)

    def _limited_logical_operator_search(self, max_weight: int = 6) -> Optional[int]:
        """
        Limited search for logical operators in graph states.

        Only searches up to max_weight for computational efficiency.

        Args:
            max_weight: Maximum operator weight to search

        Returns:
            Weight of minimum-weight logical operator found, or None if none found
        """
        n = self.n_vertices

        for target_weight in range(1, min(max_weight, n) + 1):
            # Try X-type operators first (more common in graph states)
            for positions in itertools.combinations(range(n), target_weight):
                # Try pure X operator
                ops = ['I'] * n
                for pos in positions:
                    ops[pos] = 'X'
                candidate = PauliOperator(n, tuple(ops), phase=0)

                if all(candidate.commutes_with(s) for s in self.stabilizers):
                    if not self._is_trivial_in_stabilizer_group(candidate):
                        return target_weight

                # Try pure Z operator
                ops = ['I'] * n
                for pos in positions:
                    ops[pos] = 'Z'
                candidate = PauliOperator(n, tuple(ops), phase=0)

                if all(candidate.commutes_with(s) for s in self.stabilizers):
                    if not self._is_trivial_in_stabilizer_group(candidate):
                        return target_weight

                # Try Y operators (less common but possible)
                if target_weight <= 3:
                    ops = ['I'] * n
                    for pos in positions:
                        ops[pos] = 'Y'
                    candidate = PauliOperator(n, tuple(ops), phase=0)

                    if all(candidate.commutes_with(s) for s in self.stabilizers):
                        if not self._is_trivial_in_stabilizer_group(candidate):
                            return target_weight

        return None

    def _is_trivial_in_stabilizer_group(self, pauli: PauliOperator) -> bool:
        """
        Quick check if Pauli is in stabilizer group.

        Args:
            pauli: PauliOperator to check

        Returns:
            True if operator is in the stabilizer group
        """
        # Check if it exactly matches any stabilizer
        for s in self.stabilizers:
            if pauli.operators == s.operators:
                return True

        stab_binary = self._stabilizers_to_binary()
        if len(stab_binary) == 0:
            return all(op == 'I' for op in pauli.operators)

        # Convert to binary and check row space
        pauli_binary = []
        for op in pauli.operators:
            if op == 'I':
                pauli_binary.extend([0, 0])
            elif op == 'X':
                pauli_binary.extend([1, 0])
            elif op == 'Z':
                pauli_binary.extend([0, 1])
            elif op == 'Y':
                pauli_binary.extend([1, 1])

        pauli_vec = np.array(pauli_binary, dtype=int)

        # Gaussian elimination check
        augmented = np.vstack([stab_binary, pauli_vec.reshape(1, -1)])
        matrix = augmented.copy()
        rows, cols = matrix.shape
        pivot_row = 0

        for col in range(cols):
            found_pivot = False
            for row in range(pivot_row, rows):
                if matrix[row, col] == 1:
                    if row != pivot_row:
                        matrix[[row, pivot_row]] = matrix[[pivot_row, row]]
                    found_pivot = True
                    break

            if not found_pivot:
                continue

            for row in range(rows):
                if row != pivot_row and matrix[row, col] == 1:
                    matrix[row] = (matrix[row] + matrix[pivot_row]) % 2

            pivot_row += 1
            if pivot_row >= rows:
                break

        return np.all(matrix[-1] == 0)

    def _graph_state_distance_with_logical_qubits(self) -> int:
        """
        For graph states with k > 0, compute distance using graph properties
        and limited Pauli operator search.

        Returns:
            Estimated code distance
        """
        n = self.n_vertices
        rank = self._compute_stabilizer_rank()
        k = n - rank

        # Bound 1: Minimum stabilizer weight (lower bound)
        min_stab_weight = min(s.weight() for s in self.stabilizers) if self.stabilizers else 1

        # Bound 2: Graph connectivity bounds
        if nx.is_connected(self.graph):
            vertex_conn = nx.node_connectivity(self.graph)
            graph_bound = vertex_conn + 1
        else:
            components = list(nx.connected_components(self.graph))
            graph_bound = min(len(c) for c in components) if len(components) > 1 else 2

        # Bound 3: Minimum degree + 1
        degrees = [self.graph.degree(v) for v in self.graph.nodes()]
        min_degree = min(degrees) if degrees else 0
        degree_bound = min_degree + 1

        # Bound 4: Search for small-weight logical operators
        search_bound = self._limited_logical_operator_search(max_weight=min(6, n))

        estimated_distance = max(
            min(min_stab_weight, graph_bound, degree_bound),
            2 if k > 0 else 1
        )

        # If we found a logical operator, that's the true distance
        if search_bound is not None and search_bound < estimated_distance:
            estimated_distance = search_bound

        # Singleton bound: d ≤ n - k + 1
        singleton_bound = n - k + 1
        estimated_distance = min(estimated_distance, singleton_bound)

        return estimated_distance

    def _enhanced_distance_estimation(self) -> int:
        """
        FAST graph-state distance calculation.

        Uses graph properties instead of expensive operator searches.

        Returns:
            Estimated code distance
        """
        if 'distance' in self._distance_cache:
            return self._distance_cache['distance']

        n = self.n_vertices

        # Empty graph
        if self.graph.number_of_edges() == 0:
            self._distance_cache['distance'] = 1
            return 1

        rank = self._compute_stabilizer_rank()
        k = n - rank

        # FAST PATH for k=0 (most common for graph states)
        if k == 0:
            # Distance = minimum stabilizer weight
            # For graph states: weight(K_v) = 1 + degree(v)
            if self.stabilizers:
                distance = min(s.weight() for s in self.stabilizers)
            else:
                distance = 1
            self._distance_cache['distance'] = distance
            return distance

        # FAST PATH for k>0: Use graph bounds without expensive search
        distance = self._fast_graph_bounds()
        self._distance_cache['distance'] = distance
        return distance

    def _fast_graph_bounds(self) -> int:
        """
        Fast distance bounds using graph properties only.

        No expensive Pauli operator searches.

        Returns:
            Conservative distance estimate
        """
        n = self.n_vertices
        rank = self._compute_stabilizer_rank()
        k = n - rank

        # Bound 1: Minimum stabilizer weight
        min_stab_weight = min(s.weight() for s in self.stabilizers) if self.stabilizers else 1

        # Bound 2: Graph connectivity
        if nx.is_connected(self.graph):
            try:
                vertex_conn = nx.node_connectivity(self.graph)
                conn_bound = min(vertex_conn + 1, n)
            except:
                conn_bound = 2
        else:
            components = list(nx.connected_components(self.graph))
            if len(components) > 1:
                conn_bound = min(len(c) for c in components)
            else:
                conn_bound = 2

        # Bound 3: Minimum degree + 1
        degrees = dict(self.graph.degree())
        if degrees:
            min_degree = min(degrees.values())
            degree_bound = min_degree + 1
        else:
            degree_bound = 1

        # Bound 4: Recognize special structures
        structure_bound = self._recognize_graph_structure_distance()

        # Conservative estimate
        distance = min(min_stab_weight, conn_bound, degree_bound)

        if structure_bound is not None:
            distance = max(distance, structure_bound)

        # Singleton bound
        singleton_bound = n - k + 1
        distance = min(distance, singleton_bound)

        # Ensure reasonable
        distance = max(1, min(distance, n))

        return distance

    def _recognize_graph_structure_distance(self) -> Optional[int]:
        """
        Fast recognition of common graph structures with known distances.

        Returns:
            Known distance for recognized structure, or None
        """
        n = self.n_vertices
        m = self.graph.number_of_edges()

        # Complete graph K_n: distance = n
        if m == n * (n - 1) // 2:
            return n

        # Cycle C_n: distance = 2 (for n >= 3)
        if m == n and nx.is_connected(self.graph):
            degrees = list(dict(self.graph.degree()).values())
            if all(d == 2 for d in degrees):
                return 2

        # Path graph: distance = 2
        if m == n - 1 and nx.is_connected(self.graph):
            degrees = list(dict(self.graph.degree()).values())
            if degrees.count(1) == 2 and all(d <= 2 for d in degrees):
                return 2

        # Star graph: distance = 2
        if m == n - 1 and nx.is_connected(self.graph):
            degrees = list(dict(self.graph.degree()).values())
            if max(degrees) == n - 1:
                return 2

        # Grid graphs (2D): distance ≈ 3-4
        if self._is_grid_like():
            return 3

        # Regular graphs with degree d
        degrees = list(dict(self.graph.degree()).values())
        if len(set(degrees)) == 1 and degrees[0] >= 3:
            return min(degrees[0], n // 2)

        return None

    def _is_grid_like(self) -> bool:
        """Fast check if graph resembles a 2D grid."""
        if not nx.is_connected(self.graph):
            return False

        degrees = list(dict(self.graph.degree()).values())
        degree_counts = {d: degrees.count(d) for d in set(degrees)}

        # Grid has mostly degree 4, with some degree 2 or 3 at boundaries
        high_degree = sum(degree_counts.get(d, 0) for d in [3, 4])
        return high_degree >= len(degrees) * 0.6

    def is_mbqc_compatible(self) -> bool:
        """
        Check if graph state is suitable for MBQC (measurement-based QC).

        Criteria:
        1. Connected (enables universal computation)
        2. Sufficient entanglement (degree >= 2 typically)

        Returns:
            True if compatible with MBQC
        """
        if not nx.is_connected(self.graph):
            return False
        avg_degree = np.mean([self.graph.degree(i) for i in self.graph.nodes()])
        return avg_degree >= 2.0 and self.graph.number_of_edges() >= self.n_vertices

    def is_2d_embeddable(self) -> bool:
        """
        Check if graph can be embedded in 2D plane (planarity test).

        Critical for hardware implementations on 2D chip architectures.

        Returns:
            True if graph is planar
        """
        try:
            return nx.check_planarity(self.graph)[0]
        except:
            return False

    def copy(self):
        """Create a deep copy of this graph state."""
        new_state = EnhancedGraphState(self.n_vertices)
        new_state.graph = self.graph.copy()
        new_state._update_stabilizers()
        return new_state

    def verify_commutation(self) -> bool:
        """
        Verify all stabilizers commute with each other.

        Returns:
            True if all stabilizers mutually commute
        """
        for i, stab1 in enumerate(self.stabilizers):
            for stab2 in self.stabilizers[i+1:]:
                if not stab1.commutes_with(stab2):
                    return False
        return True

    def get_circuit_description(self) -> str:
        """
        Describe the circuit that prepares this graph state.

        Returns:
            Human-readable description of preparation circuit
        """
        lines = []
        lines.append(f"Graph State Preparation Circuit ({self.n_vertices} qubits):")
        lines.append("-" * 50)
        lines.append("1. Initialize all qubits in |+⟩ state:")
        lines.append(f"   Apply H to qubits 0-{self.n_vertices-1}")
        lines.append("")
        lines.append("2. Apply CZ gates for edges:")

        if self.graph.number_of_edges() == 0:
            lines.append("   No edges (product state |+⟩^⊗n)")
        else:
            for i, (u, v) in enumerate(self.graph.edges()):
                lines.append(f"   Step {i+1}: CZ between qubit {u} and qubit {v}")

        return "\n".join(lines)

    def describe(self) -> str:
        """
        Full description of the graph state.

        Returns:
            Detailed multi-line description
        """
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"GRAPH STATE: {self.n_vertices} vertices, {self.graph.number_of_edges()} edges")
        lines.append(f"{'='*60}")

        lines.append("\nGraph edges: " + str(list(self.graph.edges())))

        lines.append("\nStabilizer generators (K_v = X_v ∏_u Z_u):")
        for v, stab in enumerate(self.stabilizers):
            neighbors = list(self.graph.neighbors(v))
            neighbor_str = f" (neighbors: {neighbors})" if neighbors else " (isolated)"
            lines.append(f"  K_{v} = {stab}{neighbor_str}")

        lines.append(f"\nAll stabilizers commute: {self.verify_commutation()}")
        lines.append("\n" + self.get_circuit_description())

        return "\n".join(lines)

    def is_bipartite(self) -> bool:
        """
        Check if graph state is bipartite.

        Important for MBQC and photonic implementations.

        Returns:
            True if graph is bipartite
        """
        try:
            return nx.is_bipartite(self.graph)
        except:
            return False

    def check_regularity(self) -> Tuple[bool, float]:
        """
        Check if graph is regular (all vertices have same degree).

        Returns:
            (is_regular, degree_variance)
        """
        degrees = [d for _, d in self.graph.degree()]

        if len(degrees) == 0:
            return False, 0.0

        degree_variance = np.var(degrees)
        is_regular = degree_variance < 0.01

        return is_regular, degree_variance

    def compute_girth(self) -> int:
        """
        Compute girth (length of shortest cycle) of graph.

        Important for LDPC code properties.

        Returns:
            Length of shortest cycle, or inf if no cycles
        """
        try:
            cycles = nx.cycle_basis(self.graph)
            if not cycles:
                return float('inf')
            return min(len(cycle) for cycle in cycles)
        except:
            return 0

    def get_structural_properties(self) -> Dict[str, Any]:
        """
        Get comprehensive structural properties.

        Returns:
            Dictionary with all structural properties
        """
        n, k, d = self.code_parameters()

        properties = {
            'code_params': (n, k, d),
            'n_vertices': n,
            'n_edges': self.graph.number_of_edges(),
            'rate': k / n if n > 0 else 0,
            'relative_distance': d / n if n > 0 else 0,
            'bipartite': self.is_bipartite(),
            'planar': self.is_2d_embeddable(),
            'mbqc_compatible': self.is_mbqc_compatible(),
        }

        regular, degree_var = self.check_regularity()
        properties['regular'] = regular
        properties['degree_variance'] = degree_var

        # Graph statistics
        degrees = [d for _, d in self.graph.degree()]
        properties['min_degree'] = min(degrees) if degrees else 0
        properties['max_degree'] = max(degrees) if degrees else 0
        properties['avg_degree'] = np.mean(degrees) if degrees else 0

        # Connectivity
        if nx.is_connected(self.graph):
            properties['connected'] = True
            properties['vertex_connectivity'] = nx.node_connectivity(self.graph)
            properties['edge_connectivity'] = nx.edge_connectivity(self.graph)
        else:
            properties['connected'] = False
            properties['vertex_connectivity'] = 0
            properties['edge_connectivity'] = 0

        properties['girth'] = self.compute_girth()
        properties['diameter'] = nx.diameter(self.graph) if properties['connected'] else float('inf')

        return properties
