"""
Stabilizer group representation for quantum error correction codes.

This module provides the StabilizerGroup class for representing and analyzing
stabilizer codes through their generator sets.
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from .pauli import PauliOperator
from .graph_state import EnhancedGraphState


class StabilizerGroup:
    """
    Stabilizer group representation with comprehensive analysis capabilities.

    A stabilizer group is defined by a set of commuting Pauli operators
    (generators) that stabilize a quantum error correction code.

    Attributes:
        n_qubits: Number of qubits
        generators: List of stabilizer generators (Pauli operators)
        logical_operators: Dictionary of logical X and Z operators

    Examples:
        >>> sg = StabilizerGroup(5)
        >>> sg.add_generator(PauliOperator.from_string("XZZII"))
        >>> n, k, d = sg.code_parameters()
    """

    def __init__(self, n_qubits: int):
        """
        Initialize stabilizer group.

        Args:
            n_qubits: Number of qubits in the code
        """
        self.n_qubits = n_qubits
        self.generators: List[PauliOperator] = []
        self.logical_operators: Dict[str, List[PauliOperator]] = {'X': [], 'Z': []}
        self._source_graph_state: Optional[EnhancedGraphState] = None

    def add_generator(self, pauli: PauliOperator) -> bool:
        """
        Add a stabilizer generator with validation.

        Validates that:
        1. Number of qubits matches
        2. Phase is 0 (+1 eigenvalue)
        3. Commutes with all existing generators
        4. Not already in the generator list

        Args:
            pauli: PauliOperator to add as generator

        Returns:
            True if generator was added, False otherwise

        Raises:
            ValueError: If number of qubits doesn't match
        """
        if pauli.n_qubits != self.n_qubits:
            raise ValueError(
                f"Pauli has {pauli.n_qubits} qubits, expected {self.n_qubits}"
            )

        # Must have +1 eigenvalue (phase = 0)
        if pauli.phase != 0:
            return False

        # Reject -I (phase = 2, all identity)
        if all(op == 'I' for op in pauli.operators) and pauli.phase == 2:
            return False

        # Must commute with all existing generators
        for gen in self.generators:
            if not pauli.commutes_with(gen):
                return False

        # Don't add duplicates
        if pauli in self.generators:
            return False

        self.generators.append(pauli)
        return True

    def from_graph_state(self, graph_state: EnhancedGraphState) -> None:
        """
        Initialize stabilizer group from a graph state.

        Extracts stabilizer generators and logical operators from the
        graph state structure.

        Args:
            graph_state: EnhancedGraphState to convert
        """
        self.generators = []
        self.n_qubits = graph_state.n_vertices
        self._source_graph_state = graph_state

        # Add all stabilizers
        for stab in graph_state.stabilizers:
            self.add_generator(stab)

        # Get logical operators if they exist
        self.logical_operators = graph_state._find_logical_operators()

    def validate(self) -> bool:
        """
        Validate stabilizer group properties.

        Checks that:
        1. All generators mutually commute
        2. All have eigenvalue +1 (phase = 0)
        3. At most n generators

        Returns:
            True if all validation checks pass
        """
        # Check all generators commute
        for i, gen1 in enumerate(self.generators):
            for gen2 in self.generators[i+1:]:
                if not gen1.commutes_with(gen2):
                    return False

        # Check all have eigenvalue +1
        for gen in self.generators:
            if gen.phase != 0:
                return False

        # Check at most n generators
        if len(self.generators) > self.n_qubits:
            return False

        return True

    def code_parameters(self) -> Tuple[int, int, int]:
        """
        Calculate quantum error correction code parameters.

        Returns:
            Tuple of (n, k, d) where:
            - n: number of physical qubits
            - k: number of logical qubits (n - number of generators)
            - d: code distance (minimum weight of generators, simplified)
        """
        n = self.n_qubits
        k = n - len(self.generators)
        d = min(gen.weight() for gen in self.generators) if self.generators else 1
        return (n, k, d)

    def get_latex_report(self) -> str:
        """
        Generate comprehensive LaTeX report for publication.

        Includes code parameters, stabilizer generators, logical operators,
        weight distribution, and generator matrix.

        Returns:
            LaTeX formatted string
        """
        n, k, d = self.code_parameters()

        latex = []
        latex.append("% Stabilizer Code Report")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\caption{Stabilizer Code Properties}")
        latex.append("\\begin{tabular}{|l|c|}")
        latex.append("\\hline")
        latex.append("\\textbf{Property} & \\textbf{Value} \\\\")
        latex.append("\\hline")
        latex.append(f"Code parameters $[[n, k, d]]$ & $[[{n}, {k}, {d}]]$ \\\\")
        latex.append(f"Number of generators & {len(self.generators)} \\\\")
        latex.append(f"Code rate $k/n$ & ${k}/{n} = {k/n:.4f}$ \\\\")
        latex.append(f"Relative distance $d/n$ & ${d}/{n} = {d/n:.4f}$ \\\\")
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")

        # Stabilizer generators
        latex.append("\\subsection{Stabilizer Generators}")
        latex.append("\\begin{align}")
        for i, gen in enumerate(self.generators):
            pauli_str = self._format_pauli_latex(gen)
            latex.append(f"S_{{{i+1}}} &= {pauli_str} \\\\")
        latex.append("\\end{align}")
        latex.append("")

        # Logical operators if k > 0
        if k > 0:
            latex.append("\\subsection{Logical Operators}")
            latex.append("\\begin{align}")

            for i, x_op in enumerate(self.logical_operators.get('X', [])):
                pauli_str = self._format_pauli_latex(x_op)
                latex.append(f"\\bar{{X}}_{{{i+1}}} &= {pauli_str} \\\\")

            for i, z_op in enumerate(self.logical_operators.get('Z', [])):
                pauli_str = self._format_pauli_latex(z_op)
                latex.append(f"\\bar{{Z}}_{{{i+1}}} &= {pauli_str} \\\\")

            latex.append("\\end{align}")
            latex.append("")

        # Weight distribution
        weights = [gen.weight() for gen in self.generators]
        weight_dist = {}
        for w in weights:
            weight_dist[w] = weight_dist.get(w, 0) + 1

        latex.append("\\subsection{Weight Distribution}")
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append("\\begin{tabular}{|c|c|c|}")
        latex.append("\\hline")
        latex.append("\\textbf{Weight} & \\textbf{Count} & \\textbf{Percentage} \\\\")
        latex.append("\\hline")

        for weight in sorted(weight_dist.keys()):
            count = weight_dist[weight]
            percentage = 100 * count / len(self.generators) if self.generators else 0
            latex.append(f"{weight} & {count} & {percentage:.1f}\\% \\\\")

        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")

        # Generator matrix (for small codes)
        if len(self.generators) <= 10:
            latex.append("\\subsection{Generator Matrix}")
            latex.append("\\begin{equation}")
            latex.append("G = \\begin{pmatrix}")

            for gen in self.generators:
                row = []
                for op in gen.operators:
                    if op == 'I':
                        row.append('I')
                    else:
                        row.append(op)
                latex.append(" & ".join(row) + " \\\\")

            latex.append("\\end{pmatrix}")
            latex.append("\\end{equation}")

        return "\n".join(latex)

    def _format_pauli_latex(self, pauli: PauliOperator) -> str:
        """
        Format a Pauli operator for LaTeX.

        Args:
            pauli: PauliOperator to format

        Returns:
            LaTeX formatted string
        """
        result = []
        for i, op in enumerate(pauli.operators):
            if op != 'I':
                result.append(f"{op}_{{{i+1}}}")

        if not result:
            return "I"

        return " ".join(result)

    def get_research_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive research summary as dictionary.

        Useful for programmatic analysis and data collection.

        Returns:
            Dictionary containing code parameters, statistics, and properties
        """
        n, k, d = self.code_parameters()

        weights = [gen.weight() for gen in self.generators]

        summary = {
            'code_parameters': {'n': n, 'k': k, 'd': d},
            'code_rate': k/n if n > 0 else 0,
            'relative_distance': d/n if n > 0 else 0,
            'n_generators': len(self.generators),
            'weight_statistics': {
                'min': min(weights) if weights else 0,
                'max': max(weights) if weights else 0,
                'mean': np.mean(weights) if weights else 0,
                'std': np.std(weights) if weights else 0
            },
            'weight_distribution': dict(zip(*np.unique(weights, return_counts=True))) if weights else {},
            'has_logical_qubits': k > 0,
            'n_logical_x_operators': len(self.logical_operators.get('X', [])),
            'n_logical_z_operators': len(self.logical_operators.get('Z', []))
        }

        return summary

    def get_source_graph_state(self) -> Optional[EnhancedGraphState]:
        """
        Get the source graph state if this stabilizer group was created from one.

        Returns:
            EnhancedGraphState or None
        """
        return self._source_graph_state


def graph_state_to_stabilizer_group(state: EnhancedGraphState) -> StabilizerGroup:
    """
    Convert an EnhancedGraphState into a StabilizerGroup.

    Takes the vertex-stabilizer set {K_v} from the graph state and creates
    a StabilizerGroup with those generators. Preserves reference to source
    graph state for circuit generation.

    Args:
        state: EnhancedGraphState to convert

    Returns:
        StabilizerGroup initialized from the graph state

    Examples:
        >>> graph = EnhancedGraphState(5)
        >>> graph.add_edge(0, 1)
        >>> sg = graph_state_to_stabilizer_group(graph)
        >>> sg.n_qubits
        5
    """
    sg = StabilizerGroup(n_qubits=state.n_vertices)
    sg.from_graph_state(state)
    return sg
