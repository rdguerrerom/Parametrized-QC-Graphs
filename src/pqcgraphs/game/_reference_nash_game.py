"""
Nash equilibrium game for quantum error correction code discovery.

This module implements the code discovery algorithm using Nash equilibrium
game dynamics with simulated annealing and population-based search.
"""
from __future__ import annotations
import numpy as np
import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from .objectives import GameObjective


@dataclass
class StrategySnapshot:
    """
    Snapshot of game state at a particular iteration.

    Attributes:
        iteration: Iteration number
        graph_edges: List of edges in the graph
        code_params: (n, k, d) code parameters
        objectives: Dictionary mapping objective names to values
        total_reward: Combined reward across all objectives
        nash_gap: Distance from Nash equilibrium
        temperature: Simulated annealing temperature
        best_reward: Best reward seen so far
    """
    iteration: int
    graph_edges: List[Tuple[int, int]]
    code_params: Tuple[int, int, int]
    objectives: Dict[str, float]
    total_reward: float
    nash_gap: float = 0.0
    temperature: float = 1.0
    best_reward: float = 0.0


class EnhancedNashGame:
    """
    Enhanced Nash equilibrium game with:
    - Simulated annealing for exploration
    - Edge addition AND removal
    - Population-based diversity
    - Proper Nash gap calculation

    The game discovers quantum error correction codes by treating code
    design as a multi-player game where players optimize different objectives.

    Attributes:
        n_qubits: Number of qubits in the code
        objectives: List of GameObjective instances
        n_players: Number of players (typically 2)
        hardware_topology: Hardware constraint ('unconstrained' or '2d_grid')
        population_size: Number of candidate solutions maintained
        state: Current best state
        best_state: Best state found so far
        history: List of StrategySnapshot instances

    Examples:
        >>> from game.objectives import define_small_objectives
        >>> objectives = define_small_objectives()
        >>> game = EnhancedNashGame(n_qubits=5, objectives=objectives)
        >>> best_code = game.enhanced_best_response_dynamics(max_iterations=50)
        >>> n, k, d = best_code.code_parameters()
        >>> print(f"Found code: [[{n}, {k}, {d}]]")
    """

    def __init__(
        self,
        n_qubits: int,
        objectives: List[GameObjective],
        n_players: int = 2,
        hardware_topology: str = 'unconstrained',
        population_size: int = 5,
        target_distance: int = None,
        distance_range: List[int] = None
    ):
        """
        Initialize Nash game.

        Args:
            n_qubits: Number of qubits in the code
            objectives: List of GameObjective instances
            n_players: Number of players in the game
            hardware_topology: 'unconstrained' or '2d_grid'
            population_size: Number of candidate solutions to maintain
            target_distance: If specified, only accept codes with exactly this distance
            distance_range: If specified, only accept codes with distance in this list
        """
        self.n_qubits = n_qubits
        self.objectives = objectives
        self.n_players = n_players
        self.hardware_topology = hardware_topology
        self.population_size = population_size
        self.target_distance = target_distance
        self.distance_range = distance_range

        # Initialize population
        self.population = [self._initialize_state() for _ in range(population_size)]
        self.state = self.population[0]

        self.history: List[StrategySnapshot] = []
        self.iteration = 0
        self.best_state = self.state.copy()
        self.best_reward = float('-inf')

        # Annealing parameters
        self.temperature = 2.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.1

    def _initialize_state(self):
        """Create diverse initial states."""
        from quantum import EnhancedGraphState

        state = EnhancedGraphState(self.n_qubits)

        # Random strategy: ring, star, random, or grid
        strategy = random.choice(['ring', 'star', 'random', 'grid'])

        if strategy == 'ring':
            for i in range(self.n_qubits):
                u, v = i, (i + 1) % self.n_qubits
                if self._check_hardware_constraint(u, v):
                    try:
                        state.add_edge(u, v)
                    except:
                        pass

        elif strategy == 'star':
            center = self.n_qubits // 2
            for i in range(self.n_qubits):
                if i != center and self._check_hardware_constraint(center, i):
                    try:
                        state.add_edge(center, i)
                    except:
                        pass

        elif strategy == 'grid':
            grid_size = int(np.ceil(np.sqrt(self.n_qubits)))
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    row_i, col_i = i // grid_size, i % grid_size
                    row_j, col_j = j // grid_size, j % grid_size
                    if abs(row_i - row_j) + abs(col_i - col_j) == 1:
                        if self._check_hardware_constraint(i, j):
                            try:
                                state.add_edge(i, j)
                            except:
                                pass

        else:  # random
            n_edges = random.randint(self.n_qubits, self.n_qubits * 2)
            for _ in range(n_edges):
                u, v = random.sample(range(self.n_qubits), 2)
                if self._check_hardware_constraint(u, v):
                    try:
                        state.add_edge(u, v)
                    except:
                        pass

        return state

    def _check_hardware_constraint(self, u: int, v: int) -> bool:
        """
        Check if edge (u,v) satisfies hardware constraints.

        Args:
            u: First vertex
            v: Second vertex

        Returns:
            True if edge is allowed by hardware topology
        """
        if self.hardware_topology == 'unconstrained':
            return True
        elif self.hardware_topology == '2d_grid':
            grid_size = int(np.ceil(np.sqrt(self.n_qubits)))
            row_u, col_u = u // grid_size, u % grid_size
            row_v, col_v = v // grid_size, v % grid_size
            # Allow nearest and next-nearest neighbors
            return abs(row_u - row_v) + abs(col_u - col_v) <= 2
        return True

    def _check_distance_constraint(self, distance: int) -> bool:
        """
        Check if distance satisfies the configured constraints.

        Args:
            distance: The code distance to check

        Returns:
            True if distance meets constraints, False otherwise
        """
        # If target_distance is specified, check exact match
        if self.target_distance is not None:
            return distance == self.target_distance

        # If distance_range is specified, check if in range
        if self.distance_range is not None:
            return distance in self.distance_range

        # No constraints specified, accept all
        return True

    def _evaluate_state(self, state) -> float:
        """
        Evaluate a graph state using all objectives.

        Args:
            state: EnhancedGraphState to evaluate

        Returns:
            Weighted sum of objective values
        """
        total = 0.0
        for obj in self.objectives:
            try:
                value = obj.eval_func(state)
                total += obj.weight * value
            except Exception as e:
                # Penalize states that cause evaluation errors
                total += -500
        return total

    def _generate_candidate_moves(self, state, n_candidates: int = 150):
        """
        Generate diverse candidate moves: add, remove, swap edges.

        Args:
            state: Current EnhancedGraphState
            n_candidates: Number of candidate moves to generate

        Returns:
            List of (move_type, move_data, new_state, reward) tuples
        """
        candidates = []

        # Edge additions
        available_edges = [
            (u, v) for u in range(self.n_qubits)
            for v in range(u+1, self.n_qubits)
            if not state.graph.has_edge(u, v)
            and self._check_hardware_constraint(u, v)
        ]

        if available_edges:
            for edge in random.sample(
                available_edges,
                min(n_candidates // 2, len(available_edges))
            ):
                new_state = state.copy()
                try:
                    new_state.add_edge(*edge)
                    reward = self._evaluate_state(new_state)
                    candidates.append(('add', edge, new_state, reward))
                except:
                    pass

        # Edge removals
        existing_edges = list(state.graph.edges())
        if existing_edges:
            for edge in random.sample(
                existing_edges,
                min(n_candidates // 4, len(existing_edges))
            ):
                new_state = state.copy()
                try:
                    new_state.remove_edge(*edge)
                    reward = self._evaluate_state(new_state)
                    candidates.append(('remove', edge, new_state, reward))
                except:
                    pass

        # Edge swaps (remove one, add another)
        if existing_edges and available_edges:
            n_swaps = min(n_candidates // 4, len(existing_edges))
            for _ in range(n_swaps):
                remove_edge = random.choice(existing_edges)
                add_edge = random.choice(available_edges)
                new_state = state.copy()
                try:
                    new_state.remove_edge(*remove_edge)
                    new_state.add_edge(*add_edge)
                    reward = self._evaluate_state(new_state)
                    candidates.append(('swap', (remove_edge, add_edge), new_state, reward))
                except:
                    pass

        return candidates

    def _accept_move(self, current_reward: float, new_reward: float) -> bool:
        """
        Simulated annealing acceptance criterion.

        Always accepts improving moves. Accepts worse moves with
        probability exp(Δ/T) where Δ is reward change and T is temperature.

        Args:
            current_reward: Current state reward
            new_reward: Proposed state reward

        Returns:
            True if move should be accepted
        """
        if new_reward > current_reward:
            return True

        # Accept worse moves with probability based on temperature
        delta = new_reward - current_reward
        probability = np.exp(delta / self.temperature)
        return random.random() < probability

    def enhanced_best_response_dynamics(
        self,
        max_iterations: int = 50
    ):
        """
        Run enhanced Nash dynamics with exploration.

        Uses population-based search with simulated annealing to discover
        high-quality quantum error correction codes.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            EnhancedGraphState representing best code found

        Notes:
            - Maintains population of diverse solutions
            - Uses simulated annealing for exploration
            - Cools temperature over time
            - Early stopping if converged to high-quality solution
        """
        print(f"    Starting enhanced Nash equilibrium search...")
        print(f"    Population size: {self.population_size}, Temperature: {self.temperature:.2f}")

        # Print distance filtering info
        if self.target_distance is not None:
            print(f"    Distance filter: target_distance = {self.target_distance}")
        elif self.distance_range is not None:
            print(f"    Distance filter: distance_range = {self.distance_range}")

        for iteration in range(max_iterations):
            self.iteration = iteration

            # Evolve each member of population
            for pop_idx, state in enumerate(self.population):
                current_reward = self._evaluate_state(state)

                # Generate and evaluate candidates (reduced for GPU performance)
                candidates = self._generate_candidate_moves(state, n_candidates=10)

                if candidates:
                    # Sort by reward
                    candidates.sort(key=lambda x: x[3], reverse=True)

                    # Select move (best or accept with annealing)
                    best_move = candidates[0]
                    move_type, move_data, new_state, new_reward = best_move

                    # Accept move based on annealing
                    if self._accept_move(current_reward, new_reward):
                        self.population[pop_idx] = new_state

                        # Update global best (with distance filtering)
                        if new_reward > self.best_reward:
                            n, k, d = new_state.code_parameters()

                            if self._check_distance_constraint(d):
                                self.best_reward = new_reward
                                self.best_state = new_state.copy()
                                print(f"      ** New best at iteration {iteration}: [[{n},{k},{d}]], reward={new_reward:.1f}")
                            else:
                                # Log rejected codes
                                if self.target_distance is not None:
                                    print(f"      Rejected (d={d}, target={self.target_distance}): [[{n},{k},{d}]], reward={new_reward:.1f}")
                                elif self.distance_range is not None:
                                    print(f"      Rejected (d={d} not in {self.distance_range}): [[{n},{k},{d}]], reward={new_reward:.1f}")

            # Cool down temperature
            self.temperature = max(
                self.min_temperature,
                self.temperature * self.cooling_rate
            )

            # Use best from population as current state
            population_rewards = [self._evaluate_state(s) for s in self.population]
            best_pop_idx = np.argmax(population_rewards)
            self.state = self.population[best_pop_idx]

            # Calculate Nash gap (difference from best possible single move)
            current_reward = population_rewards[best_pop_idx]
            candidates = self._generate_candidate_moves(self.state, n_candidates=10)
            if candidates:
                best_single_move_reward = max(c[3] for c in candidates)
                nash_gap = abs(best_single_move_reward - current_reward)
            else:
                nash_gap = 0.0

            # Save snapshot
            n, k, d = self.state.code_parameters()
            obj_values = {obj.name: obj.eval_func(self.state) for obj in self.objectives}

            snapshot = StrategySnapshot(
                iteration=iteration,
                graph_edges=list(self.state.graph.edges()),
                code_params=(n, k, d),
                objectives=obj_values,
                total_reward=current_reward,
                nash_gap=nash_gap,
                temperature=self.temperature,
                best_reward=self.best_reward
            )
            self.history.append(snapshot)

            # Progress report
            if iteration % 10 == 0:
                print(f"      Iteration {iteration}: best [[{n},{k},{d}]], "
                      f"reward={current_reward:.1f}, temp={self.temperature:.3f}, gap={nash_gap:.2f}")

            # Early stopping if converged and quality is good
            if nash_gap < 0.5 and d >= 4 and iteration > 20:
                print(f"      Converged at iteration {iteration}")
                break

        print(f"    Final best: [[{self.best_state.code_parameters()}]], reward={self.best_reward:.1f}")
        return self.best_state


__all__ = [
    'StrategySnapshot',
    'EnhancedNashGame',
]
