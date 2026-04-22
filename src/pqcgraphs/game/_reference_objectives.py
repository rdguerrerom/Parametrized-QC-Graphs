"""
Objective functions for quantum error correction code discovery.

This module defines objective functions used by the Nash game to guide
code discovery toward desired properties (distance, hardware compatibility,
MBQC suitability, etc.).
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Callable
from dataclasses import dataclass


@dataclass
class GameObjective:
    """
    Objective function for code discovery game.

    Attributes:
        name: Human-readable name
        weight: Relative importance (typically 0.0-1.0)
        eval_func: Function that evaluates a graph state
    """
    name: str
    weight: float
    eval_func: Callable


def create_aggressive_distance_objective(min_distance: int = 4):
    """
    Aggressively reward high distance codes.

    Prioritizes distance above all else, with exponential rewards
    for distance and strong penalties for codes below min_distance.

    Args:
        min_distance: Minimum acceptable distance

    Returns:
        Objective function

    Examples:
        >>> obj = create_aggressive_distance_objective(min_distance=5)
        >>> GameObjective(name="High Distance", weight=1.0, eval_func=obj)
    """
    def objective(state) -> float:
        from quantum import EnhancedGraphState
        if not isinstance(state, EnhancedGraphState):
            return -1000.0

        n, k, d = state.code_parameters()

        # Strong penalties
        if d < min_distance:
            return -1000.0 * (min_distance - d)
        if k == 0:
            return -500.0
        if k < 0:
            return -2000.0

        # Exponential rewards for distance
        distance_reward = (d ** 3) * (1 + k/max(n, 1))

        # Bonus for connectivity
        if nx.is_connected(state.graph):
            distance_reward *= 1.5

        # Penalty for excessive edges
        max_edges = n * (n - 1) // 4
        if state.graph.number_of_edges() > max_edges:
            edge_penalty = (state.graph.number_of_edges() - max_edges) * 10
        else:
            edge_penalty = 0

        return distance_reward - edge_penalty

    return objective


def create_hardware_objective():
    """
    Hardware-adapted: low degree + high distance.

    Optimizes for hardware constraints by penalizing high degree vertices
    (which require many physical connections) while maintaining high distance.

    Returns:
        Objective function

    Notes:
        - Penalizes maximum degree (hardest connection to implement)
        - Penalizes average degree (overall complexity)
        - Rewards regular graphs (uniform resource requirements)
    """
    def objective(state) -> float:
        from quantum import EnhancedGraphState
        if not isinstance(state, EnhancedGraphState):
            return -500.0

        n, k, d = state.code_parameters()

        if d < 3:
            return -500.0
        if k == 0:
            return -200.0

        degrees = [state.graph.degree(i) for i in state.graph.nodes()]
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        # Reward distance, penalize high degree
        score = (d ** 2.5) * (1 + 0.5 * k)
        score -= 5 * max_degree  # Penalize maximum degree
        score -= 2 * avg_degree  # Penalize average degree

        # Bonus for regular graphs
        if len(set(degrees)) == 1:
            score *= 1.3

        return score

    return objective


def create_rate_distance_objective():
    """
    Maximize k * d product (rate-distance tradeoff).

    Balances encoding rate (k/n) with error protection (d).
    Particularly useful for finding codes with good rate-distance tradeoff.

    Returns:
        Objective function

    Notes:
        - Maximizes k × d product
        - Bonus for balanced codes (rate ≈ 0.25-0.5)
        - Penalties for k=0 or d<3
    """
    def objective(state) -> float:
        from quantum import EnhancedGraphState
        if not isinstance(state, EnhancedGraphState):
            return -300.0

        n, k, d = state.code_parameters()

        if d < 3:
            return -300.0
        if k <= 0:
            return -150.0

        # Maximize k * d with bonus for balance
        product = k * d
        rate = k / max(n, 1)

        # Bonus for balanced codes (rate near 0.25-0.5)
        if 0.2 <= rate <= 0.5:
            product *= 1.5

        return product * 10

    return objective


def create_graph_state_cluster_objective():
    """
    Optimize for cluster states (good for MBQC).

    Cluster states are a class of graph states particularly suitable for
    measurement-based quantum computation. They typically have:
    - Regular degree distribution
    - Bipartite structure

    Returns:
        Objective function

    References:
        Raussendorf et al., "Measurement-based quantum computation on cluster states"
    """
    def objective(state) -> float:
        from quantum import EnhancedGraphState
        if not isinstance(state, EnhancedGraphState):
            return -300.0

        n, k, d = state.code_parameters()

        if d < 3:
            return -300.0
        if k == 0:
            return -200.0

        # Reward regular degree (cluster states have this)
        degrees = [state.graph.degree(v) for v in state.graph.nodes()]
        degree_variance = np.var(degrees) if degrees else 0
        regularity_bonus = 100.0 / (1.0 + degree_variance)

        # Reward bipartite structure (many cluster states are bipartite)
        try:
            is_bipartite = nx.is_bipartite(state.graph)
            bipartite_bonus = 50.0 if is_bipartite else 0.0
        except:
            bipartite_bonus = 0.0

        base_score = d * d * (1 + k)
        return base_score + regularity_bonus + bipartite_bonus

    return objective


def create_graph_state_surface_like_objective():
    """
    Find graph states with surface-code-like properties.

    Surface codes are the leading candidate for fault-tolerant quantum
    computation. This objective finds graph states with similar properties:
    - 2D embeddable (planar)
    - Uniform degree ≈ 4
    - Balanced k and d

    Returns:
        Objective function

    Notes:
        While true surface codes have specific structure, this finds
        graph states with similar desirable properties.
    """
    def objective(state) -> float:
        from quantum import EnhancedGraphState
        if not isinstance(state, EnhancedGraphState):
            return -500.0

        n, k, d = state.code_parameters()

        if d < 4:
            return -500.0
        if k == 0:
            return -300.0

        # Reward 2D embeddability (like surface codes)
        planar_bonus = 200.0 if state.is_2d_embeddable() else 0.0

        # Reward uniform degree ≈ 4 (like surface codes)
        degrees = [state.graph.degree(v) for v in state.graph.nodes()]
        avg_degree = np.mean(degrees) if degrees else 0
        degree_penalty = abs(avg_degree - 4.0) * 20.0

        # Reward balanced k and d
        balance_score = min(k, d) * 50.0

        base_score = d * d * k
        return base_score + planar_bonus - degree_penalty + balance_score

    return objective


def create_graph_state_connectivity_objective():
    """
    Optimize graph connectivity for robust codes.

    High connectivity makes codes more robust to local errors.
    This objective rewards:
    - High vertex connectivity (hard to cut the graph)
    - High edge connectivity
    - Good distance
    - Efficient edge usage

    Returns:
        Objective function

    Notes:
        Vertex connectivity: minimum number of vertices to remove to disconnect
        Edge connectivity: minimum number of edges to remove to disconnect
    """
    def objective(state) -> float:
        from quantum import EnhancedGraphState
        if not isinstance(state, EnhancedGraphState):
            return -400.0

        n, k, d = state.code_parameters()

        if d < 3:
            return -400.0

        # Reward high vertex connectivity (harder to cut)
        if nx.is_connected(state.graph):
            vertex_conn = nx.node_connectivity(state.graph)
            edge_conn = nx.edge_connectivity(state.graph)
            connectivity_bonus = (vertex_conn + edge_conn) * 30.0
        else:
            connectivity_bonus = -200.0

        # Reward distance
        distance_score = d ** 2.5 * (1 + 0.5 * k)

        # Penalize excessive edges (want efficient codes)
        edge_penalty = max(0, state.graph.number_of_edges() - 2 * n) * 5.0

        return distance_score + connectivity_bonus - edge_penalty

    return objective


def create_custom_objective(
    name: str,
    min_distance: int = 3,
    min_rate: float = 0.0,
    prefer_planar: bool = False,
    prefer_regular: bool = False,
    max_degree: int = None
):
    """
    Create a custom objective with specified preferences.

    Convenience function for creating custom objectives without
    writing a full objective function.

    Args:
        name: Name for the objective
        min_distance: Minimum distance (hard constraint)
        min_rate: Minimum rate k/n (hard constraint)
        prefer_planar: Bonus for planar graphs
        prefer_regular: Bonus for regular graphs
        max_degree: Penalty for degrees above this

    Returns:
        Objective function

    Examples:
        >>> obj = create_custom_objective(
        ...     name="Hardware-friendly",
        ...     min_distance=5,
        ...     prefer_planar=True,
        ...     max_degree=4
        ... )
    """
    def objective(state) -> float:
        from quantum import EnhancedGraphState
        if not isinstance(state, EnhancedGraphState):
            return -500.0

        n, k, d = state.code_parameters()

        # Hard constraints
        if d < min_distance:
            return -1000.0 * (min_distance - d)

        rate = k / max(n, 1)
        if rate < min_rate:
            return -500.0

        # Base score
        score = d ** 2 * (1 + k)

        # Planarity bonus
        if prefer_planar and state.is_2d_embeddable():
            score += 100.0

        # Regularity bonus
        if prefer_regular:
            degrees = [state.graph.degree(v) for v in state.graph.nodes()]
            degree_var = np.var(degrees) if degrees else 0
            score += 50.0 / (1.0 + degree_var)

        # Max degree penalty
        if max_degree is not None:
            degrees = [state.graph.degree(v) for v in state.graph.nodes()]
            if degrees:
                actual_max = max(degrees)
                if actual_max > max_degree:
                    score -= (actual_max - max_degree) * 50.0

        return score

    return objective


# Pre-defined objective configurations
def define_enhanced_objectives():
    """
    Define a comprehensive set of objectives for multi-objective optimization.

    Returns:
        List of GameObjective instances

    Usage:
        >>> objectives = define_enhanced_objectives()
        >>> for obj in objectives:
        ...     print(f"{obj.name}: weight={obj.weight}")
    """
    return [
        GameObjective(
            name="High Distance",
            weight=1.0,
            eval_func=create_aggressive_distance_objective(min_distance=4)
        ),
        GameObjective(
            name="Hardware Compatible",
            weight=0.8,
            eval_func=create_hardware_objective()
        ),
        GameObjective(
            name="Rate-Distance Balance",
            weight=0.9,
            eval_func=create_rate_distance_objective()
        ),
        GameObjective(
            name="MBQC Cluster",
            weight=0.7,
            eval_func=create_graph_state_cluster_objective()
        ),
        GameObjective(
            name="Surface-like",
            weight=0.85,
            eval_func=create_graph_state_surface_like_objective()
        ),
        GameObjective(
            name="High Connectivity",
            weight=0.75,
            eval_func=create_graph_state_connectivity_objective()
        ),
    ]


def define_small_objectives():
    """
    Define a smaller set of objectives for quick testing.

    Returns:
        List of GameObjective instances
    """
    return [
        GameObjective(
            name="High Distance",
            weight=1.0,
            eval_func=create_aggressive_distance_objective(min_distance=3)
        ),
        GameObjective(
            name="Hardware Compatible",
            weight=0.8,
            eval_func=create_hardware_objective()
        ),
    ]


__all__ = [
    'GameObjective',
    'create_aggressive_distance_objective',
    'create_hardware_objective',
    'create_rate_distance_objective',
    'create_graph_state_cluster_objective',
    'create_graph_state_surface_like_objective',
    'create_graph_state_connectivity_objective',
    'create_custom_objective',
    'define_enhanced_objectives',
    'define_small_objectives',
]
