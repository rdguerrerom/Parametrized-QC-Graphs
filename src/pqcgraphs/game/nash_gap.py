"""Per-player Nash gap δ_Nash.

A Nash equilibrium of the weighted-potential game is a circuit topology
such that, for every player P_i, no unilateral deviation by P_i (here:
any move proposed from the move set) strictly improves P_i's own
weighted contribution to the potential while holding the others fixed.

The *per-player gap* for player i at state dag is:

    δ_i(dag) = max_{d ∈ neighbours(dag)}  ( payoff_i(d) − payoff_i(dag) )

and is ≥ 0 by construction (the trivial move "keep dag" gives 0).

The overall Nash gap is

    δ_Nash(dag) = max_i δ_i(dag).

Report this alongside the raw per-player deviations — Tier1.md calls out
that "equilibrium analysis shows which moves were blocked by which player,
explaining the emergent gate structure".

Because the move set is shared across players, we can evaluate it once,
then for each candidate compute each player's payoff delta. All n_players
gaps are then maxima over the same (typically ~30–100) candidate list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..dag.circuit_dag import CircuitDAG
from .players import NashPlayer


@dataclass
class NashGapReport:
    per_player: Dict[str, float]       # δ_i per player name
    delta_nash: float                  # max over players
    best_move_name: str                # the name of the move that attained δ_Nash
    best_player: str                   # which player benefits most from that move
    n_candidates: int                  # how many candidates were evaluated

    def is_equilibrium(self, tol: float = 1e-6) -> bool:
        return self.delta_nash <= tol


def compute_nash_gap(
    players: Sequence[NashPlayer],
    dag: CircuitDAG,
    candidates: Sequence,  # List[Tuple[str, CircuitDAG]]
) -> NashGapReport:
    """Evaluate δ_i for each player over the pre-generated candidate set.

    Candidate cost: each candidate evaluates each player's scorer once. With
    ~50 candidates and 4 players on an RTX 4060 this is ≤ 1 s total for
    n_qubits ≤ 6 circuits.
    """
    base_payoffs = {p.name: p.payoff(dag) for p in players}
    per_player_delta: Dict[str, float] = {p.name: 0.0 for p in players}
    best_move_per_player: Dict[str, str] = {p.name: "(no improving move)" for p in players}

    for move_name, cand in candidates:
        for p in players:
            try:
                candidate_payoff = p.payoff(cand)
            except Exception:  # noqa: BLE001
                # A candidate that breaks a scorer (e.g. stabilizer-entropy
                # limit) is not a valid deviation for THAT player; skip.
                continue
            delta = candidate_payoff - base_payoffs[p.name]
            if delta > per_player_delta[p.name]:
                per_player_delta[p.name] = delta
                best_move_per_player[p.name] = move_name

    if per_player_delta:
        best_player = max(per_player_delta, key=per_player_delta.get)
        delta_nash = per_player_delta[best_player]
        best_move_name = best_move_per_player[best_player]
    else:
        best_player = ""
        delta_nash = 0.0
        best_move_name = ""

    return NashGapReport(
        per_player=per_player_delta,
        delta_nash=delta_nash,
        best_move_name=best_move_name,
        best_player=best_player,
        n_candidates=len(candidates),
    )


__all__ = ["NashGapReport", "compute_nash_gap"]
