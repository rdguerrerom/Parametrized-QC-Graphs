"""Move generators for the Tier-1 Nash best-response loop.

Each move type receives the current `CircuitDAG` + the target `Topology`
and returns a list of (move_name, candidate_dag) pairs. The caller (the
Nash engine) evaluates the potential on each candidate and chooses
according to the simulated-annealing criterion.

Move set (per Tier1.md sec. 1):
  - add_gate    : append a new gate
  - remove_gate : drop an existing op
  - retype_gate : swap gate type (arity-preserving)
  - rewire_gate : move an op to a different qubit pair
  - perturb_theta : Gaussian perturbation of a parameter

We intentionally generate a bounded sample of each move type to keep the
candidate list manageable; for n_qubits = 6 the full move space is already
> 500 per iteration.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from ..dag.circuit_dag import CircuitDAG
from ..dag.gate_types import GATE_SPECS, gate_spec
from ..dag.topology import Topology


Move = Tuple[str, CircuitDAG]


def _random_theta(rng: random.Random) -> float:
    return rng.uniform(0.0, 2.0 * np.pi)


def add_gate_candidates(
    dag: CircuitDAG,
    topology: Topology,
    *,
    n_per_type: int = 2,
    allow_non_native: bool = True,
    rng: Optional[random.Random] = None,
) -> List[Move]:
    """Propose new-gate insertions. Both native and non-native moves
    are generated; f4 will penalise the latter.
    """
    rng = rng or random.Random()
    out: List[Move] = []

    for gate_name, spec in GATE_SPECS.items():
        # Limit per-type population
        for _ in range(n_per_type):
            if spec.arity == 1:
                q = rng.randrange(dag.n_qubits)
                qubits = (q,)
            else:
                # Prefer native pairs, but occasionally propose non-native
                if allow_non_native and rng.random() < 0.2:
                    u = rng.randrange(dag.n_qubits)
                    v = rng.randrange(dag.n_qubits)
                    if u == v:
                        continue
                    qubits = (u, v)
                else:
                    pairs = list(topology.allowed_pairs())
                    if not pairs:
                        continue
                    u, v = rng.choice(pairs)
                    # allow both orderings
                    if rng.random() < 0.5:
                        u, v = v, u
                    qubits = (u, v)
            cand = dag.copy()
            try:
                theta = _random_theta(rng) if spec.is_parametric else None
                cand.append_gate(gate_name, qubits, theta=theta)
            except ValueError:
                continue
            out.append((f"add:{gate_name}{qubits}", cand))
    return out


def remove_gate_candidates(
    dag: CircuitDAG,
    *,
    max_candidates: int = 10,
    rng: Optional[random.Random] = None,
) -> List[Move]:
    rng = rng or random.Random()
    if dag.n_ops == 0:
        return []
    ops = list(dag.op_ids)
    rng.shuffle(ops)
    out: List[Move] = []
    for op_id in ops[:max_candidates]:
        cand = dag.copy()
        try:
            cand.remove_op(op_id)
        except (KeyError, ValueError):
            continue
        out.append((f"remove:{op_id}", cand))
    return out


def retype_gate_candidates(
    dag: CircuitDAG,
    *,
    max_candidates: int = 10,
    rng: Optional[random.Random] = None,
) -> List[Move]:
    rng = rng or random.Random()
    if dag.n_ops == 0:
        return []
    out: List[Move] = []
    ops = list(dag.ops())
    rng.shuffle(ops)
    for op in ops[:max_candidates]:
        for candidate_name, spec in GATE_SPECS.items():
            if candidate_name == op.gate_name:
                continue
            if spec.arity != len(op.qubits):
                continue
            cand = dag.copy()
            try:
                theta = _random_theta(rng) if spec.is_parametric else None
                cand.retype_op(op.node_id, candidate_name, new_theta=theta)
            except ValueError:
                continue
            out.append((f"retype:{op.node_id}->{candidate_name}", cand))
            break  # one swap per op per batch
    return out


def rewire_gate_candidates(
    dag: CircuitDAG,
    topology: Topology,
    *,
    max_candidates: int = 10,
    rng: Optional[random.Random] = None,
) -> List[Move]:
    rng = rng or random.Random()
    if dag.n_ops == 0:
        return []
    out: List[Move] = []
    ops = [op for op in dag.ops() if len(op.qubits) == 2]
    rng.shuffle(ops)
    allowed = list(topology.allowed_pairs())
    for op in ops[:max_candidates]:
        if not allowed:
            break
        new_pair = rng.choice(allowed)
        if set(new_pair) == set(op.qubits):
            continue
        cand = dag.copy()
        try:
            cand.rewire_op(op.node_id, new_pair)
        except ValueError:
            continue
        out.append((f"rewire:{op.node_id}->{new_pair}", cand))
    return out


def perturb_theta_candidates(
    dag: CircuitDAG,
    *,
    max_candidates: int = 12,
    sigma: float = 0.3,
    rng: Optional[random.Random] = None,
) -> List[Move]:
    """Theta-only perturbations.

    Critically: these candidates share the *exact* structure of `dag` (same
    gate schedule, same n_params), so every downstream JIT-cached scorer
    (QFIM, magic, performance) hits a warm trace. This is the cheap
    move type and should dominate the Nash candidate budget.

    To maximise diversity with minimal cost we allow sampling >1 perturbation
    per parameter: if n_params < max_candidates we wrap around.
    """
    rng = rng or random.Random()
    if dag.n_params == 0:
        return []
    out: List[Move] = []
    for _ in range(max_candidates):
        idx = rng.randrange(dag.n_params)
        cand = dag.copy()
        delta = rng.gauss(0.0, sigma)
        try:
            cand.perturb_theta(idx, delta)
        except (IndexError, ValueError):
            continue
        out.append((f"perturb:θ[{idx}]+={delta:+.3f}", cand))
    return out


def bounded_candidates(
    dag: CircuitDAG,
    topology: Topology,
    *,
    n_theta: int = 12,
    n_add: int = 4,
    n_remove: int = 2,
    n_retype: int = 2,
    n_rewire: int = 2,
    rng: Optional[random.Random] = None,
) -> List[Move]:
    """Bounded candidate batch biased toward structure-preserving moves.

    Why this is the default for the Nash engine:

    - Each *structurally unique* DAG triggers a fresh JIT trace in the GPU
      objectives (QFIM, magic, performance share a structural cache key).
      First-time JIT compilation on the RTX 4060 is ~5–15 s per structure.
    - θ-perturb moves preserve the structure exactly, so all n_theta
      candidates hit the same warm trace. They should therefore dominate
      the budget.
    - Structural moves (add/remove/retype/rewire) each introduce a fresh
      structure; we cap each type at a small number (2–3) per iteration so
      at most ~8 new JIT traces per step instead of ~60.

    The totals below (default 20 candidates, 60% θ-perturb) empirically give
    ~3–5× step-time improvement for n_qubits ≤ 6 circuits with ≥ 5 params.
    """
    rng = rng or random.Random()
    out: List[Move] = []

    # Adaptive re-allocation: if there are no parameters to perturb, all
    # n_theta slots would be wasted. Redirect them to add_gate so the search
    # can actually grow the circuit from empty. Once parameters exist, the
    # full n_theta budget resumes and keeps the JIT caches warm.
    if dag.n_params == 0:
        effective_n_add = max(n_add, n_theta // 2)
        effective_n_theta = 0
    else:
        effective_n_add = n_add
        effective_n_theta = n_theta

    out.extend(perturb_theta_candidates(dag, max_candidates=effective_n_theta, rng=rng))
    # Generate more add candidates than we need and sample down, so we cover
    # a wider swath of gate types rather than the first few in GATE_SPECS.
    add_pool = add_gate_candidates(dag, topology, n_per_type=1, rng=rng)
    rng.shuffle(add_pool)
    out.extend(add_pool[:effective_n_add])
    out.extend(remove_gate_candidates(dag, max_candidates=n_remove, rng=rng))
    out.extend(retype_gate_candidates(dag, max_candidates=n_retype, rng=rng))
    out.extend(rewire_gate_candidates(dag, topology, max_candidates=n_rewire, rng=rng))
    return out


def all_candidates(
    dag: CircuitDAG,
    topology: Topology,
    *,
    rng: Optional[random.Random] = None,
) -> List[Move]:
    """Concatenate all move types (legacy unrestricted batch).

    Use `bounded_candidates` instead in performance-sensitive loops — this
    function exists for completeness but will generate ~50–70 candidates per
    call, each structural one triggering a JIT trace if not already cached.
    """
    rng = rng or random.Random()
    return (
        add_gate_candidates(dag, topology, rng=rng)
        + remove_gate_candidates(dag, rng=rng)
        + retype_gate_candidates(dag, rng=rng)
        + rewire_gate_candidates(dag, topology, rng=rng)
        + perturb_theta_candidates(dag, rng=rng)
    )


__all__ = [
    "add_gate_candidates",
    "remove_gate_candidates",
    "retype_gate_candidates",
    "rewire_gate_candidates",
    "perturb_theta_candidates",
    "bounded_candidates",
    "all_candidates",
]
