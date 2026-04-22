"""Reference-state factories for seeding a Nash architecture search.

The Nash potential Φ(dag) depends on the output state, which depends on the
DAG *including* its initial preparation. Starting from the literal |0⟩^n
state is rarely appropriate: for MaxCut/QAOA the canonical reference state
is |+⟩^n (Hadamard on every qubit), and for electronic-structure VQE it is
the Hartree-Fock state.

Starting from |0⟩^n makes several objectives vanish identically — f3
(performance on MaxCut) is zero because ⟨Z_iZ_j⟩|0⟩ = 1, f2 (magic) is
zero because |0⟩ is a stabilizer state, f1 (QFIM rank) is zero because
there are no parameters yet. With all rewards zero and f4 (hardware) only
strictly positive, the empty DAG is the Nash equilibrium regardless of
weights. That result is physically correct but uninformative.

The factories below produce DAGs matching the task at hand so the
potential has a meaningful gradient from iteration 0.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .circuit_dag import CircuitDAG
from .topology import Topology


def plus_layer(n_qubits: int) -> CircuitDAG:
    """Hadamard layer: |+⟩^⊗n = H^⊗n |0⟩^⊗n.

    The canonical QAOA / MaxCut reference state. No parameters, but for
    MaxCut the performance objective f3 is already non-zero (⟨H⟩ = |E|/2)
    from this starting point, giving the Nash search real f3 gradient.
    """
    dag = CircuitDAG(n_qubits)
    for q in range(n_qubits):
        dag.append_gate("h", (q,))
    return dag


def hartree_fock_h2() -> CircuitDAG:
    """Hartree-Fock initial state for the 4-qubit H₂ STO-3G Jordan-Wigner encoding.

    Two lowest spin-orbitals occupied (qubits 0 and 1 under the TensorCircuit
    qubit-0-MSB convention). Matches the state that `hamiltonians.py`
    evaluates to -1.1167 Ha, i.e. the HF energy. VQE starting from here
    measures correlation energy directly.
    """
    dag = CircuitDAG(4)
    dag.append_gate("x", (0,))
    dag.append_gate("x", (1,))
    return dag


def qaoa_warm_layer(
    topology: Topology,
    *,
    gamma_init: float = 0.1,
    beta_init: float = 0.1,
) -> CircuitDAG:
    """QAOA-style ansatz: |+⟩^⊗n → problem layer (rzz per edge) → mixer layer (rx).

    Uses the topology's allowed pairs for the entangling layer. Provides
    both parameterized structure (for f1) and physical entanglement (for
    f2 / f3 on MaxCut-like objectives). The γ / β defaults match the
    small-angle regime that is the natural QAOA warm-start.
    """
    dag = CircuitDAG(topology.n_qubits)
    for q in range(topology.n_qubits):
        dag.append_gate("h", (q,))
    for (u, v) in sorted(topology.pairs):
        dag.append_gate("rzz", (u, v), theta=float(gamma_init))
    for q in range(topology.n_qubits):
        dag.append_gate("rx", (q,), theta=float(beta_init))
    return dag


def hardware_efficient(
    n_qubits: int,
    n_layers: int,
    topology: Topology,
    *,
    seed: int = 0,
) -> CircuitDAG:
    """Hardware-efficient ansatz: alternating ry + rzz layers on native pairs.

    Typical VQE baseline structure. The Nash search can then compress
    (remove), retype, or rewire these gates rather than having to grow the
    circuit from nothing.
    """
    rng = np.random.default_rng(seed)
    dag = CircuitDAG(n_qubits)
    pairs = sorted(topology.pairs)
    for _ in range(n_layers):
        for q in range(n_qubits):
            dag.append_gate("ry", (q,), theta=float(rng.uniform(0.0, 2 * np.pi)))
        for (u, v) in pairs:
            dag.append_gate("rzz", (u, v), theta=float(rng.uniform(0.0, 2 * np.pi)))
    return dag


def empty(n_qubits: int) -> CircuitDAG:
    """Literal |0⟩^n starting point.

    Equivalent to `CircuitDAG(n_qubits)`. Exists here so that callers can
    parameterise over initial-state factories using a uniform signature,
    and so the contrast with reference-state starts is explicit.
    """
    return CircuitDAG(n_qubits)


def uccsd_seed_lih() -> CircuitDAG:
    """Trotterized UCCSD-singlet skeleton for 6-qubit LiH active space.

    Loads the cached gate list from `data/circuits/lih_uccsd_seed.json`
    (produced by `scripts/generate_lih_uccsd_seed.py`) and replays it into
    a CircuitDAG. Includes the HF preparation (X_0, X_1) followed by the
    full Trotter-1 / first-order UCCSD gate sequence. Parametric gates
    carry a small random initial θ; Nash's θ-GD inner loop is expected
    to refine these to the UCCSD amplitude values that recover the
    active-space correlation energy.
    """
    import json as _json
    from pathlib import Path as _Path
    path = _Path(__file__).resolve().parents[3] / "data" / "circuits" / "lih_uccsd_seed.json"
    with open(path) as f:
        payload = _json.load(f)
    dag = CircuitDAG(int(payload["n_qubits"]))
    for (name, qubits, theta) in payload["operations"]:
        qtup = tuple(qubits)
        if theta is None:
            dag.append_gate(name, qtup)
        else:
            dag.append_gate(name, qtup, theta=float(theta))
    return dag


def hartree_fock_lih() -> CircuitDAG:
    """HF reference state for the 6-qubit reduced LiH STO-3G Hamiltonian.

    After frozen-core (Li 1s²) and frozen-virtual (Li 2p_{x,y}) reduction,
    the active space contains 2 electrons in 6 spin-orbitals (Li 2s, Li
    2p_z, H 1s) ordered as (MO₁↑, MO₁↓, MO₄↑, MO₄↓, MO₅↑, MO₅↓) where MO_i
    denote the SCF molecular orbitals sorted by energy. HF fills the two
    lowest active spin-orbitals (MO₁ doubly occupied — the bonding σ of Li
    2s / H 1s). In qubit-0-MSB ordering this is the computational basis
    state |110000⟩: X on qubits 0 and 1.
    """
    dag = CircuitDAG(6)
    dag.append_gate("x", (0,))
    dag.append_gate("x", (1,))
    return dag


__all__ = [
    "plus_layer",
    "hartree_fock_h2",
    "hartree_fock_lih",
    "uccsd_seed_lih",
    "qaoa_warm_layer",
    "hardware_efficient",
    "empty",
]
