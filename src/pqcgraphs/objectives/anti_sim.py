"""f2: anti-classical-simulability objective.

Operationalised via the per-qubit stabilizer Renyi entropy M_2 of the output
state (Leone, Oliviero, Hamma, PRL 128, 050402 (2022)). M_2 is zero on any
stabilizer state and strictly positive for magic-carrying states; the
larger M_2 the harder the classical simulation via stabilizer methods.

Normalisation: we divide by n_qubits so f2 scales intensively and is
comparable across circuits of different widths. M_2 is bounded above by
log_2((2^n + 1)/2) in bits, so the per-qubit version is bounded but not by
a width-independent constant; we accept that — f2 is a *direction* (bigger
is better), not a unit-normalised score.

Limitation
----------
The exact 4^n enumeration in `magic_jax.stabilizer_renyi_entropy` caps out
at n_qubits = 10 on 8 GB. Beyond that f2 will deliberately refuse
(`NotImplementedError`) rather than silently switch to a Monte-Carlo
estimator — see module notes in `gpu.magic_jax`.
"""
from __future__ import annotations

from ..dag.circuit_dag import CircuitDAG
from ..gpu.magic_jax import stabilizer_renyi_entropy


def f2_anti_sim(dag: CircuitDAG) -> float:
    """Return M_2(|psi(theta)>) / n_qubits (bits / qubit).

    Raises NotImplementedError for n_qubits > 10 (propagated from
    `stabilizer_renyi_entropy`).
    """
    n = dag.n_qubits
    if n <= 0:
        return 0.0
    m2 = stabilizer_renyi_entropy(dag)
    return m2 / float(n)


__all__ = ["f2_anti_sim"]
