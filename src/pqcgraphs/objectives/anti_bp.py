"""f1: anti-barren-plateau objective.

Per Tier1.md sec. 1, P_1 maximizes a proxy for the variance of the loss
gradient. The operationalisation we use here is the "effective dimension of
the accessible state manifold" at the current theta point, i.e. the rank of
the Quantum Fisher Information Matrix.

Rationale
---------
- The QFIM is the Fubini-Study metric tensor of the parameterised-state
  manifold. Its rank equals the local dimension of the tangent space, so
  the number of independent physical directions the gradient can probe.
- Ragone et al. (Nat. Commun. 15, 7172, 2024) tie Var[dC] to dim(DLA) for
  Hamiltonian-variational ansaetze; the QFIM rank coincides with dim(DLA)
  on generic theta (see docs/circuit-dag-generalization.md sec. 5), but
  is defined at any theta and is the natural local quantity.
- We normalise by n_params so f1 is in [0, 1] and is invariant under the
  trivial "pad with more parameters" move — an un-normalised rank would be
  monotonically growing by adding rotations even when those rotations are
  redundant.

Higher f1 = more-informative gradients = less-BP-prone architecture.
"""
from __future__ import annotations

from ..dag.circuit_dag import CircuitDAG
from ..gpu.qfim_effdim import effective_dimension


def f1_anti_bp(dag: CircuitDAG) -> float:
    """Return the normalised effective dimension of the QFIM at dag.thetas.

    Value: rank(QFIM) / n_params, in [0.0, 1.0]. Zero if the DAG has no
    parametric gates (nothing to optimize, trivially barren).
    """
    p = dag.n_params
    if p <= 0:
        return 0.0
    r = effective_dimension(dag)
    return float(r) / float(p)


__all__ = ["f1_anti_bp"]
