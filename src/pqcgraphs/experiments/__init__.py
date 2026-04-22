"""End-to-end experiments validating the Tier-1 framework.

E1: abelian embedding -> QFIM rank = DLA dim = |E|
E2: VQE H2 on IBM heavy-hex vs HEA baseline
E3: (w_anti_bp, w_anti_sim) Pareto sweep on MaxCut
E4: Topology ablation (heavy_hex / grid / rydberg) on MaxCut
"""
from . import (
    exp1_abelian_embedding,
    exp2_h2_heavy_hex,
    exp3_weight_sweep,
    exp4_topology_ablation,
    exp_c3_lih_vqe,
    exp_d1_tfim_scaling,
    exp_f4_ablation,
    exp_f6_vs_sa_dqas,
)

__all__ = [
    "exp1_abelian_embedding",
    "exp2_h2_heavy_hex",
    "exp3_weight_sweep",
    "exp4_topology_ablation",
    "exp_c3_lih_vqe",
    "exp_d1_tfim_scaling",
    "exp_f4_ablation",
    "exp_f6_vs_sa_dqas",
]
