<p align="center">
  <img src="Logo.png" alt="NeuroTechNet S.A.S." width="480">
</p>

# Parametrized-QC-Graphs

**Nash-equilibrium architecture search for parameterized quantum circuits.**

A four-player potential game over circuit directed acyclic graphs (DAGs)
addresses the Cerezo et al. 2025 barren-plateau / classical-simulability
tension. Four players — trainability, non-stabilizerness, task performance,
and hardware cost — own restricted action sets (append, remove, retype,
rewire) on the shared circuit DAG. A block-coordinate ε-Nash residual
δ_Nash certifies that no single player can improve unilaterally.

This repository contains the full framework, experimental evidence, and
the submission-ready manuscript.

## Sponsor & acknowledgment

This work was supported by **NeuroTechNet S.A.S.**, Bogotá, Colombia, who
provided financial support and computational resources. All code and data
in this repository are released under MPL-2.0 (see `LICENSE`) to enable
reproduction and extension by the scientific community.

## Highlights

- **Pareto frontier on MaxCut K₄** — a single weight sweep traces the full
  trainability/simulability trade-off from a Clifford endpoint
  `(M₂/n, ⟨H⟩) = (0, 4.00)` to a non-Clifford endpoint `(0.48, 3.30)`.
- **Chemistry seeding on LiH/STO-3G** — starting from a 58-gate Givens-
  doubles ansatz, the Nash search returns a 48-operation, depth-25
  circuit retaining **97.7 %** of the correlation energy while jointly
  controlling trainability, non-stabilizerness, and hardware cost.
- **Five-seed scaling on TFIM at n = 4, 6, 8** with tight 95 % bootstrap
  confidence intervals; wall clock per iteration scales linearly at
  ~4.7 s/qubit on a single NVIDIA RTX 4060.

Honest limitations are disclosed in the manuscript: paired Wilcoxon
`p = 0.50/0.22/0.50` across three topologies on head-to-head vs. a
simulated-annealing baseline (no statistically significant advantage),
no hardware demonstration, and no n-scaling of the Pareto frontier.

## Repository layout

```
.
├── src/pqcgraphs/           core framework (DAG, gates, objectives, Nash game)
│   ├── dag/                 CircuitDAG + gate types + topologies + lowering
│   ├── quantum/             stabilizer primitives
│   ├── gpu/                 JAX/TensorCircuit evaluators (QFIM, M₂, Hamiltonians)
│   ├── objectives/          f₁ anti-BP, f₂ anti-sim, f₃ performance, f₄ hardware
│   ├── game/                PQCNashGame, players, moves, δ_Nash certificate
│   ├── experiments/         E1–F6 + D1 + C3 experiment modules
│   └── baselines/           simulated-annealing baseline for head-to-head
├── scripts/                 experiment runners, plot scripts, utilities
├── tests/                   unit + integration tests
├── data/                    cached Hamiltonians (H₂, LiH) + UCCSD seed DAG
├── results/                 experiment outputs (JSON, one per figure/claim)
├── figures/                 publication figures (PDF + PNG)
├── Overleaf/                self-contained manuscript bundle for Overleaf/arXiv
├── docs/
│   ├── manuscript.{tex,pdf} submitted manuscript (Quantum format)
│   ├── Tier1.md             framework specification
│   └── circuit-dag-generalization.md   DAG-as-adjacency-matrix theory
├── Logo.png                 NeuroTechNet acknowledgment logo
├── LICENSE                  Mozilla Public License 2.0
└── pyproject.toml           install config
```

## Hardware target

Single NVIDIA RTX 4060 (8 GB, compute capability 8.9). All experiments
in the manuscript were produced on this hardware.

## Quick start

```bash
# Clone
git clone https://github.com/rdguerrerom/Parametrized-QC-Graphs.git
cd Parametrized-QC-Graphs

# Install into a Python 3.11+ environment with CUDA-enabled JAX
pip install -e '.[dev]'

# Verify GPU / environment
python scripts/verify_env.py
python scripts/verify_gpu.py

# Run the full test suite
pytest tests/

# Regenerate all four publication figures from cached results
python scripts/plot_all.py
```

## Reproducing individual experiments

Each experiment in the manuscript is a single module under
`src/pqcgraphs/experiments/`:

| Experiment | Module                            | Output                                  |
|------------|-----------------------------------|-----------------------------------------|
| E1         | `exp1_abelian_embedding.py`       | `results/exp1_abelian.json`             |
| E2 (H₂)    | `exp2_h2_heavy_hex.py`            | `results/exp2_h2_heavy_hex.json`        |
| E3 (Pareto)| `exp3_weight_sweep.py`            | `results/exp3_weight_sweep.json`        |
| E4         | `exp4_topology_ablation.py`       | `results/exp4_topology_ablation.json`   |
| C3 (LiH)   | `exp_c3_lih_vqe.py`               | `results/exp_c3_lih_vqe.json`           |
| D1 (TFIM)  | `exp_d1_tfim_scaling.py`          | `results/exp_d1_tfim_scaling.json`      |
| F6 (vs SA) | `exp_f6_vs_sa_dqas.py`            | `results/exp_f6_vs_sa_dqas.json`        |

Multi-seed reruns:

```bash
python scripts/run_multi_seed.py --both --seeds 5
```

LiH with the Givens-doubles seed:

```bash
python scripts/generate_lih_hamiltonian.py     # one-time: cache Hamiltonian
python scripts/run_lih_givens_adam.py          # Adam on 58-gate Givens ansatz
python scripts/run_lih_givens_nash.py          # Nash structure-search on same seed
```

## Citation

The manuscript is available in `docs/manuscript.pdf`. If you use this
framework, please cite the arXiv preprint (to be assigned) and the
published version.

```bibtex
@unpublished{guerrero2026pqcgraphs,
  author = {Guerrero, Ruben Dario},
  title  = {A four-player potential game for barren-plateau-aware quantum
            ansatz design},
  year   = {2026},
  note   = {Submitted to {Quantum}},
}
```

## License

Code and data in this repository are released under the **Mozilla Public
License 2.0** (see `LICENSE`). MPL-2.0 is a file-level copyleft license:
modifications to MPL-licensed source files must remain under MPL, while
new files in a combined work may carry any compatible license.

## Author

Ruben Dario Guerrero — NeuroTechNet S.A.S., Bogotá, Colombia
[rudaguerman@gmail.com](mailto:rudaguerman@gmail.com)
