# Claude Code Context — Parametrized-QC-Graphs

## CUDA Hardware

**Available GPU:** NVIDIA GeForce RTX 4060 (only CUDA-capable device)

- Compute Capability: 8.9
- CUDA Cores: 3072
- Memory: 8 GB GDDR6
- Memory Bandwidth: 272 GB/s

All GPU-accelerated computations must target this hardware.

## Conda Environment

Reuse the existing shared env: `pytorch-env`. Already contains:

- Python 3.11, numpy, scipy, networkx 3.3, matplotlib
- `tensorcircuit` 1.3.0
- `jax` 0.8.2 with CUDA-enabled `jaxlib` (`CudaDevice(id=0)` available)

Optional dev deps (install with `pip install -e '.[dev]'` in `pytorch-env`):

- `pytest` — test runner
- `stim` — Clifford-circuit oracle for validation

Verify the env with `python scripts/verify_env.py`.

## Project Provenance

This repo is fully self-contained. Core primitives (`quantum/`, `game/nash_game.py`,
`game/objectives.py`, `metrology/tensorcircuit_qfim_optimized.py`) are **copied
verbatim** from `~/Research/Science/Projects/STABILIZER_GAMES` (tested and
proven-correct). Divergence from that project is expected as the Tier-1 PQC
extension develops; do not attempt to re-import or sync changes.

## Scientific Context

- Top-level spec: `docs/Tier1.md` (4-player Nash game over PQC DAGs).
- Embedding theory: `docs/circuit-dag-generalization.md` (graph state ⊂ PQC DAG).
- Related theorems (abelian DLA bound): `~/Research/Science/Projects/STABILIZER_GAMES/DLA/Stabilizer-Rank-from-DLA.md`.

## Implementation Rules

- **No placeholders.** Every function must be end-to-end wired in v1. Use
  `raise NotImplementedError("specific reason")` if truly blocked; never
  silently return dummy values.
- **Real Hamiltonians for f3 (performance).** H₂ STO-3G 4-qubit Pauli sum at
  bond 0.7414 Å is hardcoded in `src/pqcgraphs/objectives/hamiltonians.py`;
  MaxCut Hamiltonian builds directly from `networkx.Graph`.
- **Single source of truth for state lowering.** `CircuitDAG → tc.Circuit`
  lives in `src/pqcgraphs/dag/lowering.py` and is the only path to
  TensorCircuit. All GPU objectives consume it via JIT.
