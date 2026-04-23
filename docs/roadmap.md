# The Nash framework programme — roadmap

This document names the research programme built around the
four-player potential game over parameterized-quantum-circuit DAGs
introduced in the manuscript *A four-player potential game for
barren-plateau-aware quantum ansatz design* (submitted to *Quantum*,
April 2026; arXiv preprint concurrent).

The framework itself is a single conceptual contribution — a DAG
generalization of graph-state adjacency with a per-player restricted
action set and a block-coordinate ε-Nash residual. The forthcoming
papers listed below extend that contribution along independent axes.
They are planned as a coherent programme rather than unrelated
follow-ups, with cross-citation intended.

This roadmap is public; each work package below is pursued locally
and becomes visible in this repository only when the corresponding
paper is submitted. The branch and squash-merge conventions that
enforce this "append-only `main`" discipline are documented in
[`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Published / submitted

| # | Title | Target | Status |
|---|-------|--------|--------|
| 0 | *A four-player potential game for barren-plateau-aware quantum ansatz design* | Quantum | Submitted, arXiv preprint posted |

## Forthcoming

Priority order reflects a realistic 1–2 papers per year cadence for a
single-author programme on consumer-grade GPU hardware.

### 1. Reference implementation paper

A standalone description of the `pqcgraphs` codebase as a reusable
library for parameterised-quantum-circuit architecture search. Covers
the DAG mutation primitives, the per-player restricted-action move set,
the GPU-accelerated quantum Fisher-information and stabilizer Rényi
estimators, and the JIT-cache-aware simulated-annealing + gradient-
descent driver. Target: **SciPost Codebases** (preferred) or **Journal
of Open Source Software**.

### 2. Chemistry-aware primitives and the LiH potential-energy surface

First-class Givens and particle-number-conserving primitives added to
the DAG move catalogue, followed by a Nash-equilibrium scan of the LiH
STO-3G active space across the dissociation coordinate. The question
is whether chemistry-aware primitives close the 2.3 % correlation gap
observed in the main paper's single-geometry demonstration, and
whether Nash-compressed ansätze remain robust under bond stretching
where barren-plateau effects deepen. Target: **Quantum**; **PRX
Quantum** if the result beats ADAPT-VQE gate counts at chemical
accuracy across the full surface.

### 3. Scaling of the Pareto frontier

Reproduction of the non-stabilizerness / energy Pareto frontier on
MaxCut over 3-regular random graphs at n = 6, 8, 10 qubits.
The claim under test is that the frontier shape predicted at n = 4
persists as the system grows — a necessary condition before the
framework's Pareto-navigation interpretation extends beyond the toy
regime. Target: **Quantum** or **New Journal of Physics**.

### 4. Chemistry baselines — Nash vs ADAPT-VQE, UCCSD, k-UpCCGSD

Matched-budget head-to-head between Nash-compressed circuits and the
three dominant chemistry-aware variational ansätze on LiH, BeH₂, and
the H₂O active space. The question is whether Nash's multi-axis
control (trainability, non-stabilizerness, hardware cost simultaneous
with energy) yields circuits that are meaningfully different from
ADAPT-VQE family ansätze at matched correlation recovery. Pursued
only after package 2 lands. Target: **Journal of Chemical Theory and
Computation**.

### 5. Barren-plateau / simulability theorem

A speculative theoretical background pursuit: under what conditions
on the four-player potential does the resulting Nash equilibrium
enjoy a gradient-variance lower bound alongside a lower bound on
non-stabilizerness? A formal result would close the Cerezo–Larocca
tension for this framework class. No committed deliverable; held in
reserve. Venue depends on result: potentially **Quantum**, **PRX
Quantum**, or a physics-theory archive paper.

### 6. Hardware demonstration

Implementation of a Nash-compressed ansatz on superconducting
(IBM Eagle or Heron) or trapped-ion (Quantinuum H2) hardware, for a
chemistry or optimization task already studied numerically in this
programme. Gated on cloud or collaboration access. Target: **npj
Quantum Information**.

## Explicit non-goals

Readers may wonder whether certain natural extensions belong here.
They do not:

- **A Nature-family paper**. The framework is methodological and
  incremental relative to the field; pursuing Nature, Nature
  Physics, or Science as a primary target is not planned.
- **A single-objective neural-architecture-search variant**. The
  four-player factorization is the contribution; collapsing it to
  one objective would reduce the framework to a reimplementation of
  existing approaches.
- **A quantum-cryptography or QEC application**. The DAG formalism
  could support these but they are outside the programme's scope.

## Citation ask

Each forthcoming paper will cross-cite the others explicitly so the
programme is visible as a connected contribution rather than a set
of isolated results. Readers building on any single paper are asked
to consider the programme framing when attributing the underlying
framework.

## Contact

Correspondence on this roadmap: Ruben Dario Guerrero,
NeuroTechNet S.A.S., rudaguerman@gmail.com.
