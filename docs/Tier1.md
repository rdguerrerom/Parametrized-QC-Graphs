
## Tier 1 — Direct Attack Surfaces: Strong Fit, Immediate Traction

---

### 1. Ansatz Architecture Design Under the BP / Classical Simulability Tension

**The roadmap problem.** The Cerezo et al. 2025 result (Nat. Commun. 16, 7907) establishes the central dilemma: provably BP-free architectures tend to be classically simulable via the Lie-algebraic dimension argument, while architectures that escape classical simulability tend to exhibit exponential gradient concentration. The roadmap identifies this as the field's defining open problem and states explicitly that "exactly how to integrate variational principles and quantum computing remains to be answered." No existing method provides a principled way to navigate the boundary between these two failure modes.

**The Nash formulation.** The shared discrete state is the parameterized circuit topology — the directed acyclic graph (DAG) of gate layers, connectivity pattern, and gate types. Players are:

- **P₁ (Anti-BP player):** Maximizes the variance of the loss gradient, operationalized via a proxy such as the Frobenius norm of the Lie algebra DLA(θ) or the effective dimension of the accessible state manifold. High payoff when gradients are informative.
- **P₂ (Anti-simulability player):** Maximizes classical simulation cost, operationalized via non-stabilizerness (magic) content, T-gate count, or entanglement entropy of mid-circuit states. High payoff when the circuit is hard to simulate classically.
- **P₃ (Performance player):** Minimizes the loss on the target task for a set of training instances.
- **P₄ (Hardware player):** Penalizes circuit depth, non-native gate count, and connectivity violations for the target device topology.

The potential function Φ = w₁f₁ + w₂f₂ + w₃f₃ - w₄f₄ is a weighted potential game by construction. Nash equilibria of this game correspond to circuit topologies where no single objective can be improved without degrading another — the precise definition of a Pareto-balanced architecture. The Nash gap quantifies how far any discovered architecture is from this balance point.

**What equilibrium means physically.** A Nash equilibrium here is a circuit topology where: simultaneously increasing DLA dimension (moving toward full unitary group) would push the architecture into the classically simulable regime; simultaneously increasing magic/complexity would either cause BP or violate hardware constraints; and any further performance gain would require sacrificing trainability or device compatibility. This is the "sweet spot" the roadmap says must exist but provides no method to find.

**Your methodological edge.** The existing approach is essentially warm-start heuristics applied to fixed ansatz families (hardware-efficient, chemically-inspired, MERA-inspired) with no convergence certificate and no mechanism for understanding why a particular topology sits at the boundary. Your framework:
- Provides δ_Nash as a certificate that the discovered architecture is genuinely balanced, not just a lucky initialization
- Makes the tradeoff mechanistically interpretable: equilibrium analysis shows which moves were blocked by which player, explaining the emergent gate structure
- Allows the weight vector (w₁, w₂, w₃, w₄) to encode hardware-specific priors without changing the algorithm — the IBM heavy-hex, Rydberg all-to-all, and superconducting grid cases all become objective reconfiguration problems

**Immediate connection to your PRA work.** Your paper already handles n = 100 graph state codes with hardware constraints (∆(G) ≤ 3 for 2D grids). A parameterized circuit DAG is a natural generalization of the graph state adjacency matrix: edges become gate connections, vertex degrees become qubit fan-in constraints, and the stabilizer rank calculation becomes a circuit expressibility proxy. The infrastructure transfers almost directly.

