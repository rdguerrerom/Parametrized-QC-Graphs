# Circuit DAGs as Generalizations of Graph State Adjacency Matrices

## 1. Graph State Adjacency Matrix — Standard Formulation

Given an undirected graph $G = (V, E)$ with $n = |V|$ vertices and adjacency matrix $\Gamma$, the graph state is:

$$|G\rangle = \prod_{(i,j) \in E} \text{CZ}_{ij} \, |+\rangle^{\otimes n}$$

Equivalently, using $\Gamma_{ij} \in \{0,1\}$ (symmetric, zero diagonal):

$$|G\rangle = \prod_{i < j} \text{CZ}_{ij}^{\,\Gamma_{ij}} \, |+\rangle^{\otimes n}$$

The graph state is the unique $+1$ eigenstate of $n$ stabilizer generators:

$$K_a = X_a \prod_{b \in N(a)} Z_b, \quad a = 1, \ldots, n$$

where $N(a) = \{b : \Gamma_{ab} = 1\}$ is the neighborhood of vertex $a$.

**Key references:**
- Hein, Eisert, Briegel, "Multiparty entanglement in graph states," Phys. Rev. A 69, 062311 (2004)
- Hein et al., "Entanglement in graph states and its applications," arXiv:quant-ph/0602096 (2006)
- Raussendorf, Briegel, "A one-way quantum computer," Phys. Rev. Lett. 86, 5188 (2001)


## 2. Circuit DAG Representation

A parameterized quantum circuit is represented as a directed acyclic graph $D = (V, E_d)$ with three node types:

- **Input nodes** $V_{\text{in}}$: one per qubit, representing the wire's entry point
- **Operation nodes** $V_{\text{op}} = \{g_1, \ldots, g_L\}$: each gate $U_l(\theta_l)$
- **Output nodes** $V_{\text{out}}$: one per qubit, representing the wire's exit point

Directed edges $E_d \subseteq V \times V \times Q$ carry wire labels. An edge $(g_l, g_m, q_i) \in E_d$ means qubit $q_i$ flows from the output of gate $g_l$ to the input of gate $g_m$.

The parameterized unitary is $U(\boldsymbol{\theta}) = \prod_{l=1}^{L} U_l(\theta_l)$.

**References:**
- Qiskit DAGCircuit: https://docs.quantum.ibm.com/api/qiskit/qiskit.dagcircuit.DAGCircuit
- Iten et al., "Introduction to UniversalQCompiler," arXiv:1904.01072 (2019)
- AltGraph, "Redesigning quantum circuits using generative graph models," GLSVLSI 2024, arXiv:2403.12979


## 3. The Embedding Map $\iota: G \mapsto D_G$

A graph state $|G\rangle$ defined by $\Gamma$ maps to a circuit DAG $D_G$ as follows:

| Graph state $G = (V, E)$ | Circuit DAG $D$ |
|---|---|
| Vertex $v \in V$ (qubit) | Qubit wire $q_v \in V_q$ |
| Edge $(i,j) \in E$ | Operation node $\text{CZ}_{ij} \in V_{\text{op}}$ with wire-edges on $q_i, q_j$ |
| Vertex degree $\deg(v)$ | Fan-in/fan-out of qubit wire $q_v$ |
| Adjacency matrix $\Gamma$ (symmetric) | Bipartite incidence structure (directed) |
| All gates are CZ | Gate set unrestricted: $\{R_x, R_y, R_z, \text{CNOT}, \text{CZ}, \ldots\}$ |
| No parameters | Parameters $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_L)$ |

The embedding is well-defined and injective: every graph state has a unique (up to CZ ordering) circuit DAG.


## 4. What the Generalization Relaxes

1. **Undirected to Directed.** The graph state adjacency matrix $\Gamma$ is symmetric ($\text{CZ}_{ij} = \text{CZ}_{ji}$). The circuit DAG introduces temporal ordering, breaking this symmetry.

2. **Single gate type to Heterogeneous gate set.** Graph states use only CZ gates (after Hadamard layer). A PQC DAG allows arbitrary parameterized gates $U_l(\theta_l) \in \{R_x(\theta), R_y(\theta), R_z(\theta), \text{CNOT}, \text{CZ}, \ldots\}$.

3. **Fixed structure to Parameterized.** $\Gamma_{ij} \in \{0,1\}$ is binary. In a PQC, continuous parameters $\theta_l \in [0, 2\pi)$ modulate each gate.

4. **Single layer to Multi-layer (depth).** The graph state circuit has depth 2 (Hadamard + CZ layer). A PQC DAG has arbitrary depth $L$, with the same qubit pair potentially interacting multiple times.


## 5. Expressibility, Stabilizer Rank, and DLA Dimension

### Stabilizer Rank

The stabilizer rank $\chi(|\psi\rangle)$ is the minimum number of stabilizer states needed to decompose $|\psi\rangle$:

$$\chi(|\psi\rangle) = \min\left\{ k : |\psi\rangle = \sum_{i=1}^{k} c_i |\phi_i\rangle,\; |\phi_i\rangle \in \text{STAB}_n \right\}$$

Classical simulation cost scales as $O(\chi^2 \cdot \text{poly}(n))$ (Bravyi-Gosset).

### DLA Dimension

The dynamical Lie algebra of a PQC with generators $\{iH_1, \ldots, iH_L\}$ is:

$$\mathfrak{g} = \text{Lie}(iH_1, \ldots, iH_L) = \text{span}\{iH_l, [iH_l, iH_m], [[iH_l, iH_m], iH_k], \ldots\}$$

Ragone et al. (Nat. Commun. 2024) proved that gradient variance satisfies $\text{Var}[\partial_l C] \in \Theta(1/\dim(\mathfrak{g}))$.

### The Expressibility Spectrum

| Regime | $\chi$ | $\dim(\mathfrak{g})$ | Simulable? | BPs? |
|---|---|---|---|---|
| Pure Clifford (graph states) | 1 | poly($n$) | Yes | No |
| Clifford + few T gates | grows with T-count | starts growing | Harder | Emerging |
| Full PQC, many non-Clifford | exponential | $\Omega(4^n)$ possible | No | Yes |

### The BP / Simulability Tension (Cerezo et al. 2025)

The central result: provable absence of barren plateaus implies the loss landscape is classically simulable (in a wide class of cases). If $\dim(\mathfrak{g}) \in O(\text{poly}(n))$, gradients are informative but the circuit is simulable. If $\dim(\mathfrak{g}) \in \Omega(\exp(n))$, the circuit may offer quantum advantage but gradients concentrate exponentially.

Graph states sit at the pure Clifford end ($\chi = 1$). The PQC DAG generalization moves along this spectrum by introducing parameterized non-Clifford gates.


## 6. Open Problem: Stabilizer Rank and DLA Dimension

The link between stabilizer rank $\chi$ and DLA dimension $\dim(\mathfrak{g})$ is **not yet formalized** in the literature. A natural conjecture:

$$\log \chi\bigl(U(\boldsymbol{\theta})|+\rangle^{\otimes n}\bigr) \leq f\bigl(\dim(\mathfrak{g})\bigr)$$

for some monotone $f$. Formalizing this connection is a potential contribution of this project.


## 7. References

### Graph States
- Hein, Eisert, Briegel, Phys. Rev. A 69, 062311 (2004). arXiv:quant-ph/0307130
- Hein et al., arXiv:quant-ph/0602096 (2006)

### BP / Simulability
- Cerezo et al., Nat. Commun. 16, 7907 (2025)
- Ragone et al., Nat. Commun. 15, 7172 (2024)
- Holmes, Sharma, Cerezo, Coles, PRX Quantum 3, 010313 (2022)

### Circuit Expressibility
- Sim, Johnson, Aspuru-Guzik, Adv. Quantum Technol. 2, 1900070 (2019). arXiv:1905.10876

### Stabilizer Rank and Simulation
- Bravyi, Smith, Smolin, Phys. Rev. X 6, 021043 (2016). arXiv:1506.01396
- Bravyi, Browne, Calpin, Campbell, Gosset, Howard, Quantum 3, 181 (2019)

### Circuit Topology as Generalized Graph
- AltGraph, GLSVLSI 2024. arXiv:2403.12979
- Du et al., arXiv:2405.08100 (2024) — GNNs for PQC expressibility estimation
