"""One-shot builder: UCCSD skeleton circuit for the 6-qubit LiH active space.

Generates the Trotterized singlet UCCSD ansatz structure (HF + excitation
gate sequence) and serialises it as a list of (gate_name, qubits, theta)
tuples. The initial θ values are small (0.05) — Nash's inner-loop θ-GD
will tune them. Cached in `data/circuits/lih_uccsd_seed.json` so runtime
code doesn't need openfermion.

Active space: 2 electrons in 3 spatial orbitals (Li 2s, Li 2pz, H 1s),
matching the reduction used in lih_sto3g_hamiltonian(). HF state has
MO-0 doubly occupied → qubits 0 (α) and 1 (β) filled.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from openfermion.circuits import (
    trotterize_exp_qubop_to_qasm,
    uccsd_singlet_generator,
    uccsd_singlet_paramsize,
)
from openfermion.transforms import jordan_wigner

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "circuits" / "lih_uccsd_seed.json"


def _parse_qasm_line(line: str):
    """Return (gate_name, qubits_tuple, theta_or_None) for one openfermion
    QASM line. The emitter uses: H, Rx, Ry, Rz, CNOT.
    """
    parts = line.split()
    gate = parts[0]
    if gate == "H":
        return ("h", (int(parts[1]),), None)
    if gate == "CNOT":
        return ("cnot", (int(parts[1]), int(parts[2])), None)
    if gate in ("Rx", "Ry", "Rz"):
        theta = float(parts[1])
        q = int(parts[2])
        return (gate.lower(), (q,), theta)
    if gate in ("X", "Y", "Z"):
        return (gate.lower(), (int(parts[1]),), None)
    raise ValueError(f"Unrecognised QASM line: {line!r}")


def main() -> None:
    n_active_q = 6
    n_active_e = 2
    n_params = uccsd_singlet_paramsize(n_active_q, n_active_e)
    print(f"UCCSD singlet param size (6q, 2e): {n_params}")

    # Small non-zero seed amplitudes so Trotterization emits the full gate
    # sequence. The ACTUAL parameter values will be optimised by Nash's
    # θ-gradient inner loop.
    seed_theta = 0.05
    packed_amps = [seed_theta] * n_params

    # Build the UCCSD generator (anti-Hermitian fermion operator) and
    # JW-transform. Note: uccsd_singlet_generator emits T - T^† with the
    # i factor already folded in if anti_hermitian=True; we want the
    # Hermitian part so the Trotter emitter can exponentiate -i·H·t.
    # openfermion convention: pass anti_hermitian=True (default) to get
    # the (T − T^†) operator; jordan_wigner yields an imaginary Hermitian
    # QubitOperator, which trotterize_exp_qubop_to_qasm exponentiates as
    # exp(−it·H). For UCC we actually want exp(T−T^†) ≈ exp(−i·i(T−T^†)),
    # so we multiply by −i to get a Hermitian op and pass evolution_time=1.
    ferm_generator = uccsd_singlet_generator(
        packed_amps, n_active_q, n_active_e, anti_hermitian=True
    )
    qubit_generator = jordan_wigner(ferm_generator)
    # qubit_generator is anti-Hermitian (coefs are pure imaginary). We
    # want to expose the Hermitian part H such that exp(−i·H) = exp(T−T†).
    # So H = i·(T−T†) (Hermitian, real coefficients after the i-multiply).
    hermitian = 1j * qubit_generator  # real-coef Hermitian operator
    hermitian.compress()

    print(f"Qubit generator has {len(list(hermitian.terms))} Pauli terms.")

    # Trotterize (order 1, 1 step): emits QASM for exp(−i·H·t).
    qasm_lines = list(trotterize_exp_qubop_to_qasm(
        hermitian, evolution_time=1.0, trotter_number=1, trotter_order=1
    ))

    # Prepend the HF reference preparation.
    dag_ops = [("x", (0,), None), ("x", (1,), None)]
    for line in qasm_lines:
        dag_ops.append(_parse_qasm_line(line))

    # Reset parametric θ values to small random (scaled by 0.01) so Nash
    # starts near identity — the UCCSD *structure* is the seed, not its
    # specific angles.
    rng = np.random.default_rng(0)
    final_ops = []
    for (name, qubits, theta) in dag_ops:
        if theta is None:
            final_ops.append([name, list(qubits), None])
        else:
            final_ops.append([name, list(qubits), float(rng.normal(0.0, 0.01))])

    n_parametric = sum(1 for op in final_ops if op[2] is not None)
    print(f"Emitted DAG: {len(final_ops)} gates total, {n_parametric} "
          f"parametric (θ-tunable). HF-prep + UCCSD-skeleton.")

    payload = {
        "description": "UCCSD singlet skeleton for 6-qubit LiH active space "
                       "(frozen-core + Li 2p_{x,y} frozen). Includes HF prep "
                       "(X_0, X_1) then Trotterized UCCSD structure.",
        "n_qubits": n_active_q,
        "n_electrons": n_active_e,
        "uccsd_param_count": n_params,
        "n_gates": len(final_ops),
        "n_parametric_gates": n_parametric,
        "operations": final_ops,  # list of [gate_name, [qubits], theta_or_None]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"cached → {OUT}")


if __name__ == "__main__":
    main()
