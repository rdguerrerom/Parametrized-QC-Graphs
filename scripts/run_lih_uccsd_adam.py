"""UCCSD-skeleton + Adam θ-optimization for LiH at R=1.545 A.

This is the chemistry-aware complement to the hardware-generic Nash result.
Nash (HE gate set) converges to HF with 4 ops; UCCSD (chemistry primitives)
reaches the active-space ground state with 602 ops + 184 parameters via
400 Adam steps. Structure-preserving optimisation, since removing any gate
from the UCCSD Trotter block breaks its function — Nash structural moves
are not applicable here; the cross-gate-set comparison is the scientific
point.

Writes `results/exp_c3_lih_uccsd_nash.json`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tensorcircuit as tc
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import (
    freeze_orbitals,
    get_fermion_operator,
    jordan_wigner,
)
from openfermionpyscf import run_pyscf

from pqcgraphs.dag.initial_states import uccsd_seed_lih
from pqcgraphs.dag.lowering import _apply_op
from pqcgraphs.objectives import lih_sto3g_reference_energies

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    # Build dense active-space Hamiltonian for JAX-friendly expectation.
    mol = MolecularData(
        [("Li", (0, 0, 0)), ("H", (0, 0, 1.545))],
        "sto-3g", multiplicity=1, charge=0,
    )
    mol = run_pyscf(mol, run_scf=True)
    q_ham = jordan_wigner(freeze_orbitals(
        get_fermion_operator(mol.get_molecular_hamiltonian()),
        [0, 1], [4, 5, 6, 7]
    ))
    H_dense = jnp.asarray(get_sparse_operator(q_ham, n_qubits=6).todense())

    refs = lih_sto3g_reference_energies()
    E_HF = refs["E_HF"]; E_GS = refs["E_ground_active"]
    print(f"[refs] E_HF={E_HF:.5f}  E_GS={E_GS:.5f}  corr={(E_HF - E_GS) * 1000:.2f} mHa",
          flush=True)

    dag = uccsd_seed_lih()
    print(f"[seed] n_ops={dag.n_ops}  n_params={dag.n_params}  depth={dag.depth()}",
          flush=True)

    # Parametric-schedule representation for JIT.
    sched = []
    pidx = 0
    for node in dag.nodes.values():
        if node.is_parametric:
            sched.append((node.gate_name, node.qubits, pidx)); pidx += 1
        else:
            sched.append((node.gate_name, node.qubits, -1))
    theta_init = jnp.asarray(np.asarray(dag.thetas), dtype=jnp.float64)

    def state_fn(theta):
        c = tc.Circuit(6)
        for name, qubits, pi in sched:
            _apply_op(c, name, qubits, theta, pi)
        return c.state()

    def energy_fn(theta):
        psi = state_fn(theta)
        return jnp.real(jnp.vdot(psi, H_dense @ psi))

    grad_fn = jax.jit(jax.value_and_grad(energy_fn))

    print("[warmup] compiling grad_fn...", flush=True)
    t0 = time.perf_counter()
    E0, _ = grad_fn(theta_init)
    E0.block_until_ready()
    print(f"[warmup] done in {time.perf_counter() - t0:.1f}s  E0={float(E0):.6f}",
          flush=True)

    # Adam
    theta = theta_init.copy()
    lr = 0.02
    m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t0 = time.perf_counter()
    for step in range(400):
        E, g = grad_fn(theta)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (step + 1))
        vh = v / (1 - b2 ** (step + 1))
        theta = theta - lr * mh / (jnp.sqrt(vh) + eps)
        if step % 50 == 0:
            print(f"  step={step:>3d}  E={float(E):.6f}  "
                  f"ΔHF={(float(E) - E_HF) * 1000:+6.2f} mHa  "
                  f"ΔGS={(float(E) - E_GS) * 1000:+6.2f} mHa", flush=True)

    E_final, _ = grad_fn(theta)
    E_final = float(E_final)
    wall = time.perf_counter() - t0
    corr = (E_HF - E_final) / (E_HF - E_GS)
    print(f"[final] E={E_final:.6f}  corr={100 * corr:.1f}%  wall={wall:.1f}s",
          flush=True)

    out = {
        "name": "C3_lih_uccsd_theta_opt",
        "description": (
            "Trotterized UCCSD-singlet skeleton (from openfermion), refined by "
            "Adam θ-gradient descent. Chemistry-aware complement to the "
            "hardware-generic Nash result on LiH: Nash (HE gate set) reaches "
            "HF with 4 ops; UCCSD (chemistry primitives) reaches E_GS with "
            "602 ops and 184 θ. Structure-preserving optimization — Nash's "
            "structural moves break UCCSD Trotter blocks by construction, so "
            "they are not applicable here."
        ),
        "n_qubits": 6,
        "n_ops": int(dag.n_ops),
        "n_params": int(dag.n_params),
        "n_uccsd_amplitudes": 5,
        "depth": int(dag.depth()),
        "optimizer": "Adam",
        "lr": 0.02,
        "n_steps": 400,
        "initial_energy_Ha": float(E0),
        "final_energy_Ha": E_final,
        "fraction_correlation_recovered": float(corr),
        "references": {
            "HF_energy_Ha": E_HF,
            "active_ground_energy_Ha": E_GS,
            "FCI_full_energy_Ha": refs["E_FCI_full"],
        },
        "wall_time_s": float(wall),
    }
    out_path = ROOT / "results" / "exp_c3_lih_uccsd_nash.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
