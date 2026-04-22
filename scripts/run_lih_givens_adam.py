"""Adam on Givens-doubles LiH seed. Final energy saved to JSON.

Uses the hand-synthesised `lih_givens_doubles_seed` (58 gates, 16 θs —
HF + two paired double excitations). Serves both as:
  (a) a sanity check that the minimal chemistry-aware ansatz can break
      the HF plateau,
  (b) the baseline against which Nash structure search is later compared.
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

from pqcgraphs.dag.chem_ansatze import lih_givens_doubles_seed
from pqcgraphs.dag.lowering import _apply_op
from pqcgraphs.objectives import lih_sto3g_reference_energies

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    tc.set_backend("jax"); tc.set_dtype("complex128")
    mol = MolecularData([("Li", (0, 0, 0)), ("H", (0, 0, 1.545))],
                         "sto-3g", multiplicity=1, charge=0)
    mol = run_pyscf(mol, run_scf=True)
    q_ham = jordan_wigner(freeze_orbitals(
        get_fermion_operator(mol.get_molecular_hamiltonian()),
        [0, 1], [4, 5, 6, 7],
    ))
    H_dense = jnp.asarray(get_sparse_operator(q_ham, n_qubits=6).todense())

    refs = lih_sto3g_reference_energies()
    E_HF, E_GS = refs["E_HF"], refs["E_ground_active"]
    print(f"[refs] E_HF={E_HF:.5f}  E_GS={E_GS:.5f}  corr_gap={(E_HF-E_GS)*1000:.2f} mHa",
          flush=True)

    dag = lih_givens_doubles_seed()
    print(f"[seed] n_ops={dag.n_ops}  n_params={dag.n_params}  depth={dag.depth()}",
          flush=True)

    # Walk op nodes in topological order; map to (gate_name, qubits, theta_idx).
    sched = []
    for n in dag.ops():
        tidx = n.param_idx if n.param_idx is not None else -1
        sched.append((n.gate_name, n.qubits, tidx))
    theta_init = jnp.asarray(np.asarray(dag.thetas), dtype=jnp.float64)

    def state_fn(t):
        c = tc.Circuit(6)
        for name, qs, pi in sched:
            _apply_op(c, name, qs, t, pi)
        return c.state()

    def energy_fn(t):
        psi = state_fn(t); return jnp.real(jnp.vdot(psi, H_dense @ psi))

    grad_fn = jax.jit(jax.value_and_grad(energy_fn))

    print("[warmup] JIT...", flush=True); t0 = time.perf_counter()
    E0, _ = grad_fn(theta_init); E0.block_until_ready()
    print(f"[warmup] done {time.perf_counter()-t0:.2f}s  E0={float(E0):.6f}",
          flush=True)

    th = theta_init.copy(); lr = 0.05
    m = jnp.zeros_like(th); v = jnp.zeros_like(th); b1, b2, eps = 0.9, 0.999, 1e-8
    t0 = time.perf_counter()
    for step in range(500):
        E, g = grad_fn(th)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (step + 1))
        vh = v / (1 - b2 ** (step + 1))
        th = th - lr * mh / (jnp.sqrt(vh) + eps)
        if step % 50 == 0 or step == 499:
            dHF = (float(E) - E_HF) * 1000
            dGS = (float(E) - E_GS) * 1000
            print(f"  step={step:>3d}  E={float(E):.6f}  ΔHF={dHF:+6.2f} mHa  ΔGS={dGS:+6.2f} mHa",
                  flush=True)

    Ef, _ = grad_fn(th); Ef = float(Ef)
    wall = time.perf_counter() - t0
    corr = (E_HF - Ef) / (E_HF - E_GS)
    print(f"[final] E={Ef:.6f}  corr_recovered={100*corr:.1f}%  wall={wall:.1f}s",
          flush=True)

    out = {
        "name": "C3_lih_givens_doubles_adam",
        "description": (
            "Minimal chemistry-aware ansatz: HF + 2 paired DoubleExcitation "
            "gates (HOMO→LUMO1 and HOMO→LUMO2) on qubits (0,1,2,3) and "
            "(0,1,4,5). PennyLane-verified decomposition, 58 gates, 16 "
            "independent θ parameters. Adam θ-gradient descent, 500 steps, "
            "lr=0.05."
        ),
        "n_qubits": 6,
        "n_ops": int(dag.n_ops),
        "n_params": int(dag.n_params),
        "depth": int(dag.depth()),
        "optimizer": "Adam",
        "lr": 0.05, "n_steps": 500,
        "initial_energy_Ha": float(E0),
        "final_energy_Ha": Ef,
        "fraction_correlation_recovered": float(corr),
        "references": {
            "HF_energy_Ha": E_HF,
            "active_ground_energy_Ha": E_GS,
            "FCI_full_energy_Ha": refs["E_FCI_full"],
        },
        "wall_time_s": float(wall),
    }
    out_path = ROOT / "results" / "exp_c3_lih_givens_adam.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
