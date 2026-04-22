"""One-shot builder: LiH STO-3G → 6-qubit reduced Pauli sum.

Run this ONCE to populate `data/hamiltonians/lih_sto3g_6q.json`. The runtime
code in `src/pqcgraphs/gpu/hamiltonians.py` only reads that JSON, so the
`pqcgraphs` package has no openfermion / pyscf dependency.

Reduction scheme: Jordan-Wigner with frozen core (Li 1s²) and frozen
Li 2p_{x,y} virtuals (non-bonding for linear LiH on the z-axis). Active
space: Li(2s, 2p_z), H(1s) → 6 spin-orbitals, 2 electrons.

Dependency: `pip install openfermion openfermionpyscf` (only needed to
regenerate; not a runtime requirement).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import (
    freeze_orbitals,
    get_fermion_operator,
    jordan_wigner,
)
from openfermion.utils import count_qubits
from openfermionpyscf import run_pyscf
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "hamiltonians" / "lih_sto3g_6q.json"


def main(bond_length: float = 1.545) -> None:
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_length))]
    mol = MolecularData(geometry, "sto-3g", multiplicity=1, charge=0)
    mol = run_pyscf(mol, run_scf=True, run_fci=True)

    ham = get_fermion_operator(mol.get_molecular_hamiltonian())
    # Spatial-orbital ordering (pyscf default, STO-3G):
    #   0 = Li 1s,  1 = Li 2s,  2 = Li 2px,  3 = Li 2py,  4 = Li 2pz,  5 = H 1s
    # Spin-orbital indices = 2·spatial and 2·spatial+1.
    frozen_core_spin = [0, 1]                # Li 1s² (doubly occupied, core)
    frozen_virt_spin = [4, 5, 6, 7]          # Li 2px, 2py (non-bonding)
    ham_reduced = freeze_orbitals(ham, frozen_core_spin, frozen_virt_spin)

    q_ham = jordan_wigner(ham_reduced)
    n_qubits = count_qubits(q_ham)
    assert n_qubits == 6, f"Expected 6 qubits after reduction, got {n_qubits}."

    H = get_sparse_operator(q_ham, n_qubits=n_qubits)
    e_active = float(eigsh(H, k=1, which="SA", return_eigenvectors=False)[0].real)

    terms = []
    for op_tuple, coef in q_ham.terms.items():
        pstr = ["I"] * n_qubits
        for qi, axis in op_tuple:
            pstr[qi] = axis
        if abs(coef.imag) > 1e-10:
            raise ValueError(f"Non-real coefficient {coef} for term {op_tuple}")
        terms.append(["".join(pstr), float(coef.real)])

    payload = {
        "molecule": "LiH",
        "basis": "sto-3g",
        "bond_length_angstrom": float(bond_length),
        "frozen_core_spin_orbitals": frozen_core_spin,
        "frozen_virtual_spin_orbitals": frozen_virt_spin,
        "mapping": "jordan_wigner",
        "n_qubits": int(n_qubits),
        "n_active_electrons": 2,
        "e_hf": float(mol.hf_energy),
        "e_fci": float(mol.fci_energy),
        "e_active_ground": e_active,
        "terms": terms,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"cached {len(terms)} Pauli terms → {OUT}")
    print(f"E_HF = {mol.hf_energy:.6f}  E_FCI_full = {mol.fci_energy:.6f}  "
          f"E_active = {e_active:.6f}  Ha")


if __name__ == "__main__":
    main()
