"""Gate type registry for CircuitDAG nodes.

A gate type is defined by (name, arity, is_parametric). The registry is the
single authority on what gate names the DAG accepts; lowering to TensorCircuit
and validity checks both read it.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class GateSpec:
    """Metadata for a gate type.

    name: canonical identifier (matches tc.Circuit method when possible).
    arity: number of qubits the gate acts on.
    is_parametric: whether the gate takes a continuous parameter θ.
    is_clifford: whether the gate (with its default params, if any) preserves
                 the stabilizer group — used by objectives/anti_sim.
    native_on: set of hardware topology tags where this gate is considered
               native (zero cost for f4). Empty = native everywhere.
    """
    name: str
    arity: int
    is_parametric: bool
    is_clifford: bool
    native_on: FrozenSet[str] = frozenset()


# Canonical gate set supported in v1. Names chosen to match tc.Circuit methods
# for a clean 1:1 lowering mapping.
GATE_SPECS: Dict[str, GateSpec] = {
    # Single-qubit Clifford
    "h":   GateSpec("h",   arity=1, is_parametric=False, is_clifford=True),
    "s":   GateSpec("s",   arity=1, is_parametric=False, is_clifford=True),
    "sd":  GateSpec("sd",  arity=1, is_parametric=False, is_clifford=True),
    "x":   GateSpec("x",   arity=1, is_parametric=False, is_clifford=True),
    "y":   GateSpec("y",   arity=1, is_parametric=False, is_clifford=True),
    "z":   GateSpec("z",   arity=1, is_parametric=False, is_clifford=True),
    # Single-qubit non-Clifford (T introduces magic)
    "t":   GateSpec("t",   arity=1, is_parametric=False, is_clifford=False),
    "td":  GateSpec("td",  arity=1, is_parametric=False, is_clifford=False),
    # Single-qubit parametric rotations (generically non-Clifford)
    "rx":  GateSpec("rx",  arity=1, is_parametric=True,  is_clifford=False),
    "ry":  GateSpec("ry",  arity=1, is_parametric=True,  is_clifford=False),
    "rz":  GateSpec("rz",  arity=1, is_parametric=True,  is_clifford=False),
    # Two-qubit Clifford entanglers (native on superconducting / IBM)
    "cz":     GateSpec("cz",     arity=2, is_parametric=False, is_clifford=True,
                       native_on=frozenset({"heavy_hex", "grid"})),
    "cnot":   GateSpec("cnot",   arity=2, is_parametric=False, is_clifford=True,
                       native_on=frozenset({"heavy_hex", "grid"})),
    # Two-qubit parametric entanglers (generically non-Clifford)
    "rzz": GateSpec("rzz", arity=2, is_parametric=True, is_clifford=False),
    "rxx": GateSpec("rxx", arity=2, is_parametric=True, is_clifford=False,
                    native_on=frozenset({"rydberg"})),
    "ryy": GateSpec("ryy", arity=2, is_parametric=True, is_clifford=False),
}


def gate_spec(name: str) -> GateSpec:
    """Look up a gate spec by name; raise on unknown gates."""
    try:
        return GATE_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown gate {name!r}. Known: {sorted(GATE_SPECS)}"
        ) from exc


def is_native(gate_name: str, topology_tag: str) -> bool:
    """True iff `gate_name` is native on hardware `topology_tag`.

    Unrestricted single-qubit gates are considered native everywhere.
    """
    spec = gate_spec(gate_name)
    if spec.arity == 1:
        return True
    return topology_tag in spec.native_on
