"""TensorCircuit + JAX backend bootstrap for GPU objectives.

Importing this module sets the TensorCircuit backend to JAX and the default
dtype to complex128, once and only once per Python process. Subsequent
imports are no-ops (guard variable). All GPU objective modules (f1, f2, f3)
must import this module before touching `tensorcircuit` or `jax` so the
backend is in a known state.

Precision policy
----------------
We use complex128 everywhere. The RTX 4060 has 8 GB RAM; for the circuit
sizes this project targets (n <= 14 qubits, <= ~50 parameters), complex128
is comfortably within budget (state vector of 2^14 complex128 = 256 kB;
QFIM Jacobian stack of 50 × 2^14 complex128 = 12.8 MB). If this ever has
to drop to complex64, change the `_DTYPE` constant below and update every
module header that documents the precision choice.
"""
from __future__ import annotations

import os
from typing import Dict, List

# Silence TF C++ logging before JAX/TC pull anything in.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_DTYPE = "complex128"
_BACKEND_TAG = "jax"

# Module-level guard: setup runs exactly once, even across re-imports.
_INITIALIZED: bool = False


def _initialize() -> None:
    """Idempotent: configure tc + jax exactly once per process."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    import tensorcircuit as tc  # noqa: F401  (side effects)
    import jax  # noqa: F401

    tc.set_backend(_BACKEND_TAG)
    tc.set_dtype(_DTYPE)

    # Make JAX default to double precision too; otherwise jax.jacrev on a
    # complex128 circuit output can silently demote real intermediates.
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", True)

    _INITIALIZED = True


_initialize()


def check_gpu() -> Dict[str, object]:
    """Report GPU availability and device list.

    Returns a dict with keys:
      - gpu_available: bool
      - n_gpus: int
      - devices: list of stringified JAX devices
      - backend: str ('jax')
      - dtype: str ('complex128')
      - platform: 'gpu' | 'cpu'
    """
    import jax

    devices = jax.devices()
    gpu_devices: List[object] = [d for d in devices if getattr(d, "platform", "cpu") == "gpu"]
    return {
        "gpu_available": len(gpu_devices) > 0,
        "n_gpus": len(gpu_devices),
        "devices": [str(d) for d in devices],
        "backend": _BACKEND_TAG,
        "dtype": _DTYPE,
        "platform": "gpu" if gpu_devices else "cpu",
    }


__all__ = ["check_gpu"]
