"""Verify pytorch-env has required deps and GPU visibility."""
from __future__ import annotations
import sys


def main() -> int:
    ok = True

    def check(name: str, importer):
        nonlocal ok
        try:
            mod = importer()
            ver = getattr(mod, "__version__", "?")
            print(f"  [OK]  {name} {ver}")
            return mod
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {name}: {exc}")
            ok = False
            return None

    print("Required:")
    check("numpy", lambda: __import__("numpy"))
    check("scipy", lambda: __import__("scipy"))
    check("networkx", lambda: __import__("networkx"))
    check("matplotlib", lambda: __import__("matplotlib"))
    tc = check("tensorcircuit", lambda: __import__("tensorcircuit"))
    jax = check("jax", lambda: __import__("jax"))

    print("\nOptional (install with `pip install -e '.[dev]'`):")
    check("pytest", lambda: __import__("pytest"))
    check("stim", lambda: __import__("stim"))

    if jax is not None:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            print(f"\n  [OK]  GPU visible to JAX: {[str(d) for d in gpu_devices]}")
        else:
            print(f"\n  [WARN] No GPU devices; running on {devices}")
            ok = False

    if tc is not None and jax is not None:
        tc.set_backend("jax")
        tc.set_dtype("complex128")
        print("  [OK]  tensorcircuit backend set to jax / complex128")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
