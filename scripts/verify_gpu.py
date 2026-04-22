#!/usr/bin/env python3
"""
GPU verification script for TensorCircuit QFIM.

This script checks if JAX and TensorCircuit can access GPU acceleration
and provides diagnostic information for troubleshooting.

Usage:
    conda activate pytorch-env
    python metrology/verify_gpu.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("GPU Verification for TensorCircuit QFIM")
print("=" * 70)
print()

# Step 1: Check JAX installation
print("Step 1: Checking JAX installation...")
try:
    import jax
    import jax.numpy as jnp
    print("✓ JAX successfully imported")
    print(f"  JAX version: {jax.__version__}")
except ImportError as e:
    print(f"✗ JAX import failed: {e}")
    print("\nInstall JAX:")
    print("  pip install --upgrade jax jaxlib")
    sys.exit(1)

print()

# Step 2: Check GPU availability
print("Step 2: Checking GPU availability...")
try:
    devices = jax.devices()
    print(f"  Total devices: {len(devices)}")

    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
    cpu_devices = [d for d in devices if d.device_kind == 'cpu']

    if gpu_devices:
        print(f"✓ GPU available: {len(gpu_devices)} GPU(s) detected")
        for i, gpu in enumerate(gpu_devices):
            print(f"  - GPU {i}: {gpu}")
    else:
        print("✗ No GPU detected")
        print(f"  CPU devices: {len(cpu_devices)}")
        for cpu in cpu_devices:
            print(f"  - {cpu}")
        print("\nTo enable GPU:")
        print("  1. Check CUDA version: nvidia-smi")
        print("  2. Install CUDA-enabled JAX:")
        print("     pip install --upgrade 'jax[cuda12]'  # For CUDA 12.x")
        print("     pip install --upgrade 'jax[cuda11]'  # For CUDA 11.x")

except Exception as e:
    print(f"✗ Device check failed: {e}")
    sys.exit(1)

print()

# Step 3: Test GPU computation
if gpu_devices:
    print("Step 3: Testing GPU computation...")
    try:
        # Simple GPU test
        x = jnp.ones((1000, 1000))
        y = jnp.dot(x, x)
        result = jnp.sum(y)

        print(f"✓ GPU computation successful")
        print(f"  Test result: {result:.2f} (expected: 1000000.00)")

        # Check default device
        default_device = jax.devices()[0]
        print(f"  Default device: {default_device}")

    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        print("  JAX may not be properly configured for GPU")
else:
    print("Step 3: Skipping GPU computation test (no GPU available)")

print()

# Step 4: Check TensorCircuit
print("Step 4: Checking TensorCircuit installation...")
try:
    import tensorcircuit as tc
    print("✓ TensorCircuit successfully imported")
    print(f"  TensorCircuit version: {tc.__version__}")

    # Check backend
    backend = tc.get_backend()
    print(f"  Current backend: {backend}")

    # Set JAX backend
    tc.set_backend("jax")
    print("✓ TensorCircuit using JAX backend")

except ImportError as e:
    print(f"✗ TensorCircuit import failed: {e}")
    print("\nInstall TensorCircuit:")
    print("  pip install tensorcircuit[jax]")
    sys.exit(1)

print()

# Step 5: Run minimal TensorCircuit circuit on GPU
if gpu_devices:
    print("Step 5: Testing TensorCircuit circuit on GPU...")
    try:
        import tensorcircuit as tc
        tc.set_backend("jax")
        tc.set_dtype("complex128")

        # Create simple circuit
        n_qubits = 10
        c = tc.Circuit(n_qubits)
        for i in range(n_qubits):
            c.h(i)
        for i in range(n_qubits - 1):
            c.cnot(i, i + 1)

        state = c.state()

        print(f"✓ TensorCircuit circuit executed on GPU")
        print(f"  Circuit: {n_qubits} qubits, {2 * n_qubits - 1} gates")
        print(f"  State vector shape: {state.shape}")
        print(f"  State norm: {jnp.linalg.norm(state):.6f} (expected: 1.000000)")

    except Exception as e:
        print(f"✗ TensorCircuit GPU test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("Step 5: Skipping TensorCircuit GPU test (no GPU available)")

print()

# Step 6: Performance comparison
print("Step 6: Quick performance test...")
try:
    import time
    import tensorcircuit as tc
    tc.set_backend("jax")
    tc.set_dtype("complex128")

    n_qubits = 12

    # Warmup
    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.h(i)
    _ = c.state()

    # Timed run
    t0 = time.time()
    c = tc.Circuit(n_qubits)
    for i in range(n_qubits):
        c.h(i)
    for i in range(n_qubits - 1):
        c.cnot(i, i + 1)
    state = c.state()
    elapsed = time.time() - t0

    device_type = "GPU" if gpu_devices else "CPU"
    print(f"✓ Performance test completed")
    print(f"  Device: {device_type}")
    print(f"  Time for n={n_qubits} circuit: {elapsed:.4f}s")

    if gpu_devices:
        print(f"  Expected GPU time: ~0.001-0.01s")
        if elapsed > 0.1:
            print("  WARNING: Slower than expected. Check GPU utilization:")
            print("    nvidia-smi  # Should show Python process using GPU")
    else:
        print(f"  Expected CPU time: ~0.01-0.1s")

except Exception as e:
    print(f"✗ Performance test failed: {e}")

print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if gpu_devices:
    print("✓ GPU acceleration is ENABLED")
    print(f"  {len(gpu_devices)} GPU(s) available")
    print("\nYou can now run TensorCircuit QFIM with GPU acceleration:")
    print("  python metrology/test_tensorcircuit_qfim.py --large")
    print("\nExpected performance:")
    print("  n=15 qubits: ~0.05s")
    print("  n=20 qubits: ~0.2s")
    print("  n=25 qubits: ~1s")
else:
    print("✗ GPU acceleration is DISABLED (using CPU)")
    print("\nTo enable GPU acceleration:")
    print("  1. Check NVIDIA drivers: nvidia-smi")
    print("  2. Install CUDA-enabled JAX:")
    print("     pip install --upgrade 'jax[cuda12]'  # For CUDA 12.x")
    print("     pip install --upgrade 'jax[cuda11]'  # For CUDA 11.x")
    print("  3. Re-run this script: python metrology/verify_gpu.py")
    print("\nCurrent performance (CPU):")
    print("  n=12 qubits: ~0.7s")
    print("  n=15 qubits: ~1.2s")
    print("  n=20 qubits: Limited by memory")

print("=" * 70)
print()
